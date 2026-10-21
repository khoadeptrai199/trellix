"""
Benchmarks trellis/modules/quantized_linear_metal.py on the real sparse_structure_flow_model
(555.5M params): memory footprint and isolated forward-pass speed, quantized vs original fp32.
"""
import os, sys, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import torch
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules.quantized_linear_metal import quantize_model_linears, available

print(f"Metal available: {available()}\n")

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)
model = pipeline.models['sparse_structure_flow_model']
model.eval()

total_params = sum(p.numel() for p in model.parameters())
linear_params = sum(p.numel() for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) for p in m.parameters())
print(f"total params: {total_params/1e6:.1f}M, in nn.Linear layers: {linear_params/1e6:.1f}M ({100*linear_params/total_params:.1f}%)")

fp32_bytes = linear_params * 4
int8_bytes = linear_params * 1 + linear_params  # weights 1 byte + scale is per-row (negligible) -- approx
print(f"nn.Linear weight memory: fp32={fp32_bytes/1e6:.1f}MB, int8 (approx)={linear_params*1/1e6:.1f}MB (+ negligible per-row scales)\n")

B = 1
x = torch.randn(B, 8, 16, 16, 16, device=device)
t = torch.full((B,), 500.0, device=device)
cond = torch.randn(B, 1374, 1024, device=device)

with torch.no_grad():
    # Baseline fp32 timing
    for _ in range(3):
        model(x, t, cond)
    torch.mps.synchronize()
    N = 10
    t0 = time.time()
    for _ in range(N):
        model(x, t, cond)
    torch.mps.synchronize()
    t_fp32 = (time.time() - t0) / N
    print(f"fp32: {t_fp32*1000:.1f}ms/forward pass")

    n_quantized = quantize_model_linears(model)
    print(f"quantized {n_quantized} nn.Linear layers\n")

    out_q = model(x, t, cond)
    print(f"quantized forward pass output shape: {out_q.shape}, finite: {torch.isfinite(out_q).all().item()}")

    for _ in range(3):
        model(x, t, cond)
    torch.mps.synchronize()
    t0 = time.time()
    for _ in range(N):
        model(x, t, cond)
    torch.mps.synchronize()
    t_int8 = (time.time() - t0) / N
    print(f"int8:  {t_int8*1000:.1f}ms/forward pass")

speedup = t_fp32 / t_int8
print(f"\nspeedup: {speedup:.2f}x")
print(f"memory reduction (nn.Linear weights only): {fp32_bytes/(linear_params*1):.2f}x")
