"""
Benchmarks trellis/modules/quantized_ffn_matmul2d.py on the real sparse_structure_flow_model
(555.5M params): replaces every dense FeedForwardNet (fc1->GELU->fc2) with the fused, zero-copy,
matmul2d-based quantized version and times a full forward pass against fp32. Same test shape as
every other benchmark script in this port.
"""
import os, sys, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import torch
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules.transformer.blocks import FeedForwardNet
from trellis.modules.quantized_ffn_matmul2d import quantize_model_ffns_matmul2d, available

print(f"matmul2d available: {available()}\n")

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)
model = pipeline.models['sparse_structure_flow_model']
model.eval()

n_ffn = sum(1 for m in model.modules() if isinstance(m, FeedForwardNet))
ffn_params = sum(p.numel() for m in model.modules() if isinstance(m, FeedForwardNet) for p in m.parameters())
total_params = sum(p.numel() for p in model.parameters())
print(f"total params: {total_params/1e6:.1f}M, in FeedForwardNet blocks: {ffn_params/1e6:.1f}M "
      f"({100*ffn_params/total_params:.1f}%) across {n_ffn} blocks\n")

B = 1
x = torch.randn(B, 8, 16, 16, 16, device=device)
t = torch.full((B,), 500.0, device=device)
cond = torch.randn(B, 1374, 1024, device=device)

with torch.no_grad():
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

    n_quantized = quantize_model_ffns_matmul2d(model)
    print(f"quantized {n_quantized} FeedForwardNet blocks\n")

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
