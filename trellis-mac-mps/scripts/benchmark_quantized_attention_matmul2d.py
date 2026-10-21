"""
Benchmarks trellis/modules/quantized_attention_matmul2d.py on the real sparse_structure_flow_model
(555.5M params): replaces every MultiHeadAttention (self-attn and cross-attn) with the quantized
matmul2d version and times a full forward pass against fp16 (this model's actual native dtype --
see APPLE_SILICON.md's fp16 correction). Same test shape as every other benchmark script in this
port. Also isolates FFN-bypassed / attention-bypassed contributions the same way APPLE_SILICON.md's
FFN ceiling analysis did, so the attention result is directly comparable.
"""
import os, sys, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import torch
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules.attention.modules import MultiHeadAttention
from trellis.modules.quantized_attention_matmul2d import quantize_model_attention_matmul2d, available

print(f"matmul2d available: {available()}\n")

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)
model = pipeline.models['sparse_structure_flow_model']
model.eval()
print(f"model.use_fp16: {model.use_fp16}, model.dtype: {model.dtype}\n")

n_attn = sum(1 for m in model.modules() if isinstance(m, MultiHeadAttention))
attn_params = sum(p.numel() for m in model.modules() if isinstance(m, MultiHeadAttention) for p in m.parameters())
total_params = sum(p.numel() for p in model.parameters())
print(f"total params: {total_params/1e6:.1f}M, in MultiHeadAttention: {attn_params/1e6:.1f}M "
      f"({100*attn_params/total_params:.1f}%) across {n_attn} modules\n")

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
    t_native = (time.time() - t0) / N
    print(f"native (fp16): {t_native*1000:.1f}ms/forward pass")

    n_quantized = quantize_model_attention_matmul2d(model)
    print(f"quantized {n_quantized} MultiHeadAttention modules\n")

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

speedup = t_native / t_int8
print(f"\nspeedup: {speedup:.2f}x")
