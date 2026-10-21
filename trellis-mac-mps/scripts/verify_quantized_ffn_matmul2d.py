"""
Correctness check for trellis/modules/quantized_ffn_matmul2d.py (fused fc1 -> GELU -> fc2 via
matmul2d, zero-copy dispatch).
"""
import os, sys
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
import torch
from trellis.modules.transformer.blocks import FeedForwardNet
from trellis.modules.quantized_ffn_matmul2d import QuantizedFeedForwardNetMatmul2D, available

print(f"matmul2d available: {available()}\n")
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
if device != 'mps':
    print("no MPS device -- fast path can't be exercised, skipping")
    sys.exit(0)
all_ok = True


def check(name, channels, mlp_ratio, batch, seed=0):
    global all_ok
    torch.manual_seed(seed)
    ffn = FeedForwardNet(channels, mlp_ratio=mlp_ratio)
    with torch.no_grad():
        for lin in (ffn.mlp[0], ffn.mlp[2]):
            lin.weight.normal_(0, 0.02)
            lin.bias.normal_(0, 0.02)

    x_cpu = torch.randn(batch, channels) * 0.5
    ref = ffn(x_cpu)  # fp32 reference on CPU

    qffn = QuantizedFeedForwardNetMatmul2D.from_float(ffn).to(device)
    out = qffn(x_cpu.to(device)).to('cpu')

    err = (out - ref).abs()
    rel_error = (err / (ref.abs() + 1e-6)).median().item()
    # int8 quantization on BOTH linears plus fp16 matmul2d accumulation -- real, compounding error,
    # looser bound than a single quantized linear.
    ok = err.max().item() < 0.15 and rel_error < 0.25
    print(f"[{name}] channels={channels} hidden={ffn.mlp[0].out_features} batch={batch}")
    print(f"[{name}] max abs error: {err.max().item():.4f}, median rel error: {rel_error:.4f} (PASS: {ok})\n")
    all_ok = all_ok and ok


check("small", channels=64, mlp_ratio=4.0, batch=4)
check("mlp-shape", channels=1024, mlp_ratio=4.0, batch=16)
check("odd-batch", channels=1024, mlp_ratio=4.0, batch=17)  # not a multiple of tile size
check("wide-mlp", channels=1024, mlp_ratio=4.0, batch=4096)

print("=== " + ("ALL FUSED FFN CHECKS PASSED" if all_ok else "SOME CHECKS FAILED") + " ===")
sys.exit(0 if all_ok else 1)
