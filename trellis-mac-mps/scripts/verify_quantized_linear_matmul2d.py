"""
Correctness check for trellis/modules/quantized_linear_matmul2d.py (the Metal 4 matmul2d-based
quantized Linear, separate from quantized_linear_metal.py's hand-rolled kernel).
"""
import sys
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
import torch
import torch.nn as nn
from trellis.modules.quantized_linear_matmul2d import QuantizedLinearMatmul2D, available

print(f"matmul2d available: {available()}\n")
all_ok = True


def check(name, in_features, out_features, batch, has_bias=True, seed=0):
    global all_ok
    torch.manual_seed(seed)
    linear = nn.Linear(in_features, out_features, bias=has_bias)
    with torch.no_grad():
        linear.weight.normal_(0, 0.02)
        if has_bias:
            linear.bias.normal_(0, 0.02)

    x = torch.randn(batch, in_features) * 0.5
    ref = linear(x)
    qlin = QuantizedLinearMatmul2D.from_float(linear)
    out = qlin(x)

    err = (out - ref).abs()
    rel_error = (err / (ref.abs() + 1e-6)).median().item()
    # fp16 accumulation (matmul2d path) adds real but small extra error on top of int8
    # quantization's own error -- looser bound than the fp32-accumulating hand kernel.
    ok = err.max().item() < 0.08 and rel_error < 0.2
    print(f"[{name}] in={in_features} out={out_features} batch={batch}")
    print(f"[{name}] max abs error: {err.max().item():.4f}, median rel error: {rel_error:.4f} (PASS: {ok})\n")
    all_ok = all_ok and ok


check("small", in_features=64, out_features=64, batch=4)
check("mlp-shape", in_features=1024, out_features=4096, batch=16)
check("attn-shape", in_features=1024, out_features=3072, batch=4096)
check("no-bias", in_features=256, out_features=512, batch=8, has_bias=False)
check("odd-batch", in_features=1024, out_features=1024, batch=17)  # not a multiple of 64 (tile M)

print("=== " + ("ALL MATMUL2D QUANTIZED LINEAR CHECKS PASSED" if all_ok else "SOME CHECKS FAILED") + " ===")
sys.exit(0 if all_ok else 1)
