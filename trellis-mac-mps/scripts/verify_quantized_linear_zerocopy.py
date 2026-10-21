"""
Correctness check for trellis/modules/quantized_linear_zerocopy.py (torch.mps.compile_shader,
zero-CPU-round-trip dispatch of the same v5 kernel as quantized_linear_metal.py).

Same two-part check as verify_quantized_linear.py:
1. Kernel-vs-fallback: does the Metal kernel correctly implement int8-times-scale dequantized
   matmul? Compared against this module's own PyTorch dequant fallback.
2. Quantized-vs-fp32: is int8 weight-only quantization a reasonable approximation of the original
   layer? Real but bounded error expected.

Unlike verify_quantized_linear.py, tensors are moved to 'mps' explicitly -- the zero-copy fast
path only activates when weights and activations are both already resident there (see this
module's forward()); on CPU it silently uses the same fallback path being compared against, which
would make check #1 vacuous.
"""
import sys
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
import torch
import torch.nn as nn
from trellis.modules.quantized_linear_zerocopy import QuantizedLinearZeroCopy, available

print(f"compile_shader available: {available()}\n")
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
if device != 'mps':
    print("no MPS device -- fast path can't be exercised, skipping")
    sys.exit(0)
all_ok = True


def check(name, in_features, out_features, batch, has_bias=True, seed=0):
    global all_ok
    torch.manual_seed(seed)
    linear = nn.Linear(in_features, out_features, bias=has_bias)
    with torch.no_grad():
        linear.weight.normal_(0, 0.02)
        if has_bias:
            linear.bias.normal_(0, 0.02)

    x_cpu = torch.randn(batch, in_features) * 0.5
    ref = linear(x_cpu)  # fp32 reference, computed on CPU

    qlin = QuantizedLinearZeroCopy.from_float(linear).to(device)
    x = x_cpu.to(device)

    w_dequant = (qlin.weight_int8.to(torch.float32) * qlin.scale[:, None])
    fallback_out = (x.to('cpu') @ w_dequant.to('cpu').T)
    if qlin.bias is not None:
        fallback_out = fallback_out + qlin.bias.to('cpu')

    metal_out = qlin(x).to('cpu')

    kernel_diff = (metal_out - fallback_out).abs().max().item()
    quant_error = (metal_out - ref).abs()
    rel_error = (quant_error / (ref.abs() + 1e-6)).median().item()

    kernel_ok = kernel_diff < 1e-3
    quant_ok = quant_error.max().item() < 0.05 and rel_error < 0.15

    print(f"[{name}] in={in_features} out={out_features} batch={batch}")
    print(f"[{name}] kernel-vs-fallback max diff: {kernel_diff:.6f} (PASS: {kernel_ok})")
    print(f"[{name}] quantized-vs-fp32 max abs error: {quant_error.max().item():.4f}, median rel error: {rel_error:.4f} (PASS: {quant_ok})")
    print(f"[{name}] PASS: {kernel_ok and quant_ok}\n")
    all_ok = all_ok and kernel_ok and quant_ok


check("small", in_features=64, out_features=64, batch=4)
check("mlp-shape", in_features=1024, out_features=4096, batch=16)
check("attn-shape", in_features=1024, out_features=3072, batch=4096)
check("no-bias", in_features=256, out_features=512, batch=8, has_bias=False)
check("odd-batch", in_features=1024, out_features=1024, batch=17)  # not a multiple of BM=64

print("=== " + ("ALL ZEROCOPY QUANTIZED LINEAR CHECKS PASSED" if all_ok else "SOME CHECKS FAILED") + " ===")
sys.exit(0 if all_ok else 1)
