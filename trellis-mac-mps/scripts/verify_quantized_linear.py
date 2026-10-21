"""
Correctness check for trellis/modules/quantized_linear_metal.py.

Two separate things being checked, deliberately kept apart:
1. Does the Metal kernel correctly implement int8-times-scale dequantized matmul? Checked by
   comparing it against QuantizedLinearMetal's own PyTorch fallback path (same quantized weights,
   different execution engine) -- these should match near-exactly (float32 accumulation order
   differences only), since both are computing the identical dequantized matmul.
2. Is int8 weight-only quantization itself a reasonable approximation of the original fp32
   nn.Linear? Checked by comparing against the real fp32 layer -- this WILL show real error (that's
   the nature of quantization), the question is whether it's small enough to be usable, not whether
   it's zero.
"""
import sys
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
import torch
import torch.nn as nn
from trellis.modules.quantized_linear_metal import QuantizedLinearMetal, available, _init, _quantized_matmul_metal

print(f"Metal available: {available()}\n")
all_ok = True


def check(name, in_features, out_features, batch, has_bias=True, seed=0):
    global all_ok
    torch.manual_seed(seed)
    linear = nn.Linear(in_features, out_features, bias=has_bias)
    # Realistic-ish weight distribution (small, roughly normal, like a trained model's weights --
    # not the default init's larger scale) so quantization error is representative.
    with torch.no_grad():
        linear.weight.normal_(0, 0.02)
        if has_bias:
            linear.bias.normal_(0, 0.02)

    x = torch.randn(batch, in_features) * 0.5

    ref = linear(x)
    qlin = QuantizedLinearMetal.from_float(linear)

    # Force the PyTorch dequant fallback explicitly (bypass Metal) for the kernel-vs-fallback check.
    w_dequant = (qlin.weight_int8.to(torch.float32) * qlin.scale[:, None])
    fallback_out = x @ w_dequant.T
    if qlin.bias is not None:
        fallback_out = fallback_out + qlin.bias

    metal_out = qlin(x)  # uses Metal automatically if available

    kernel_diff = (metal_out - fallback_out).abs().max().item()
    quant_error = (metal_out - ref).abs()
    rel_error = (quant_error / (ref.abs() + 1e-6)).median().item()

    kernel_ok = kernel_diff < 1e-3  # should be tiny -- same computation, different engine
    quant_ok = quant_error.max().item() < 0.05 and rel_error < 0.15  # real but bounded quantization error

    print(f"[{name}] in={in_features} out={out_features} batch={batch}")
    print(f"[{name}] kernel-vs-fallback max diff: {kernel_diff:.6f} (PASS: {kernel_ok})")
    print(f"[{name}] quantized-vs-fp32 max abs error: {quant_error.max().item():.4f}, median rel error: {rel_error:.4f} (PASS: {quant_ok})")
    print(f"[{name}] PASS: {kernel_ok and quant_ok}\n")
    all_ok = all_ok and kernel_ok and quant_ok


check("small", in_features=64, out_features=64, batch=4)
check("mlp-shape", in_features=1024, out_features=4096, batch=16)  # matches this model's mlp_ratio=4 expansion
check("attn-shape", in_features=1024, out_features=3072, batch=4096)  # matches to_qkv at seq_len=16^3
check("no-bias", in_features=256, out_features=512, batch=8, has_bias=False)

print("=== " + ("ALL QUANTIZED LINEAR CHECKS PASSED" if all_ok else "SOME CHECKS FAILED") + " ===")
sys.exit(0 if all_ok else 1)
