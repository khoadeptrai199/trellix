"""
Correctness check for trellis/modules/quantized_attention_matmul2d.py: quantized MultiHeadAttention
(self-attention w/ qk_rms_norm=True, cross-attention w/o) via matmul2d, zero-copy dispatch.
"""
import os, sys
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
import torch
from trellis.modules.attention.modules import MultiHeadAttention
from trellis.modules.quantized_attention_matmul2d import QuantizedMultiHeadAttentionMatmul2D, available

print(f"matmul2d available: {available()}\n")
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
if device != 'mps':
    print("no MPS device -- fast path can't be exercised, skipping")
    sys.exit(0)
all_ok = True


def _init_linear(lin, seed_offset=0):
    with torch.no_grad():
        lin.weight.normal_(0, 0.02)
        if lin.bias is not None:
            lin.bias.normal_(0, 0.02)


def check_self(name, channels, num_heads, L, batch=1, seed=0):
    global all_ok
    torch.manual_seed(seed)
    mha = MultiHeadAttention(channels, num_heads=num_heads, type="self", qk_rms_norm=True)
    _init_linear(mha.to_qkv)
    _init_linear(mha.to_out)
    with torch.no_grad():
        mha.q_rms_norm.gamma.normal_(1.0, 0.02)
        mha.k_rms_norm.gamma.normal_(1.0, 0.02)

    x_cpu = torch.randn(batch, L, channels) * 0.5
    ref = mha(x_cpu)  # fp32 reference on CPU

    qmha = QuantizedMultiHeadAttentionMatmul2D.from_float(mha).to(device)
    out = qmha(x_cpu.to(device)).to('cpu')

    err = (out - ref).abs()
    rel_error = (err / (ref.abs() + 1e-6)).median().item()
    ok = err.max().item() < 0.2 and rel_error < 0.3
    print(f"[{name}] self-attn channels={channels} heads={num_heads} L={L} batch={batch}")
    print(f"[{name}] max abs error: {err.max().item():.4f}, median rel error: {rel_error:.4f} (PASS: {ok})\n")
    all_ok = all_ok and ok


def check_cross(name, channels, ctx_channels, num_heads, L, Lkv, batch=1, seed=0):
    global all_ok
    torch.manual_seed(seed)
    mha = MultiHeadAttention(channels, num_heads=num_heads, ctx_channels=ctx_channels, type="cross", qk_rms_norm=False)
    _init_linear(mha.to_q)
    _init_linear(mha.to_kv)
    _init_linear(mha.to_out)

    x_cpu = torch.randn(batch, L, channels) * 0.5
    ctx_cpu = torch.randn(batch, Lkv, ctx_channels) * 0.5
    ref = mha(x_cpu, ctx_cpu)  # fp32 reference on CPU

    qmha = QuantizedMultiHeadAttentionMatmul2D.from_float(mha).to(device)
    out = qmha(x_cpu.to(device), ctx_cpu.to(device)).to('cpu')

    err = (out - ref).abs()
    rel_error = (err / (ref.abs() + 1e-6)).median().item()
    ok = err.max().item() < 0.2 and rel_error < 0.3
    print(f"[{name}] cross-attn channels={channels} ctx_channels={ctx_channels} heads={num_heads} "
          f"L={L} Lkv={Lkv} batch={batch}")
    print(f"[{name}] max abs error: {err.max().item():.4f}, median rel error: {rel_error:.4f} (PASS: {ok})\n")
    all_ok = all_ok and ok


check_self("small", channels=64, num_heads=4, L=8)
check_self("real-shape", channels=1024, num_heads=16, L=512)
check_self("odd-L", channels=1024, num_heads=16, L=513)  # not a multiple of matmul2d's row tile
check_self("wide-L", channels=1024, num_heads=16, L=4096)

check_cross("small", channels=64, ctx_channels=64, num_heads=4, L=8, Lkv=6)
check_cross("real-shape", channels=1024, ctx_channels=1024, num_heads=16, L=512, Lkv=1374)
check_cross("odd-L", channels=1024, ctx_channels=1024, num_heads=16, L=513, Lkv=1374)
check_cross("diff-ctx-channels", channels=1024, ctx_channels=768, num_heads=16, L=512, Lkv=257)

# Unsupported-config guards: these should raise NotImplementedError, not silently misbehave.
guard_ok = True
try:
    QuantizedMultiHeadAttentionMatmul2D.from_float(
        MultiHeadAttention(64, num_heads=4, type="self", qk_rms_norm=False))
    guard_ok = False
    print("[guard] FAILED: self-attn without qk_rms_norm should have raised NotImplementedError")
except NotImplementedError:
    print("[guard] self-attn without qk_rms_norm correctly raised NotImplementedError")

try:
    QuantizedMultiHeadAttentionMatmul2D.from_float(
        MultiHeadAttention(64, num_heads=4, type="cross", qk_rms_norm=True))
    guard_ok = False
    print("[guard] FAILED: cross-attn with qk_rms_norm should have raised NotImplementedError")
except NotImplementedError:
    print("[guard] cross-attn with qk_rms_norm correctly raised NotImplementedError")

try:
    QuantizedMultiHeadAttentionMatmul2D.from_float(
        MultiHeadAttention(64, num_heads=4, type="self", use_rope=True))
    guard_ok = False
    print("[guard] FAILED: use_rope=True should have raised NotImplementedError")
except NotImplementedError:
    print("[guard] use_rope=True correctly raised NotImplementedError")
print()

all_ok = all_ok and guard_ok
print("=== " + ("ALL QUANTIZED ATTENTION CHECKS PASSED" if all_ok else "SOME CHECKS FAILED") + " ===")
sys.exit(0 if all_ok else 1)
