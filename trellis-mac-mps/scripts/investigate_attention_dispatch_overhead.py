"""
Investigates the "leading unconfirmed hypothesis" flagged in APPLE_SILICON.md's attention section:
that per-Linear amax()/rescale/half-cast dispatch overhead (not absorbed by torch.compile) explains
why quantized attention is ~14.5% slower than native fp16 attention on its own true (bypass-
measured) contribution. Real shapes only (per this project's own hard-won lesson from the FFN
cache-gap investigation): self-attn num_rows=4096/channels=1024/num_heads=16/head_dim=64, cross-attn
adds Lkv=1374 -- all previously captured directly from the real model, not guessed.

Method: isolate self-attention and cross-attention each on their own (not the whole model), compare
(a) the real native fp16 MultiHeadAttention module (the actual thing running in the real model,
verbatim) against (b) QuantizedMultiHeadAttentionMatmul2D, both compiled where applicable, at 240
calls each cycling a few distinct weight sets (cache effect already refuted -- doesn't matter how
many). Then breaks the quantized eager (uncompiled) chain into its individual sub-steps with
torch.mps.synchronize() after each, to see directly where time is actually going -- rather than
guessing further.
"""
import os, sys, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import torch
import torch.nn.functional as F
from trellis.modules.attention.modules import MultiHeadAttention
from trellis.modules.attention.full_attn import scaled_dot_product_attention
import trellis.modules.quantized_attention_matmul2d as qattn

print(f"matmul2d available: {qattn.available()}\n")
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

channels, num_heads, head_dim = 1024, 16, 64
B, L, Lkv = 1, 4096, 1374
N_CALLS = 60
N_DISTINCT = 4


def bench(fn, n_calls, warmup=6):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.time()
    for _ in range(n_calls):
        fn()
    torch.mps.synchronize()
    return (time.time() - t0) / n_calls * 1000


# ---------- self-attention ----------
print("=" * 70)
print("SELF-ATTENTION")
print("=" * 70)

x_self = torch.randn(B, L, channels, device=device, dtype=torch.float16)

native_self_blocks = []
quant_self_blocks = []
for i in range(N_DISTINCT):
    torch.manual_seed(i)
    m = MultiHeadAttention(channels, num_heads=num_heads, type="self", qk_rms_norm=True)
    m = m.to(device).half().eval()
    native_self_blocks.append(m)
    quant_self_blocks.append(qattn.QuantizedMultiHeadAttentionMatmul2D.from_float(m).to(device).eval())

with torch.no_grad():
    idx = {'i': 0}
    def call_native_self():
        m = native_self_blocks[idx['i'] % N_DISTINCT]; idx['i'] += 1
        return m(x_self)
    ms_native_self = bench(call_native_self, N_CALLS)
    print(f"native fp16 MultiHeadAttention (self):        {ms_native_self:.3f} ms/call")

    idx = {'i': 0}
    def call_quant_self():
        m = quant_self_blocks[idx['i'] % N_DISTINCT]; idx['i'] += 1
        return m(x_self)
    ms_quant_self = bench(call_quant_self, N_CALLS)
    print(f"quantized matmul2d (self), compiled:          {ms_quant_self:.3f} ms/call")
    print(f"ratio: {ms_quant_self/ms_native_self:.3f}x\n")

# ---------- cross-attention ----------
print("=" * 70)
print("CROSS-ATTENTION")
print("=" * 70)

x_cross = torch.randn(B, L, channels, device=device, dtype=torch.float16)
ctx_cross = torch.randn(B, Lkv, channels, device=device, dtype=torch.float16)

native_cross_blocks = []
quant_cross_blocks = []
for i in range(N_DISTINCT):
    torch.manual_seed(100 + i)
    m = MultiHeadAttention(channels, num_heads=num_heads, ctx_channels=channels, type="cross", qk_rms_norm=False)
    m = m.to(device).half().eval()
    native_cross_blocks.append(m)
    quant_cross_blocks.append(qattn.QuantizedMultiHeadAttentionMatmul2D.from_float(m).to(device).eval())

with torch.no_grad():
    idx = {'i': 0}
    def call_native_cross():
        m = native_cross_blocks[idx['i'] % N_DISTINCT]; idx['i'] += 1
        return m(x_cross, ctx_cross)
    ms_native_cross = bench(call_native_cross, N_CALLS)
    print(f"native fp16 MultiHeadAttention (cross):       {ms_native_cross:.3f} ms/call")

    idx = {'i': 0}
    def call_quant_cross():
        m = quant_cross_blocks[idx['i'] % N_DISTINCT]; idx['i'] += 1
        return m(x_cross, ctx_cross)
    ms_quant_cross = bench(call_quant_cross, N_CALLS)
    print(f"quantized matmul2d (cross), compiled:         {ms_quant_cross:.3f} ms/call")
    print(f"ratio: {ms_quant_cross/ms_native_cross:.3f}x\n")

# ---------- sub-step breakdown, self-attention, eager (uncompiled) ----------
print("=" * 70)
print("SELF-ATTENTION SUB-STEP BREAKDOWN (eager, uncompiled, one weight set)")
print("=" * 70)

mha = native_self_blocks[0]
qmha = quant_self_blocks[0]
r = qmha._resident_weights(device)
w_qkv_int8, scale_qkv, bias_qkv, w_out_int8, scale_out, bias_out, q_gamma, k_gamma = r

with torch.no_grad():
    x2d = x_self.reshape(-1, channels)
    num_rows = x2d.shape[0]

    def step_rescale1():
        global x_half, x_scale
        x_amax = x2d.abs().amax().clamp(min=1e-8)
        x_scale = x_amax / 256.0
        x_half = (x2d.contiguous() / x_scale).to(torch.float16)
    ms = bench(step_rescale1, N_CALLS)
    print(f"  amax + rescale + half-cast (x):       {ms:.4f} ms")

    def step_mm1():
        global qkv2d
        qkv2d = torch.ops.trellis.matmul2d_linear_quantized(x_half, w_qkv_int8, num_rows, channels, channels * 3)
    ms = bench(step_mm1, N_CALLS)
    print(f"  matmul2d call #1 (to_qkv):             {ms:.4f} ms")

    def step_scalebias1():
        global qkv2d_sb
        qkv2d_sb = qkv2d * (scale_qkv * x_scale) + bias_qkv
    ms = bench(step_scalebias1, N_CALLS)
    print(f"  scale + bias (qkv):                    {ms:.4f} ms")

    def step_reshape():
        global q, k, v
        qkv = qkv2d_sb.reshape(B, L, 3, num_heads, head_dim)
        q, k, v = qkv.unbind(dim=2)
    ms = bench(step_reshape, N_CALLS)
    print(f"  reshape + unbind:                      {ms:.4f} ms")

    def step_rmsnorm():
        global qn, kn
        rms_scale = head_dim ** 0.5
        qn = (F.normalize(q.float(), dim=-1) * q_gamma * rms_scale).to(torch.float32)
        kn = (F.normalize(k.float(), dim=-1) * k_gamma * rms_scale).to(torch.float32)
    ms = bench(step_rmsnorm, N_CALLS)
    print(f"  RMSNorm (q, k):                        {ms:.4f} ms")

    def step_sdpa():
        global h
        h = scaled_dot_product_attention(qn, kn, v)
    ms = bench(step_sdpa, N_CALLS)
    print(f"  scaled_dot_product_attention:          {ms:.4f} ms")

    def step_rescale2():
        global h_half, h_scale
        h2d = h.reshape(num_rows, channels)
        h_amax = h2d.abs().amax().clamp(min=1e-8)
        h_scale = h_amax / 256.0
        h_half = (h2d.contiguous() / h_scale).to(torch.float16)
    ms = bench(step_rescale2, N_CALLS)
    print(f"  amax + rescale + half-cast (h):        {ms:.4f} ms")

    def step_mm2():
        global out2d
        out2d = torch.ops.trellis.matmul2d_linear_quantized(h_half, w_out_int8, num_rows, channels, channels)
    ms = bench(step_mm2, N_CALLS)
    print(f"  matmul2d call #2 (to_out):             {ms:.4f} ms")

    def step_scalebias2():
        global out2d_sb
        out2d_sb = out2d * (scale_out * h_scale) + bias_out
    ms = bench(step_scalebias2, N_CALLS)
    print(f"  scale + bias (out):                    {ms:.4f} ms")

# ---------- native self-attention sub-step breakdown, for direct comparison ----------
print()
print("=" * 70)
print("NATIVE SELF-ATTENTION SUB-STEP BREAKDOWN (eager, for comparison)")
print("=" * 70)
with torch.no_grad():
    def step_native_qkv():
        global qkv_n
        qkv_n = mha.to_qkv(x_self)
    ms = bench(step_native_qkv, N_CALLS)
    print(f"  to_qkv (native fp16 Linear):            {ms:.4f} ms")

    def step_native_reshape():
        global qn2, kn2, vn2
        qkv = qkv_n.reshape(B, L, 3, num_heads, head_dim)
        q2, k2, vn2 = qkv.unbind(dim=2)
        qn2 = mha.q_rms_norm(q2)
        kn2 = mha.k_rms_norm(k2)
    ms = bench(step_native_reshape, N_CALLS)
    print(f"  reshape/unbind + RMSNorm:               {ms:.4f} ms")

    def step_native_sdpa():
        global hn
        hn = scaled_dot_product_attention(qn2, kn2, vn2)
    ms = bench(step_native_sdpa, N_CALLS)
    print(f"  scaled_dot_product_attention:           {ms:.4f} ms")

    def step_native_out():
        global outn
        outn = mha.to_out(hn.reshape(B, L, -1))
    ms = bench(step_native_out, N_CALLS)
    print(f"  to_out (native fp16 Linear):             {ms:.4f} ms")
