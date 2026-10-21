"""
Tests attention_qkt_batched / attention_av_batched (trellis/modules/metal_kernels/
quantized_linear_matmul2d.metal) -- the fix for the 2.8x-slower-than-bmm finding in
scripts/investigate_matmul2d_dtype_combos.py's QK^T test: instead of a Python loop calling
matmul2d once per head (16 separate dispatches), the head index lives in the dispatch grid's z
dimension (tgid.z), so all heads execute as ONE dispatchThreads call per kernel. Softmax stays a
native torch op between the two kernels (not attempting online-softmax/flash-attention tiling here
-- that's a separate, harder question, isolated on purpose).

Step 1: correctness on a SMALL synthetic shape first (per this port's own verification
methodology) -- attention_av_batched's transpose convention (no transpose, unlike every other
kernel in this file) is new and unverified until measured, not assumed correct from reading the
descriptor params.
Step 2: speed at the real self-attention shape (B=1, H=16, L=4096, D=64) against native SDPA and
against the original per-head Python-loop version, so the "does proper batching close the gap"
question has a direct, three-way comparison.
"""
import os, sys, time, math
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import torch

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
METALLIB = 'trellis/modules/metal_kernels/quantized_linear_matmul2d.metallib'
lib = torch.mps.load_metallib(METALLIB)
print("loaded metallib OK\n")


def ceil_div(n, m):
    return (n + m - 1) // m


def qkt_dims(L, H):
    group_size = (128, 1, 1)
    return (ceil_div(L, 32) * group_size[0], ceil_div(L, 64) * group_size[1], H), group_size


def av_dims(L, D, H):
    group_size = (128, 1, 1)
    return (ceil_div(D, 32) * group_size[0], ceil_div(L, 64) * group_size[1], H), group_size


def batched_attention(q, k, v, H, L, D):
    """q, k, v: [H, L, D] half tensors on device. Returns [H, L, D] float32."""
    scale = 1.0 / math.sqrt(D)
    scores = torch.empty(H, L, L, device=device, dtype=torch.float32)
    threads, group_size = qkt_dims(L, H)
    lib.attention_qkt_batched(q, k, scores, L, D, threads=threads, group_size=group_size)

    probs = torch.softmax(scores * scale, dim=-1).to(torch.float16)

    out = torch.empty(H, L, D, device=device, dtype=torch.float32)
    threads2, group_size2 = av_dims(L, D, H)
    lib.attention_av_batched(probs, v, out, L, D, threads=threads2, group_size=group_size2)
    return out


# ---------- Step 1: correctness, small shape ----------
print("=" * 70)
print("CORRECTNESS: small synthetic shape vs. naive fp32 CPU reference")
print("=" * 70)

H_small, L_small, D_small = 3, 37, 16  # deliberately not tile-aligned (64/32 tiles)
torch.manual_seed(0)
q_cpu = torch.randn(H_small, L_small, D_small) * 0.5
k_cpu = torch.randn(H_small, L_small, D_small) * 0.5
v_cpu = torch.randn(H_small, L_small, D_small) * 0.5

scale = 1.0 / math.sqrt(D_small)
ref_scores = torch.einsum('hld,hmd->hlm', q_cpu, k_cpu) * scale
ref_probs = torch.softmax(ref_scores, dim=-1)
ref_out = torch.einsum('hlm,hmd->hld', ref_probs, v_cpu)

q_dev = q_cpu.to(torch.float16).to(device)
k_dev = k_cpu.to(torch.float16).to(device)
v_dev = v_cpu.to(torch.float16).to(device)

with torch.no_grad():
    out = batched_attention(q_dev, k_dev, v_dev, H_small, L_small, D_small).to('cpu')

err = (out - ref_out).abs()
rel = (err / (ref_out.abs() + 1e-6)).median().item()
print(f"max abs err: {err.max().item():.4f}, median rel err: {rel:.4f}")
correctness_ok = err.max().item() < 0.1 and rel < 0.15
print(f"CORRECTNESS: {'PASS' if correctness_ok else 'FAIL'}\n")

if not correctness_ok:
    print("Correctness failed -- not proceeding to speed benchmark on a kernel that's wrong.")
    sys.exit(1)

# ---------- Step 2: speed at the real shape ----------
print("=" * 70)
print("SPEED at real self-attention shape (B=1, H=16, L=4096, D=64)")
print("=" * 70)

H, L, D = 16, 4096, 64
torch.manual_seed(0)
q = torch.randn(H, L, D, device=device, dtype=torch.float16)
k = torch.randn(H, L, D, device=device, dtype=torch.float16)
v = torch.randn(H, L, D, device=device, dtype=torch.float16)

N_CALLS = 30


def bench(fn):
    for _ in range(6):
        fn()
    torch.mps.synchronize()
    t0 = time.time()
    for _ in range(N_CALLS):
        fn()
    torch.mps.synchronize()
    return (time.time() - t0) / N_CALLS * 1000


def call_batched():
    return batched_attention(q, k, v, H, L, D)


def call_native_sdpa():
    q4 = q.unsqueeze(0)  # [1, H, L, D]
    k4 = k.unsqueeze(0)
    v4 = v.unsqueeze(0)
    return torch.nn.functional.scaled_dot_product_attention(q4, k4, v4)


def call_perhead_loop():
    # The original per-head Python-loop version (matmul2d_half_half), for a direct three-way
    # comparison -- reuses the same kernel from investigate_matmul2d_dtype_combos.py.
    threads, group_size = ceil_div(L, 32) * 128, ceil_div(L, 64)
    thr = (ceil_div(L, 32) * 128, ceil_div(L, 64))
    scores = torch.empty(H, L, L, device=device, dtype=torch.float32)
    for h in range(H):
        lib.matmul2d_half_half(q[h], k[h], scores[h], L, D, L, threads=thr, group_size=(128, 1))
    probs = torch.softmax(scores / math.sqrt(D), dim=-1).to(torch.float16)
    out = torch.empty(H, L, D, device=device, dtype=torch.float32)
    thr2 = (ceil_div(D, 32) * 128, ceil_div(L, 64))
    for h in range(H):
        lib.matmul2d_half_half(probs[h], v[h].contiguous(), out[h], L, L, D, threads=thr2, group_size=(128, 1))
    return out


with torch.no_grad():
    ms_batched = bench(call_batched)
    ms_native = bench(call_native_sdpa)
    ms_perhead = bench(call_perhead_loop)

print(f"batched-dispatch matmul2d attention: {ms_batched:.4f} ms/call")
print(f"native scaled_dot_product_attention: {ms_native:.4f} ms/call")
print(f"per-head Python-loop matmul2d:       {ms_perhead:.4f} ms/call")
print()
print(f"batched vs native:   {ms_native/ms_batched:.3f}x")
print(f"batched vs per-head: {ms_perhead/ms_batched:.3f}x (>1 means batching helped)")
