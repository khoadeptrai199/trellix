"""
Tests attention_flash_online_softmax (trellis/modules/metal_kernels/quantized_linear_matmul2d.metal)
-- the real online-softmax/flash-attention-style kernel, following up on
scripts/investigate_batched_attention_matmul2d.py's negative result (proper dispatch batching alone
gave almost no speedup, 1.016x, meaning dispatch count wasn't the dominant cost). This kernel keeps
the score tile and running softmax statistics in threadgroup memory instead of writing the full
L x L score matrix to device memory, testing whether THAT was the real bottleneck.

Step 1: correctness on a small synthetic shape first, MANDATORY here more than anywhere else in
this port -- this kernel has real synchronization complexity (threadgroup_barrier placement,
online-softmax numerical bookkeeping, multiply_accumulate mode never used elsewhere in this port)
that's easy to get subtly wrong. A pass here is necessary but not sufficient (this port's own
methodology); a fail here means stopping before trusting any speed number.
Step 2: only if correctness passes, speed at the real shape (H=16, L=4096, D=64) against native
SDPA and against the batched (non-flash) version this replaces.
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


def flash_dims(L, H):
    group_size = (128, 1, 1)
    return (group_size[0], ceil_div(L, 64), H), group_size


def flash_attention(q, k, v, H, L, D):
    """q, k, v: [H, L, D] half tensors on device. Returns [H, L, D] float32."""
    scale = 1.0 / math.sqrt(D)
    out = torch.zeros(H, L, D, device=device, dtype=torch.float32)
    threads, group_size = flash_dims(L, H)
    lib.attention_flash_online_softmax(q, k, v, out, L, scale, threads=threads, group_size=group_size)
    return out


# ---------- Step 1: correctness, small shape ----------
print("=" * 70)
print("CORRECTNESS: small synthetic shape vs. naive fp32 CPU reference")
print("=" * 70)

# D is hardcoded to 64 in the kernel (this model's real head_dim) -- can't vary it in this test.
# L deliberately not a multiple of 64 (query-block size) or 32 (kv-block size) to exercise the
# boundary-guard (global_row < L) and the loop's final partial KV block.
for H_small, L_small in [(1, 64), (2, 96), (3, 100)]:
    D_small = 64
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
        out = flash_attention(q_dev, k_dev, v_dev, H_small, L_small, D_small).to('cpu')

    err = (out - ref_out).abs()
    rel = (err / (ref_out.abs() + 1e-6)).median().item()
    ok = err.max().item() < 0.1 and rel < 0.15
    print(f"H={H_small} L={L_small}: max abs err={err.max().item():.4f}, median rel err={rel:.4f} "
          f"({'PASS' if ok else 'FAIL'})")
    if not ok:
        print("\nCorrectness FAILED -- not proceeding to speed benchmark on a wrong kernel.")
        print("(This is still a real, reportable result -- see APPLE_SILICON.md.)")
        sys.exit(1)

print("\nAll correctness checks PASSED.\n")

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


def call_flash():
    return flash_attention(q, k, v, H, L, D)


def call_native_sdpa():
    q4 = q.unsqueeze(0)
    k4 = k.unsqueeze(0)
    v4 = v.unsqueeze(0)
    return torch.nn.functional.scaled_dot_product_attention(q4, k4, v4)


with torch.no_grad():
    ms_flash = bench(call_flash)
    ms_native = bench(call_native_sdpa)

print(f"flash (online-softmax, threadgroup-resident) matmul2d attention: {ms_flash:.4f} ms/call")
print(f"native scaled_dot_product_attention:                             {ms_native:.4f} ms/call")
print()
print(f"flash vs native: {ms_native/ms_flash:.3f}x")
print(f"(for reference: batched (non-flash, full L x L in device memory) was 15.04ms, "
      f"per-head loop was 15.28ms)")
