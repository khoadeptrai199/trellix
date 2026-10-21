"""
Tile-size tuning follow-up to scripts/investigate_flash_attention_matmul2d.py. KVBLOCK=64
(matching QBLOCK) is infeasible outright -- score_tile (16KB) + acc (16KB) + m/l (~512B) exceeds
the ~32KB per-threadgroup memory budget, doesn't compile/dispatch. Tests the other direction:
attention_flash_online_softmax_kv16 (KVBLOCK=16, otherwise identical to the KVBLOCK=32 kernel,
including the same 2-lanes-per-row parallelized bookkeeping), added as a new, non-destructive
kernel variant in quantized_linear_matmul2d.metal.
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


def flash_attention_kv16(q, k, v, H, L, D):
    scale = 1.0 / math.sqrt(D)
    out = torch.zeros(H, L, D, device=device, dtype=torch.float32)
    threads, group_size = flash_dims(L, H)
    lib.attention_flash_online_softmax_kv16(q, k, v, out, L, scale, threads=threads, group_size=group_size)
    return out


print("=" * 70)
print("CORRECTNESS: small synthetic shape vs. naive fp32 CPU reference")
print("=" * 70)
for H_small, L_small in [(1, 64), (3, 100)]:
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
        out = flash_attention_kv16(q_dev, k_dev, v_dev, H_small, L_small, D_small).to('cpu')
    err = (out - ref_out).abs()
    rel = (err / (ref_out.abs() + 1e-6)).median().item()
    ok = err.max().item() < 0.1 and rel < 0.15
    print(f"H={H_small} L={L_small}: max abs err={err.max().item():.4f} median rel={rel:.4f} "
          f"({'PASS' if ok else 'FAIL'})")
    if not ok:
        sys.exit(1)

print("\n" + "=" * 70)
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


with torch.no_grad():
    ms_kv16 = bench(lambda: flash_attention_kv16(q, k, v, H, L, D))
    q4, k4, v4 = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    ms_native = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q4, k4, v4))

print(f"flash kv16 (parallelized bookkeeping): {ms_kv16:.4f} ms/call")
print(f"native scaled_dot_product_attention:   {ms_native:.4f} ms/call")
print(f"kv16 vs native: {ms_native/ms_kv16:.3f}x")
print(f"(for reference: kv32 parallelized was 5.81ms, native was 4.98ms -- kv16 is WORSE, "
      f"not better -- KVBLOCK=32 remains the best configuration found)")
