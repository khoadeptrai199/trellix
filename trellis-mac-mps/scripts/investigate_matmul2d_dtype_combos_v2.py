"""
Follow-up to scripts/investigate_matmul2d_dtype_combos.py -- that script tested 3 of the ~46 operand
type combinations matmul2d's header documents (half x int8_t -> float, half x half -> float,
int8_t x int8_t -> int32); this tests three more, chosen for being genuinely distinct questions
rather than closing out the full combinatorial table (most of the other ~40 are float/uint8/int4
permutations that don't test a new mechanism):

  - bfloat x bfloat -> float:  bfloat has float32's exponent range, unlike half (whose limited
    range is exactly why this port's compute functions carry a dynamic amax()-based rescale before
    every half cast). If bfloat matches half's matmul2d speed, it's a candidate for removing that
    rescale overhead entirely -- a different question from int8 vs half.
  - bfloat x int8_t -> float:  same weight-only-quantization shape as the production kernel, but
    bfloat activation instead of half -- tests whether "a half operand is the bottleneck" (found in
    the first script) is specific to `half` or general to any narrow left operand.
  - half x half -> half:       smaller destination write (2 bytes/elem vs. 4) than every kernel
    tested so far -- tests whether destination width, not just operand types, matters.

int4b_format (4-bit weights) is NOT tested here: its bit-packing convention isn't discoverable in
the on-disk SDK headers (a Metal compiler builtin, not textually documented), and the first script's
own finding -- the LEFT (activation) operand's width is what's slow, not the weight's -- makes it an
unlikely candidate to change that mechanism even if the packing were worked out. Left as a flagged,
not a silently skipped, gap.
"""
import os, sys, time
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


def dims(out_features, num_rows):
    group_size = (128, 1)
    num_tg_x = ceil_div(out_features, 32)
    num_tg_y = ceil_div(num_rows, 64)
    return (num_tg_x * group_size[0], num_tg_y * group_size[1]), group_size


num_rows, in_features, out_features = 4096, 1024, 3072
threads, group_size = dims(out_features, num_rows)

# ---------- correctness ----------
print("=" * 70)
print("CORRECTNESS: each new combination vs. an fp32 CPU reference")
print("=" * 70)

torch.manual_seed(0)
x_cpu = torch.randn(num_rows, in_features) * 0.5
w_cpu = torch.randn(out_features, in_features) * 0.02
ref = x_cpu @ w_cpu.t()

row_absmax = w_cpu.abs().amax(dim=1).clamp(min=1e-8)
w_scale = row_absmax / 127.0
w_int8_cpu = (w_cpu / w_scale[:, None]).round().clamp(-127, 127).to(torch.int8)
w_int8 = w_int8_cpu.to(device)
w_scale_dev = w_scale.to(device)

x_bf16 = x_cpu.to(torch.bfloat16).to(device)
w_bf16 = w_cpu.to(torch.bfloat16).to(device)
x_half = x_cpu.to(torch.float16).to(device)
w_half = w_cpu.to(torch.float16).to(device)

with torch.no_grad():
    y_bf_bf = torch.empty(num_rows, out_features, device=device, dtype=torch.float32)
    lib.matmul2d_bfloat_bfloat(x_bf16, w_bf16, y_bf_bf, num_rows, in_features, out_features,
                                threads=threads, group_size=group_size)
    y_bf_bf = y_bf_bf.to('cpu')

    y_bf_i8 = torch.empty(num_rows, out_features, device=device, dtype=torch.float32)
    lib.matmul2d_bfloat_int8(x_bf16, w_int8, y_bf_i8, num_rows, in_features, out_features,
                              threads=threads, group_size=group_size)
    y_bf_i8 = (y_bf_i8 * w_scale_dev).to('cpu')

    y_hhh = torch.empty(num_rows, out_features, device=device, dtype=torch.float16)
    lib.matmul2d_half_half_half(x_half, w_half, y_hhh, num_rows, in_features, out_features,
                                 threads=threads, group_size=group_size)
    y_hhh = y_hhh.to(torch.float32).to('cpu')

for name, y in [("bfloat x bfloat -> float", y_bf_bf),
                ("bfloat x int8_t -> float", y_bf_i8),
                ("half x half -> half", y_hhh)]:
    rel = ((y - ref).abs() / (ref.abs() + 1e-6)).median().item()
    print(f"{name}: median rel err = {rel:.4f}")

# ---------- speed ----------
print()
print("=" * 70)
print(f"SPEED at real to_qkv shape (num_rows={num_rows}, in_features={in_features}, out_features={out_features})")
print("=" * 70)

N_CALLS = 60


def bench(fn):
    for _ in range(6):
        fn()
    torch.mps.synchronize()
    t0 = time.time()
    for _ in range(N_CALLS):
        fn()
    torch.mps.synchronize()
    return (time.time() - t0) / N_CALLS * 1000


def call_bf_bf():
    y = torch.empty(num_rows, out_features, device=device, dtype=torch.float32)
    lib.matmul2d_bfloat_bfloat(x_bf16, w_bf16, y, num_rows, in_features, out_features,
                                threads=threads, group_size=group_size)


def call_bf_i8():
    y = torch.empty(num_rows, out_features, device=device, dtype=torch.float32)
    lib.matmul2d_bfloat_int8(x_bf16, w_int8, y, num_rows, in_features, out_features,
                              threads=threads, group_size=group_size)


def call_hhh():
    y = torch.empty(num_rows, out_features, device=device, dtype=torch.float16)
    lib.matmul2d_half_half_half(x_half, w_half, y, num_rows, in_features, out_features,
                                 threads=threads, group_size=group_size)


def call_native_linear():
    torch.nn.functional.linear(x_half, w_half)


def call_native_linear_bf16():
    torch.nn.functional.linear(x_bf16, w_bf16)


ms_bf_bf = bench(call_bf_bf)
ms_bf_i8 = bench(call_bf_i8)
ms_hhh = bench(call_hhh)
ms_native = bench(call_native_linear)
ms_native_bf16 = bench(call_native_linear_bf16)

print(f"matmul2d bfloat x bfloat -> float : {ms_bf_bf:.4f} ms/call")
print(f"matmul2d bfloat x int8_t -> float : {ms_bf_i8:.4f} ms/call")
print(f"matmul2d half x half -> half      : {ms_hhh:.4f} ms/call")
print(f"native fp16 nn.functional.linear  : {ms_native:.4f} ms/call")
print(f"native bf16 nn.functional.linear  : {ms_native_bf16:.4f} ms/call")
print()
print(f"bfloat x bfloat vs native fp16 Linear: {ms_native/ms_bf_bf:.3f}x")
print(f"bfloat x int8_t vs native fp16 Linear: {ms_native/ms_bf_i8:.3f}x")
print(f"half x half -> half vs native fp16 Linear: {ms_native/ms_hhh:.3f}x")
print(f"(for reference, from the first script: half x int8_t -> float was 0.7992ms, "
      f"half x half -> float was 0.7952ms, int8_t x int8_t -> int32 was 0.4086ms)")
