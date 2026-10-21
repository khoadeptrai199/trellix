"""
Tests whether matmul2d's OPERAND TYPE COMBINATION (not just relaxed_precision, already tested null
in scripts/investigate_relaxed_precision.py) matters for speed on this machine (Apple M5 Max).
matmul2d supports many combinations (see the table at the top of MPPTensorOpsMatMul2d.h); this
compares:
  - matmul2d_linear_quantized:  half x int8_t -> float   (current production kernel, weight-only
                                                           quantization, activation stays half)
  - matmul2d_int8_int8:         int8_t x int8_t -> int32 (fully quantized, no int8->half widen)
  - matmul2d_half_half:         half x half -> float      (proxy for "what matmul2d looks like at
                                                           native nn.Linear's own dtypes")
All three test kernels are in trellis/modules/metal_kernels/quantized_linear_matmul2d.metal,
investigation-only, not wired into any production module. Same real to_qkv shape as every other
investigation in this port: num_rows=4096, in_features=1024, out_features=3072.
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

# ---------- correctness ----------
print("=" * 70)
print("CORRECTNESS: each combination vs. an fp32 CPU reference")
print("=" * 70)

torch.manual_seed(0)
x_cpu = torch.randn(num_rows, in_features) * 0.5
w_cpu = torch.randn(out_features, in_features) * 0.02
ref = x_cpu @ w_cpu.t()

# --- half x int8_t -> float (existing production kernel) ---
row_absmax = w_cpu.abs().amax(dim=1).clamp(min=1e-8)
w_scale = row_absmax / 127.0
w_int8_cpu = (w_cpu / w_scale[:, None]).round().clamp(-127, 127).to(torch.int8)
x_amax = x_cpu.abs().amax().clamp(min=1e-8)
x_scale = x_amax / 256.0
x_half = (x_cpu / x_scale).to(torch.float16).to(device)
w_int8 = w_int8_cpu.to(device)
w_scale_dev = w_scale.to(device)

with torch.no_grad():
    y = torch.empty(num_rows, out_features, device=device, dtype=torch.float32)
    threads, group_size = dims(out_features, num_rows)
    lib.matmul2d_linear_quantized(x_half, w_int8, y, num_rows, in_features, out_features,
                                   threads=threads, group_size=group_size)
    y = (y * (w_scale_dev * x_scale.to(device))).to('cpu')
rel = ((y - ref).abs() / (ref.abs() + 1e-6)).median().item()
print(f"half x int8_t -> float:  median rel err = {rel:.4f}")

# --- half x half -> float (unquantized, proxy for native dtypes through matmul2d) ---
x_half_full = x_cpu.to(torch.float16).to(device)
w_half_full = w_cpu.to(torch.float16).to(device)
with torch.no_grad():
    y2 = torch.empty(num_rows, out_features, device=device, dtype=torch.float32)
    lib.matmul2d_half_half(x_half_full, w_half_full, y2, num_rows, in_features, out_features,
                            threads=threads, group_size=group_size)
    y2 = y2.to('cpu')
rel2 = ((y2 - ref).abs() / (ref.abs() + 1e-6)).median().item()
print(f"half x half -> float:    median rel err = {rel2:.4f}")

# --- int8_t x int8_t -> int32 (fully quantized: both activation and weight) ---
x_row_absmax = x_cpu.abs().amax(dim=1).clamp(min=1e-8)  # per-row activation scale
xa_scale = x_row_absmax / 127.0
x_int8_cpu = (x_cpu / xa_scale[:, None]).round().clamp(-127, 127).to(torch.int8)
x_int8 = x_int8_cpu.to(device)
xa_scale_dev = xa_scale.to(device)
with torch.no_grad():
    y3 = torch.empty(num_rows, out_features, device=device, dtype=torch.int32)
    lib.matmul2d_int8_int8(x_int8, w_int8, y3, num_rows, in_features, out_features,
                            threads=threads, group_size=group_size)
    y3 = (y3.to(torch.float32) * (w_scale_dev[None, :] * xa_scale_dev[:, None])).to('cpu')
rel3 = ((y3 - ref).abs() / (ref.abs() + 1e-6)).median().item()
print(f"int8_t x int8_t -> int32: median rel err = {rel3:.4f}  (activation ALSO quantized -- "
      f"more error expected, this is the real accuracy cost of full W8A8)\n")

# ---------- speed ----------
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


def call_half_int8():
    y = torch.empty(num_rows, out_features, device=device, dtype=torch.float32)
    lib.matmul2d_linear_quantized(x_half, w_int8, y, num_rows, in_features, out_features,
                                   threads=threads, group_size=group_size)


def call_half_half():
    y = torch.empty(num_rows, out_features, device=device, dtype=torch.float32)
    lib.matmul2d_half_half(x_half_full, w_half_full, y, num_rows, in_features, out_features,
                            threads=threads, group_size=group_size)


def call_int8_int8():
    y = torch.empty(num_rows, out_features, device=device, dtype=torch.int32)
    lib.matmul2d_int8_int8(x_int8, w_int8, y, num_rows, in_features, out_features,
                            threads=threads, group_size=group_size)


def call_native_linear():
    lin_w = w_half_full  # [out_features, in_features], already fp16 on device
    torch.nn.functional.linear(x_half_full, lin_w)


ms_half_int8 = bench(call_half_int8)
ms_half_half = bench(call_half_half)
ms_int8_int8 = bench(call_int8_int8)
ms_native = bench(call_native_linear)

print(f"matmul2d half x int8_t -> float   (current production): {ms_half_int8:.4f} ms/call")
print(f"matmul2d half x half   -> float                        : {ms_half_half:.4f} ms/call")
print(f"matmul2d int8_t x int8_t -> int32 (full W8A8)           : {ms_int8_int8:.4f} ms/call")
print(f"native fp16 nn.functional.linear                        : {ms_native:.4f} ms/call")
print()
print(f"int8_t x int8_t vs half x int8_t speedup: {ms_half_int8/ms_int8_int8:.3f}x")
print(f"int8_t x int8_t vs native fp16 Linear speedup:           {ms_native/ms_int8_int8:.3f}x")
print(f"matmul2d half x half vs native fp16 Linear:              {ms_native/ms_half_half:.3f}x")
