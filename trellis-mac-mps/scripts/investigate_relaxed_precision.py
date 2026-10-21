"""
Tests matmul2d_linear_quantized_relaxed (relaxed_precision=true, new kernel added in
quantized_linear_matmul2d.metal, exact copy of matmul2d_linear_quantized otherwise) against the
existing matmul2d_linear_quantized (relaxed_precision=false) kernel already in production use.
Directly grounded in Apple's own MPPTensorOpsMatMul2d.h header, which documents this flag as
letting "implementation sacrifice accuracy for performance" -- never tried until now. Checks both
correctness (vs. a CPU fp32 reference, same error-bound methodology as every other verify_*.py in
this port) and speed, at the real shapes captured earlier (num_rows=4096, channels=1024,
hidden/out_features=3072 -- the actual to_qkv shape).
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


def run_kernel(fn_name, x_half, w_int8, num_rows, in_features, out_features):
    y = torch.empty(num_rows, out_features, device=device, dtype=torch.float32)
    threads, group_size = dims(out_features, num_rows)
    getattr(lib, fn_name)(x_half, w_int8, y, num_rows, in_features, out_features,
                           threads=threads, group_size=group_size)
    return y


# ---------- correctness ----------
print("=" * 70)
print("CORRECTNESS: relaxed vs non-relaxed vs fp32 CPU reference")
print("=" * 70)


def check(name, num_rows, in_features, out_features, seed=0):
    torch.manual_seed(seed)
    x_cpu = torch.randn(num_rows, in_features) * 0.5
    w_cpu = torch.randn(out_features, in_features) * 0.02

    ref = x_cpu @ w_cpu.t()  # fp32 CPU reference, unquantized

    row_absmax = w_cpu.abs().amax(dim=1).clamp(min=1e-8)
    scale = row_absmax / 127.0
    w_int8_cpu = (w_cpu / scale[:, None]).round().clamp(-127, 127).to(torch.int8)

    x_amax = x_cpu.abs().amax().clamp(min=1e-8)
    x_scale = x_amax / 256.0
    x_half = (x_cpu / x_scale).to(torch.float16).to(device)
    w_int8 = w_int8_cpu.to(device)
    scale_dev = scale.to(device)

    with torch.no_grad():
        y_normal = run_kernel('matmul2d_linear_quantized', x_half, w_int8, num_rows, in_features, out_features)
        y_normal = (y_normal * (scale_dev * x_scale.to(device))).to('cpu')

        y_relaxed = run_kernel('matmul2d_linear_quantized_relaxed', x_half, w_int8, num_rows, in_features, out_features)
        y_relaxed = (y_relaxed * (scale_dev * x_scale.to(device))).to('cpu')

    err_normal = (y_normal - ref).abs()
    err_relaxed = (y_relaxed - ref).abs()
    rel_normal = (err_normal / (ref.abs() + 1e-6)).median().item()
    rel_relaxed = (err_relaxed / (ref.abs() + 1e-6)).median().item()
    diff_between = (y_normal - y_relaxed).abs().max().item()

    print(f"[{name}] shape=({num_rows},{in_features})x({out_features},{in_features})")
    print(f"  non-relaxed: max abs err={err_normal.max().item():.4f}, median rel err={rel_normal:.4f}")
    print(f"  relaxed:     max abs err={err_relaxed.max().item():.4f}, median rel err={rel_relaxed:.4f}")
    print(f"  max abs diff between the two kernels' outputs: {diff_between:.6f}\n")


check("small", num_rows=16, in_features=64, out_features=64)
check("real-shape (to_qkv)", num_rows=4096, in_features=1024, out_features=3072)
check("odd-batch", num_rows=17, in_features=64, out_features=64)

# ---------- speed ----------
print("=" * 70)
print("SPEED: relaxed vs non-relaxed at real to_qkv shape")
print("=" * 70)

num_rows, in_features, out_features = 4096, 1024, 3072
x_half = torch.randn(num_rows, in_features, device=device, dtype=torch.float16)
w_int8 = torch.randint(-127, 128, (out_features, in_features), dtype=torch.int8, device=device)

N_CALLS = 60


def bench(fn_name, n_calls, warmup=6):
    for _ in range(warmup):
        run_kernel(fn_name, x_half, w_int8, num_rows, in_features, out_features)
    torch.mps.synchronize()
    t0 = time.time()
    for _ in range(n_calls):
        run_kernel(fn_name, x_half, w_int8, num_rows, in_features, out_features)
    torch.mps.synchronize()
    return (time.time() - t0) / n_calls * 1000


ms_normal = bench('matmul2d_linear_quantized', N_CALLS)
ms_relaxed = bench('matmul2d_linear_quantized_relaxed', N_CALLS)
print(f"non-relaxed: {ms_normal:.4f} ms/call")
print(f"relaxed:     {ms_relaxed:.4f} ms/call")
print(f"speedup from relaxed_precision: {ms_normal/ms_relaxed:.3f}x")
