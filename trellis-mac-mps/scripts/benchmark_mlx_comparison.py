"""
Direct MLX baseline, closing a gap flagged in this project's own Limitations (Part I, and again
in the paper): "we compare only against PyTorch's native MPS backend... though we do use MLX's
published kernel source as qualitative corroborating evidence." This upgrades that from
qualitative (reading sdpa_vector.h) to quantitative: real, timed MLX operations at the exact same
shapes already used throughout this paper's kernel-level benchmarks, run back-to-back against
PyTorch MPS on the same machine (Apple M5 Max) in the same process.

Two comparisons, matching the paper's own reference shapes:
  1. Linear/matmul at the real to_qkv shape (num_rows=4096, in_features=1024, out_features=3072)
     -- Section 4.3's reference native-fp16-Linear baseline (0.4123ms).
  2. Self-attention SDPA at the real shape (H=16, L=4096, D=64) -- Section 4.6's reference native
     SDPA baseline (4.98ms).

MLX and PyTorch don't share tensors/memory, so this can't literally reuse the pipeline's own
tensors -- both frameworks get independently-constructed random tensors at the same shape/dtype,
which is the correct comparison for "does the native kernel/dispatch path perform differently,"
not a correctness comparison (already established as out of scope; this is a speed-only baseline).
"""
import time
import torch
import mlx.core as mx
import mlx.nn as nn

print(f"MLX version: {mx.__version__}")
print(f"PyTorch MPS available: {torch.backends.mps.is_available()}\n")

torch.manual_seed(0)
mx.random.seed(0)

N_WARMUP = 15
N_TRIALS = 20


def bench_torch(fn, *args):
    for _ in range(N_WARMUP):
        fn(*args)
    torch.mps.synchronize()
    samples = []
    for _ in range(N_TRIALS):
        t0 = time.time()
        fn(*args)
        torch.mps.synchronize()
        samples.append(time.time() - t0)
    return samples


def bench_mlx(fn, *args):
    for _ in range(N_WARMUP):
        mx.eval(fn(*args))
    samples = []
    for _ in range(N_TRIALS):
        t0 = time.time()
        mx.eval(fn(*args))
        samples.append(time.time() - t0)
    return samples


def stats(samples):
    import statistics
    mean = statistics.mean(samples)
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return mean, std


print("=== 1. Linear/matmul at the real to_qkv shape (4096 x 1024 -> 3072) ===")
num_rows, in_features, out_features = 4096, 1024, 3072

x_torch = torch.randn(num_rows, in_features, device='mps', dtype=torch.float16)
w_torch = torch.randn(out_features, in_features, device='mps', dtype=torch.float16)
b_torch = torch.randn(out_features, device='mps', dtype=torch.float16)


def torch_linear(x, w, b):
    return torch.nn.functional.linear(x, w, b)


torch_samples = bench_torch(torch_linear, x_torch, w_torch, b_torch)
t_mean, t_std = stats(torch_samples)
print(f"  PyTorch MPS (fp16):  {t_mean*1000:.4f} +/- {t_std*1000:.4f} ms")

x_mlx = mx.random.normal((num_rows, in_features)).astype(mx.float16)
w_mlx = mx.random.normal((out_features, in_features)).astype(mx.float16)
b_mlx = mx.random.normal((out_features,)).astype(mx.float16)


def mlx_linear(x, w, b):
    return x @ w.T + b


mlx_samples = bench_mlx(mlx_linear, x_mlx, w_mlx, b_mlx)
m_mean, m_std = stats(mlx_samples)
print(f"  MLX (fp16):          {m_mean*1000:.4f} +/- {m_std*1000:.4f} ms")
print(f"  Ratio (PyTorch/MLX): {t_mean/m_mean:.3f}x  ({'MLX faster' if m_mean < t_mean else 'PyTorch faster'})")

print("\n=== 2. Self-attention SDPA at the real shape (B=1, H=16, L=4096, D=64) ===")
B, H, L, D = 1, 16, 4096, 64

q_torch = torch.randn(B, H, L, D, device='mps', dtype=torch.float16)
k_torch = torch.randn(B, H, L, D, device='mps', dtype=torch.float16)
v_torch = torch.randn(B, H, L, D, device='mps', dtype=torch.float16)


def torch_sdpa(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


torch_samples = bench_torch(torch_sdpa, q_torch, k_torch, v_torch)
t_mean, t_std = stats(torch_samples)
print(f"  PyTorch MPS native SDPA (fp16): {t_mean*1000:.4f} +/- {t_std*1000:.4f} ms")

q_mlx = mx.random.normal((B, H, L, D)).astype(mx.float16)
k_mlx = mx.random.normal((B, H, L, D)).astype(mx.float16)
v_mlx = mx.random.normal((B, H, L, D)).astype(mx.float16)
scale = 1.0 / (D ** 0.5)


def mlx_sdpa(q, k, v):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)


mlx_samples = bench_mlx(mlx_sdpa, q_mlx, k_mlx, v_mlx)
m_mean, m_std = stats(mlx_samples)
print(f"  MLX fast.scaled_dot_product_attention (fp16): {m_mean*1000:.4f} +/- {m_std*1000:.4f} ms")
print(f"  Ratio (PyTorch/MLX): {t_mean/m_mean:.3f}x  ({'MLX faster' if m_mean < t_mean else 'PyTorch faster'})")
