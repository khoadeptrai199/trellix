"""
A third generalization data point, extending the existing "does the framework pattern hold at a
different shape" check (scripts/benchmark_mlx_comparison.py's Llama-2-7B-like profile) to a
genuinely different model *family*, not just a different LLM config: ViT-L/14 (CLIP's vision
encoder), a standard multi-head-attention vision transformer.

Real, published ViT-L/14 config (hidden_size=1024, 16 heads, head_dim=64, matching CLIP/OpenCLIP's
actual released architecture): same channel/head dimensions as TRELLIS's own shape, but a
dramatically shorter sequence length (257 = 256 patches + 1 CLS token for 224x224 images at a
14x14 patch size, vs. TRELLIS's 4096) -- a different attention FLOP/memory regime (short-sequence,
quadratic-cost-is-tiny) that TRELLIS's own shape and the Llama-2-7B-like shape don't test.

Covers PyTorch MPS and MLX. CoreML was tried and deliberately excluded (see
scripts/benchmark_coreml_comparison.py, kept on disk but not committed/cited): its predict() API's
measured latency is dominated by data-marshalling overhead, not compute, per a size-matched
trivial-compute control -- not a fair comparison against these two frameworks' in-process,
zero-copy calls. llama.cpp/GGML is not re-run for this shape either: rebuilding
scripts/ggml_bench.cpp's from-source llama.cpp checkout is a substantial standalone step already
covered by the existing Llama-2-7B-like generalization point, and is not repeated here for a
Linear-op-and-SDPA-only third shape sample -- noted as a scoping choice, not a silent omission.

Reports both raw sample std. dev. and the 95% CI (1.96*std/sqrt(n)) used elsewhere in this
project's paper (e.g. the window-size sweep), after an earlier draft's inconsistency where one
table's prose called raw std "confidence interval" -- fixed by actually reporting both here.
"""
import time
import statistics
import math
import torch
import mlx.core as mx

print(f"MLX version: {mx.__version__}")
print(f"PyTorch MPS available: {torch.backends.mps.is_available()}\n")

torch.manual_seed(0)
mx.random.seed(0)

N_WARMUP = 15
N_TRIALS = 20


def stats(samples):
    mean = statistics.mean(samples)
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(len(samples))
    return mean, std, ci95


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


print("=== ViT-L/14 (CLIP) shape profile: hidden=1024, heads=16, head_dim=64, L=257 ===\n")

print("--- 1. Linear/matmul at the ViT-L/14 QKV-projection shape (257 x 1024 -> 3072) ---")
num_rows, in_features, out_features = 257, 1024, 3072

x_torch = torch.randn(num_rows, in_features, device='mps', dtype=torch.float16)
w_torch = torch.randn(out_features, in_features, device='mps', dtype=torch.float16)
b_torch = torch.randn(out_features, device='mps', dtype=torch.float16)
torch_samples = bench_torch(lambda x, w, b: torch.nn.functional.linear(x, w, b), x_torch, w_torch, b_torch)
t_mean, t_std, t_ci = stats(torch_samples)
print(f"  PyTorch MPS (fp16):  {t_mean*1000:.4f} ms (std {t_std*1000:.4f}, 95% CI +/-{t_ci*1000:.4f})")

x_mlx = mx.random.normal((num_rows, in_features)).astype(mx.float16)
w_mlx = mx.random.normal((out_features, in_features)).astype(mx.float16)
b_mlx = mx.random.normal((out_features,)).astype(mx.float16)
mlx_samples = bench_mlx(lambda x, w, b: x @ w.T + b, x_mlx, w_mlx, b_mlx)
m_mean, m_std, m_ci = stats(mlx_samples)
print(f"  MLX (fp16):          {m_mean*1000:.4f} ms (std {m_std*1000:.4f}, 95% CI +/-{m_ci*1000:.4f})")
print(f"  Ratio (PyTorch/MLX): {t_mean/m_mean:.3f}x  ({'MLX faster' if m_mean < t_mean else 'PyTorch faster'})")

print("\n--- 2. Self-attention SDPA at the ViT-L/14 shape (H=16, L=257, D=64) ---")
B, H, L, D = 1, 16, 257, 64

q_torch = torch.randn(B, H, L, D, device='mps', dtype=torch.float16)
k_torch = torch.randn(B, H, L, D, device='mps', dtype=torch.float16)
v_torch = torch.randn(B, H, L, D, device='mps', dtype=torch.float16)
torch_samples = bench_torch(torch.nn.functional.scaled_dot_product_attention, q_torch, k_torch, v_torch)
t_mean, t_std, t_ci = stats(torch_samples)
print(f"  PyTorch MPS native SDPA (fp16): {t_mean*1000:.4f} ms (std {t_std*1000:.4f}, 95% CI +/-{t_ci*1000:.4f})")

q_mlx = mx.random.normal((B, H, L, D)).astype(mx.float16)
k_mlx = mx.random.normal((B, H, L, D)).astype(mx.float16)
v_mlx = mx.random.normal((B, H, L, D)).astype(mx.float16)
scale = 1.0 / (D ** 0.5)
mlx_samples = bench_mlx(lambda q, k, v: mx.fast.scaled_dot_product_attention(q, k, v, scale=scale), q_mlx, k_mlx, v_mlx)
m_mean, m_std, m_ci = stats(mlx_samples)
print(f"  MLX fast.scaled_dot_product_attention (fp16): {m_mean*1000:.4f} ms (std {m_std*1000:.4f}, 95% CI +/-{m_ci*1000:.4f})")
print(f"  Ratio (PyTorch/MLX): {t_mean/m_mean:.3f}x  ({'MLX faster' if m_mean < t_mean else 'PyTorch faster'})")
print(f"  MLX coefficient of variation: {100*m_std/m_mean:.1f}% (flagged in the paper as anomalously")
print(f"  high relative to this project's other measurements -- see Discussion.)")
