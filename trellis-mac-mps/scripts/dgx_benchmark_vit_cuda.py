"""
CUDA-only (DGX pod, H200). Cross-hardware companion to
scripts/benchmark_architecture_generalization_vit.py (Apple M5 Max, MPS + MLX): same ViT-L/14
(CLIP) reference shape -- hidden 1024, 16 heads, head_dim 64, L=257 (256 patches + 1 CLS token,
real published config) -- benchmarked via PyTorch's own native CUDA path on an NVIDIA H200.

Purpose: the M5 Max result found MLX's long-sequence attention advantage over PyTorch MPS
reverses at this short-sequence shape (PyTorch faster). This script asks whether that reversal is
an Apple-Silicon/MPS-backend-specific phenomenon or whether short sequences generally erode
whatever advantage a non-default attention path has -- by checking PyTorch's own alternate CUDA
SDPA backends (FLASH_ATTENTION, EFFICIENT_ATTENTION, CUDNN_ATTENTION, MATH) against each other at
the identical shape, on a completely different vendor's hardware. MLX itself does not run on CUDA,
so it cannot be included directly here; this is a same-shape, same-op, different-hardware/
different-backend-selection check, not a literal MLX rerun.

Kept as a separate file per this project's own platform-separation convention: MPS/MLX (Mac)
scripts and CUDA (DGX) scripts never share a file or branch on device.
"""
import time
import statistics
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend

device = 'cuda'
assert torch.cuda.is_available(), "This script requires a CUDA device (run on the DGX pod)."
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}\n")

torch.manual_seed(0)

N_WARMUP = 15
N_TRIALS = 20


def stats(samples):
    mean = statistics.mean(samples)
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return mean, std


def bench(fn, *args):
    for _ in range(N_WARMUP):
        fn(*args)
    torch.cuda.synchronize()
    samples = []
    for _ in range(N_TRIALS):
        t0 = time.time()
        fn(*args)
        torch.cuda.synchronize()
        samples.append(time.time() - t0)
    return samples


print("=== ViT-L/14 (CLIP) shape profile: hidden=1024, heads=16, head_dim=64, L=257 ===\n")

print("--- 1. Linear/matmul at the ViT-L/14 QKV-projection shape (257 x 1024 -> 3072) ---")
num_rows, in_features, out_features = 257, 1024, 3072

x = torch.randn(num_rows, in_features, device=device, dtype=torch.float16)
w = torch.randn(out_features, in_features, device=device, dtype=torch.float16)
b = torch.randn(out_features, device=device, dtype=torch.float16)
samples = bench(lambda x, w, b: torch.nn.functional.linear(x, w, b), x, w, b)
mean, std = stats(samples)
print(f"  PyTorch CUDA (fp16, H200): {mean*1000:.4f} +/- {std*1000:.4f} ms")

print("\n--- 2. Self-attention SDPA at the ViT-L/14 shape (H=16, L=257, D=64) ---")
B, H, L, D = 1, 16, 257, 64
q = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
k = torch.randn(B, H, L, D, device=device, dtype=torch.float16)
v = torch.randn(B, H, L, D, device=device, dtype=torch.float16)

# Default (backend auto-selected by PyTorch).
samples = bench(torch.nn.functional.scaled_dot_product_attention, q, k, v)
mean, std = stats(samples)
print(f"  PyTorch CUDA SDPA, default backend (fp16, H200): {mean*1000:.4f} +/- {std*1000:.4f} ms")

# Explicit backend sweep, to check whether the M5 Max reversal (short sequences favor the
# straightforward path over a more specialized kernel) shows up as a similar backend-selection
# effect on CUDA, where multiple real SDPA backends actually exist (unlike MPS, where they were
# previously found to all resolve to the same implementation).
for name, backend in [
    ("MATH", SDPBackend.MATH),
    ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
    ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
    ("CUDNN_ATTENTION", SDPBackend.CUDNN_ATTENTION),
]:
    try:
        with sdpa_kernel(backend):
            samples = bench(torch.nn.functional.scaled_dot_product_attention, q, k, v)
        mean, std = stats(samples)
        print(f"  PyTorch CUDA SDPA, {name}: {mean*1000:.4f} +/- {std*1000:.4f} ms")
    except Exception as e:
        print(f"  PyTorch CUDA SDPA, {name}: FAILED ({type(e).__name__}: {e})")
