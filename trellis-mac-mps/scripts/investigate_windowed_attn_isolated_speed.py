"""
Isolates the windowed self-attention implementation's own wall-clock cost (full vs. windowed, same
real shape, no full-model overhead) -- stage-3-style isolated benchmark, per this project's own
methodology, applied to the pre-existing `sparse_windowed_scaled_dot_product_self_attention` path
rather than a new custom kernel.

Motivation: research.md Part II's bypass measurement found self-attention is genuinely the larger
share of slat_flow_model's compute (29.3%, 91.15ms of 310.60ms) -- a real ceiling of up to 1.42x if
self-attention were free. But the actual interleaved-windowed-attention result only reached 1.05x,
capturing only ~15 of the possible ~91ms. This isolates whether the windowed implementation itself
is fast at the real problem scale, or whether it's already close to (or worse than) full attention
even with no full-model overhead in the way -- the same "kernel-quality gap, not algorithmic
ceiling" pattern Part I already found for custom attention kernels.
"""
import os, sys, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import torch
from trellis.pipelines import TrellisImageTo3DPipeline
import trellis.modules.sparse as sp

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}\n")

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)
model = pipeline.models['slat_flow_model']
model.eval()

# Use the real model's own first-block self_attn module directly (real weights, real config --
# qk_rms_norm, num_heads, channels -- not a guessed reconstruction), and the real model's own
# channel count / dtype, so this isolated test can't diverge from the real shape the way an
# earlier FFN isolated benchmark once did (APPLE_SILICON.md's "always capture real shapes"
# lesson).
self_attn = model.blocks[0].self_attn
print(f"channels={self_attn.channels}, num_heads={self_attn.num_heads}, "
      f"qk_rms_norm={self_attn.qk_rms_norm}, dtype={model.dtype}\n")

torch.manual_seed(0)


def make_sparse_tensor(n, extent, channels, dtype):
    pts = torch.unique(torch.randint(0, extent, (n, 3)), dim=0)
    coords = torch.cat([torch.zeros(pts.shape[0], 1, dtype=torch.int32), pts.int()], dim=1).to(device)
    feats = torch.randn(coords.shape[0], channels, device=device, dtype=dtype)
    return sp.SparseTensor(feats=feats, coords=coords)


# real per-block active-voxel count from the same test input used in the bypass-split
# investigation (~2982 voxels) -- same shape, so this is directly comparable to that result.
x = make_sparse_tensor(3000, model.resolution, self_attn.channels, model.dtype)
print(f"num active voxels in test input: {x.feats.shape[0]}\n")


def sync():
    if device == 'mps':
        torch.mps.synchronize()


def bench(label, n_reps=50, n_warmup=5):
    with torch.no_grad():
        for _ in range(n_warmup):
            self_attn(x)
        sync()
        t0 = time.time()
        for _ in range(n_reps):
            self_attn(x)
        sync()
        elapsed = (time.time() - t0) / n_reps
    print(f"  {label}: {elapsed*1000:.4f} ms/call")
    return elapsed


print("=== Isolated self-attention, full vs. windowed, real shape/dtype/weights ===")
self_attn.attn_mode = "full"
t_full = bench(f"full ({x.feats.shape[0]} tokens, O(N^2))")

results = {}
for window_size in [4, 8, 16]:
    self_attn.attn_mode = "windowed"
    self_attn.window_size = window_size
    self_attn.shift_window = 0
    t_win = bench(f"windowed, window_size={window_size:2d}, shift=0")
    results[window_size] = t_win

self_attn.attn_mode = "full"

print("\n=== Summary ===")
print(f"  full attention:  {t_full*1000:.4f} ms/call (baseline)")
for window_size, t_win in results.items():
    print(f"  windowed W={window_size:2d}: {t_win*1000:.4f} ms/call, "
          f"{t_full/t_win:.3f}x vs full attention")
