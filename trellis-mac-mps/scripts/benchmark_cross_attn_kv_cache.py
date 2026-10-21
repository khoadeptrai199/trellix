"""
Benchmarks trellis/modules/attention/cross_kv_cache.py (stages 3-4 of this project's own
verification protocol; correctness is scripts/verify_cross_attn_kv_cache.py, stages 1-2).

Stage 3: isolated per-call timing of a single cross_attn.to_kv Linear, cached vs uncached, at the
real production shape.
Stage 4: the real full-model benchmark -- the only number that actually matters, per this
project's own "isolated and real-model speed diverge, never report isolated as real" lesson.
Simulates a real guidance-interval CFG sampling loop (cond + neg_cond called every step) on the
real sparse_structure_flow_model (555.5M params).
"""
import os, sys, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import torch
import torch.nn as nn
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules.attention.cross_kv_cache import (
    enable_cross_attention_kv_cache,
    disable_cross_attention_kv_cache,
    clear_cross_attention_kv_cache,
    cross_attention_kv_cache_stats,
)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}\n")

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)
model = pipeline.models['sparse_structure_flow_model']
model.eval()

import glob
from PIL import Image
image_paths = glob.glob('/Users/vinayak/Downloads/TRELLIS-main/assets/**/*.png', recursive=True) + \
              glob.glob('/Users/vinayak/Downloads/TRELLIS-main/assets/**/*.jpg', recursive=True)
image = Image.open(image_paths[0]).convert("RGBA")
real_cond = pipeline.get_cond([image])['cond']
real_neg_cond = torch.zeros_like(real_cond)

B = 1
x = torch.randn(B, 8, 16, 16, 16, device=device)
t = torch.full((B,), 500.0, device=device)


def sync():
    if device == 'mps':
        torch.mps.synchronize()


# ---------------------------------------------------------------------------
# Stage 3: isolated per-call benchmark of a single to_kv Linear.
# ---------------------------------------------------------------------------
print("=== Stage 3: isolated to_kv benchmark ===")
first_block = model.blocks[0]
to_kv = first_block.cross_attn.to_kv
N = 200
# to_kv only ever sees `cond` after the model's own `cond.type(self.dtype)` cast (see
# cross_kv_cache.py's docstring) -- match that here, or MPS's matmul kernel asserts on a
# fp32-input x fp16-weight mismatch that the real forward path never actually hits.
real_cond_cast = real_cond.type(model.dtype)

with torch.no_grad():
    for _ in range(5):
        to_kv(real_cond_cast)
    sync()
    t0 = time.time()
    for _ in range(N):
        to_kv(real_cond_cast)
    sync()
    t_uncached = (time.time() - t0) / N
print(f"  uncached: {t_uncached*1000:.4f} ms/call")

enable_cross_attention_kv_cache(model)
clear_cross_attention_kv_cache(model)
with torch.no_grad():
    to_kv(real_cond_cast)  # prime (1 miss)
    sync()
    t0 = time.time()
    for _ in range(N):
        to_kv(real_cond_cast)
    sync()
    t_cached = (time.time() - t0) / N
print(f"  cached:   {t_cached*1000:.4f} ms/call")
print(f"  isolated speedup: {t_uncached/t_cached:.2f}x (NOT the real number -- see Stage 4)")
disable_cross_attention_kv_cache(model)

# ---------------------------------------------------------------------------
# Stage 4: real full-model benchmark, simulating a real CFG guidance-interval sampling loop.
# ---------------------------------------------------------------------------
print("\n=== Stage 4: real full-model benchmark (simulated 25-step CFG loop) ===")
STEPS = 25

with torch.no_grad():
    for _ in range(2):
        model(x, t, real_cond)
        model(x, t, real_neg_cond)
    sync()
    t0 = time.time()
    for _ in range(STEPS):
        model(x, t, real_cond)
        model(x, t, real_neg_cond)
    sync()
    t_baseline = time.time() - t0
print(f"  baseline (no cache): {t_baseline*1000:.1f}ms for {STEPS} steps ({t_baseline/STEPS*1000:.2f}ms/step)")

n_patched = enable_cross_attention_kv_cache(model)
print(f"  patched {n_patched} cross_attn.to_kv Linears")
with torch.no_grad():
    clear_cross_attention_kv_cache(model)
    for _ in range(2):
        model(x, t, real_cond)
        model(x, t, real_neg_cond)
    sync()
    clear_cross_attention_kv_cache(model)
    t0 = time.time()
    for _ in range(STEPS):
        model(x, t, real_cond)
        model(x, t, real_neg_cond)
    sync()
    t_cached = time.time() - t0
hits, misses = cross_attention_kv_cache_stats(model)
print(f"  cached: {t_cached*1000:.1f}ms for {STEPS} steps ({t_cached/STEPS*1000:.2f}ms/step)")
print(f"  hits={hits}, misses={misses}")

speedup = t_baseline / t_cached
print(f"\n  REAL full-model speedup: {speedup:.3f}x")
disable_cross_attention_kv_cache(model)
