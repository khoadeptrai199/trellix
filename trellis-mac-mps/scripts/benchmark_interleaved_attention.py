"""
Real full-model wall-clock benchmark of interleaved full/windowed attention, BEFORE investing in
the distillation pipeline -- deliberately checking the speedup ceiling first, per this project's
own established discipline (this is exactly how the FFN-caching ~11%-time-share ceiling was caught
early in Part I, rather than discovered only after building the full technique). Quality is not
measured here at all (that needs the distillation/ablation pipeline in research.md Part II) --
this only answers "is there a real wall-clock reason to keep pursuing this," on the real
slat_flow_model (555.5M-class params, 24 blocks), across the full_attention_interval sweep.
"""
import os, sys, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import torch
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.modules.sparse.attention.interleave_windowed import (
    configure_interleaved_attention,
    restore_full_attention,
    attention_mode_summary,
)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}\n")

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)
model = pipeline.models['slat_flow_model']
model.eval()
n_blocks = len(model.blocks)
print(f"slat_flow_model: {n_blocks} blocks\n")

import glob
from PIL import Image
image_paths = glob.glob('/Users/vinayak/Downloads/TRELLIS-main/assets/**/*.png', recursive=True) + \
              glob.glob('/Users/vinayak/Downloads/TRELLIS-main/assets/**/*.jpg', recursive=True)
image = Image.open(image_paths[0]).convert("RGBA")
real_cond = pipeline.get_cond([image])['cond']

# A real SLat latent at inference time has irregular sparse coords at the model's real
# resolution/channel count -- build a representative one the same way scripts/verify_*.py do.
import trellis.modules.sparse as sp
torch.manual_seed(0)


def make_sparse_tensor(n, extent, channels):
    pts = torch.unique(torch.randint(0, extent, (n, 3)), dim=0)
    coords = torch.cat([torch.zeros(pts.shape[0], 1, dtype=torch.int32), pts.int()], dim=1).to(device)
    feats = torch.randn(coords.shape[0], channels, device=device)
    return sp.SparseTensor(feats=feats, coords=coords)


x = make_sparse_tensor(3000, model.resolution, model.in_channels)
t = torch.full((1,), 500.0, device=device)


def sync():
    if device == 'mps':
        torch.mps.synchronize()


import statistics


def bench(label, n_trials=8, n_warmup=3):
    """
    Statistical-rigor upgrade: earlier runs of this script reported one mean-of-10 number per
    config (flagged in research.md's own limitations list as a real gap -- "mostly single-run
    wall-clock, no repeated trials with confidence intervals"). This times each individual forward
    pass as its own sample (not just one aggregate mean over a rep loop) and reports mean, std, and
    a 95% CI across n_trials independent samples, so run-to-run variance is visible instead of
    hidden inside a single averaged number.
    """
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x, t, real_cond)
        sync()
        samples = []
        for _ in range(n_trials):
            t0 = time.time()
            model(x, t, real_cond)
            sync()
            samples.append(time.time() - t0)
    mean = statistics.mean(samples)
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    ci95 = 1.96 * std / (len(samples) ** 0.5)
    print(f"  {label}: {mean*1000:.2f} +/- {std*1000:.2f} ms/forward (95% CI +/- {ci95*1000:.2f}, n={n_trials})")
    return {'mean': mean, 'std': std, 'ci95': ci95, 'samples': samples}


print(f"num active voxels in test input: {x.feats.shape[0]}\n")

# window_size=16 added after scripts/investigate_windowed_attn_isolated_speed.py found the
# windowed-attention implementation is actually SLOWER than full attention in isolation at
# window_size=8 (0.483x) and much slower at 4 (0.152x), but genuinely FASTER at 16 (1.466x) --
# testing whether that isolated advantage shows up at the full-model level too, not just assuming
# window_size=8 (the value already trusted elsewhere in this codebase for the VAE decoders, but
# never benchmarked in isolation until now) is the right choice for this different use case.
WINDOW_SIZES = [8, 16, 32]
K_SWEEP = [1, 2, 3, 4, 6, 8, 12, 24]
results = {}
for W in WINDOW_SIZES:
    print(f"\n--- window_size={W} ---")
    for K in K_SWEEP:
        assignment = configure_interleaved_attention(model, full_attention_interval=K, window_size=W)
        n_windowed = sum(1 for v in assignment.values() if v == "windowed")
        label = f"K={K:2d} ({n_windowed}/{n_blocks} windowed)"
        results[(W, K)] = bench(label)

restore_full_attention(model)

print("\n=== Summary (speedup relative to K=1, i.e. the unmodified pretrained baseline) ===")
print("(speedup CI via standard error propagation on the ratio of two means, not re-derived from raw samples)\n")
for W in WINDOW_SIZES:
    baseline = results[(W, 1)]
    print(f"\n  window_size={W}:")
    for K in K_SWEEP:
        r = results[(W, K)]
        speedup = baseline['mean'] / r['mean']
        rel_err = ((baseline['std']/baseline['mean'])**2 + (r['std']/r['mean'])**2) ** 0.5 if r['mean'] > 0 and baseline['mean'] > 0 else 0.0
        speedup_ci95 = 1.96 * speedup * rel_err
        print(f"    K={K:2d}: {r['mean']*1000:6.2f} +/- {r['std']*1000:.2f} ms, "
              f"speedup {speedup:.3f}x +/- {speedup_ci95:.3f} (95% CI)")

import json
summary_path = '/Users/vinayak/Downloads/TRELLIS-main/outputs/benchmark_interleaved_attention_stats.json'
with open(summary_path, 'w') as f:
    json.dump({
        f"W{W}_K{K}": {'mean_ms': results[(W,K)]['mean']*1000, 'std_ms': results[(W,K)]['std']*1000,
                       'ci95_ms': results[(W,K)]['ci95']*1000, 'samples_ms': [s*1000 for s in results[(W,K)]['samples']]}
        for W in WINDOW_SIZES for K in K_SWEEP
    }, f, indent=2)
print(f"\nSaved full per-trial data -> {summary_path}")
