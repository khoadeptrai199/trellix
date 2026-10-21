"""
Tests CachedCfgFlowEulerGuidanceIntervalSampler (trellis/pipelines/samplers/flow_euler_cached_cfg.py)
-- caches the classifier-free-guidance negative-branch prediction and reuses it for
`neg_cache_interval` consecutive steps, instead of recomputing every step within the guidance
interval (confirmed directly in guidance_interval_mixin.py: the model is called twice per step --
cond and neg_cond -- whenever t is in cfg_interval, which this pipeline's real config sets to
[0.5, 1.0]). This is a genuinely new experiment for TRELLIS's specific rectified-flow +
guidance-interval combination, not a known-working technique being reapplied (see the module
docstring in flow_euler_cached_cfg.py).

Measures, for neg_cache_interval in [1 (baseline, no caching), 2, 3, 4], same seed/image
throughout, at the real pipeline's default 25 steps:
  - wall-clock time for the full sample_sparse_structure + sample_slat + decode_slat + render
  - actual negative-branch recompute count (from the sampler's own counters) vs. the naive
    "N model calls" estimate, so the speedup claim is grounded in a real count, not assumed
  - the same quantitative structural-fidelity proxy used in investigate_step_count.py (voxel IoU
    vs. the neg_cache_interval=1 baseline) plus a rendered preview per interval for visual check
"""
import os, sys, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import torch
import numpy as np
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.pipelines.samplers.flow_euler_cached_cfg import CachedCfgFlowEulerGuidanceIntervalSampler
from trellis.utils import render_utils

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
out_dir = "/Users/vinayak/Downloads/TRELLIS-main/outputs"
os.makedirs(out_dir, exist_ok=True)

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)

# Swap in the cached sampler for both stages, matching the real checkpoint's own sampler args
# (sigma_min=1e-05, confirmed directly from pipeline.json) -- same sampler class, only the negative-
# branch caching behavior differs from the production FlowEulerGuidanceIntervalSampler.
sigma_min = pipeline.sparse_structure_sampler.sigma_min
ss_params = dict(pipeline.sparse_structure_sampler_params)
slat_sigma_min = pipeline.slat_sampler.sigma_min
slat_params = dict(pipeline.slat_sampler_params)
print(f"sparse_structure sampler params: {ss_params}")
print(f"slat sampler params: {slat_params}\n")

pipeline.sparse_structure_sampler = CachedCfgFlowEulerGuidanceIntervalSampler(sigma_min=sigma_min)
pipeline.slat_sampler = CachedCfgFlowEulerGuidanceIntervalSampler(sigma_min=slat_sigma_min)

image = Image.open("assets/example_image/T.png")
image_p = pipeline.preprocess_image(image)
cond = pipeline.get_cond([image_p])
print("cond ready\n")

CACHE_INTERVALS = [1, 2, 3, 4]
SEED = 1
baseline_coords = None
results = []


def run_one(interval):
    torch.manual_seed(SEED)
    ss_p = {**ss_params, 'neg_cache_interval': interval} if False else ss_params
    # neg_cache_interval is a constructor kwarg, not a sample() kwarg -- set on the instance.
    pipeline.sparse_structure_sampler.neg_cache_interval = interval
    pipeline.slat_sampler.neg_cache_interval = interval

    t0 = time.time()
    coords = pipeline.sample_sparse_structure(cond, 1, ss_params)
    torch.mps.synchronize()
    t_ss = time.time() - t0
    ss_recompute = pipeline.sparse_structure_sampler.neg_recompute_count
    ss_reuse = pipeline.sparse_structure_sampler.neg_reuse_count

    t1 = time.time()
    slat = pipeline.sample_slat(cond, coords, slat_params)
    torch.mps.synchronize()
    t_slat = time.time() - t1
    slat_recompute = pipeline.slat_sampler.neg_recompute_count
    slat_reuse = pipeline.slat_sampler.neg_reuse_count

    t2 = time.time()
    out = pipeline.decode_slat(slat, formats=['mesh'])
    t_decode = time.time() - t2

    mesh = out['mesh'][0]
    n_verts = mesh.vertices.shape[0] if mesh.success else 0
    n_faces = mesh.faces.shape[0] if mesh.success else 0

    coords_set = set(map(tuple, coords[:, 1:].cpu().numpy().tolist()))
    global baseline_coords
    if interval == 1:
        baseline_coords = coords_set
        iou = 1.0
    else:
        inter = len(coords_set & baseline_coords)
        union = len(coords_set | baseline_coords)
        iou = inter / union if union > 0 else 0.0

    t3 = time.time()
    frame = render_utils.render_frames(
        mesh,
        *render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics([0.8], [0.4], 2, 40, device=mesh.vertices.device),
        options={'resolution': 512},
    )
    normal_img = (frame['normal'][0]).astype(np.uint8)
    preview_path = f"{out_dir}/cfgcache_{interval}interval.png"
    Image.fromarray(normal_img).save(preview_path)
    t_render = time.time() - t3

    total = t_ss + t_slat + t_decode + t_render
    total_recompute = ss_recompute + slat_recompute
    total_reuse = ss_reuse + slat_reuse
    print(f"interval={interval}: sample_sparse_structure={t_ss:.2f}s sample_slat={t_slat:.2f}s "
          f"decode={t_decode:.2f}s render={t_render:.2f}s TOTAL={total:.2f}s | "
          f"neg-branch recomputed={total_recompute} reused={total_reuse} | "
          f"mesh: {n_verts}v/{n_faces}f | voxel IoU vs interval=1: {iou:.4f} | -> {preview_path}",
          flush=True)
    results.append({'interval': interval, 'total': total, 'recompute': total_recompute,
                     'reuse': total_reuse, 'n_verts': n_verts, 'n_faces': n_faces, 'iou': iou})


for interval in CACHE_INTERVALS:
    with torch.no_grad():
        run_one(interval)
    torch.mps.empty_cache()

print("\n=== SUMMARY ===")
baseline_total = results[0]['total']
for r in results:
    speedup = baseline_total / r['total']
    print(f"interval={r['interval']}: {r['total']:.2f}s ({speedup:.2f}x vs no-caching baseline), "
          f"neg-branch calls saved: {r['reuse']}/{r['recompute']+r['reuse']}, "
          f"voxel IoU={r['iou']:.4f}, {r['n_verts']}v/{r['n_faces']}f")
