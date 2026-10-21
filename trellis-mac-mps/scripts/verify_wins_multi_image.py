"""
Broader verification of the two real wins found this session -- both were only checked on ONE
example image/seed (assets/example_image/T.png, seed=1) so far, flagged as a caveat in
APPLE_SILICON.md each time. Repeats both at their most aggressive tested setting, across 3
additional, deliberately varied example images (a simple inorganic object, an organic creature, a
mechanical vehicle) x 2 seeds each:

1. Step-count reduction: 25 steps (baseline) vs. 8 steps.
2. CFG difference-caching: neg_cache_interval=1 (baseline) vs. 4
   (DiffCachedCfgFlowEulerGuidanceIntervalSampler).

For each (image, seed, technique) combination: voxel IoU vs. that same image/seed's own 25-step/
interval=1 baseline, mesh vertex/face counts (the "did structure actually change" proxy that
proved more reliable than IoU alone in the original single-image tests), and a rendered preview
for visual spot-checking.
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
from trellis.pipelines.samplers.flow_euler_cached_cfg import DiffCachedCfgFlowEulerGuidanceIntervalSampler
from trellis.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
from trellis.utils import render_utils

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
out_dir = "/Users/vinayak/Downloads/TRELLIS-main/outputs/multi_verify"
os.makedirs(out_dir, exist_ok=True)

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)

ss_sigma_min = pipeline.sparse_structure_sampler.sigma_min
ss_params = dict(pipeline.sparse_structure_sampler_params)
slat_sigma_min = pipeline.slat_sampler.sigma_min
slat_params = dict(pipeline.slat_sampler_params)

IMAGES = {
    'vase': 'assets/example_image/object2_vase.png',
    'dragon': 'assets/example_image/typical_creature_dragon.png',
    'biplane': 'assets/example_image/typical_vehicle_biplane.png',
}
SEEDS = [1, 2]

results = []


def run_config(cond, seed, sampler_pair, ss_p, slat_p):
    pipeline.sparse_structure_sampler, pipeline.slat_sampler = sampler_pair
    torch.manual_seed(seed)
    with torch.no_grad():
        coords = pipeline.sample_sparse_structure(cond, 1, ss_p)
        slat = pipeline.sample_slat(cond, coords, slat_p)
        out = pipeline.decode_slat(slat, formats=['mesh'])
    mesh = out['mesh'][0]
    n_verts = mesh.vertices.shape[0] if mesh.success else 0
    n_faces = mesh.faces.shape[0] if mesh.success else 0
    coords_set = set(map(tuple, coords[:, 1:].cpu().numpy().tolist()))
    return mesh, n_verts, n_faces, coords_set


def render_preview(mesh, path):
    frame = render_utils.render_frames(
        mesh,
        *render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics([0.8], [0.4], 2, 40, device=mesh.vertices.device),
        options={'resolution': 512},
    )
    Image.fromarray((frame['normal'][0]).astype(np.uint8)).save(path)


for img_name, img_path in IMAGES.items():
    image = Image.open(img_path)
    image_p = pipeline.preprocess_image(image)
    cond = pipeline.get_cond([image_p])
    print(f"\n=== {img_name} ({img_path}) ===", flush=True)

    for seed in SEEDS:
        t0 = time.time()
        native_pair = (FlowEulerGuidanceIntervalSampler(sigma_min=ss_sigma_min),
                       FlowEulerGuidanceIntervalSampler(sigma_min=slat_sigma_min))
        mesh_base, v_base, f_base, coords_base = run_config(cond, seed, native_pair, ss_params, slat_params)
        t_base = time.time() - t0
        render_preview(mesh_base, f"{out_dir}/{img_name}_seed{seed}_baseline25.png")
        print(f"  seed={seed} baseline (25 steps): {t_base:.1f}s, {v_base}v/{f_base}f", flush=True)

        t0 = time.time()
        step8_ss_params = {**ss_params, 'steps': 8}
        step8_slat_params = {**slat_params, 'steps': 8}
        native_pair2 = (FlowEulerGuidanceIntervalSampler(sigma_min=ss_sigma_min),
                        FlowEulerGuidanceIntervalSampler(sigma_min=slat_sigma_min))
        mesh_8, v_8, f_8, coords_8 = run_config(cond, seed, native_pair2, step8_ss_params, step8_slat_params)
        t_8 = time.time() - t0
        render_preview(mesh_8, f"{out_dir}/{img_name}_seed{seed}_steps8.png")
        inter = len(coords_8 & coords_base)
        union = len(coords_8 | coords_base)
        iou_8 = inter / union if union > 0 else 0.0
        speedup_8 = t_base / t_8
        print(f"  seed={seed} 8 steps: {t_8:.1f}s ({speedup_8:.2f}x), {v_8}v/{f_8}f, "
              f"voxel IoU vs baseline={iou_8:.4f}", flush=True)

        t0 = time.time()
        diff_pair = (DiffCachedCfgFlowEulerGuidanceIntervalSampler(sigma_min=ss_sigma_min, neg_cache_interval=4),
                     DiffCachedCfgFlowEulerGuidanceIntervalSampler(sigma_min=slat_sigma_min, neg_cache_interval=4))
        mesh_cfg, v_cfg, f_cfg, coords_cfg = run_config(cond, seed, diff_pair, ss_params, slat_params)
        t_cfg = time.time() - t0
        render_preview(mesh_cfg, f"{out_dir}/{img_name}_seed{seed}_cfgdiff4.png")
        inter = len(coords_cfg & coords_base)
        union = len(coords_cfg | coords_base)
        iou_cfg = inter / union if union > 0 else 0.0
        speedup_cfg = t_base / t_cfg
        print(f"  seed={seed} CFG-diff interval=4: {t_cfg:.1f}s ({speedup_cfg:.2f}x), {v_cfg}v/{f_cfg}f, "
              f"voxel IoU vs baseline={iou_cfg:.4f}", flush=True)

        results.append({
            'image': img_name, 'seed': seed,
            'v_base': v_base, 'f_base': f_base, 't_base': t_base,
            'v_8': v_8, 'f_8': f_8, 't_8': t_8, 'speedup_8': speedup_8, 'iou_8': iou_8,
            'v_cfg': v_cfg, 'f_cfg': f_cfg, 't_cfg': t_cfg, 'speedup_cfg': speedup_cfg, 'iou_cfg': iou_cfg,
        })
        torch.mps.empty_cache()

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"{'image':10s} {'seed':5s} {'base v/f':16s} {'8steps v/f':16s} {'8step speedup':14s} {'8step IoU':10s} "
      f"{'cfg4 v/f':16s} {'cfg4 speedup':13s} {'cfg4 IoU':10s}")
for r in results:
    print(f"{r['image']:10s} {r['seed']:<5d} {str(r['v_base'])+'/'+str(r['f_base']):16s} "
          f"{str(r['v_8'])+'/'+str(r['f_8']):16s} {r['speedup_8']:<14.2f} {r['iou_8']:<10.4f} "
          f"{str(r['v_cfg'])+'/'+str(r['f_cfg']):16s} {r['speedup_cfg']:<13.2f} {r['iou_cfg']:<10.4f}")

avg_speedup_8 = sum(r['speedup_8'] for r in results) / len(results)
avg_iou_8 = sum(r['iou_8'] for r in results) / len(results)
avg_speedup_cfg = sum(r['speedup_cfg'] for r in results) / len(results)
avg_iou_cfg = sum(r['iou_cfg'] for r in results) / len(results)
print(f"\nAverage across {len(results)} (image, seed) combos:")
print(f"  8-step:        {avg_speedup_8:.2f}x speedup, {avg_iou_8:.4f} avg voxel IoU")
print(f"  CFG-diff int=4: {avg_speedup_cfg:.2f}x speedup, {avg_iou_cfg:.4f} avg voxel IoU")
