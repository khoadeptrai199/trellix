"""
Tests whether TRELLIS's actual sampler (FlowEulerGuidanceIntervalSampler, confirmed directly from
the real microsoft/TRELLIS-image-large checkpoint's pipeline.json -- both sparse_structure_sampler
and slat_sampler default to 25 steps) tolerates fewer steps without a fancy solver swap, since
rectified flow is specifically designed for straighter ODE paths. No retraining, no kernel changes
-- just the existing `steps` param, per pipeline.run()'s own sparse_structure_sampler_params/
slat_sampler_params kwargs.

Measures, for step counts [25 (baseline), 20, 15, 10, 8], same seed/image throughout:
  - wall-clock time for sample_sparse_structure + sample_slat (the two diffusion sampling stages)
  - a quantitative structural-fidelity proxy: IoU of occupied sparse-structure voxel coordinates
    against the 25-step baseline (falsifiable, not a vibes-based comparison)
  - a rendered single-view mesh normal preview per step count, saved to outputs/, for visual check
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
from trellis.utils import render_utils

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
out_dir = "/Users/vinayak/Downloads/TRELLIS-main/outputs"
os.makedirs(out_dir, exist_ok=True)

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)

image = Image.open("assets/example_image/T.png")
image_p = pipeline.preprocess_image(image)
cond = pipeline.get_cond([image_p])
print("cond ready\n")

STEP_COUNTS = [25, 20, 15, 10, 8]
SEED = 1
baseline_coords = None
results = []

# pipeline.run() is @torch.no_grad()-decorated, but the sub-stages called individually below
# (sample_sparse_structure/sample_slat/decode_slat) are NOT -- calling them directly without this
# wrapper retains a full autograd graph through a 600M+ param forward pass every iteration, which
# balloons MPS allocation into the tens of GB by the second loop pass (hit exactly this: 63GB
# allocated, OOM, on iteration 2 before this fix). Real bug, not a flaky OOM -- always wrap direct
# sub-stage calls in torch.no_grad() outside of pipeline.run() itself.
torch.mps.empty_cache()


def run_one(steps):
    torch.manual_seed(SEED)
    t0 = time.time()
    coords = pipeline.sample_sparse_structure(cond, 1, {'steps': steps})
    torch.mps.synchronize()
    t_ss = time.time() - t0

    t1 = time.time()
    slat = pipeline.sample_slat(cond, coords, {'steps': steps})
    torch.mps.synchronize()
    t_slat = time.time() - t1

    t2 = time.time()
    out = pipeline.decode_slat(slat, formats=['mesh'])
    t_decode = time.time() - t2

    mesh = out['mesh'][0]
    n_verts = mesh.vertices.shape[0] if mesh.success else 0
    n_faces = mesh.faces.shape[0] if mesh.success else 0

    # Quantitative structural fidelity: IoU of occupied voxel coordinates vs. the 25-step baseline.
    coords_set = set(map(tuple, coords[:, 1:].cpu().numpy().tolist()))  # drop batch column
    global baseline_coords
    if steps == 25:
        baseline_coords = coords_set
        iou = 1.0
    else:
        inter = len(coords_set & baseline_coords)
        union = len(coords_set | baseline_coords)
        iou = inter / union if union > 0 else 0.0

    # Render a single-view normal preview for visual inspection.
    t3 = time.time()
    frame = render_utils.render_frames(
        mesh,
        *render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics([0.8], [0.4], 2, 40, device=mesh.vertices.device),
        options={'resolution': 512},
    )
    normal_img = (frame['normal'][0]).astype(np.uint8)
    preview_path = f"{out_dir}/stepcheck_{steps}steps.png"
    Image.fromarray(normal_img).save(preview_path)
    t_render = time.time() - t3

    total = t_ss + t_slat + t_decode + t_render
    print(f"steps={steps:2d}: sample_sparse_structure={t_ss:.2f}s sample_slat={t_slat:.2f}s "
          f"decode={t_decode:.2f}s render={t_render:.2f}s TOTAL={total:.2f}s | "
          f"mesh: {n_verts}v/{n_faces}f | voxel IoU vs 25-step: {iou:.4f} | -> {preview_path}",
          flush=True)
    results.append({'steps': steps, 't_ss': t_ss, 't_slat': t_slat, 't_decode': t_decode,
                     't_render': t_render, 'total': total, 'n_verts': n_verts, 'n_faces': n_faces,
                     'iou': iou})


for steps in STEP_COUNTS:
    with torch.no_grad():
        run_one(steps)
    torch.mps.empty_cache()

print("\n=== SUMMARY ===")
baseline_total = results[0]['total']
for r in results:
    speedup = baseline_total / r['total']
    print(f"steps={r['steps']:2d}: {r['total']:.2f}s ({speedup:.2f}x vs 25-step baseline), "
          f"voxel IoU={r['iou']:.4f}, {r['n_verts']}v/{r['n_faces']}f")
