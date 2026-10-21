"""
Benchmarks _fill_holes (postprocessing_utils.py) at several rasterization resolutions on the same
real, decimated mesh, to check whether dropping fill_holes_resolution below the pipeline's default
of 1024 is a safe way to speed up hole-filling (see APPLE_SILICON.md's "Repeated .to('cpu') calls"
section for background -- the tri-buffer cache/sync-batching fix that resolution independent
didn't help because at 1024x1024 the Metal kernel's own O(H*W*F) compute dominates, not the sync).

_fill_holes only ever reads back `face_id` and a `mask > 0.95` threshold per view -- it doesn't
need a full-resolution image -- so this checks both:
  1. speed at each resolution
  2. whether the resulting per-face visibility count (and therefore which faces get removed) is
     materially different from the 1024 baseline

Uses seed=1 on object2_vase.png (matches scripts/run_real_pipeline.py's vase run) so the mesh
geometry is identical to the numbers already recorded in APPLE_SILICON.md.
"""
import os, sys, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import numpy as np
import torch
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils as pp
import pyvista as pv

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"=== device: {device} ===", flush=True)

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)

image = Image.open("assets/example_image/object2_vase.png")
outputs = pipeline.run(image, seed=1, formats=['mesh'])
mesh = outputs['mesh'][0]
print(f"raw mesh: {mesh.vertices.shape[0]}v / {mesh.faces.shape[0]}f", flush=True)

# Decimate once (matches run_real_pipeline.py's simplify=0.95) so every resolution below rasterizes
# the exact same ~35K-face mesh -- isolates the resolution variable.
vertices_np = mesh.vertices.cpu().numpy()
faces_np = mesh.faces.cpu().numpy()
pvmesh = pv.PolyData(vertices_np, np.concatenate([np.full((faces_np.shape[0], 1), 3), faces_np], axis=1))
pvmesh = pvmesh.decimate(0.95)
vertices_np, faces_np = pvmesh.points, pvmesh.faces.reshape(-1, 4)[:, 1:]
print(f"decimated mesh: {vertices_np.shape[0]}v / {faces_np.shape[0]}f\n", flush=True)

verts_t = torch.tensor(vertices_np, dtype=torch.float32).to(device)
faces_t = torch.tensor(faces_np.astype(np.int32)).to(device)

NUM_VIEWS = 1000
RESOLUTIONS = [1024, 512, 256, 128]

results = {}
for res in RESOLUTIONS:
    t0 = time.time()
    out_v, out_f = pp._fill_holes(
        verts_t.clone(), faces_t.clone(),
        max_hole_size=0.04,
        max_hole_nbe=int(250 * np.sqrt(1 - 0.95)),
        resolution=res,
        num_views=NUM_VIEWS,
        verbose=False,
    )
    elapsed = time.time() - t0
    results[res] = {'time': elapsed, 'faces_kept': out_f.shape[0]}
    print(f"resolution={res:5d}: {elapsed:7.1f}s, faces kept after hole-fill: {out_f.shape[0]} / {faces_t.shape[0]}", flush=True)

print("\n=== Summary vs 1024 baseline ===")
base = results[1024]
for res in RESOLUTIONS:
    r = results[res]
    speedup = base['time'] / r['time']
    face_delta = r['faces_kept'] - base['faces_kept']
    face_delta_pct = 100.0 * face_delta / base['faces_kept']
    print(f"res={res:5d}  time={r['time']:7.1f}s  speedup={speedup:5.2f}x  "
          f"faces_kept={r['faces_kept']:6d}  delta_vs_1024={face_delta:+6d} ({face_delta_pct:+.2f}%)")
