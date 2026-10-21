"""
Profiles metal_rasterizer_binned.rasterize() against metal_rasterizer.rasterize() (brute force)
on the REAL decimated vase mesh (outputs/vase_decimated_mesh.npz, see
scripts/save_real_mesh_for_bench.py) instead of a synthetic random-triangle mesh -- random
triangles (3 independently random vertices) are mostly huge/screen-spanning and a bad proxy for a
real mesh's small, locally-coherent faces, which is what tile binning is actually meant to exploit.

Also breaks binned rasterize() down into its stages (CPU bin-build vs Metal upload/dispatch/
readback) to see where time actually goes if it's not winning.
"""
import sys, time
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
import numpy as np
import torch
from trellis.renderers import metal_rasterizer as brute
from trellis.renderers import metal_rasterizer_binned as binned

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
data = np.load("outputs/vase_decimated_mesh.npz")
vertices_np, faces_np = data['vertices'], data['faces']
V, F = vertices_np.shape[0], faces_np.shape[0]
print(f"real mesh: {V}v / {F}f, device={device}\n")

verts = torch.tensor(vertices_np, dtype=torch.float32)
verts_h = torch.cat([verts, torch.ones(V, 1)], dim=-1).unsqueeze(0).to(device)
tri = torch.tensor(faces_np, dtype=torch.int32).to(device)

for H, W in [(1024, 1024), (512, 512)]:
    t0 = time.time()
    r_brute, _ = brute.rasterize(None, verts_h, tri, (H, W))
    if device == 'mps': torch.mps.synchronize()
    t_brute = time.time() - t0

    t0 = time.time()
    r_binned, _ = binned.rasterize(None, verts_h, tri, (H, W))
    if device == 'mps': torch.mps.synchronize()
    t_binned = time.time() - t0

    match = torch.equal(r_brute, r_binned)
    print(f"[{H}x{W}] brute: {t_brute:.3f}s, binned: {t_binned:.3f}s, speedup: {t_brute/t_binned:.2f}x, match: {match}")

# Stage breakdown for binned at 1024x1024, using its internal helpers directly
print("\n--- binned rasterize() stage breakdown @ 1024x1024 ---")
H, W = 1024, 1024
pos0 = verts_h[0]
w = pos0[:, 3].clamp(min=1e-6)
ndc = pos0[:, :3] / w[:, None]
px = (ndc[:, 0] * 0.5 + 0.5) * W
py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * H

t0 = time.time()
px_np = px.detach().to('cpu').to(torch.float32).numpy()
py_np = py.detach().to('cpu').to(torch.float32).numpy()
tri_np = tri.detach().to('cpu').to(torch.int32).numpy()
if device == 'mps': torch.mps.synchronize()
t_sync = time.time() - t0
print(f"sync px/py/tri to CPU: {t_sync:.4f}s")

t0 = time.time()
tile_offsets, bin_tri_ids, ntx, nty = binned._build_bins(px_np, py_np, tri_np, W, H)
t_bin = time.time() - t0
print(f"CPU bin-build (_build_bins): {t_bin:.4f}s -- {ntx}x{nty} tiles, {bin_tri_ids.shape[0]} total bin entries "
      f"({bin_tri_ids.shape[0] / F:.2f} avg tiles/triangle)")

t0 = time.time()
r_binned2, _ = binned.rasterize(None, verts_h, tri, (H, W))
if device == 'mps': torch.mps.synchronize()
t_full = time.time() - t0
print(f"full binned rasterize() call: {t_full:.4f}s (bin-build is {100*t_bin/t_full:.1f}% of that)")
