import os
import sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trellis')
def fake_pkg(name, path):
    m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m; return m
fake_pkg('trellis', _root)
fake_pkg('trellis.renderers', _root + '/renderers')  # bypass renderers/__init__.py's lazy-import machinery
# trellis.representations.__init__ eagerly imports Strivec/Octree/Gaussian; only need .mesh here.
_repr_pkg = fake_pkg('trellis.representations', _root + '/representations')
import trellis.representations.mesh as mesh_mod
_repr_pkg.MeshExtractResult = mesh_mod.MeshExtractResult

import torch
from trellis.renderers.mesh_renderer import MeshRenderer, intrinsics_to_projection
from trellis.representations.mesh.cube2mesh import MeshExtractResult

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}\n")

# unit cube centered at origin, 8 verts / 12 tris (outward-facing winding doesn't matter -- the
# rasterizer's inside test is winding-agnostic, verified in the unit test)
V = torch.tensor([
    [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
    [-0.5, -0.5,  0.5], [0.5, -0.5,  0.5], [0.5, 0.5,  0.5], [-0.5, 0.5,  0.5],
], dtype=torch.float32, device=device)
F = torch.tensor([
    [0,1,2],[0,2,3], [4,6,5],[4,7,6], [0,4,5],[0,5,1],
    [1,5,6],[1,6,2], [2,6,7],[2,7,3], [3,7,4],[3,4,0],
], dtype=torch.int64, device=device)
mesh = MeshExtractResult(vertices=V, faces=F)

# Camera at world origin, looking straight down +z (identity extrinsics -- world coords ARE
# camera coords, sidesteps any look-at rotation convention ambiguity). Cube translated to z=3
# so it's in front of the camera within [near, far].
cube_at = V.clone()
cube_at[:, 2] += 3.0
mesh = MeshExtractResult(vertices=cube_at, faces=F)

extrinsics = torch.eye(4, device=device)
intrinsics = torch.tensor([
    [0.7, 0.0, 0.5],
    [0.0, 0.7, 0.5],
    [0.0, 0.0, 1.0],
], device=device)  # OpenCV-normalized: fx=fy=0.7, principal point at image center

renderer = MeshRenderer(rendering_options={"resolution": 128, "near": 0.1, "far": 10.0, "ssaa": 1}, device=device)
out = renderer.render(mesh, extrinsics, intrinsics, return_types=["mask", "depth", "normal"])

all_ok = True
mask = out["mask"]
depth = out["depth"]
normal = out["normal"]
print(f"mask: shape={tuple(mask.shape)} dtype={mask.dtype} device={mask.device}")
print(f"depth: shape={tuple(depth.shape)} device={depth.device}")
print(f"normal: shape={tuple(normal.shape)} device={normal.device}")

covered_frac = mask.float().mean().item()
print(f"\ncovered pixel fraction: {covered_frac:.4f} (expect roughly a centered square-ish blob, well under 1.0, well over 0)")
all_ok &= 0.05 < covered_frac < 0.9

# cube spans world [-0.5,0.5]+z=3 -> camera-space z in [2.5, 3.5]; depth output is interpolated
# camera-space z (vertices_camera[...,2]), so covered-pixel depths should fall in that range.
covered = mask.bool()
if covered.dim() == 3:  # [1,H,W] if a channel dim survived squeeze
    covered = covered[0]
d_covered = depth[covered] if depth.dim() == 2 else depth.squeeze()[covered]
print(f"depth range at covered pixels: [{d_covered.min().item():.3f}, {d_covered.max().item():.3f}] (expect within [2.5, 3.5])")
depth_ok = d_covered.min().item() >= 2.4 and d_covered.max().item() <= 3.6
all_ok &= depth_ok
print(f"depth range plausible: {depth_ok}")

# silhouette should be roughly centered: compute centroid of covered pixels in normalized coords
ys, xs = torch.meshgrid(torch.arange(128, device=device), torch.arange(128, device=device), indexing='ij')
cy = ys[covered].float().mean().item() / 128
cx = xs[covered].float().mean().item() / 128
print(f"silhouette centroid (normalized): ({cx:.3f}, {cy:.3f}) (expect near (0.5, 0.5))")
centroid_ok = abs(cx - 0.5) < 0.1 and abs(cy - 0.5) < 0.1
all_ok &= centroid_ok
print(f"silhouette centered: {centroid_ok}")

finite_ok = torch.isfinite(depth[covered]).all().item() and torch.isfinite(normal).all().item()
all_ok &= finite_ok
print(f"finite depth/normal: {finite_ok}")

# ssaa path (F.interpolate downsample) sanity check
renderer_ssaa = MeshRenderer(rendering_options={"resolution": 128, "near": 0.1, "far": 10.0, "ssaa": 2}, device=device)
out_ssaa = renderer_ssaa.render(mesh, extrinsics, intrinsics, return_types=["mask"])
ssaa_ok = out_ssaa["mask"].shape[-1] == 128 and torch.isfinite(out_ssaa["mask"]).all().item()
all_ok &= ssaa_ok
print(f"\nssaa=2 path shape/finite ok: {ssaa_ok}, shape={tuple(out_ssaa['mask'].shape)}")

print("\n" + ("ALL MESH RENDERER MPS CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
