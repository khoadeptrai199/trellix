import os
import sys, types, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trellis')
def fake_pkg(name, path):
    m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m; return m
fake_pkg('trellis', _root)
_repr_pkg = fake_pkg('trellis.representations', _root + '/representations')
from trellis.representations.gaussian.gaussian_model import Gaussian
_repr_pkg.Gaussian = Gaussian
_repr_pkg.Strivec = object
from trellis.representations.octree.octree_dfs import DfsOctree
_repr_pkg.Octree = DfsOctree
import trellis.representations.mesh as mesh_mod
_repr_pkg.MeshExtractResult = mesh_mod.MeshExtractResult
from trellis.representations.mesh.cube2mesh import MeshExtractResult

import numpy as np
import torch
import trimesh
import trimesh.visual
from PIL import Image

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"device: {device}\n")
torch.manual_seed(0)

from trellis.utils import postprocessing_utils as pp
from trellis.utils.render_utils import render_multiview

# --- build a synthetic colored-sphere Gaussian representation ---
N = 4000
aabb = [-1, -1, -1, 2, 2, 2]
g = Gaussian(aabb=aabb, sh_degree=0, scaling_activation="softplus", device=device)
dirs = torch.randn(N, 3, device=device)
dirs = dirs / dirs.norm(dim=-1, keepdim=True)
radius = 0.6
xyz = dirs * radius
g.from_xyz(xyz)
g.from_scaling(torch.full((N, 3), 0.04, device=device))
g.from_rotation(torch.tensor([[1., 0., 0., 0.]], device=device).expand(N, 4).contiguous())
g.from_opacity(torch.full((N, 1), 0.98, device=device))
colors = (dirs + 1) / 2  # color by normal direction, gives a visibly-varying sphere
g.from_features(colors.unsqueeze(1))

# --- matching mesh (icosphere, same radius) ---
tm = trimesh.creation.icosphere(subdivisions=2, radius=radius)
mesh = MeshExtractResult(
    vertices=torch.tensor(tm.vertices, dtype=torch.float32, device=device),
    faces=torch.tensor(tm.faces, dtype=torch.int64, device=device),
)
print(f"mesh: {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces; gaussians: {N}\n")

# --- reproduce to_glb()'s logic with reduced view counts / resolution for tractable runtime ---
simplify = 0.5
t0 = time.time()
vertices = mesh.vertices.cpu().numpy()
faces = mesh.faces.cpu().numpy()

vertices, faces = pp.postprocess_mesh(
    vertices, faces,
    simplify=True, simplify_ratio=simplify,
    fill_holes=True,
    fill_holes_max_hole_size=0.04,
    fill_holes_max_hole_nbe=int(250 * np.sqrt(1 - simplify)),
    fill_holes_resolution=128, fill_holes_num_views=20,
    verbose=True, device=device,
)
print(f"after postprocess: {vertices.shape[0]} verts, {faces.shape[0]} faces  ({time.time()-t0:.1f}s)")

vertices, faces, uvs = pp.parametrize_mesh(vertices, faces)
print(f"after parametrize: {vertices.shape[0]} verts, {faces.shape[0]} faces, uv {uvs.shape}  ({time.time()-t0:.1f}s)")

observations, extrinsics, intrinsics = render_multiview(g, resolution=256, nviews=12)
print(f"rendered {len(observations)} views  ({time.time()-t0:.1f}s)")
masks = [np.any(observation > 0, axis=-1) for observation in observations]
extrinsics = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
intrinsics = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]

texture = pp.bake_texture(
    vertices, faces, uvs,
    observations, masks, extrinsics, intrinsics,
    texture_size=512, mode='opt', lambda_tv=0.01,
    verbose=True, device=device,
)
print(f"baked texture {texture.shape}  ({time.time()-t0:.1f}s)")
texture_img = Image.fromarray(texture)

vertices_rot = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
material = trimesh.visual.material.PBRMaterial(
    roughnessFactor=1.0, baseColorTexture=texture_img,
    baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8)
)
out_mesh = trimesh.Trimesh(vertices_rot, faces, visual=trimesh.visual.TextureVisuals(uv=uvs, material=material))
print(f"\nfinal trimesh: {out_mesh.vertices.shape[0]} verts, {out_mesh.faces.shape[0]} faces  ({time.time()-t0:.1f}s total)")

import tempfile, os
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "out.glb")
    out_mesh.export(path)
    size = os.path.getsize(path)
    print(f"exported glb: {size} bytes")

ok = out_mesh.vertices.shape[0] > 0 and out_mesh.faces.shape[0] > 0 and size > 1000
print("\n" + ("FULL PIPELINE PASSED" if ok else "FULL PIPELINE FAILED"))
