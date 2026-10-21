import sys, types
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
_root = '/Users/vinayak/Downloads/TRELLIS-main/trellis'
def fake_pkg(name, path):
    m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m; return m
fake_pkg('trellis', _root)
_repr_pkg = fake_pkg('trellis.representations', _root + '/representations')
import trellis.representations.mesh as mesh_mod
_repr_pkg.MeshExtractResult = mesh_mod.MeshExtractResult
_repr_pkg.Gaussian = None  # placeholder, real import below
from trellis.representations.gaussian.gaussian_model import Gaussian
_repr_pkg.Gaussian = Gaussian
_repr_pkg.Strivec = object  # unused stub, postprocessing_utils only needs it for a type hint
from trellis.representations.octree.octree_dfs import DfsOctree
_repr_pkg.Octree = DfsOctree

import numpy as np
import torch
import trimesh

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"device: {device}\n")

from trellis.utils import postprocessing_utils as pp

# icosphere-ish test mesh via trimesh (real, non-trivial topology, unlike the cube)
tm = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
vertices = tm.vertices.astype(np.float32)
faces = tm.faces.astype(np.int64)
print(f"test mesh: {vertices.shape[0]} verts, {faces.shape[0]} faces")

print("\n=== parametrize_mesh (xatlas) ===")
v2, f2, uvs = pp.parametrize_mesh(vertices, faces)
print(f"output: {v2.shape[0]} verts, {f2.shape[0]} faces, uvs {uvs.shape}")
print(f"uv range: [{uvs.min():.3f}, {uvs.max():.3f}]")
ok1 = uvs.shape[1] == 2 and np.isfinite(uvs).all() and f2.shape[0] > 0
print(f"PASS: {ok1}")

print("\n=== postprocess_mesh (pyvista decimate + pymeshfix/igraph hole-fill via _fill_holes on MPS) ===")
v3, f3 = pp.postprocess_mesh(
    vertices, faces,
    simplify=True, simplify_ratio=0.5,
    fill_holes=True, fill_holes_resolution=64, fill_holes_num_views=20,
    verbose=True, device=device,
)
print(f"output: {v3.shape[0]} verts, {f3.shape[0]} faces")
ok2 = v3.shape[0] > 0 and f3.shape[0] > 0 and np.isfinite(v3).all()
print(f"PASS: {ok2}")

print("\n" + ("STAGE 1 CHECKS PASSED" if (ok1 and ok2) else "SOME CHECKS FAILED"))
