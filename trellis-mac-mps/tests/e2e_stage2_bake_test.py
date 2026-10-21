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

import numpy as np
import torch
import trimesh

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"device: {device}\n")
np.random.seed(0)

from trellis.utils import postprocessing_utils as pp

tm = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
vertices = tm.vertices.astype(np.float32)
faces = tm.faces.astype(np.int64)
_, faces2, uvs = pp.parametrize_mesh(vertices, faces)
vertices2 = vertices[np.unique(faces)]  # placeholder; real vmapping already applied inside parametrize_mesh
vertices2, faces2, uvs = pp.parametrize_mesh(vertices, faces)  # redo cleanly
print(f"parametrized: {vertices2.shape[0]} verts, {faces2.shape[0]} faces, uv shape {uvs.shape}")

# synthetic multi-view "observations": color the sphere by a simple function of vertex position
# projected per-view isn't trivial to fake perfectly, so instead render solid-colored views based
# on face index parity -- good enough to exercise the optimize-a-texture-from-views pipeline.
res = 64
n_views = 6
observations = []
masks = []
extrinsics = []
intrinsics = []
for i in range(n_views):
    img = np.zeros((res, res, 3), dtype=np.uint8)
    color = [(i * 40) % 200 + 30, (i * 80) % 200 + 30, (i * 120) % 200 + 30]  # never (0,0,0)
    img[16:48, 16:48] = color
    observations.append(img)
    masks.append((img.sum(-1) > 0))
    extr = np.eye(4, dtype=np.float32)
    extr[0, 3] = 0.1 * i
    extrinsics.append(extr)
    intr = np.array([[0.7, 0, 0.5], [0, 0.7, 0.5], [0, 0, 1.0]], dtype=np.float32)
    intrinsics.append(intr)

print("\n=== bake_texture (mode='opt', the gradient-descent path using our dr.texture) ===")
t0 = time.time()
texture = pp.bake_texture(
    vertices2, faces2, uvs,
    observations, masks, extrinsics, intrinsics,
    texture_size=256, mode='opt', lambda_tv=0.01,
    verbose=True, device=device,
)
elapsed = time.time() - t0
print(f"\nbake_texture (opt, 2500 steps) took {elapsed:.1f}s")
print(f"texture shape: {texture.shape}, dtype: {texture.dtype}, range: [{texture.min()}, {texture.max()}]")
ok_opt = texture.shape == (256, 256, 3) and texture.dtype == np.uint8 and texture.std() > 0

print("\n=== bake_texture (mode='fast', nearest-neighbor scatter path) ===")
texture_fast = pp.bake_texture(
    vertices2, faces2, uvs,
    observations, masks, extrinsics, intrinsics,
    texture_size=256, mode='fast',
    verbose=True, device=device,
)
print(f"texture_fast shape: {texture_fast.shape}, range: [{texture_fast.min()}, {texture_fast.max()}]")
ok_fast = texture_fast.shape == (256, 256, 3) and texture_fast.std() > 0

print("\n" + ("STAGE 2 CHECKS PASSED" if (ok_opt and ok_fast) else "SOME CHECKS FAILED"))
