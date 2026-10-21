"""
postprocess_mesh's pyvista decimate(0.95) call silently no-op'd on the dragon mesh (658136 faces
in, 658136 faces out) while working fine on vase (692142 -> 34603, a real ~95% reduction). This
investigates why, using CPU only (the dragon pipeline run may still be using MPS for the
multiview-render/texture-bake stages when this runs).

Checks, in order:
  1. Does the raw decoded mesh have degenerate (zero-area) triangles, duplicate vertices, or
     non-manifold edges (an edge shared by != 2 triangles)? vtkQuadricDecimation (what pyvista's
     default .decimate() wraps) can refuse to simplify around non-manifold/boundary geometry.
  2. Does decimate_pro() (vtkDecimatePro, a different algorithm/pyvista method) behave differently
     on the same mesh?
  3. Does trimesh's independent quadric decimation (open3d-free, via
     trimesh.Trimesh.simplify_quadric_decimation if available, else skip) reduce it, to tell apart
     "this mesh is fundamentally hard to decimate" from "pyvista/VTK's specific algorithm bails out
     on it".
"""
import os, sys
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import numpy as np
import torch
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
import pyvista as pv

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"=== device: {device} (forced, to avoid GPU contention with the live run) ===", flush=True)

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)

image = Image.open("assets/example_image/typical_creature_dragon.png")
outputs = pipeline.run(image, seed=1, formats=['mesh'])
mesh = outputs['mesh'][0]
vertices_np = mesh.vertices.cpu().numpy()
faces_np = mesh.faces.cpu().numpy()
print(f"raw mesh: {vertices_np.shape[0]}v / {faces_np.shape[0]}f\n", flush=True)

np.savez("outputs/dragon_raw_mesh.npz", vertices=vertices_np.astype(np.float32), faces=faces_np.astype(np.int32))

# === Check 1: degenerate triangles, duplicate vertices, non-manifold edges ===
v0, v1, v2 = vertices_np[faces_np[:, 0]], vertices_np[faces_np[:, 1]], vertices_np[faces_np[:, 2]]
cross = np.cross(v1 - v0, v2 - v0)
areas = 0.5 * np.linalg.norm(cross, axis=1)
n_degenerate = int((areas < 1e-12).sum())
print(f"[check 1] degenerate (near-zero-area) triangles: {n_degenerate} / {faces_np.shape[0]} "
      f"({100*n_degenerate/faces_np.shape[0]:.3f}%)")

uniq_verts, inv, counts = np.unique(vertices_np, axis=0, return_inverse=True, return_counts=True)
n_dup_verts = vertices_np.shape[0] - uniq_verts.shape[0]
print(f"[check 1] duplicate vertex positions: {n_dup_verts} (mesh has {vertices_np.shape[0]} verts, "
      f"{uniq_verts.shape[0]} unique positions)")

edges = np.concatenate([
    faces_np[:, [0, 1]], faces_np[:, [1, 2]], faces_np[:, [2, 0]],
], axis=0)
edges_sorted = np.sort(edges, axis=1)
_, edge_counts = np.unique(edges_sorted, axis=0, return_counts=True)
n_manifold = int((edge_counts == 2).sum())
n_boundary = int((edge_counts == 1).sum())
n_nonmanifold = int((edge_counts > 2).sum())
print(f"[check 1] edges: {n_manifold} manifold (shared by exactly 2 faces), {n_boundary} boundary "
      f"(1 face, i.e. a hole), {n_nonmanifold} non-manifold (>2 faces sharing an edge)")

# === Check 2: pyvista decimate() (vtkQuadricDecimation) vs decimate_pro() (vtkDecimatePro) ===
pvmesh = pv.PolyData(vertices_np, np.concatenate([np.full((faces_np.shape[0], 1), 3), faces_np], axis=1))
print(f"\n[check 2] pyvista mesh: is_manifold={pvmesh.is_manifold}, n_open_edges={pvmesh.n_open_edges}")

qd = pvmesh.decimate(0.95)
print(f"[check 2] decimate() [vtkQuadricDecimation], target_reduction=0.95: "
      f"{pvmesh.n_faces_strict} -> {qd.n_faces_strict} faces "
      f"({100*(1 - qd.n_faces_strict/pvmesh.n_faces_strict):.1f}% actually reduced)")

try:
    dp = pvmesh.decimate_pro(0.95)
    print(f"[check 2] decimate_pro() [vtkDecimatePro], target_reduction=0.95: "
          f"{pvmesh.n_faces_strict} -> {dp.n_faces_strict} faces "
          f"({100*(1 - dp.n_faces_strict/pvmesh.n_faces_strict):.1f}% actually reduced)")
except Exception as e:
    print(f"[check 2] decimate_pro() raised: {e}")

# === Check 3: does splitting non-manifold-adjacent vs clean regions change the picture? ===
# If most degenerate/non-manifold structure clusters in a few components (e.g. thin
# spikes/horns/claws typical of a dragon mesh), report component count as a hint.
try:
    conn = pvmesh.connectivity()
    n_components = len(np.unique(conn['RegionId']))
    print(f"\n[check 3] connected components in raw mesh: {n_components}")
except Exception as e:
    print(f"[check 3] connectivity() raised: {e}")

# === Check 4: independent decimator (trimesh) for comparison ===
import trimesh
tmesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)
if hasattr(tmesh, 'simplify_quadric_decimation'):
    try:
        target_faces = int(faces_np.shape[0] * 0.05)
        tsimplified = tmesh.simplify_quadric_decimation(face_count=target_faces)
        print(f"\n[check 4] trimesh simplify_quadric_decimation(target={target_faces}): "
              f"{faces_np.shape[0]} -> {tsimplified.faces.shape[0]} faces")
    except Exception as e:
        print(f"\n[check 4] trimesh simplify_quadric_decimation raised: {e}")
else:
    print("\n[check 4] trimesh.Trimesh has no simplify_quadric_decimation in this version -- skipped")

print("\n=== done ===")
