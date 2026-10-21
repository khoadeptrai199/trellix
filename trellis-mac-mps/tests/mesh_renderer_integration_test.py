import os, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'

import sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trellis')
def fake_pkg(name, path):
    m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m; return m

fake_pkg('trellis', _root)
fake_pkg('trellis.modules', _root + '/modules')
fake_pkg('trellis.models', _root + '/models')
fake_pkg('trellis.models.structured_latent_vae', _root + '/models/structured_latent_vae')
fake_pkg('trellis.renderers', _root + '/renderers')
_repr_pkg = fake_pkg('trellis.representations', _root + '/representations')
import trellis.representations.mesh as mesh_mod
_repr_pkg.MeshExtractResult = mesh_mod.MeshExtractResult

import torch
import trellis.modules.sparse as sp
from trellis.models.structured_latent_vae.decoder_mesh import SLatMeshDecoder
from trellis.renderers.mesh_renderer import MeshRenderer

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}\n")
torch.manual_seed(0)


def make_sparse_tensor(batch_sizes, extent, channels):
    coords_list = []
    for b, n in enumerate(batch_sizes):
        pts = torch.randint(0, extent, (n, 3))
        pts = torch.unique(pts, dim=0)
        bcol = torch.full((pts.shape[0], 1), b, dtype=torch.int32)
        coords_list.append(torch.cat([bcol, pts.int()], dim=1))
    coords = torch.cat(coords_list, dim=0).to(device)
    feats = torch.randn(coords.shape[0], channels, device=device)
    return sp.SparseTensor(feats=feats, coords=coords)


dec = SLatMeshDecoder(
    resolution=16, model_channels=256, latent_channels=8, num_blocks=2,
    num_heads=4, mlp_ratio=2, attn_mode="swin", window_size=8, pe_mode="ape",
    representation_config={"use_color": False},
).to(device)
dec.eval()

x = make_sparse_tensor([100], extent=16, channels=8)
with torch.no_grad():
    mesh = dec(x)[0]
print(f"decoded mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")

# normalize/center/scale the (randomly-weighted, so not a coherent shape) mesh into a
# camera-friendly range so it's actually framed in view
v = mesh.vertices
center = (v.min(0).values + v.max(0).values) / 2
scale = (v.max(0).values - v.min(0).values).max().clamp(min=1e-6)
mesh.vertices = (v - center) / scale  # now roughly in [-0.5, 0.5]
mesh.vertices = mesh.vertices + torch.tensor([0, 0, 3.0], device=device)  # push in front of camera
mesh.face_normal = mesh.comput_face_normals(mesh.vertices, mesh.faces)

extrinsics = torch.eye(4, device=device)
intrinsics = torch.tensor([[0.7, 0, 0.5], [0, 0.7, 0.5], [0, 0, 1.0]], device=device)
renderer = MeshRenderer(rendering_options={"resolution": 512, "near": 0.1, "far": 10.0, "ssaa": 1}, device=device)

t0 = time.time()
out = renderer.render(mesh, extrinsics, intrinsics, return_types=["mask", "depth", "normal"])
if device == 'mps':
    torch.mps.synchronize()
elapsed = time.time() - t0

mask = out["mask"]
covered = mask.float().mean().item()
print(f"render time @ 512x512, {mesh.faces.shape[0]} faces: {elapsed:.2f}s")
print(f"covered pixel fraction: {covered:.4f}")
print(f"mask finite: {torch.isfinite(mask).all().item()}, normal finite: {torch.isfinite(out['normal']).all().item()}")

ok = 0.0 < covered < 1.0 and torch.isfinite(mask).all().item() and torch.isfinite(out['normal']).all().item()
print("\n" + ("INTEGRATION TEST PASSED" if ok else "INTEGRATION TEST FAILED"))
