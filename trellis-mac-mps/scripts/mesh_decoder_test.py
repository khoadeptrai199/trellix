import os
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'

import sys, types
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
_root = '/Users/vinayak/Downloads/TRELLIS-main/trellis'

def fake_pkg(name, path):
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m
    return m

_trellis_pkg = fake_pkg('trellis', _root)
_modules_pkg = fake_pkg('trellis.modules', _root + '/modules')
_models_pkg = fake_pkg('trellis.models', _root + '/models')
_slat_vae_pkg = fake_pkg('trellis.models.structured_latent_vae', _root + '/models/structured_latent_vae')
# trellis/representations/__init__.py eagerly imports Strivec/Octree/Gaussian too -- unrelated to
# mesh decoding. Bypass it, then real-import the (portable, already-verified) .mesh submodule and
# expose MeshExtractResult on the fake parent so `from ...representations import MeshExtractResult`
# (used inside decoder_mesh.py) resolves correctly.
_repr_pkg = fake_pkg('trellis.representations', _root + '/representations')
import trellis.representations.mesh as mesh_mod
_repr_pkg.MeshExtractResult = mesh_mod.MeshExtractResult

import torch
from trellis.models.structured_latent_vae.decoder_mesh import SLatMeshDecoder

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}\n")
torch.manual_seed(0)


def make_sparse_tensor(batch_sizes, extent, channels):
    import trellis.modules.sparse as sp
    coords_list = []
    for b, n in enumerate(batch_sizes):
        pts = torch.randint(0, extent, (n, 3))
        pts = torch.unique(pts, dim=0)
        bcol = torch.full((pts.shape[0], 1), b, dtype=torch.int32)
        coords_list.append(torch.cat([bcol, pts.int()], dim=1))
    coords = torch.cat(coords_list, dim=0).to(device)
    feats = torch.randn(coords.shape[0], channels, device=device)
    return sp.SparseTensor(feats=feats, coords=coords)


print("=== SLatMeshDecoder: sparse conv backbone + SparseSubdivide upsampling + FlexiCubes extraction ===")
dec = SLatMeshDecoder(
    resolution=16,
    model_channels=256,  # SparseSubdivideBlock3d hardcodes num_groups=32, needs model_channels//8 divisible by 32
    latent_channels=8,
    num_blocks=2,
    num_heads=4,
    mlp_ratio=2,
    attn_mode="swin",
    window_size=8,
    pe_mode="ape",
    representation_config={"use_color": False},
).to(device)
dec.eval()

# confirm the device-sync fix: mesh_extractor starts on the safe 'cpu' construction default...
print(f"  mesh_extractor.device before forward: {dec.mesh_extractor.device}")
assert torch.device(dec.mesh_extractor.device).type == 'cpu'

x = make_sparse_tensor([80, 65], extent=16, channels=8)
with torch.no_grad():
    meshes = dec(x)

# ...and gets re-pointed to the real device the first time it's actually used.
print(f"  mesh_extractor.device after forward:  {dec.mesh_extractor.device}")
assert torch.device(dec.mesh_extractor.device).type == device

assert len(meshes) == 2, f"expected 2 meshes (batch size 2), got {len(meshes)}"

all_ok = True
for i, mesh in enumerate(meshes):
    v, f = mesh.vertices, mesh.faces
    ok = (
        v.device.type == device and f.device.type == device
        and torch.isfinite(v).all().item()
        and torch.isfinite(mesh.face_normal).all().item()
    )
    print(f"  mesh[{i}]: success={mesh.success} vertices={tuple(v.shape)} faces={tuple(f.shape)} "
          f"device={v.device} finite={torch.isfinite(v).all().item()} -> {'OK' if ok else 'FAIL'}")
    all_ok &= ok

print("\n" + ("ALL MESH DECODER MPS CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
