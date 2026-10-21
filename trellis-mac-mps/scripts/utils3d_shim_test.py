import sys, types
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
_root = '/Users/vinayak/Downloads/TRELLIS-main/trellis'
def fake_pkg(name, path):
    m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m; return m
fake_pkg('trellis', _root)
fake_pkg('trellis.renderers', _root + '/renderers')

from trellis.renderers.nvdiffrast_shim import install
install()

import torch
import utils3d

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"device: {device}")
print(f"nvdiffrast.torch resolved to: {sys.modules['nvdiffrast.torch']}\n")

ctx = utils3d.torch.RastContext(backend='cuda', device=device)
print(f"RastContext created OK, nvd_ctx: {ctx.nvd_ctx}")

# simple quad (2 triangles), unit UVs
V = torch.tensor([[-0.5, -0.5, 3.0], [0.5, -0.5, 3.0], [0.5, 0.5, 3.0], [-0.5, 0.5, 3.0]], device=device)
F_ = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64, device=device)
uvs = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], device=device)

view = torch.eye(4, device=device)
intrinsics = torch.tensor([[0.7, 0, 0.5], [0, 0.7, 0.5], [0, 0, 1.0]], device=device)
projection = utils3d.torch.intrinsics_to_perspective(intrinsics, 0.1, 10.0)

rast = utils3d.torch.rasterize_triangle_faces(
    ctx, V[None], F_, 128, 128, uv=uvs[None], view=view, projection=projection
)
print(f"rasterize_triangle_faces keys: {list(rast.keys())}")
mask = rast['mask']
uv_map = rast['uv']
print(f"mask shape: {tuple(mask.shape)}, covered fraction: {mask.float().mean().item():.4f}")
print(f"uv shape: {tuple(uv_map.shape)}, finite: {torch.isfinite(uv_map).all().item()}")

ok1 = mask.float().mean().item() > 0.01 and torch.isfinite(uv_map).all().item()
print(f"[rasterize_triangle_faces via utils3d shim] PASS: {ok1}\n")

# now the exact call pattern bake_texture uses: dr.texture(texture, uv, uv_dr)[0]
import nvdiffrast.torch as dr
texture_img = torch.rand(1, 64, 64, 3, device=device, requires_grad=True)
uv_dr = rast['uv_dr']
sampled = dr.texture(texture_img, uv_map, uv_dr)
print(f"dr.texture(...) output shape: {tuple(sampled.shape)}, finite: {torch.isfinite(sampled).all().item()}")
loss = sampled.sum()
loss.backward()
ok2 = torch.isfinite(sampled).all().item() and texture_img.grad is not None and torch.isfinite(texture_img.grad).all().item()
print(f"[dr.texture direct call, matching bake_texture's usage] PASS: {ok2}")

print("\n" + ("ALL SHIM CHECKS PASSED" if (ok1 and ok2) else "SOME CHECKS FAILED"))
