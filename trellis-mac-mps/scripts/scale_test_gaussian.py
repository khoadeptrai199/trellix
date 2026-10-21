import sys, types, time, math
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
_root = '/Users/vinayak/Downloads/TRELLIS-main/trellis'
def fake_pkg(name, path):
    m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m; return m
fake_pkg('trellis', _root)
_repr_pkg = fake_pkg('trellis.representations', _root + '/representations')
from trellis.representations.gaussian.gaussian_model import Gaussian
_repr_pkg.Gaussian = Gaussian
fake_pkg('trellis.renderers', _root + '/renderers')

import torch
import trellis.renderers.metal_gaussian_rasterizer as metal_impl
from trellis.renderers.gaussian_render import intrinsics_to_projection

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"device: {device}, metal available: {metal_impl.available()}\n", flush=True)
torch.manual_seed(0)

# match the real run's scale exactly: 478,336 splats, 1024x1024 (bake_texture's actual resolution)
N = 478336
aabb = [-1, -1, -1, 2, 2, 2]
g = Gaussian(aabb=aabb, sh_degree=0, scaling_activation="softplus", device=device)
dirs = torch.randn(N, 3, device=device)
dirs = dirs / dirs.norm(dim=-1, keepdim=True)
radii_scatter = torch.rand(N, device=device) ** 0.5
xyz = dirs * radii_scatter.unsqueeze(-1) * 0.4
g.from_xyz(xyz)
g.from_scaling(torch.full((N, 3), 0.01, device=device))
g.from_rotation(torch.tensor([[1., 0., 0., 0.]], device=device).expand(N, 4).contiguous())
g.from_opacity(torch.rand(N, 1, device=device) * 0.8 + 0.15)
colors = (dirs + 1) / 2
g.from_features(colors.unsqueeze(1))

resolution, near, far = 1024, 0.1, 10.0
extrinsics = torch.eye(4, device=device)
intrinsics = torch.tensor([[0.7, 0, 0.5], [0, 0.7, 0.5], [0, 0, 1.0]], device=device)
perspective = intrinsics_to_projection(intrinsics, near, far)
focalx, focaly = intrinsics[0, 0], intrinsics[1, 1]
fovx = 2 * torch.atan(0.5 / focalx)
fovy = 2 * torch.atan(0.5 / focaly)

settings = metal_impl.GaussianRasterizationSettings(
    image_height=resolution, image_width=resolution,
    tanfovx=math.tan(fovx.item() * 0.5), tanfovy=math.tan(fovy.item() * 0.5),
    kernel_size=0.1, subpixel_offset=None,
    bg=torch.tensor([0.1, 0.1, 0.1], device=device), scale_modifier=1.0,
    viewmatrix=extrinsics.T.contiguous(), projmatrix=(perspective @ extrinsics).T.contiguous(),
    sh_degree=g.active_sh_degree, campos=torch.inverse(extrinsics)[:3, 3],
    prefiltered=False, debug=False,
)

print(f"testing at REAL scale: N={N}, resolution={resolution}x{resolution}", flush=True)
t0 = time.time()
img, radii = metal_impl.rasterize_gaussians(
    g.get_xyz, g.get_opacity, settings, shs=g.get_features, scales=g.get_scaling, rotations=g.get_rotation,
)
if device == 'mps':
    torch.mps.synchronize()
elapsed = time.time() - t0
print(f"single view rendered in {elapsed:.2f}s (previously: 545.11s on the old unbatched-sync code)", flush=True)
print(f"finite: {torch.isfinite(img).all().item()}, shape: {tuple(img.shape)}", flush=True)

# second call to see steady-state (first call may include one-time MPS graph compilation)
t0 = time.time()
img2, _ = metal_impl.rasterize_gaussians(
    g.get_xyz, g.get_opacity, settings, shs=g.get_features, scales=g.get_scaling, rotations=g.get_rotation,
)
if device == 'mps':
    torch.mps.synchronize()
elapsed2 = time.time() - t0
print(f"second view rendered in {elapsed2:.2f}s (steady-state estimate)", flush=True)
print(f"projected time for 100 views: {elapsed2*100/60:.1f} minutes", flush=True)
