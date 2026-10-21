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
import trellis.renderers.gaussian_rasterizer_pytorch as pytorch_impl
import trellis.renderers.metal_gaussian_rasterizer as metal_impl
from trellis.renderers.gaussian_render import intrinsics_to_projection

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"device: {device}")
print(f"metal available: {metal_impl.available()}\n")
torch.manual_seed(0)

N = 50000
aabb = [-1, -1, -1, 2, 2, 2]
g = Gaussian(aabb=aabb, sh_degree=0, scaling_activation="softplus", device=device)
dirs = torch.randn(N, 3, device=device)
dirs = dirs / dirs.norm(dim=-1, keepdim=True)
radii_scatter = torch.rand(N, device=device) ** 0.5
xyz = dirs * radii_scatter.unsqueeze(-1) * 0.6
g.from_xyz(xyz)
g.from_scaling(torch.full((N, 3), 0.02, device=device))
g.from_rotation(torch.tensor([[1., 0., 0., 0.]], device=device).expand(N, 4).contiguous())
g.from_opacity(torch.rand(N, 1, device=device) * 0.8 + 0.15)
colors = (dirs + 1) / 2
g.from_features(colors.unsqueeze(1))

resolution, near, far = 256, 0.1, 10.0
extrinsics = torch.eye(4, device=device)
intrinsics = torch.tensor([[0.7, 0, 0.5], [0, 0.7, 0.5], [0, 0, 1.0]], device=device)
perspective = intrinsics_to_projection(intrinsics, near, far)
focalx, focaly = intrinsics[0, 0], intrinsics[1, 1]
fovx = 2 * torch.atan(0.5 / focalx)
fovy = 2 * torch.atan(0.5 / focaly)

settings = pytorch_impl.GaussianRasterizationSettings(
    image_height=resolution, image_width=resolution,
    tanfovx=math.tan(fovx.item() * 0.5), tanfovy=math.tan(fovy.item() * 0.5),
    kernel_size=0.1, subpixel_offset=None,
    bg=torch.tensor([0.1, 0.1, 0.1], device=device), scale_modifier=1.0,
    viewmatrix=extrinsics.T.contiguous(), projmatrix=(perspective @ extrinsics).T.contiguous(),
    sh_degree=g.active_sh_degree, campos=torch.inverse(extrinsics)[:3, 3],
    prefiltered=False, debug=False,
)

t0 = time.time()
img_pytorch, _ = pytorch_impl.rasterize_gaussians(
    g.get_xyz, g.get_opacity, settings, shs=g.get_features, scales=g.get_scaling, rotations=g.get_rotation,
)
t_pytorch = time.time() - t0
print(f"pure-pytorch: {t_pytorch:.2f}s, shape {tuple(img_pytorch.shape)}, finite {torch.isfinite(img_pytorch).all().item()}")

t0 = time.time()
img_metal, _ = metal_impl.rasterize_gaussians(
    g.get_xyz, g.get_opacity, settings, shs=g.get_features, scales=g.get_scaling, rotations=g.get_rotation,
)
t_metal = time.time() - t0
print(f"metal:        {t_metal:.2f}s, shape {tuple(img_metal.shape)}, finite {torch.isfinite(img_metal).all().item()}")

diff = (img_pytorch - img_metal).abs()
print(f"\nmax abs diff: {diff.max().item():.6f}")
print(f"mean abs diff: {diff.mean().item():.6f}")
print(f"speedup: {t_pytorch/t_metal:.1f}x")

ok = diff.max().item() < 0.01
print("\n" + ("PASS" if ok else "FAIL"))
