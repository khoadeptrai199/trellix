import sys, types
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
_root = '/Users/vinayak/Downloads/TRELLIS-main/trellis'
def fake_pkg(name, path):
    m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m; return m
fake_pkg('trellis', _root)
fake_pkg('trellis.renderers', _root + '/renderers')
_repr_pkg = fake_pkg('trellis.representations', _root + '/representations')

import torch
from trellis.renderers.gaussian_render import GaussianRenderer, render as low_render, intrinsics_to_projection
from trellis.representations.gaussian.gaussian_model import Gaussian
import trellis.renderers.gaussian_rasterizer_pytorch as gr
import math

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"device: {device}\n")
torch.manual_seed(0)

# === Test 1: brute-force sequential-compositing reference vs chunked cumprod implementation ===
# Build a tiny scene by hand directly at the rasterize_gaussians() level (bypass camera/SH plumbing
# so this isolates JUST the alpha-compositing math).
H, W = 32, 32
N = 7
px = torch.rand(N, device=device) * W
py = torch.rand(N, device=device) * H
z = torch.linspace(1.0, 7.0, N, device=device)  # distinct depths
inv_a = torch.full((N,), 0.02, device=device)
inv_b = torch.zeros(N, device=device)
inv_c = torch.full((N,), 0.02, device=device)
opacity = torch.rand(N, device=device) * 0.8 + 0.1
color = torch.rand(N, 3, device=device)

order = torch.argsort(z)
px, py, inv_a, inv_b, inv_c, opacity, color = [t[order] for t in (px, py, inv_a, inv_b, inv_c, opacity, color)]

xs = (torch.arange(W, device=device, dtype=torch.float32) + 0.5).view(1, 1, W)
ys = (torch.arange(H, device=device, dtype=torch.float32) + 0.5).view(1, H, 1)

# chunked cumprod path (chunk_size=3, forces multiple chunks to exercise cross-chunk transmittance carry-over)
accum_color = torch.zeros(H, W, 3, device=device)
trans = torch.ones(H, W, device=device)
chunk_size = 3
for start in range(0, N, chunk_size):
    end = min(start + chunk_size, N)
    B = end - start
    dx = xs - px[start:end].view(B, 1, 1)
    dy = ys - py[start:end].view(B, 1, 1)
    maha2 = inv_a[start:end].view(B,1,1)*dx*dx + 2*inv_b[start:end].view(B,1,1)*dx*dy + inv_c[start:end].view(B,1,1)*dy*dy
    alpha = (opacity[start:end].view(B,1,1) * torch.exp(-0.5*maha2)).clamp(max=0.9999)
    chunk_color, chunk_trans_end = gr._composite_chunk(alpha, color[start:end].view(B,1,1,3))
    accum_color += trans.unsqueeze(-1) * chunk_color
    trans = trans * chunk_trans_end
chunked_result = accum_color  # (ignore background term for this isolated comparison)

# brute-force fully sequential reference (one gaussian at a time, explicit Python loop, no chunking/cumprod at all)
ref_color = torch.zeros(H, W, 3, device=device)
ref_trans = torch.ones(H, W, device=device)
for i in range(N):
    dx = xs - px[i]
    dy = ys - py[i]
    maha2 = inv_a[i]*dx*dx + 2*inv_b[i]*dx*dy + inv_c[i]*dy*dy
    alpha = (opacity[i] * torch.exp(-0.5*maha2)).clamp(max=0.9999)[0]
    ref_color += ref_trans.unsqueeze(-1) * alpha.unsqueeze(-1) * color[i].view(1,1,3)
    ref_trans = ref_trans * (1 - alpha)

diff = (chunked_result - ref_color).abs().max().item()
print(f"[compositing math] max abs diff, chunked+cumprod vs fully-sequential reference: {diff:.3e}")
ok1 = diff < 1e-4
print(f"[compositing math] PASS: {ok1}\n")

# === Test 2: full GaussianRenderer pipeline -- single Gaussian at scene center, check it appears ===
aabb = [-1, -1, -1, 2, 2, 2]
g = Gaussian(aabb=aabb, sh_degree=0, scaling_activation="softplus", device=device)
g.from_xyz(torch.tensor([[0.5]], device=device).expand(1, 3) * 0 + torch.tensor([[0.5, 0.5, 0.5]], device=device))  # center of aabb -> world (0,0,3)... compute below
# aabb is [-1,-1,-1, 2,2,2] meaning get_xyz = _xyz*aabb[3:]+aabb[:3]; we want world xyz=(0,0,3).
# from_xyz stores _xyz = (xyz - aabb[:3])/aabb[3:], so pass world xyz directly:
world_xyz = torch.tensor([[0.0, 0.0, 3.0]], device=device)
g.from_xyz(world_xyz)
g.from_scaling(torch.tensor([[0.05, 0.05, 0.05]], device=device))
g.from_rotation(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device))
g.from_opacity(torch.tensor([[0.95]], device=device))
g.from_features(torch.tensor([[[1.0], [0.2], [0.1]]], device=device).permute(0, 2, 1) * 0 + torch.tensor([[[1.0, 0.2, 0.1]]], device=device))

extrinsics = torch.eye(4, device=device)
intrinsics = torch.tensor([[0.7, 0, 0.5], [0, 0.7, 0.5], [0, 0, 1.0]], device=device)
renderer = GaussianRenderer(rendering_options={"resolution": 128, "near": 0.1, "far": 10.0, "ssaa": 1, "bg_color": [0.0, 0.0, 0.0]})
out = renderer.render(g, extrinsics, intrinsics)
img = out['color']  # [3,H,W]
print(f"[single gaussian] image shape: {tuple(img.shape)}, finite: {torch.isfinite(img).all().item()}")

center_color = img[:, 64, 64]
corner_color = img[:, 5, 5]
print(f"[single gaussian] color at image center: {center_color.tolist()} (expect near red-ish [1.0,0.2,0.1] * opacity)")
print(f"[single gaussian] color at corner (background): {corner_color.tolist()} (expect [0,0,0])")
ok2 = torch.isfinite(img).all().item() and center_color[0].item() > 0.5 and corner_color.abs().max().item() < 1e-3
print(f"[single gaussian] PASS: {ok2}\n")

# === Test 3: occlusion -- nearer opaque gaussian should dominate a farther one at the same pixel ===
g2 = Gaussian(aabb=aabb, sh_degree=0, scaling_activation="softplus", device=device)
g2.from_xyz(torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]], device=device))
g2.from_scaling(torch.tensor([[0.1, 0.1, 0.1]] * 2, device=device))
g2.from_rotation(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device=device))
g2.from_opacity(torch.tensor([[0.999], [0.999]], device=device))
g2.from_features(torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]], device=device))  # near=red, far=green
out2 = renderer.render(g2, extrinsics, intrinsics)
center2 = out2['color'][:, 64, 64]
print(f"[occlusion] color at center (near=red opaque, far=green): {center2.tolist()} (expect mostly red)")
ok3 = center2[0].item() > center2[1].item()
print(f"[occlusion] PASS: {ok3}")

print("\n" + ("ALL GAUSSIAN RASTERIZER CHECKS PASSED" if (ok1 and ok2 and ok3) else "SOME CHECKS FAILED"))
