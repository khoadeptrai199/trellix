"""
Correctness check for octree_renderer_metal.py against the already-verified pure-PyTorch
octree_renderer_mps.py (scripts/verify_octree_renderer_mps.py) -- same algorithm, translated to
Metal Shading Language, so output should match closely (allowing for float32 accumulation order
differences between the two implementations, not exact bit-identity like the triangle rasterizer's
binned-vs-brute comparison, since this kernel restructures the marching loop's control flow).
"""
import sys, time
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
import torch
from trellis.representations import Strivec
from trellis.renderers import octree_renderer_mps as pt_impl
from trellis.renderers import octree_renderer_metal as metal_impl
from trellis.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Metal available: {metal_impl.available()}")
print(f"device: {device}\n")
all_ok = True


def make_multi_voxel_strivec(coords, resolution=8, rank=4, dim=8, device=device, seed=0):
    torch.manual_seed(seed)
    s = Strivec(resolution=resolution, aabb=[-0.5, -0.5, -0.5, 1, 1, 1], sh_degree=0, rank=rank, dim=dim, device=device)
    s.density_shift = 0.0
    n = len(coords)
    pos = torch.tensor([[(c[0] + 0.5) / resolution, (c[1] + 0.5) / resolution, (c[2] + 0.5) / resolution] for c in coords], device=device)
    s.position = pos
    s.depth = torch.full((n, 1), int(torch.tensor(resolution).log2().round().item()), dtype=torch.uint8, device=device)
    s.trivec = torch.randn((n, rank, 3, dim), device=device) * 0.5 + 1.0
    s.density = torch.rand((n, rank), device=device) * 10 + 2
    s.features_dc = (torch.rand((n, rank, 1, 3), device=device) - 0.5) * 4
    return s


def compare(name, strivec, yaw, pitch, resolution=96, r=2.0, fov=40.0, bg=(0.1, 0.15, 0.2)):
    global all_ok
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics([yaw], [pitch], r, fov, device=device)
    bg_t = torch.tensor(bg, dtype=torch.float32, device=device)

    t0 = time.time()
    ret_pt = pt_impl.render_trivec(strivec, extrinsics[0], intrinsics[0], resolution, bg_t)
    t_pt = time.time() - t0

    t0 = time.time()
    ret_metal = metal_impl.render_trivec(strivec, extrinsics[0], intrinsics[0], resolution, bg_t)
    t_metal = time.time() - t0

    color_diff = (ret_pt['color'] - ret_metal['color']).abs()
    alpha_diff = (ret_pt['alpha'] - ret_metal['alpha']).abs()
    max_color_diff = color_diff.max().item()
    max_alpha_diff = alpha_diff.max().item()
    mean_color_diff = color_diff.mean().item()

    ok = max_color_diff < 0.05 and max_alpha_diff < 0.05 and mean_color_diff < 0.01
    speedup = t_pt / t_metal if t_metal > 0 else float('inf')
    print(f"[{name}] pytorch: {t_pt:.3f}s, metal: {t_metal:.3f}s, speedup: {speedup:.1f}x")
    print(f"[{name}] max color diff: {max_color_diff:.4f}, max alpha diff: {max_alpha_diff:.4f}, mean color diff: {mean_color_diff:.5f}")
    print(f"[{name}] PASS: {ok}\n")
    all_ok = all_ok and ok
    return t_pt, t_metal


# === Test 1: sparse handful of voxels, several viewpoints ===
s1 = make_multi_voxel_strivec([(3, 3, 3), (3, 3, 4), (4, 3, 3), (2, 5, 2), (6, 1, 6)], resolution=8, rank=4, dim=8)
compare("sparse-8cells @ yaw=0", s1, 0.0, 0.0)
compare("sparse-8cells @ yaw=1.0,pitch=0.3", s1, 1.0, 0.3)

# === Test 2: a real-scale synthetic octree (similar order of magnitude to real model output) ===
torch.manual_seed(1)
coords2 = set()
while len(coords2) < 3000:
    coords2.add((torch.randint(0, 64, (1,)).item(), torch.randint(0, 64, (1,)).item(), torch.randint(0, 64, (1,)).item()))
s2 = make_multi_voxel_strivec(list(coords2), resolution=64, rank=16, dim=8, seed=2)
t_pt, t_metal = compare("real-scale-3000cells @ 256x256", s2, 0.4, 0.2, resolution=256)

print("=== " + ("ALL METAL OCTREE RENDERER CHECKS PASSED" if all_ok else "SOME CHECKS FAILED") + " ===")
sys.exit(0 if all_ok else 1)
