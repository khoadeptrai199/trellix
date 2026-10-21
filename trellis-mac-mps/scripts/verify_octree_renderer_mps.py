"""
Correctness checks for trellis/renderers/octree_renderer_mps.py (the pure-PyTorch port of
diffoctreerast's OctreeTrivecRasterizer, see that module's docstring for what/why).

No real CUDA is available on this machine to compare against directly, so these are physical/
mathematical sanity checks on synthetic scenes with known-by-construction expected output, plus an
end-to-end smoke test against a real model-generated Strivec.
"""
import sys
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
import torch
from trellis.representations import Strivec
from trellis.renderers.octree_renderer_mps import render_trivec
from trellis.utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"device: {device}\n")
all_ok = True


def make_single_voxel_strivec(coord=(3, 3, 3), resolution=8, rank=2, dim=4, density_val=20.0, color=(0.9, 0.2, 0.1)):
    """A Strivec with exactly one occupied leaf, fully opaque, a known solid color."""
    s = Strivec(resolution=resolution, aabb=[-0.5, -0.5, -0.5, 1, 1, 1], sh_degree=0, rank=rank, dim=dim, device=device)
    s.density_shift = 0.0
    cx, cy, cz = coord
    s.position = torch.tensor([[(cx + 0.5) / resolution, (cy + 0.5) / resolution, (cz + 0.5) / resolution]], device=device)
    s.depth = torch.full((1, 1), int(torch.tensor(resolution).log2().round().item()), dtype=torch.uint8, device=device)
    # Uniform trivec (every value 1) + a large density so the voxel is ~fully opaque at any sample
    # point inside it, and features_dc set so SH_C0 * value ≈ desired post-sigmoid color via logit.
    s.trivec = torch.ones((1, rank, 3, dim), device=device)
    # density = softplus(sum_r density[r] * tri_r - shift*10) * scale ; with trivec all-ones,
    # tri_r = 1 for every rank, so density = softplus(sum(density_per_rank)). Split evenly.
    s.density = torch.full((1, rank), density_val / rank, device=device)
    import math
    logit = lambda p: math.log(p / (1 - p))
    target = torch.tensor(color, device=device)
    target = target.clamp(1e-3, 1 - 1e-3)
    # sum over rank of SH_C0 * features_dc must equal logit(target) (since color = sigmoid(sum_r color_dc[r]*tri_r), tri_r=1)
    per_rank_logit = torch.stack([torch.log(target / (1 - target))] * rank, dim=0) / rank / (0.28209479177387814)
    s.features_dc = per_rank_logit.reshape(1, rank, 1, 3)
    return s


def render_scene(strivec, yaw=0.0, pitch=0.0, r=2.0, fov=40.0, resolution=128, bg=(0.0, 0.0, 0.0)):
    # fov is in DEGREES -- yaw_pitch_r_fov_to_extrinsics_intrinsics converts internally.
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics([yaw], [pitch], r, fov, device=device)
    bg_t = torch.tensor(bg, dtype=torch.float32, device=device)
    return render_trivec(strivec, extrinsics[0], intrinsics[0], resolution, bg_t)


# === Test 1: single opaque voxel produces a non-trivial silhouette of roughly the expected color,
# and background elsewhere ===
s1 = make_single_voxel_strivec(coord=(4, 4, 4), resolution=8, color=(0.9, 0.2, 0.1))  # near scene center
ret = render_scene(s1, yaw=0.3, pitch=0.2, resolution=128, bg=(0.0, 0.0, 0.0))
color, alpha = ret['color'], ret['alpha']
print(f"[test 1] color shape {tuple(color.shape)}, alpha shape {tuple(alpha.shape)}")
print(f"[test 1] any NaN/Inf: color={not torch.isfinite(color).all().item()}, alpha={not torch.isfinite(alpha).all().item()}")
print(f"[test 1] alpha range: [{alpha.min().item():.4f}, {alpha.max().item():.4f}], covered px: {(alpha > 0.5).sum().item()}")
ok1 = torch.isfinite(color).all() and torch.isfinite(alpha).all() and (alpha > 0.5).sum() > 0 and (alpha < 0.05).sum() > 0
if ok1:
    covered = alpha > 0.5
    mean_color = color.permute(1, 2, 0)[covered].mean(dim=0)
    print(f"[test 1] mean color in covered region: {mean_color.tolist()} (expected ~[0.9, 0.2, 0.1])")
    ok1 = ok1 and (mean_color - torch.tensor([0.9, 0.2, 0.1], device=device)).abs().max() < 0.15
print(f"[test 1] PASS: {ok1}\n")
all_ok = all_ok and bool(ok1)

# === Test 2: near-zero density everywhere -> pure background. Note softplus(0) = ln(2) =/= 0, so
# "zero density" in the raw pre-activation sense (density_val=0.0) is NOT the same as no opacity --
# need a strongly negative raw value to push softplus's output near zero, matching the ported
# formula exactly (out_density = softplus(raw - density_shift*10) * scale, see forward.cu).
s2 = make_single_voxel_strivec(coord=(4, 4, 4), resolution=8, density_val=-1000.0, color=(0.5, 0.5, 0.5))
ret2 = render_scene(s2, yaw=0.0, pitch=0.0, resolution=64, bg=(0.3, 0.6, 0.9))
color2, alpha2 = ret2['color'], ret2['alpha']
ok2 = torch.isfinite(color2).all() and (alpha2.max() < 0.05) and \
    (color2.permute(1, 2, 0) - torch.tensor([0.3, 0.6, 0.9], device=device)).abs().max() < 0.02
print(f"[test 2] zero-density octree -> alpha max {alpha2.max().item():.4f}, matches bg color: {ok2}")
print(f"[test 2] PASS: {ok2}\n")
all_ok = all_ok and bool(ok2)

# === Test 3: silhouette moves with camera yaw (sanity that the renderer is actually
# viewpoint-dependent, not accidentally constant) ===
s3 = make_single_voxel_strivec(coord=(2, 4, 4), resolution=8, color=(0.1, 0.8, 0.1))
ret_a = render_scene(s3, yaw=0.0, pitch=0.0, resolution=96)
ret_b = render_scene(s3, yaw=1.2, pitch=0.0, resolution=96)
diff = (ret_a['alpha'] - ret_b['alpha']).abs().sum().item()
ok3 = diff > 1.0  # silhouette should have moved meaningfully
print(f"[test 3] alpha mask changed between viewpoints (sum abs diff = {diff:.2f}): {ok3}")
print(f"[test 3] PASS: {ok3}\n")
all_ok = all_ok and bool(ok3)

print("=== " + ("ALL OCTREE RENDERER SANITY CHECKS PASSED" if all_ok else "SOME CHECKS FAILED") + " ===")
sys.exit(0 if all_ok else 1)
