import sys, types
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
_root = '/Users/vinayak/Downloads/TRELLIS-main/trellis'
# Bypass trellis/__init__.py (pulls in the whole pipelines stack / easydict etc, unrelated to this test).
_trellis_pkg = types.ModuleType('trellis'); _trellis_pkg.__path__ = [_root]; sys.modules['trellis'] = _trellis_pkg
# trellis/representations/__init__.py eagerly imports .mesh -> flexicubes (a git submodule that
# isn't checked out in this download) and .radiance_field/.octree (unrelated to Gaussian). Bypass
# both package __init__ files and import gaussian_model directly, same trick as before.
_repr_pkg = types.ModuleType('trellis.representations'); _repr_pkg.__path__ = [_root + '/representations']
sys.modules['trellis.representations'] = _repr_pkg
_gauss_pkg = types.ModuleType('trellis.representations.gaussian'); _gauss_pkg.__path__ = [_root + '/representations/gaussian']
sys.modules['trellis.representations.gaussian'] = _gauss_pkg

import torch
from trellis.representations.gaussian.gaussian_model import Gaussian

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}\n")
torch.manual_seed(0)

N = 500
aabb = [-1, -1, -1, 2, 2, 2]  # [x0,y0,z0, dx,dy,dz]

g = Gaussian(aabb=aabb, sh_degree=0, scaling_activation="softplus", device=device)

# sanity: bias tensors landed on the right device (this is exactly what the .cuda() patch fixed)
for name in ["scale_bias", "rots_bias", "opacity_bias", "aabb"]:
    t = getattr(g, name)
    print(f"  {name}: device={t.device}")
    assert str(t.device).startswith(device), f"{name} landed on {t.device}, expected {device}"

xyz = torch.rand(N, 3, device=device)
scales = torch.rand(N, 3, device=device) * 0.1 + 0.01
rots = torch.nn.functional.normalize(torch.randn(N, 4, device=device), dim=-1)
opacities = torch.rand(N, 1, device=device)
features = torch.randn(N, 3, 1, device=device)

g.from_xyz(xyz)
g.from_scaling(scales)
g.from_rotation(rots)
g.from_opacity(opacities)
g.from_features(features)

def check(name, t):
    ok = torch.isfinite(t).all().item() and str(t.device).startswith(device)
    print(f"  {name}: shape={tuple(t.shape)} device={t.device} finite={torch.isfinite(t).all().item()} -> {'OK' if ok else 'FAIL'}")
    return ok

print("\n--- property getters (exercise the patched general_utils.py device derivation) ---")
all_ok = True
all_ok &= check("get_xyz", g.get_xyz)
all_ok &= check("get_scaling", g.get_scaling)
all_ok &= check("get_rotation", g.get_rotation)
all_ok &= check("get_opacity", g.get_opacity)
all_ok &= check("get_features", g.get_features)

print("\n--- get_covariance() -- calls build_scaling_rotation/strip_symmetric in general_utils.py ---")
cov = g.get_covariance()
all_ok &= check("get_covariance", cov)

print("\n--- round-trip through save_ply / load_ply, WITH the default (Y/Z-swap) transform ---")
import tempfile, os as _os
with tempfile.TemporaryDirectory() as d:
    path = _os.path.join(d, "test.ply")
    g.save_ply(path)  # default transform=[[1,0,0],[0,0,-1],[0,1,0]]
    print(f"  saved to {path}, size={_os.path.getsize(path)} bytes")
    g2 = Gaussian(aabb=aabb, sh_degree=0, scaling_activation="softplus", device=device)
    g2.load_ply(path)  # default transform, now fixed to correctly invert save_ply's transform
    all_ok &= check("reloaded get_xyz", g2.get_xyz)
    all_ok &= check("reloaded get_rotation", g2.get_rotation)
    err_xyz = (g2.get_xyz - g.get_xyz).abs().max().item()
    # get_rotation returns a quaternion; q and -q represent the same rotation, so compare via
    # the sign-invariant distance instead of raw difference.
    q1, q2_ = g.get_rotation, g2.get_rotation
    err_rot = torch.minimum((q1 - q2_).abs().amax(dim=-1), (q1 + q2_).abs().amax(dim=-1)).max().item()
    print(f"  max abs error xyz (save->load roundtrip through transform): {err_xyz:.6e}")
    print(f"  max abs error rotation (sign-invariant, save->load roundtrip through transform): {err_rot:.6e}")
    all_ok &= err_xyz < 1e-4
    all_ok &= err_rot < 1e-3

print("\n" + ("ALL GAUSSIAN MPS CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
