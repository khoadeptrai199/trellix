import os
import sys, types, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trellis')
def fake_pkg(name, path):
    m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m; return m
fake_pkg('trellis', _root)
fake_pkg('trellis.renderers', _root + '/renderers')

import torch
import importlib.util

def load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, _root + relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

sw = load('trellis.renderers.sw_rasterizer', '/renderers/sw_rasterizer.py')
mr = load('trellis.renderers.metal_rasterizer', '/renderers/metal_rasterizer.py')

print("Metal available:", mr.available())
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"device: {device}\n")

# === Test 1: same overlapping-triangles scene used to verify sw_rasterizer originally ===
H, W = 64, 64
verts = torch.tensor([
    [-0.5, -0.5, 0.5, 1.0], [0.5, -0.5, 0.5, 1.0], [0.0, 0.5, 0.5, 1.0],
    [-1.0, -1.0, 0.9, 1.0], [1.0, -1.0, 0.9, 1.0], [1.0, 1.0, 0.9, 1.0], [-1.0, 1.0, 0.9, 1.0],
], dtype=torch.float32, device=device).unsqueeze(0)
tri = torch.tensor([[0,1,2],[3,4,5],[3,5,6]], dtype=torch.int32, device=device)

rast_ref, _ = sw.rasterize(None, verts, tri, (H, W))
rast_metal, _ = mr.rasterize(None, verts, tri, (H, W))

diff = (rast_ref - rast_metal).abs()
print(f"[unit scene] max abs diff vs sw_rasterizer: {diff.max().item():.3e}")
print(f"[unit scene] triangle_id exact match: {torch.equal(rast_ref[...,3], rast_metal[...,3])}")
ok1 = diff.max().item() < 1e-4 and torch.equal(rast_ref[...,3], rast_metal[...,3])
print(f"[unit scene] PASS: {ok1}\n")

# === Test 2: random mesh, larger scale, compare full rast + timing ===
torch.manual_seed(0)
V = 5000
F = 9000
rand_verts = torch.rand(V, 3, device=device) * 2 - 1
rand_verts[:, 2] = rand_verts[:, 2] * 0.3 + 2.0  # push in front of camera, moderate depth spread
rand_verts_h = torch.cat([rand_verts, torch.ones(V, 1, device=device)], dim=-1).unsqueeze(0)
rand_tri = torch.randint(0, V, (F, 3), dtype=torch.int32, device=device)

H2, W2 = 256, 256
t0 = time.time()
rast_ref2, _ = sw.rasterize(None, rand_verts_h, rand_tri, (H2, W2))
if device == 'mps': torch.mps.synchronize()
t_ref = time.time() - t0

t0 = time.time()
rast_metal2, _ = mr.rasterize(None, rand_verts_h, rand_tri, (H2, W2))
if device == 'mps': torch.mps.synchronize()
t_metal = time.time() - t0

diff2 = (rast_ref2 - rast_metal2).abs()
tri_id_match = (rast_ref2[...,3] == rast_metal2[...,3]).float().mean().item()
print(f"[random mesh, {V}v/{F}f @ {H2}x{W2}] sw_rasterizer time: {t_ref:.3f}s, metal time: {t_metal:.3f}s, speedup: {t_ref/t_metal:.1f}x")
print(f"[random mesh] max abs diff (non-triangle_id channels, where tri_id matches): ", end="")
match_mask = (rast_ref2[...,3] == rast_metal2[...,3]).unsqueeze(-1).expand_as(diff2)
print(f"{diff2[match_mask].max().item():.3e}")
print(f"[random mesh] triangle_id agreement fraction: {tri_id_match:.6f}")
ok2 = tri_id_match > 0.999
print(f"[random mesh] PASS: {ok2}")

print("\n" + ("ALL METAL RASTERIZER CHECKS PASSED" if (ok1 and ok2) else "SOME CHECKS FAILED"))
