import os
import sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trellis')
def fake_pkg(name, path):
    m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m; return m
fake_pkg('trellis', _root)

import torch
import importlib.util

spec = importlib.util.spec_from_file_location("sw_rasterizer", _root + "/renderers/sw_rasterizer.py")
sw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sw)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}\n")

H, W = 64, 64

# Triangle A: foreground, small, at ndc z=0.5 (nearer)
# Triangle B: background, full-screen quad (2 tris), at ndc z=0.9 (farther)
verts = torch.tensor([
    [-0.5, -0.5, 0.5, 1.0],   # 0: A corner
    [ 0.5, -0.5, 0.5, 1.0],   # 1: A corner
    [ 0.0,  0.5, 0.5, 1.0],   # 2: A apex
    [-1.0, -1.0, 0.9, 1.0],   # 3: B full-screen quad corner
    [ 1.0, -1.0, 0.9, 1.0],   # 4
    [ 1.0,  1.0, 0.9, 1.0],   # 5
    [-1.0,  1.0, 0.9, 1.0],   # 6
], dtype=torch.float32, device=device).unsqueeze(0)  # [1, V, 4]

tri = torch.tensor([
    [0, 1, 2],   # A
    [3, 4, 5],   # B tri 1
    [3, 5, 6],   # B tri 2
], dtype=torch.int32, device=device)

rast, rast_db = sw.rasterize(None, verts, tri, (H, W))
assert rast_db is None
assert rast.shape == (1, H, W, 4)

tri_id = rast[0, ..., 3].long()
zbuf = rast[0, ..., 2]

# Independent point-in-triangle check (not reusing rasterizer code) to know which pixels SHOULD
# belong to triangle A, in the same pixel-space convention the rasterizer claims to use.
def ndc_to_px(x, y):
    return (x * 0.5 + 0.5) * W, (1.0 - (y * 0.5 + 0.5)) * H

ax, ay = ndc_to_px(-0.5, -0.5)
bx, by = ndc_to_px(0.5, -0.5)
cx, cy = ndc_to_px(0.0, 0.5)

def sign(p1x, p1y, p2x, p2y, p3x, p3y):
    return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y)

ys_, xs_ = torch.meshgrid(torch.arange(H, device=device) + 0.5, torch.arange(W, device=device) + 0.5, indexing='ij')
d1 = sign(xs_, ys_, ax, ay, bx, by)
d2 = sign(xs_, ys_, bx, by, cx, cy)
d3 = sign(xs_, ys_, cx, cy, ax, ay)
has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
expected_in_A = ~(has_neg & has_pos)   # standard reference point-in-triangle test

print(f"expected pixels in triangle A: {expected_in_A.sum().item()}")
print(f"rasterizer pixels with tri_id==1 (triangle A): {(tri_id == 1).sum().item()}")
print(f"rasterizer pixels with tri_id>=2 (triangle B): {(tri_id >= 2).sum().item()}")
print(f"rasterizer background pixels (tri_id==0): {(tri_id == 0).sum().item()}")

all_ok = True

match = (tri_id == 1) == expected_in_A
mismatch_count = (~match).sum().item()
print(f"mismatched pixels vs independent reference test: {mismatch_count} / {H*W}")
all_ok &= mismatch_count == 0

# occlusion check: every pixel should be covered (B is full-screen), and where A is present it
# must win the z-test (A is nearer: ndc_z=0.5 < B's 0.9)
all_ok &= (tri_id > 0).all().item()
print(f"all pixels covered (B is full-screen): {(tri_id > 0).all().item()}")

a_pixels = tri_id == 1
b_pixels = tri_id >= 2
depth_ok = torch.allclose(zbuf[a_pixels], torch.full_like(zbuf[a_pixels], 0.5), atol=1e-4) and \
           torch.allclose(zbuf[b_pixels], torch.full_like(zbuf[b_pixels], 0.9), atol=1e-4)
print(f"depth values correct (A=0.5 nearer wins over B=0.9): {depth_ok}")
all_ok &= depth_ok

# interpolate: color = red (1,0,0) at A's verts, green (0,1,0) at B's verts -> should recover a
# solid-red patch on A and solid-green patch on B (constant color per vertex => constant interp)
attr = torch.tensor([
    [1., 0., 0.], [1., 0., 0.], [1., 0., 0.],   # A verts all red
    [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.],  # B verts all green
], device=device).unsqueeze(0)
out, out_da = sw.interpolate(attr, rast, tri)
assert out_da is None
assert out.shape == (1, H, W, 3)
red_ok = torch.allclose(out[0][a_pixels], torch.tensor([1., 0., 0.], device=device).expand(a_pixels.sum(), 3), atol=1e-4)
green_ok = torch.allclose(out[0][b_pixels], torch.tensor([0., 1., 0.], device=device).expand(b_pixels.sum(), 3), atol=1e-4)
print(f"interpolate recovers correct constant color on A (red): {red_ok}, on B (green): {green_ok}")
all_ok &= red_ok and green_ok

# antialias: identity passthrough, shape/dtype preserved
aa = sw.antialias(out, rast, verts, tri)
all_ok &= torch.equal(aa, out)
print(f"antialias identity passthrough: {torch.equal(aa, out)}")

print("\n" + ("ALL RASTERIZER UNIT TESTS PASSED" if all_ok else "SOME CHECKS FAILED"))
