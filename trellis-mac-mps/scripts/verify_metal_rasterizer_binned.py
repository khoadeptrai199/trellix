"""
Correctness + speed check for trellis/renderers/metal_rasterizer_binned.py against the (already
verified, see scripts/verify_metal_rasterizer.py) brute-force metal_rasterizer.py.

Claim being checked: binning is exact (every triangle whose AABB overlaps a tile is in that tile's
candidate list), so output must be bit-identical to the brute-force kernel for the same inputs --
this is not an approximation, so ANY mismatch here is a real bug, not an expected tradeoff.
"""
import sys, time
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
import torch
from trellis.renderers import sw_rasterizer as sw
from trellis.renderers import metal_rasterizer as brute
from trellis.renderers import metal_rasterizer_binned as binned

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Metal available (brute): {brute.available()}, (binned): {binned.available()}")
print(f"device: {device}\n")

all_ok = True


def check(name, verts, tri, H, W):
    global all_ok
    verts = verts.to(device)
    tri = tri.to(device)
    r_brute, _ = brute.rasterize(None, verts, tri, (H, W))
    r_binned, _ = binned.rasterize(None, verts, tri, (H, W))
    match = torch.equal(r_brute, r_binned)
    tri_id_match = torch.equal(r_brute[..., 3], r_binned[..., 3])
    if not match:
        diff = (r_brute - r_binned).abs()
        print(f"[{name}] MISMATCH -- max abs diff: {diff.max().item():.3e}, triangle_id match: {tri_id_match}")
        all_ok = False
    else:
        print(f"[{name}] PASS (bit-identical)")
    return match


# === Test 1: small unit scene (overlapping triangles, exercises z-buffer + winding) ===
verts = torch.tensor([
    [-0.5, -0.5, 0.5, 1.0], [0.5, -0.5, 0.5, 1.0], [0.0, 0.5, 0.5, 1.0],
    [-1.0, -1.0, 0.9, 1.0], [1.0, -1.0, 0.9, 1.0], [1.0, 1.0, 0.9, 1.0], [-1.0, 1.0, 0.9, 1.0],
], dtype=torch.float32).unsqueeze(0)
tri = torch.tensor([[0, 1, 2], [3, 4, 5], [3, 5, 6]], dtype=torch.int32)
check("unit scene 64x64", verts, tri, 64, 64)

# === Test 2: one triangle spanning the ENTIRE screen (exercises the MAX_TILE_SPAN "big triangle"
# fallback path -- must still be found in every tile it covers) ===
big_verts = torch.tensor([
    [-5.0, -5.0, 0.5, 1.0], [5.0, -5.0, 0.5, 1.0], [0.0, 5.0, 0.5, 1.0],
], dtype=torch.float32).unsqueeze(0)
big_tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)
check("full-screen triangle 256x256 (big-triangle fallback path)", big_verts, big_tri, 256, 256)

# === Test 3: random mesh, larger scale -- CORRECTNESS ONLY, not representative of real speed ===
# `torch.randint` triangles pick 3 *independently* random vertices per face, which produces mostly
# huge, screen-spanning triangles -- nothing like a real decimated mesh's small, locally-coherent
# faces. Binning's speed win depends entirely on triangles being small relative to a tile, so this
# scene is close to binning's worst case (most triangles hit the MAX_TILE_SPAN fallback and get
# added to every tile) -- expect binned to look *slower* here, that's not a regression. It's still
# useful as a correctness stress test (dense overlapping large triangles exercise the z-buffer and
# the big-triangle fallback path harder than Test 2's single triangle does).
# (tile-boundary-straddling triangles are unavoidable at this triangle density / TILE=32 ratio,
# so this also covers that case -- full bit-identical comparison either way.)
for (V, F, H2, W2) in [(5000, 9000, 256, 256), (18000, 34000, 1024, 1024)]:
    torch.manual_seed(0)
    rand_verts = torch.rand(V, 3) * 2 - 1
    rand_verts[:, 2] = rand_verts[:, 2] * 0.3 + 2.0
    rand_verts_h = torch.cat([rand_verts, torch.ones(V, 1)], dim=-1).unsqueeze(0)
    rand_tri = torch.randint(0, V, (F, 3), dtype=torch.int32)

    verts_d, tri_d = rand_verts_h.to(device), rand_tri.to(device)

    r_brute, _ = brute.rasterize(None, verts_d, tri_d, (H2, W2))
    r_binned, _ = binned.rasterize(None, verts_d, tri_d, (H2, W2))

    match = torch.equal(r_brute, r_binned)
    status = "PASS" if match else "MISMATCH"
    print(f"[random (unrealistic, worst-case) mesh {V}v/{F}f @ {H2}x{W2}] {status} (correctness only)")
    if not match:
        all_ok = False

# === Test 4: REAL mesh (locally-coherent triangles) -- this is the representative speed number.
# Requires outputs/vase_decimated_mesh.npz (see scripts/save_real_mesh_for_bench.py). Skipped if
# not present rather than failing the whole suite -- it's an artifact from a real pipeline run, not
# something this script can regenerate cheaply on its own.
import os
import numpy as np
mesh_path = "/Users/vinayak/Downloads/TRELLIS-main/outputs/vase_decimated_mesh.npz"
if os.path.exists(mesh_path):
    data = np.load(mesh_path)
    Vr, Fr = data['vertices'], data['faces']
    verts_r = torch.tensor(Vr, dtype=torch.float32)
    verts_r_h = torch.cat([verts_r, torch.ones(Vr.shape[0], 1)], dim=-1).unsqueeze(0).to(device)
    tri_r = torch.tensor(Fr, dtype=torch.int32).to(device)

    for (H2, W2) in [(1024, 1024), (512, 512)]:
        t0 = time.time()
        r_brute, _ = brute.rasterize(None, verts_r_h, tri_r, (H2, W2))
        if device == 'mps': torch.mps.synchronize()
        t_brute = time.time() - t0

        t0 = time.time()
        r_binned, _ = binned.rasterize(None, verts_r_h, tri_r, (H2, W2))
        if device == 'mps': torch.mps.synchronize()
        t_binned = time.time() - t0

        match = torch.equal(r_brute, r_binned)
        speedup = t_brute / t_binned if t_binned > 0 else float('inf')
        status = "PASS" if match else "MISMATCH"
        print(f"[REAL mesh {Vr.shape[0]}v/{Fr.shape[0]}f @ {H2}x{W2}] {status} -- "
              f"brute: {t_brute:.3f}s, binned: {t_binned:.3f}s, speedup: {speedup:.2f}x")
        if not match:
            all_ok = False
else:
    print(f"\n[real mesh speed test] SKIPPED -- {mesh_path} not found "
          f"(run scripts/save_real_mesh_for_bench.py first)")

print("\n" + ("ALL BINNED RASTERIZER CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
sys.exit(0 if all_ok else 1)
