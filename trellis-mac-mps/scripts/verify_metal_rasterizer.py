"""
Correctness check for the tri-buffer cache + batched-sync optimization in
trellis/renderers/metal_rasterizer.py (see APPLE_SILICON.md for the perf writeup).

Compares metal_rasterizer.rasterize() (MPS) against sw_rasterizer.rasterize() (CPU, used as
ground truth here since sw_rasterizer hits an unrelated pre-existing MPS backend bug --
`gather: index out of bounds` from torch.min's argmin output on MPS at this chunk scale --
on this build of PyTorch; CPU execution is unaffected and is a fine independent reference).
"""
import sys, time
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
import torch
from trellis.renderers import sw_rasterizer as sw
from trellis.renderers import metal_rasterizer as mr

print("Metal available:", mr.available())
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"device: {device}\n")

# === Test 1: small overlapping-triangles unit scene ===
H, W = 64, 64
verts = torch.tensor([
    [-0.5, -0.5, 0.5, 1.0], [0.5, -0.5, 0.5, 1.0], [0.0, 0.5, 0.5, 1.0],
    [-1.0, -1.0, 0.9, 1.0], [1.0, -1.0, 0.9, 1.0], [1.0, 1.0, 0.9, 1.0], [-1.0, 1.0, 0.9, 1.0],
], dtype=torch.float32).unsqueeze(0)
tri = torch.tensor([[0,1,2],[3,4,5],[3,5,6]], dtype=torch.int32)

rast_ref, _ = sw.rasterize(None, verts, tri, (H, W))
rast_metal, _ = mr.rasterize(None, verts.to(device), tri.to(device), (H, W))
rast_metal_cpu = rast_metal.cpu()

diff = (rast_ref - rast_metal_cpu).abs()
ok1 = diff.max().item() < 1e-4 and torch.equal(rast_ref[..., 3], rast_metal_cpu[..., 3])
print(f"[unit scene] max abs diff vs sw_rasterizer (CPU ref): {diff.max().item():.3e}")
print(f"[unit scene] triangle_id exact match: {torch.equal(rast_ref[..., 3], rast_metal_cpu[..., 3])}")
print(f"[unit scene] PASS: {ok1}\n")

# === Test 2: random mesh, larger scale ===
torch.manual_seed(0)
V, F = 5000, 9000
rand_verts = torch.rand(V, 3) * 2 - 1
rand_verts[:, 2] = rand_verts[:, 2] * 0.3 + 2.0
rand_verts_h = torch.cat([rand_verts, torch.ones(V, 1)], dim=-1).unsqueeze(0)
rand_tri = torch.randint(0, V, (F, 3), dtype=torch.int32)

H2, W2 = 256, 256
t0 = time.time()
rast_ref2, _ = sw.rasterize(None, rand_verts_h, rand_tri, (H2, W2))
t_ref = time.time() - t0

t0 = time.time()
rast_metal2, _ = mr.rasterize(None, rand_verts_h.to(device), rand_tri.to(device), (H2, W2))
if device == 'mps': torch.mps.synchronize()
t_metal = time.time() - t0
rast_metal2_cpu = rast_metal2.cpu()

tri_id_match = (rast_ref2[..., 3] == rast_metal2_cpu[..., 3]).float().mean().item()
match_mask = (rast_ref2[..., 3] == rast_metal2_cpu[..., 3]).unsqueeze(-1).expand_as(rast_ref2)
diff2 = (rast_ref2 - rast_metal2_cpu).abs()
print(f"[random mesh, {V}v/{F}f @ {H2}x{W2}] sw (CPU) time: {t_ref:.3f}s, metal (MPS) time: {t_metal:.3f}s")
print(f"[random mesh] max abs diff where triangle_id matches: {diff2[match_mask].max().item():.3e}")
print(f"[random mesh] triangle_id agreement fraction: {tri_id_match:.6f}")
ok2 = tri_id_match > 0.999
print(f"[random mesh] PASS: {ok2}")

# Note: this module previously also had a tri-buffer-cache correctness test here (Test 3), for a
# cache/sync-batching optimization to metal_rasterizer.py's rasterize(). That fix was reverted
# (see APPLE_SILICON.md and git history at 2c76a04) after real-workload benchmarking showed it gave
# no measurable speedup -- the O(H*W*F) kernel compute dominates at real resolution/triangle-count,
# not the sync it targeted. See metal_rasterizer_binned.py + verify_metal_rasterizer_binned.py for
# the fix that actually addresses that bottleneck.

all_ok = ok1 and ok2
print("\n" + ("ALL METAL RASTERIZER CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
sys.exit(0 if all_ok else 1)
