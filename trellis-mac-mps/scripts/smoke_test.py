import os
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPARSE_DEBUG'] = '0'
os.environ['ATTN_DEBUG'] = '0'  # NOTE: upstream full_attn.py's DEBUG assertion has a pre-existing bug
                                  # (compares s.coords[...] == i across all 4 columns instead of just
                                  # the batch column at index 0), unrelated to this port; leaving DEBUG
                                  # off here avoids tripping it.

import sys, types
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')

# Bypass trellis/__init__.py (which eagerly imports the whole pipelines stack, requiring
# unrelated heavy deps like easydict/utils3d that aren't needed to test the sparse module).
_root = '/Users/vinayak/Downloads/TRELLIS-main/trellis'
_trellis_pkg = types.ModuleType('trellis')
_trellis_pkg.__path__ = [_root]
sys.modules['trellis'] = _trellis_pkg
_modules_pkg = types.ModuleType('trellis.modules')
_modules_pkg.__path__ = [_root + '/modules']
sys.modules['trellis.modules'] = _modules_pkg

import torch
import trellis.modules.sparse as sp

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

torch.manual_seed(0)


def make_sparse_tensor(batch_sizes, extent, channels, extra_shape=()):
    coords_list = []
    for b, n in enumerate(batch_sizes):
        pts = torch.randint(0, extent, (n, 3))
        pts = torch.unique(pts, dim=0)  # dedup so coords are valid active-voxel set
        bcol = torch.full((pts.shape[0], 1), b, dtype=torch.int32)
        coords_list.append(torch.cat([bcol, pts.int()], dim=1))
    coords = torch.cat(coords_list, dim=0).to(device)
    n_total = coords.shape[0]
    feats = torch.randn(n_total, *extra_shape, channels, device=device)
    return sp.SparseTensor(feats=feats, coords=coords)


print("\n--- SparseConv3d (kernel_size=3, submanifold) ---")
x = make_sparse_tensor([37, 41], extent=8, channels=16)
conv = sp.SparseConv3d(16, 32, 3).to(device)
y = conv(x)
print("input feats:", x.feats.shape, "output feats:", y.feats.shape, "coords preserved:", torch.equal(x.coords, y.coords))
assert y.feats.shape == (x.feats.shape[0], 32)
assert torch.equal(x.coords, y.coords)

print("\n--- SparseConv3d (kernel_size=1) ---")
conv1 = sp.SparseConv3d(16, 8, 1).to(device)
y1 = conv1(x)
print("output feats:", y1.feats.shape)
assert y1.feats.shape == (x.feats.shape[0], 8)

print("\n--- SparseLinear / norm / nonlinearity ---")
lin = sp.SparseLinear(16, 16).to(device)
gn = sp.SparseGroupNorm(4, 16).to(device)
act = sp.SparseSiLU()
z = act(gn(lin(x)))
print("chained output:", z.feats.shape)

print("\n--- SparseDownsample / SparseUpsample ---")
down = sp.SparseDownsample(2).to(device)
up = sp.SparseUpsample(2).to(device)
xd = down(x)
xu = up(xd)
print("downsampled:", xd.feats.shape, xd.coords.shape, "upsampled back:", xu.feats.shape)
assert torch.equal(xu.coords, x.coords)

print("\n--- sparse_scaled_dot_product_attention (qkv packed, sdpa) ---")
H, C = 4, 8
qkv = make_sparse_tensor([13, 17], extent=8, channels=C, extra_shape=(3, H))
out = sp.sparse_scaled_dot_product_attention(qkv)
print("attn out:", out.feats.shape)
assert out.feats.shape == (qkv.feats.shape[0], H, C)

print("\n--- sparse_windowed_scaled_dot_product_self_attention (sdpa) ---")
qkv2 = make_sparse_tensor([50], extent=16, channels=C, extra_shape=(3, H))
wout = sp.sparse_windowed_scaled_dot_product_self_attention(qkv2, window_size=8)
print("windowed attn out:", wout.feats.shape)
assert wout.feats.shape == qkv2.feats.shape[:1] + (H, C)

print("\n--- sparse_serialized_scaled_dot_product_self_attention (sdpa, z_order + hilbert) ---")
qkv3 = make_sparse_tensor([50], extent=16, channels=C, extra_shape=(3, H))
sout_z = sp.sparse_serialized_scaled_dot_product_self_attention(qkv3, window_size=8, serialize_mode=sp.SerializeMode.Z_ORDER)
sout_h = sp.sparse_serialized_scaled_dot_product_self_attention(qkv3, window_size=8, serialize_mode=sp.SerializeMode.HILBERT)
print("serialized (z-order) out:", sout_z.feats.shape, " (hilbert) out:", sout_h.feats.shape)

print("\nALL SMOKE TESTS PASSED on device =", device)
