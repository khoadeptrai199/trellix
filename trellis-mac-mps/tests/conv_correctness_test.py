import os
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'

import sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trellis')
_trellis_pkg = types.ModuleType('trellis'); _trellis_pkg.__path__ = [_root]; sys.modules['trellis'] = _trellis_pkg
_modules_pkg = types.ModuleType('trellis.modules'); _modules_pkg.__path__ = [_root + '/modules']; sys.modules['trellis.modules'] = _modules_pkg

import torch
import trellis.modules.sparse as sp
from trellis.modules.sparse.conv.conv_pytorch import _kernel_offsets

torch.manual_seed(1234)
device = 'cpu'  # brute force reference on CPU for exactness

N = 60
Cin, Cout = 6, 5
kernel_size = (3, 3, 3)
extent = 6

coords = torch.randint(0, extent, (N, 3))
coords = torch.unique(coords, dim=0)
N = coords.shape[0]
bcol = torch.zeros(N, 1, dtype=torch.int32)
coords = torch.cat([bcol, coords.int()], dim=1)
feats = torch.randn(N, Cin)

x = sp.SparseTensor(feats=feats, coords=coords)
conv = sp.SparseConv3d(Cin, Cout, kernel_size, bias=True)
with torch.no_grad():
    y = conv(x)

# --- brute-force reference using the SAME weight/bias/offset convention the module claims to use ---
offsets = _kernel_offsets(kernel_size, (1, 1, 1))
coord_to_idx = {tuple(c.tolist()): i for i, c in enumerate(coords)}

ref_out = torch.zeros(N, Cout)
with torch.no_grad():
    for qi in range(N):
        b, cx, cy, cz = coords[qi].tolist()
        acc = torch.zeros(Cout)
        for oi, (dx, dy, dz) in enumerate(offsets):
            nb = (b, cx + dx, cy + dy, cz + dz)
            if nb in coord_to_idx:
                nj = coord_to_idx[nb]
                acc += feats[nj] @ conv.weight[oi]
        acc += conv.bias
        ref_out[qi] = acc

max_err = (y.feats - ref_out).abs().max().item()
print("max abs error vs brute-force reference:", max_err)
assert max_err < 1e-4, "Sparse conv gather/scatter logic does NOT match a brute-force reference!"
print("PASS: pure-PyTorch submanifold sparse conv matches brute-force reference exactly (same weight convention).")
