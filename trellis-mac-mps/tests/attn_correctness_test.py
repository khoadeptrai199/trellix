import os
import sys, types, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trellis')
_trellis_pkg = types.ModuleType('trellis'); _trellis_pkg.__path__ = [_root]; sys.modules['trellis'] = _trellis_pkg
_modules_pkg = types.ModuleType('trellis.modules'); _modules_pkg.__path__ = [_root + '/modules']; sys.modules['trellis.modules'] = _modules_pkg

import os
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPARSE_BACKEND'] = 'pytorch'

import torch
from trellis.modules.sparse.attention.full_attn import _sdpa_varlen

torch.manual_seed(0)
H, C = 3, 5
q_seqlen = [4, 7, 2]
kv_seqlen = [4, 7, 2]
T = sum(q_seqlen)
q = torch.randn(T, H, C)
k = torch.randn(T, H, C)
v = torch.randn(T, H, C)

out = _sdpa_varlen(q, k, v, q_seqlen, kv_seqlen)

ref = torch.zeros_like(out)
off = 0
for L in q_seqlen:
    qi, ki, vi = q[off:off+L], k[off:off+L], v[off:off+L]  # [L,H,C]
    for h in range(H):
        qh, kh, vh = qi[:, h], ki[:, h], vi[:, h]  # [L,C]
        scores = (qh @ kh.T) / math.sqrt(C)
        attn = torch.softmax(scores, dim=-1)
        ref[off:off+L, h] = attn @ vh
    off += L

max_err = (out - ref).abs().max().item()
print("max abs error vs manual softmax reference:", max_err)
assert max_err < 1e-4
print("PASS: _sdpa_varlen matches manual per-segment softmax attention.")
