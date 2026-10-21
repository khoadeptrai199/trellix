import os
import sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'trellis')
def fake_pkg(name, path):
    m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m; return m
fake_pkg('trellis', _root)
fake_pkg('trellis.renderers', _root + '/renderers')

import torch
from trellis.renderers import sw_rasterizer as sw

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"device: {device}\n")

# 2x2 texture: top-left=red, top-right=green, bottom-left=blue, bottom-right=white
# stored row-major with row 0 = TOP (image convention)
tex = torch.tensor([[
    [[1., 0., 0.], [0., 1., 0.]],   # row 0 (top):    red,  green
    [[0., 0., 1.], [1., 1., 1.]],   # row 1 (bottom): blue, white
]], device=device, requires_grad=True)  # [1,2,2,3]

# sample at the four corners in OpenGL UV convention (v=0 bottom, v=1 top), using coords
# just inside each texel center to avoid boundary ambiguity: texel centers at u,v in {0.25, 0.75}
uv = torch.tensor([[
    [[0.25, 0.75], [0.75, 0.75]],   # top-left (u=0.25,v=0.75 -> near top),  top-right
    [[0.25, 0.25], [0.75, 0.25]],   # bottom-left (v=0.25 -> near bottom),  bottom-right
]], device=device)  # [1,2,2,2]

out = sw.texture(tex, uv)
print("sampled:\n", out.detach().cpu().numpy())

expected = torch.tensor([[
    [[1., 0., 0.], [0., 1., 0.]],
    [[0., 0., 1.], [1., 1., 1.]],
]], device=device)

diff = (out - expected).abs().max().item()
print(f"\nmax abs diff vs expected (correct UV orientation): {diff:.4f}")
ok_orientation = diff < 0.05
print(f"orientation correct: {ok_orientation}")

# differentiability check
loss = out.sum()
loss.backward()
grad_ok = tex.grad is not None and torch.isfinite(tex.grad).all().item() and tex.grad.abs().sum().item() > 0
print(f"gradient flows to texture, finite, nonzero: {grad_ok}")

print("\n" + ("TEXTURE TEST PASSED" if (ok_orientation and grad_ok) else "TEXTURE TEST FAILED"))
