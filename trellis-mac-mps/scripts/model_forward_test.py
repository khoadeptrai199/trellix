import os
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'

import sys, types
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
_root = '/Users/vinayak/Downloads/TRELLIS-main/trellis'
_trellis_pkg = types.ModuleType('trellis'); _trellis_pkg.__path__ = [_root]; sys.modules['trellis'] = _trellis_pkg
_modules_pkg = types.ModuleType('trellis.modules'); _modules_pkg.__path__ = [_root + '/modules']; sys.modules['trellis.modules'] = _modules_pkg
_models_pkg = types.ModuleType('trellis.models'); _models_pkg.__path__ = [_root + '/models']; sys.modules['trellis.models'] = _models_pkg
# structured_latent_vae/__init__.py eagerly imports decoder_gs -> trellis.representations.Gaussian,
# which needs plyfile (not installed) and hardcodes .cuda() -- out of scope for this test (we're only
# exercising the sparse backbone: SLatFlowModel + SLatEncoder), so bypass that package __init__ too.
_slat_vae_pkg = types.ModuleType('trellis.models.structured_latent_vae')
_slat_vae_pkg.__path__ = [_root + '/models/structured_latent_vae']
sys.modules['trellis.models.structured_latent_vae'] = _slat_vae_pkg

import torch
import trellis.modules.sparse as sp
from trellis.models.structured_latent_flow import SLatFlowModel
from trellis.models.structured_latent_vae.encoder import SLatEncoder

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}\n")
torch.manual_seed(0)


def make_sparse_tensor(batch_sizes, extent, channels):
    coords_list = []
    for b, n in enumerate(batch_sizes):
        pts = torch.randint(0, extent, (n, 3))
        pts = torch.unique(pts, dim=0)
        bcol = torch.full((pts.shape[0], 1), b, dtype=torch.int32)
        coords_list.append(torch.cat([bcol, pts.int()], dim=1))
    coords = torch.cat(coords_list, dim=0).to(device)
    feats = torch.randn(coords.shape[0], channels, device=device)
    return sp.SparseTensor(feats=feats, coords=coords)


def check_finite(name, t):
    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    status = "OK" if (n_nan == 0 and n_inf == 0) else f"BAD (nan={n_nan}, inf={n_inf})"
    print(f"  {name}: shape={tuple(t.shape)} finite-check={status} mean={t.mean().item():.4f} std={t.std().item():.4f}")
    return n_nan == 0 and n_inf == 0


all_ok = True

print("=== SLatFlowModel (denoiser backbone: full sparse self-attn + cross-attn + conv + up/down) ===")
flow = SLatFlowModel(
    resolution=16,
    in_channels=8,
    model_channels=32,
    cond_channels=32,
    out_channels=8,
    num_blocks=2,
    num_heads=4,
    mlp_ratio=2,
    patch_size=2,
    num_io_res_blocks=2,
    io_block_channels=[16],
    pe_mode="ape",
    qk_rms_norm=True,
).to(device)
flow.eval()

x = make_sparse_tensor([120, 95], extent=16, channels=8)
t = torch.randint(0, 1000, (2,), device=device).float()
cond = torch.randn(2, 32, 32, device=device)  # [N, L_cond, cond_channels]

with torch.no_grad():
    out = flow(x, t, cond)
ok = check_finite("flow output feats", out.feats)
print(f"  coords preserved: {torch.equal(out.coords, x.coords)}")
all_ok &= ok

print("\n=== SLatEncoder, attn_mode='swin' (exercises windowed_attn.py / sdpa) ===")
enc_swin = SLatEncoder(
    resolution=16,
    in_channels=8,
    model_channels=32,
    latent_channels=8,
    num_blocks=2,
    num_heads=4,
    mlp_ratio=2,
    attn_mode="swin",
    window_size=8,
    pe_mode="ape",
).to(device)
enc_swin.eval()
x2 = make_sparse_tensor([150], extent=16, channels=8)
with torch.no_grad():
    z_swin = enc_swin(x2, sample_posterior=False)
ok = check_finite("swin-encoder latent", z_swin.feats)
all_ok &= ok

print("\n=== SLatEncoder, attn_mode='shift_window' (exercises serialized_attn.py + vox2seq_pytorch fallback) ===")
enc_shift = SLatEncoder(
    resolution=16,
    in_channels=8,
    model_channels=32,
    latent_channels=8,
    num_blocks=2,
    num_heads=4,
    mlp_ratio=2,
    attn_mode="shift_window",
    window_size=8,
    pe_mode="ape",
).to(device)
enc_shift.eval()
with torch.no_grad():
    z_shift = enc_shift(x2, sample_posterior=False)
ok = check_finite("shift_window-encoder latent", z_shift.feats)
all_ok &= ok

print("\n=== SLatEncoder, attn_mode='shift_order' (exercises hilbert curve path) ===")
enc_order = SLatEncoder(
    resolution=16,
    in_channels=8,
    model_channels=32,
    latent_channels=8,
    num_blocks=4,  # >=4 so all 4 SerializeModes (incl. HILBERT/HILBERT_TRANSPOSED) get exercised
    num_heads=4,
    mlp_ratio=2,
    attn_mode="shift_order",
    window_size=8,
    pe_mode="ape",
).to(device)
enc_order.eval()
with torch.no_grad():
    z_order = enc_order(x2, sample_posterior=False)
ok = check_finite("shift_order-encoder latent", z_order.feats)
all_ok &= ok

print("\n" + ("ALL MODEL FORWARD PASSES PASSED on device = " + device if all_ok else "SOME CHECKS FAILED"))
