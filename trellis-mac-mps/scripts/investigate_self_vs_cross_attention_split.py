"""
Splits attention's real time-share on slat_flow_model into self-attention vs. cross-attention,
using the exact same bypass-measurement technique already established in APPLE_SILICON.md's FFN
and attention ceiling analyses (zero-passthrough replacement -- attention output only ever feeds a
residual add, so a zero-output stand-in is the correct no-op, not an approximation).

Motivation: research.md Part II's interleaved-attention result only reached 1.05x real speedup
after windowing self-attention (cross-attention structurally can't be windowed by the current
SparseMultiHeadAttention class). Before deciding whether cross-attention deserves its own,
bigger/different architecture change, this answers the actual open question: which of the two is
really the dominant cost on this model?
"""
import os, sys, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'
sys.path.insert(0, '/Users/vinayak/Downloads/TRELLIS-main')
os.chdir('/Users/vinayak/Downloads/TRELLIS-main')

import torch
from trellis.pipelines import TrellisImageTo3DPipeline
import trellis.modules.sparse as sp

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}\n")

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)
model = pipeline.models['slat_flow_model']
model.eval()
n_blocks = len(model.blocks)
print(f"slat_flow_model: {n_blocks} blocks\n")

import glob
from PIL import Image
image_paths = glob.glob('/Users/vinayak/Downloads/TRELLIS-main/assets/**/*.png', recursive=True) + \
              glob.glob('/Users/vinayak/Downloads/TRELLIS-main/assets/**/*.jpg', recursive=True)
image = Image.open(image_paths[0]).convert("RGBA")
real_cond = pipeline.get_cond([image])['cond']

torch.manual_seed(0)


def make_sparse_tensor(n, extent, channels):
    pts = torch.unique(torch.randint(0, extent, (n, 3)), dim=0)
    coords = torch.cat([torch.zeros(pts.shape[0], 1, dtype=torch.int32), pts.int()], dim=1).to(device)
    feats = torch.randn(coords.shape[0], channels, device=device)
    return sp.SparseTensor(feats=feats, coords=coords)


x = make_sparse_tensor(3000, model.resolution, model.in_channels)
t = torch.full((1,), 500.0, device=device)
print(f"num active voxels in test input: {x.feats.shape[0]}\n")


def sync():
    if device == 'mps':
        torch.mps.synchronize()


def _zero_forward(x_in, context=None):
    if hasattr(x_in, 'feats'):
        return x_in.replace(torch.zeros_like(x_in.feats))
    return torch.zeros_like(x_in)


def bypass(target):
    """target: 'self', 'cross', or 'both'. Returns the list of (module, original_forward) to restore."""
    originals = []
    for block in model.blocks:
        attns = []
        if target in ('self', 'both'):
            attns.append(block.self_attn)
        if target in ('cross', 'both'):
            attns.append(block.cross_attn)
        for attn in attns:
            originals.append((attn, attn.forward))
            attn.forward = _zero_forward
    return originals


def restore(originals):
    for attn, fwd in originals:
        attn.forward = fwd


def bench(label, n_reps=10, n_warmup=3):
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x, t, real_cond)
        sync()
        t0 = time.time()
        for _ in range(n_reps):
            model(x, t, real_cond)
        sync()
        elapsed = (time.time() - t0) / n_reps
    print(f"  {label}: {elapsed*1000:.2f}ms/forward")
    return elapsed


print("=== Bypass measurement ===")
t_baseline = bench("baseline (nothing bypassed)")

originals = bypass('self')
t_no_self = bench("self-attention bypassed (zero passthrough)")
restore(originals)

originals = bypass('cross')
t_no_cross = bench("cross-attention bypassed (zero passthrough)")
restore(originals)

originals = bypass('both')
t_no_both = bench("both self- and cross-attention bypassed")
restore(originals)

self_attn_time = t_baseline - t_no_self
cross_attn_time = t_baseline - t_no_cross
both_time = t_baseline - t_no_both

print("\n=== Results ===")
print(f"  baseline:                {t_baseline*1000:.2f}ms")
print(f"  self-attention's share:  {self_attn_time*1000:.2f}ms ({100*self_attn_time/t_baseline:.1f}% of baseline)")
print(f"  cross-attention's share: {cross_attn_time*1000:.2f}ms ({100*cross_attn_time/t_baseline:.1f}% of baseline)")
print(f"  combined (both bypassed): {both_time*1000:.2f}ms ({100*both_time/t_baseline:.1f}% of baseline)")
print(f"  sum of individual shares: {(self_attn_time+cross_attn_time)*1000:.2f}ms "
      f"(vs combined {both_time*1000:.2f}ms -- large mismatch would suggest interaction/overlap effects)")
print(f"\n  self : cross ratio = {self_attn_time/cross_attn_time:.2f} : 1"
      if cross_attn_time > 0 else "\n  cross-attention bypass showed no measurable time reduction")
