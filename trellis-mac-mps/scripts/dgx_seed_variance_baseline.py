"""
CUDA-only (DGX pod). Calibration baseline for scripts/dgx_generation_quality_check.py's SLat
feature relative-error number (48.7% mean, teacher vs. interleaved-attention student, same
noise/coords). Directly tests the hypothesis that this magnitude is largely floating-point/
per-step-perturbation compounding over the 25-step sampler rather than a real effect of the
architecture change: runs the SAME unmodified teacher model TWICE per image with two DIFFERENT
noise seeds (same coords, same conditioning), and measures the identical SLat feature relative
error metric. If teacher-vs-teacher (different seed) is comparably large, the architecture-change
result isn't distinguishable from ordinary sampler sensitivity. If it's much smaller, the
architecture change is doing something real beyond noise.
"""
import os, sys, glob, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'

class _DummyO3D:
    def __getattr__(self, name):
        return _DummyO3D()
    def __call__(self, *a, **kw):
        return _DummyO3D()
if 'open3d' not in sys.modules:
    sys.modules['open3d'] = _DummyO3D()

import torch
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
import trellis.modules.sparse as sp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)
teacher = pipeline.models['slat_flow_model']
teacher.eval()

test_images = (
    [os.path.join(REPO_ROOT, 'assets', 'teaser.png')] +
    sorted(glob.glob(os.path.join(REPO_ROOT, 'assets', 'example_multi_image', '*_1.png')))
)
print(f"Testing on {len(test_images)} images (same set as the quality check): "
      f"{[os.path.basename(p) for p in test_images]}\n")

results = []
for image_path in test_images:
    name = os.path.basename(image_path)
    t0 = time.time()
    image = Image.open(image_path).convert("RGBA")
    with torch.no_grad():
        cond = pipeline.get_cond([image])
        coords = pipeline.sample_sparse_structure(cond)

        sampler_params = {**pipeline.slat_sampler_params}
        torch.manual_seed(1)
        noise_a = sp.SparseTensor(feats=torch.randn(coords.shape[0], teacher.in_channels, device=device), coords=coords)
        slat_a = pipeline.slat_sampler.sample(teacher, noise_a, **cond, **sampler_params, verbose=False).samples

        torch.manual_seed(2)
        noise_b = sp.SparseTensor(feats=torch.randn(coords.shape[0], teacher.in_channels, device=device), coords=coords)
        slat_b = pipeline.slat_sampler.sample(teacher, noise_b, **cond, **sampler_params, verbose=False).samples

        feat_rel_err = (slat_a.feats - slat_b.feats).norm() / slat_a.feats.norm()

    elapsed = time.time() - t0
    print(f"[{name}] {elapsed:.1f}s | teacher(seed=1) vs teacher(seed=2) SLat feat rel.err={feat_rel_err:.4f}")
    results.append({'name': name, 'feat_rel_err': feat_rel_err.item()})

print("\n=== Summary ===")
avg = sum(r['feat_rel_err'] for r in results) / len(results)
print(f"Mean SLat feature relative error, teacher vs. itself (different seeds only): {avg:.4f}")
print(f"For comparison, teacher vs. interleaved-attention student (same seed) was: 0.4868")

torch.save(results, os.path.join(REPO_ROOT, 'distillation_data', 'seed_variance_baseline.pt'))
print(f"\nSaved -> distillation_data/seed_variance_baseline.pt")
