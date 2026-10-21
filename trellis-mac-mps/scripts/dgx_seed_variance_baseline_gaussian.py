"""
CUDA-only (DGX pod). Calibration baseline for scripts/dgx_generation_quality_check_gaussian.py's
appearance-channel relative errors (color/SH 34.6%, opacity 15.6%, scale 40.1% mean, teacher vs.
interleaved-attention student). Same idea as scripts/dgx_seed_variance_baseline.py, which resolved
the analogous SLat-feature tension: runs the unmodified teacher TWICE per image with two different
noise seeds (same coords/conditioning), decodes both through the gaussian decoder, and measures the
same per-channel relative errors. Tells us whether the architecture-change numbers above are large
relative to this model's own seed-to-seed variance in these specific channels, or roughly in line
with it.
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
gs_decoder = pipeline.models['slat_decoder_gs']

test_images = (
    [os.path.join(REPO_ROOT, 'assets', 'teaser.png')] +
    sorted(glob.glob(os.path.join(REPO_ROOT, 'assets', 'example_multi_image', '*_1.png')))
)
print(f"Testing on {len(test_images)} images (same set as the gaussian quality check): "
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

        gs_a = gs_decoder(slat_a)[0]
        gs_b = gs_decoder(slat_b)[0]

        same_count = (gs_a.get_xyz.shape[0] == gs_b.get_xyz.shape[0])
        if same_count:
            xyz_err = (gs_a.get_xyz - gs_b.get_xyz).norm() / gs_a.get_xyz.norm()
            feat_err = (gs_a.get_features - gs_b.get_features).norm() / gs_a.get_features.norm()
            opacity_err = (gs_a.get_opacity - gs_b.get_opacity).norm() / gs_a.get_opacity.norm()
            scale_err = (gs_a.get_scaling - gs_b.get_scaling).norm() / gs_a.get_scaling.norm()
        else:
            xyz_err = feat_err = opacity_err = scale_err = float('nan')

    elapsed = time.time() - t0
    print(f"[{name}] {elapsed:.1f}s | same_count={same_count} | xyz={xyz_err:.4f} | "
          f"color/SH={feat_err:.4f} | opacity={opacity_err:.4f} | scale={scale_err:.4f}")
    results.append({
        'name': name, 'same_count': same_count,
        'xyz_rel_err': float(xyz_err), 'feat_rel_err': float(feat_err),
        'opacity_rel_err': float(opacity_err), 'scale_rel_err': float(scale_err),
    })

print("\n=== Summary (teacher vs. itself, different seeds only) ===")
valid = [r for r in results if r['same_count']]
print(f"Matching gaussian count: {len(valid)}/{len(results)}")
for metric in ['xyz_rel_err', 'feat_rel_err', 'opacity_rel_err', 'scale_rel_err']:
    vals = [r[metric] for r in valid]
    print(f"  mean {metric}: {sum(vals)/len(vals):.4f}")

print("\nFor comparison, teacher vs. interleaved-attention student (same seed) was:")
print("  xyz=0.0154, color/SH=0.3464, opacity=0.1556, scale=0.4012")

torch.save(results, os.path.join(REPO_ROOT, 'distillation_data', 'seed_variance_baseline_gaussian.pt'))
print(f"\nSaved -> distillation_data/seed_variance_baseline_gaussian.pt")
