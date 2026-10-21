"""
CUDA-only (DGX pod). Appearance-level generation-quality check via the gaussian decoder --
completes the "still missing" item flagged in research.md: the mesh-only quality check
(scripts/dgx_generation_quality_check.py) validates geometry but never touches the SLat channels
that drive color/texture, which is what the gaussian decoder (not the mesh decoder) actually
decodes. Deliberately still avoids anything needing compiled CUDA rendering extensions -- this
compares decoded gaussian PARAMETERS directly (position, color/SH features, opacity, scale), not
rendered pixels.

Since teacher and student are run with the *same* sparse-structure coords and the *same* initial
noise (same protocol as the mesh check), the decoded gaussians correspond 1:1 in the same order --
a paired per-gaussian comparison, not a nearest-neighbor match like the mesh check needed for
Chamfer distance. This is a more direct, less noisy comparison than mesh Chamfer distance.
"""
import os, sys, glob, copy, time
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
from trellis.modules.sparse.attention.interleave_windowed import configure_interleaved_attention

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}\n")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(REPO_ROOT, 'distillation_data', 'student_K8_W16.pt')

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)
teacher = pipeline.models['slat_flow_model']
teacher.eval()

ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
K, W = ckpt['full_attention_interval'], ckpt['window_size']
n_steps_trained = len(ckpt['losses'])
print(f"Loaded checkpoint: K={K}, window_size={W}, {n_steps_trained} training steps, "
      f"final loss={ckpt['losses'][-1]:.6f}\n")

student = copy.deepcopy(teacher)
assignment = configure_interleaved_attention(student, full_attention_interval=K, window_size=W)
student_sd = student.state_dict()
for key, val in ckpt['trainable_state_dict'].items():
    student_sd[key] = val.to(device).to(student_sd[key].dtype)
student.load_state_dict(student_sd)
student.eval()
print(f"Student rebuilt with {n_steps_trained}-step checkpoint\n")

test_images = (
    [os.path.join(REPO_ROOT, 'assets', 'teaser.png')] +
    sorted(glob.glob(os.path.join(REPO_ROOT, 'assets', 'example_multi_image', '*_1.png')))
)
print(f"Testing on {len(test_images)} held-out images: {[os.path.basename(p) for p in test_images]}\n")

gs_decoder = pipeline.models['slat_decoder_gs']

results = []
for image_path in test_images:
    name = os.path.basename(image_path)
    t0 = time.time()
    image = Image.open(image_path).convert("RGBA")
    with torch.no_grad():
        cond = pipeline.get_cond([image])
        coords = pipeline.sample_sparse_structure(cond)
        noise_feats = torch.randn(coords.shape[0], teacher.in_channels, device=device)

        sampler_params = {**pipeline.slat_sampler_params}
        noise_teacher = sp.SparseTensor(feats=noise_feats.clone(), coords=coords)
        slat_teacher = pipeline.slat_sampler.sample(teacher, noise_teacher, **cond, **sampler_params, verbose=False).samples
        noise_student = sp.SparseTensor(feats=noise_feats.clone(), coords=coords)
        slat_student = pipeline.slat_sampler.sample(student, noise_student, **cond, **sampler_params, verbose=False).samples

        gs_teacher = gs_decoder(slat_teacher)[0]
        gs_student = gs_decoder(slat_student)[0]

        n_teacher = gs_teacher.get_xyz.shape[0]
        n_student = gs_student.get_xyz.shape[0]
        same_count = (n_teacher == n_student)

        if same_count:
            xyz_rel_err = (gs_teacher.get_xyz - gs_student.get_xyz).norm() / gs_teacher.get_xyz.norm()
            feat_rel_err = (gs_teacher.get_features - gs_student.get_features).norm() / gs_teacher.get_features.norm()
            opacity_rel_err = (gs_teacher.get_opacity - gs_student.get_opacity).norm() / gs_teacher.get_opacity.norm()
            scale_rel_err = (gs_teacher.get_scaling - gs_student.get_scaling).norm() / gs_teacher.get_scaling.norm()
        else:
            xyz_rel_err = feat_rel_err = opacity_rel_err = scale_rel_err = float('nan')

    elapsed = time.time() - t0
    print(f"[{name}] {elapsed:.1f}s | {n_teacher} vs {n_student} gaussians (same_count={same_count}) | "
          f"xyz_err={xyz_rel_err:.4f} | color/SH_err={feat_rel_err:.4f} | "
          f"opacity_err={opacity_rel_err:.4f} | scale_err={scale_rel_err:.4f}")
    results.append({
        'name': name, 'n_teacher': n_teacher, 'n_student': n_student, 'same_count': same_count,
        'xyz_rel_err': float(xyz_rel_err), 'feat_rel_err': float(feat_rel_err),
        'opacity_rel_err': float(opacity_rel_err), 'scale_rel_err': float(scale_rel_err),
    })

print("\n=== Summary ===")
valid = [r for r in results if r['same_count']]
print(f"Matching gaussian count (teacher vs student): {len(valid)}/{len(results)}")
if valid:
    for metric in ['xyz_rel_err', 'feat_rel_err', 'opacity_rel_err', 'scale_rel_err']:
        vals = [r[metric] for r in valid]
        mean = sum(vals) / len(vals)
        print(f"  mean {metric}: {mean:.4f}")

torch.save(results, os.path.join(REPO_ROOT, 'distillation_data', 'quality_check_gaussian_results.pt'))
print(f"\nSaved -> distillation_data/quality_check_gaussian_results.pt")
