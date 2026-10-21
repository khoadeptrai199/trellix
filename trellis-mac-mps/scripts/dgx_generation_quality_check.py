"""
CUDA-only (DGX pod). Real generation-quality check for the distilled interleaved-attention
student against the frozen teacher -- the validation step research.md flagged as still missing
after the first distillation run (decreasing training loss confirms the pipeline works, not that
the student's actual generation quality matches the teacher's).

Deliberately avoids anything needing compiled CUDA rendering extensions (nvdiffrast,
diff-gaussian-rasterization) -- this pod has no CUDA toolkit (`nvcc` not found, NGC runtime-only
container), so those wouldn't build. Mesh decoding (`slat_decoder_mesh` -> FlexiCubes) is pure
PyTorch (confirmed by reading trellis/representations/mesh/flexicubes/flexicubes.py's imports),
so mesh vertex/face counts and Chamfer distance are used instead of rendered-pixel comparison --
the same style of proxy (mesh vertex-count stability) this project already trusted for verifying
CFG-diff-caching in Part I.

Test images are held out from training: assets/teaser.png and a few from
assets/example_multi_image/ -- scripts/dgx_generate_distillation_data.py only ever globbed
assets/example_image/*.png, so none of these were seen during distillation.

For each test image: run the SLat sampler twice with the *same* sparse-structure coords and the
*same* initial noise (only swapping which flow model denoises it) -- isolates the effect of the
architecture change from any other source of variation, the same "fair comparison" discipline used
throughout this project's benchmarking.
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
print(f"Loaded checkpoint: K={K}, window_size={W}, {len(ckpt['losses'])} training steps, "
      f"final loss={ckpt['losses'][-1]:.6f}\n")

student = copy.deepcopy(teacher)
assignment = configure_interleaved_attention(student, full_attention_interval=K, window_size=W)
assert assignment == ckpt['assignment'], "block assignment mismatch between checkpoint and reconstructed student"
student_sd = student.state_dict()
for key, val in ckpt['trainable_state_dict'].items():
    student_sd[key] = val.to(device).to(student_sd[key].dtype)
student.load_state_dict(student_sd)
student.eval()
print(f"Student rebuilt: {sum(1 for v in assignment.values() if v=='windowed')}/{len(assignment)} "
      f"blocks windowed, fine-tuned weights loaded (cast to {next(student.parameters()).dtype} "
      f"to match production inference dtype)\n")

test_images = (
    [os.path.join(REPO_ROOT, 'assets', 'teaser.png')] +
    sorted(glob.glob(os.path.join(REPO_ROOT, 'assets', 'example_multi_image', '*_1.png')))
)
print(f"Testing on {len(test_images)} held-out images (never seen during distillation): "
      f"{[os.path.basename(p) for p in test_images]}\n")


def chamfer_distance(v1, v2, n_samples=4000):
    """Symmetric nearest-neighbor L2 distance between two point sets (mesh vertices, subsampled).
    Pure vertex-based (not proper surface-area-weighted sampling) -- a cheap, honest proxy, not a
    publication-grade metric; sufficient to flag gross shape divergence between paired outputs."""
    def sample(v, n):
        if v.shape[0] <= n:
            return v
        idx = torch.randperm(v.shape[0], device=v.device)[:n]
        return v[idx]
    a, b = sample(v1, n_samples), sample(v2, n_samples)
    d = torch.cdist(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)
    return (d.min(dim=1).values.mean() + d.min(dim=0).values.mean()).item() / 2


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

        feat_rel_err = (slat_teacher.feats - slat_student.feats).norm() / slat_teacher.feats.norm()

        mesh_decoder = pipeline.models['slat_decoder_mesh']
        mesh_teacher = mesh_decoder(slat_teacher)[0]
        mesh_student = mesh_decoder(slat_student)[0]

        teacher_ok, student_ok = mesh_teacher.success, mesh_student.success
        cd = chamfer_distance(mesh_teacher.vertices, mesh_student.vertices) if (teacher_ok and student_ok) else float('nan')

    elapsed = time.time() - t0
    print(f"[{name}] {elapsed:.1f}s | SLat feat rel.err={feat_rel_err:.4f} | "
          f"teacher: {mesh_teacher.vertices.shape[0]}v/{mesh_teacher.faces.shape[0]}f (success={teacher_ok}) | "
          f"student: {mesh_student.vertices.shape[0]}v/{mesh_student.faces.shape[0]}f (success={student_ok}) | "
          f"chamfer={cd:.5f}")
    results.append({
        'name': name, 'feat_rel_err': feat_rel_err.item(),
        'teacher_verts': mesh_teacher.vertices.shape[0], 'teacher_faces': mesh_teacher.faces.shape[0], 'teacher_success': teacher_ok,
        'student_verts': mesh_student.vertices.shape[0], 'student_faces': mesh_student.faces.shape[0], 'student_success': student_ok,
        'chamfer': cd,
    })

print("\n=== Summary ===")
n_both_success = sum(1 for r in results if r['teacher_success'] and r['student_success'])
print(f"Both teacher and student produced a valid mesh: {n_both_success}/{len(results)}")
avg_feat_err = sum(r['feat_rel_err'] for r in results) / len(results)
print(f"Mean SLat feature relative error (teacher vs student, same noise/coords): {avg_feat_err:.4f}")
valid_cd = [r['chamfer'] for r in results if r['chamfer'] == r['chamfer']]  # filter NaN
if valid_cd:
    print(f"Mean Chamfer distance (paired, valid meshes only): {sum(valid_cd)/len(valid_cd):.5f}")
vert_ratios = [r['student_verts']/r['teacher_verts'] for r in results if r['teacher_verts'] > 0]
if vert_ratios:
    print(f"Mean student/teacher vertex-count ratio: {sum(vert_ratios)/len(vert_ratios):.3f} "
          f"(1.0 = identical structural complexity)")

torch.save(results, os.path.join(REPO_ROOT, 'distillation_data', 'quality_check_results.pt'))
print(f"\nSaved detailed results -> distillation_data/quality_check_results.pt")
