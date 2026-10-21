"""
CUDA-only (DGX pod). Generates real (cond, x_0) pairs for the interleaved-attention distillation
pipeline (research.md Part II) by running the actual pretrained teacher's own two-stage sampler
(sparse-structure stage, then SLat stage) on each available example conditioning image, rather than
using synthetic/random SLat latents. Matches the ablation plan's own stated protocol: "sampled from
the teacher's own generation trajectories... no need to touch the original TRELLIS training
data/labels at all."

Per this project's platform-separation convention (kept since the K/V-cache and interleaved-
attention work): this script never runs on the Apple Silicon side and has no MPS-specific code;
the MPS scripts never gain a CUDA branch either. Run this only on the DGX pod, in the `trellis`
conda env (see the private connection memory for how to reach it).

Output: one .pt file per input image under `distillation_data/`, each containing:
  - cond, neg_cond: the real DINOv2 image-conditioning tensors (what the model was actually
    trained/run with)
  - feats, coords: the real, NORMALIZED SLat latent (x_0) sampled by the teacher's own
    slat_sampler -- deliberately the pre-denormalization `.samples` output, not
    `pipeline.sample_slat()`'s return value (which applies `* std + mean` for the VAE decoder's
    consumption) -- training needs x_0 in the same normalized space the flow model's own training
    objective (trellis/trainers/flow_matching/flow_matching.py's `diffuse`/`get_v`) operates in.
"""
import os, sys, glob, time, argparse
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'

parser = argparse.ArgumentParser()
parser.add_argument("--image-glob", type=str, default="assets/example_image/*.png",
                     help="relative to repo root. Default matches the original 42-image training "
                          "set. Pass a different pattern to add more data WITHOUT touching the "
                          "held-out test images (assets/teaser.png, assets/example_multi_image/"
                          "*_1.png -- scripts/dgx_generation_quality_check.py's held-out set).")
cli_args = parser.parse_args()

# trellis/pipelines/__init__.py unconditionally imports TrellisTextTo3DPipeline, which imports
# open3d at module level purely for type annotations (o3d.geometry.TriangleMesh etc., never
# actually called by anything this script uses -- only TrellisImageTo3DPipeline's get_cond/
# sample_sparse_structure/sample_slat are needed here). This pod's open3d needs libGL.so.1, which
# isn't present and isn't worth chasing for an unrelated pipeline this script never touches --
# stub it so the annotation-time attribute access resolves to something instead of crashing.
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
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})\n")

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'distillation_data')
os.makedirs(OUT_DIR, exist_ok=True)

pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)

image_paths = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                             cli_args.image_glob)))
print(f"Found {len(image_paths)} images matching '{cli_args.image_glob}'\n")

for i, image_path in enumerate(image_paths):
    name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(OUT_DIR, f"{name}.pt")
    if os.path.exists(out_path):
        print(f"[{i+1}/{len(image_paths)}] {name}: already generated, skipping")
        continue

    t0 = time.time()
    image = Image.open(image_path).convert("RGBA")
    with torch.no_grad():
        cond = pipeline.get_cond([image])
        coords = pipeline.sample_sparse_structure(cond)
        flow_model = pipeline.models['slat_flow_model']
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(device),
            coords=coords,
        )
        sampler_params = {**pipeline.slat_sampler_params}
        x_0 = pipeline.slat_sampler.sample(flow_model, noise, **cond, **sampler_params, verbose=False).samples

    torch.save({
        'cond': cond['cond'].cpu(),
        'neg_cond': cond['neg_cond'].cpu(),
        'feats': x_0.feats.cpu(),
        'coords': x_0.coords.cpu(),
    }, out_path)
    n_finite = torch.isfinite(x_0.feats).all().item()
    print(f"[{i+1}/{len(image_paths)}] {name}: {x_0.feats.shape[0]} active voxels, "
          f"finite={n_finite}, {time.time()-t0:.1f}s -> {out_path}")

print(f"\nDone. {len(glob.glob(os.path.join(OUT_DIR, '*.pt')))} (cond, x_0) pairs in {OUT_DIR}")
