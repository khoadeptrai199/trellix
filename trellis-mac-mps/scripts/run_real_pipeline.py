import argparse
import json
import os, sys, time
os.environ['SPARSE_BACKEND'] = 'pytorch'
os.environ['ATTN_BACKEND'] = 'sdpa'

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from trellis.utils.render_utils import get_renderer, yaw_pitch_r_fov_to_extrinsics_intrinsics
from trellis.utils.random_utils import sphere_hammersley_sequence

parser = argparse.ArgumentParser(description="Run the full TRELLIS image-to-3D pipeline on Apple Silicon (MPS).")
parser.add_argument("image", nargs="?", default="assets/example_image/T.png", help="Path to the input image.")
parser.add_argument("out_prefix", nargs="?", default="sample", help="Output filename prefix (files land in --out-dir).")
parser.add_argument("--out-dir", default=str(REPO_ROOT / "outputs"), help="Output directory.")
parser.add_argument("--seed", type=int, default=1, help="Sampling seed.")
parser.add_argument("--simplify", type=float, default=0.95, help="Mesh decimation ratio (fraction of faces removed).")
parser.add_argument("--texture-size", type=int, default=1024, help="Baked texture resolution.")
parser.add_argument("--fill-holes-resolution", type=int, default=1024, help="Hole-filling visibility-pass rasterization resolution.")
parser.add_argument("--fill-holes-num-views", type=int, default=1000, help="Number of views for the hole-filling visibility pass.")
parser.add_argument("--multiview-resolution", type=int, default=1024, help="Resolution of the multi-view renders used for texture baking.")
parser.add_argument("--multiview-nviews", type=int, default=100, help="Number of views for texture-bake multi-view rendering.")
parser.add_argument("--preview-resolution", type=int, default=256, help="Resolution of the gaussian/mesh preview .mp4 videos.")
parser.add_argument("--preview-frames", type=int, default=30, help="Frame count of the preview .mp4 videos.")
parser.add_argument("--skip-previews", action="store_true", help="Skip the gaussian/mesh preview .mp4 renders (saves time).")
parser.add_argument("--mesh-only", action="store_true",
                    help="Export an untextured mesh GLB and skip Gaussian and texture baking.")
parser.add_argument("--include-radiance-field", action="store_true",
                     help="Also decode+render the radiance_field (octree/trivec) format. Opt-in: "
                          "correct (trellis/renderers/octree_renderer_mps.py) but ~18s/frame, "
                          "3-15x slower than the mesh/gaussian previews -- see APPLE_SILICON.md.")
parser.add_argument("--radiance-field-preview-frames", type=int, default=12,
                     help="Frame count for the radiance_field preview video (kept low by default given its per-frame cost).")
parser.add_argument("--device", default=None, help="Torch device; defaults to mps if available, else cpu.")
args = parser.parse_args()

device = args.device or ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"=== Using device: {device} ===", flush=True)

t0 = time.time()
print("Loading pipeline (downloads model on first run)...", flush=True)
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.to(device)
print(f"Pipeline loaded in {time.time()-t0:.1f}s", flush=True)

image_path = args.image
out_prefix = args.out_prefix
image = Image.open(image_path)
print(f"Loaded image: {image_path} {image.size} {image.mode}", flush=True)

if args.mesh_only and args.include_radiance_field:
    parser.error("--mesh-only cannot be combined with --include-radiance-field")
formats = ['mesh'] if args.mesh_only else ['mesh', 'gaussian'] + (['radiance_field'] if args.include_radiance_field else [])
t1 = time.time()
outputs = pipeline.run(
    image,
    seed=args.seed,
    formats=formats,
)
print(f"\nPipeline.run() completed in {time.time()-t1:.1f}s", flush=True)

mesh = outputs['mesh'][0]
print(f"mesh: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces, success={mesh.success}")
print(f"mesh device: {mesh.vertices.device}")
print(f"mesh vertices finite: {torch.isfinite(mesh.vertices).all().item()}")
print(f"mesh vertices range: [{mesh.vertices.min().item():.3f}, {mesh.vertices.max().item():.3f}]")
if not args.mesh_only:
    gaussian = outputs['gaussian'][0]
    print(f"gaussian: {gaussian.get_xyz.shape[0]} splats")
    print(f"gaussian device: {gaussian.get_xyz.device}")
    print(f"gaussian xyz finite: {torch.isfinite(gaussian.get_xyz).all().item()}")
    print(f"gaussian xyz range: [{gaussian.get_xyz.min().item():.3f}, {gaussian.get_xyz.max().item():.3f}]")
if args.include_radiance_field:
    radiance_field = outputs['radiance_field'][0]
    print(f"radiance_field: {radiance_field.position.shape[0]} occupied leaves / {radiance_field.resolution}^3 grid")

out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

if args.mesh_only:
    import trimesh
    vertices = mesh.vertices.cpu().numpy()
    faces = mesh.faces.cpu().numpy()
    finite_faces = np.isfinite(vertices[faces]).all(axis=(1, 2))
    removed = len(faces) - int(finite_faces.sum())
    if removed:
        print(f"Discarding {removed}/{len(faces)} faces with non-finite vertices", flush=True)
    if finite_faces.mean() < 0.95:
        raise RuntimeError("More than 5% of mesh faces have non-finite vertices; refusing to export")
    faces = faces[finite_faces]
    used_vertices, inverse = np.unique(faces, return_inverse=True)
    vertices = vertices[used_vertices]
    faces = inverse.reshape(-1, 3)
    if not np.isfinite(vertices).all():
        raise RuntimeError("Mesh still contains non-finite vertices")
    vertices_rot = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    result = trimesh.Trimesh(vertices=vertices_rot, faces=faces, process=False)
    output_path = os.path.join(out_dir, f"{out_prefix}.glb")
    result.export(output_path)
    print(f"Untextured GLB saved: {output_path}; faces={len(result.faces)}; watertight={result.is_watertight}", flush=True)
    sys.exit(0)

if args.include_radiance_field:
    print("\nRendering radiance-field (octree) preview -- this is the slowest preview stage, "
          "~18s/frame, see APPLE_SILICON.md...", flush=True)
    t_rf = time.time()
    rf_video = render_utils.render_video(
        radiance_field, resolution=args.preview_resolution, num_frames=args.radiance_field_preview_frames)['color']
    import imageio as _imageio_rf
    _imageio_rf.mimsave(f"{out_dir}/{out_prefix}_rf.mp4", rf_video, fps=8)
    _imageio_rf.mimsave(f"{out_dir}/{out_prefix}_rf.gif", rf_video, fps=8)
    print(f"radiance-field video rendered in {time.time()-t_rf:.1f}s -> {out_dir}/{out_prefix}_rf.mp4 (+.gif)", flush=True)

if not args.skip_previews:
    print("\nRendering gaussian video...", flush=True)
    t2 = time.time()
    video = render_utils.render_video(gaussian, resolution=args.preview_resolution, num_frames=args.preview_frames)['color']
    import imageio
    imageio.mimsave(f"{out_dir}/{out_prefix}_gs.mp4", video, fps=15)
    # Same frames, written as a .gif too -- a web UI can show this well before the full pipeline
    # (hole-filling, texture bake) finishes, since <img> autoplays an animated gif with no extra
    # markup/JS, unlike <video> which needs autoplay/loop/muted attributes to behave the same way.
    imageio.mimsave(f"{out_dir}/{out_prefix}_gs.gif", video, fps=15)
    print(f"gaussian video rendered in {time.time()-t2:.1f}s -> {out_dir}/{out_prefix}_gs.mp4 (+.gif)", flush=True)

    print("\nRendering mesh normal video...", flush=True)
    t3 = time.time()
    video = render_utils.render_video(mesh, resolution=args.preview_resolution, num_frames=args.preview_frames)['normal']
    imageio.mimsave(f"{out_dir}/{out_prefix}_mesh.mp4", video, fps=15)
    imageio.mimsave(f"{out_dir}/{out_prefix}_mesh.gif", video, fps=15)
    print(f"mesh video rendered in {time.time()-t3:.1f}s -> {out_dir}/{out_prefix}_mesh.mp4 (+.gif)", flush=True)

print("\nBaking texture + exporting glb...", flush=True)
t4 = time.time()
# Replicates postprocessing_utils.to_glb()'s internals (rather than calling it directly) so we
# can save a handful of the multi-view renders that feed bake_texture -- those are normally
# ephemeral, generated in-memory and never written to disk.
import trimesh
simplify, texture_size = args.simplify, args.texture_size
mesh_device = mesh.vertices.device
vertices = mesh.vertices.cpu().numpy()
faces = mesh.faces.cpu().numpy()

vertices, faces = postprocessing_utils.postprocess_mesh(
    vertices, faces,
    simplify=simplify > 0, simplify_ratio=simplify,
    fill_holes=True,
    fill_holes_max_hole_size=0.04,
    fill_holes_max_hole_nbe=int(250 * np.sqrt(1 - simplify)),
    fill_holes_resolution=args.fill_holes_resolution, fill_holes_num_views=args.fill_holes_num_views,
    verbose=True, device=mesh_device,
)
vertices, faces, uvs = postprocessing_utils.parametrize_mesh(vertices, faces)

# Inlined equivalent of render_utils.render_multiview() (same camera sampling, same renderer) --
# not called directly because that function only returns once every view is done. Here, each
# view is also written to a fixed "live" path + a small progress sidecar as it's produced, so a
# web UI (or anyone else polling outputs/) can show the multi-view render in progress instead of
# only after all nviews views (and everything downstream: texture bake, glb export) complete.
nviews = args.multiview_nviews
cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
mv_yaws = [cam[0] for cam in cams]
mv_pitchs = [cam[1] for cam in cams]
mv_extrinsics, mv_intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
    mv_yaws, mv_pitchs, 2, 40, device=gaussian.get_xyz.device)
mv_renderer = get_renderer(gaussian, resolution=args.multiview_resolution, bg_color=(0, 0, 0))

live_preview_path = f"{out_dir}/{out_prefix}_multiview_live.png"
live_status_path = f"{out_dir}/{out_prefix}_multiview_live.json"

observations = []
for i, (extr, intr) in enumerate(tqdm(list(zip(mv_extrinsics, mv_intrinsics)), desc='Rendering multi-view')):
    res = mv_renderer.render(gaussian, extr, intr)
    frame = np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    observations.append(frame)
    Image.fromarray(frame).save(live_preview_path)
    with open(live_status_path, 'w') as f:
        json.dump({"current": i + 1, "total": nviews}, f)
extrinsics, intrinsics = mv_extrinsics, mv_intrinsics

sample_dir = f"{out_dir}/multiview_samples/{out_prefix}"
os.makedirs(sample_dir, exist_ok=True)
sample_idx = sorted(set([0, len(observations) // 4, len(observations) // 2, 3 * len(observations) // 4, len(observations) - 1]))
for i in sample_idx:
    Image.fromarray(observations[i]).save(f"{sample_dir}/view_{i:03d}.png")
print(f"saved {len(sample_idx)} of {len(observations)} multi-view renders -> {sample_dir}/", flush=True)

masks = [np.any(observation > 0, axis=-1) for observation in observations]
extrinsics_np = [extrinsics[i].cpu().numpy() for i in range(len(extrinsics))]
intrinsics_np = [intrinsics[i].cpu().numpy() for i in range(len(intrinsics))]
texture = postprocessing_utils.bake_texture(
    vertices, faces, uvs,
    observations, masks, extrinsics_np, intrinsics_np,
    texture_size=texture_size, mode='opt',
    lambda_tv=0.01, verbose=True, device=mesh_device,
)
texture_img = Image.fromarray(texture)

vertices_rot = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
material = trimesh.visual.material.PBRMaterial(
    roughnessFactor=1.0, baseColorTexture=texture_img,
    baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8)
)
glb = trimesh.Trimesh(vertices_rot, faces, visual=trimesh.visual.TextureVisuals(uv=uvs, material=material))
glb.export(f"{out_dir}/{out_prefix}.glb")
print(f"glb exported in {time.time()-t4:.1f}s -> {out_dir}/{out_prefix}.glb", flush=True)

gaussian.save_ply(f"{out_dir}/{out_prefix}.ply")
print(f"ply saved -> {out_dir}/{out_prefix}.ply", flush=True)

print(f"\n=== TOTAL TIME: {time.time()-t0:.1f}s ===", flush=True)
print("=== DONE ===", flush=True)
