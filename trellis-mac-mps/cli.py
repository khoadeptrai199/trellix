#!/usr/bin/env python3
"""
trellis-mps command-line interface.

Subcommands:
  generate    Run the full image-to-3D pipeline on one image, streaming progress live.
  serve       Launch the local web UI (webapp/app.py).
  list        List generated assets in outputs/.
  examples    List bundled example images (assets/example_image/).

See APPLE_SILICON.md for what this port changes vs upstream TRELLIS, and README.md for setup.
"""
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs")
EXAMPLES_DIR = os.path.join(REPO_ROOT, "assets", "example_image")
PIPELINE_SCRIPT = os.path.join(REPO_ROOT, "scripts", "run_real_pipeline.py")


def cmd_generate(args):
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, PIPELINE_SCRIPT, args.image, args.out_prefix, "--out-dir", out_dir]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    if args.simplify is not None:
        cmd += ["--simplify", str(args.simplify)]
    if args.texture_size is not None:
        cmd += ["--texture-size", str(args.texture_size)]
    if args.fill_holes_resolution is not None:
        cmd += ["--fill-holes-resolution", str(args.fill_holes_resolution)]
    if args.fill_holes_num_views is not None:
        cmd += ["--fill-holes-num-views", str(args.fill_holes_num_views)]
    if args.multiview_nviews is not None:
        cmd += ["--multiview-nviews", str(args.multiview_nviews)]
    if args.skip_previews:
        cmd.append("--skip-previews")
    if args.mesh_only:
        cmd.append("--mesh-only")
    if args.include_radiance_field:
        cmd.append("--include-radiance-field")
    if args.device:
        cmd += ["--device", args.device]

    print(f"$ {' '.join(cmd)}\n", flush=True)
    # Streams stdout/stderr straight to this terminal -- same tqdm progress bars you'd see running
    # scripts/run_real_pipeline.py directly, no need to duplicate any of that rendering here.
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    sys.exit(proc.returncode)


def cmd_serve(args):
    webapp_script = os.path.join(REPO_ROOT, "webapp", "app.py")
    os.environ.setdefault("FLASK_RUN_PORT", str(args.port))
    print(f"Starting web UI on http://127.0.0.1:{args.port}", flush=True)
    proc = subprocess.run([sys.executable, webapp_script], cwd=os.path.join(REPO_ROOT, "webapp"))
    sys.exit(proc.returncode)


def cmd_list(args):
    if not os.path.isdir(OUTPUT_DIR):
        print("No outputs yet.")
        return
    glbs = sorted(f for f in os.listdir(OUTPUT_DIR) if f.endswith(".glb"))
    if not glbs:
        print("No generated assets yet.")
        return
    for glb in glbs:
        prefix = glb[:-4]
        ply_exists = os.path.exists(os.path.join(OUTPUT_DIR, f"{prefix}.ply"))
        size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, glb)) / (1024 * 1024)
        print(f"{prefix:30s}  {size_mb:6.1f}MB glb{'  + ply' if ply_exists else ''}")


def cmd_examples(args):
    for f in sorted(os.listdir(EXAMPLES_DIR)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            print(f)


def main():
    parser = argparse.ArgumentParser(prog="trellis-mps", description="TRELLIS image-to-3D, ported to run natively on Apple Silicon.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="Generate a 3D asset from an image.")
    p_gen.add_argument("image", help="Path to input image (or a filename from `trellis-mps examples`).")
    p_gen.add_argument("out_prefix", nargs="?", default="sample", help="Output filename prefix.")
    p_gen.add_argument("--out-dir", dest="out_dir", default=None, help="Output directory (default: ./outputs next to the repo).")
    p_gen.add_argument("--seed", type=int)
    p_gen.add_argument("--simplify", type=float, help="Mesh decimation ratio (default 0.95).")
    p_gen.add_argument("--texture-size", type=int, dest="texture_size")
    p_gen.add_argument("--fill-holes-resolution", type=int, dest="fill_holes_resolution")
    p_gen.add_argument("--fill-holes-num-views", type=int, dest="fill_holes_num_views")
    p_gen.add_argument("--multiview-nviews", type=int, dest="multiview_nviews")
    p_gen.add_argument("--skip-previews", action="store_true", help="Skip preview .mp4 renders (faster).")
    p_gen.add_argument("--mesh-only", action="store_true",
                       help="Export an untextured mesh GLB for printing; skips Gaussian, previews, and texture baking.")
    p_gen.add_argument("--include-radiance-field", action="store_true",
                        help="Also render the radiance_field (octree) preview. Opt-in: correct but "
                             "~18s/frame, 3-15x slower than mesh/gaussian previews -- see APPLE_SILICON.md.")
    p_gen.add_argument("--device", default=None, help="mps (default if available) or cpu.")
    p_gen.set_defaults(func=cmd_generate)

    p_serve = sub.add_parser("serve", help="Launch the local web UI.")
    p_serve.add_argument("--port", type=int, default=5050)
    p_serve.set_defaults(func=cmd_serve)

    p_list = sub.add_parser("list", help="List generated assets in outputs/.")
    p_list.set_defaults(func=cmd_list)

    p_ex = sub.add_parser("examples", help="List bundled example images.")
    p_ex.set_defaults(func=cmd_examples)

    if len(sys.argv) == 1:
        # If input image path resolves relative to examples/, resolve it there for convenience.
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    if getattr(args, "image", None) and not os.path.exists(args.image):
        candidate = os.path.join(EXAMPLES_DIR, args.image)
        if os.path.exists(candidate):
            args.image = candidate
    args.func(args)


if __name__ == "__main__":
    main()
