#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end visual test: image -> mesh -> PBR texture -> rendered report.

Calls the *production* handlers from `gradio_app` rather than reimplementing the
pipeline, so what this exercises is exactly what the UI runs — including the
backend detection, the compatibility patches, the mesh state dict and the
shape-pipeline unload before texturing.

Previews are rendered with `tests/render_preview.py`, which rasterizes through
`torch_rasterizer`. That is deliberate: a preview that comes out right is
evidence the rasterizer handles real generated geometry, not just synthetic
triangles.

Writes PNGs plus a self-contained `report.html` into --out.

    python tests/e2e_visual.py --steps 30 --octree 192
    python tests/e2e_visual.py --skip-texture       # shape only
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


class Stage:
    """One test stage with its own status, timing and captured artifacts."""

    def __init__(self, name: str, detail: str = "") -> None:
        self.name = name
        self.detail = detail
        self.status = "pending"
        self.elapsed = 0.0
        self.log = ""
        self.images: list[tuple[str, Path]] = []
        self.error = ""

    def to_dict(self) -> dict:
        return dict(name=self.name, status=self.status, elapsed=round(self.elapsed, 2),
                    detail=self.detail, error=self.error)


def render_views(mesh_path: str, out_dir: Path, stem: str, views: int, resolution: int):
    """Turntable render via torch_rasterizer. Returns (sheet_path, per_view_paths)."""
    from PIL import Image
    import render_preview as rp

    device = rp.preview_device()
    verts, faces, uv, albedo = rp.load_mesh(mesh_path)
    normals = rp.vertex_normals(verts, faces)

    tiles, paths = [], []
    for i in range(views):
        azim = 360.0 * i / views
        img = rp.render(verts, faces, normals, azim, 15.0, resolution, device,
                        uv=uv, albedo=albedo)
        tiles.append(img)
        p = out_dir / f"{stem}_view{i}.png"
        Image.fromarray(img).save(p)
        paths.append(p)

    sheet = out_dir / f"{stem}_turntable.png"
    Image.fromarray(np.concatenate(tiles, axis=1)).save(sheet)
    return sheet, paths, dict(verts=len(verts), faces=len(faces),
                              textured=albedo is not None,
                              texture_size=None if albedo is None else list(albedo.shape[:2]))


def build_report(out_dir: Path, stages: list[Stage], env: dict, extra_images: list[tuple[str, Path]]):
    """One self-contained HTML file — no external assets, so it can be moved."""
    css = """
    :root { color-scheme: dark light; }
    body { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
           margin: 0; padding: 2rem; background: #101014; color: #e6e6ea; line-height: 1.55; }
    h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
    h2 { font-size: 1.15rem; margin: 2rem 0 .75rem; border-bottom: 1px solid #2a2a33; padding-bottom: .4rem; }
    .sub { color: #9a9aa8; margin: 0 0 1.5rem; font-size: .9rem; }
    table { border-collapse: collapse; width: 100%; max-width: 900px; font-size: .9rem; }
    th, td { text-align: left; padding: .45rem .7rem; border-bottom: 1px solid #23232b; vertical-align: top; }
    th { color: #9a9aa8; font-weight: 600; }
    code { background: #1c1c24; padding: .1rem .35rem; border-radius: 4px; font-size: .85em; }
    .pill { display: inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .78rem; font-weight: 600; }
    .ok { background: #16351f; color: #7ee39b; }
    .fail { background: #3a1a1c; color: #ff9098; }
    .skip { background: #2a2a33; color: #9a9aa8; }
    figure { margin: 0 0 1.5rem; }
    figcaption { color: #9a9aa8; font-size: .82rem; margin-top: .4rem; }
    img { max-width: 100%; height: auto; border-radius: 6px; display: block;
          background: #0a0a0c; border: 1px solid #23232b; }
    .scroll { overflow-x: auto; }
    pre { background: #16161c; border: 1px solid #23232b; border-radius: 6px;
          padding: .8rem; overflow-x: auto; font-size: .8rem; color: #c8c8d4; }
    @media (prefers-color-scheme: light) {
      body { background: #fafafa; color: #1a1a1f; }
      th, td { border-color: #e4e4ea; } th { color: #6a6a78; }
      code, pre { background: #f0f0f4; color: #24242c; border-color: #e4e4ea; }
      img { background: #fff; border-color: #e4e4ea; }
      .sub { color: #6a6a78; } h2 { border-color: #e4e4ea; }
      .ok { background: #e3f6e9; color: #14702f; } .fail { background: #fdeaec; color: #a01722; }
      .skip { background: #eee; color: #6a6a78; }
    }
    """

    def pill(status: str) -> str:
        cls = {"pass": "ok", "fail": "fail"}.get(status, "skip")
        return f'<span class="pill {cls}">{html.escape(status.upper())}</span>'

    parts = [f"<style>{css}</style>",
             "<h1>Hunyuan3D 2.1 — ROCm visual test report</h1>",
             f'<p class="sub">{html.escape(env.get("generated_at", ""))}</p>']

    parts.append("<h2>Environment</h2><div class='scroll'><table>")
    for k, v in env.items():
        if k == "generated_at":
            continue
        parts.append(f"<tr><th>{html.escape(k)}</th><td><code>{html.escape(str(v))}</code></td></tr>")
    parts.append("</table></div>")

    parts.append("<h2>Stages</h2><div class='scroll'><table>")
    parts.append("<tr><th>Stage</th><th>Result</th><th>Time</th><th>Notes</th></tr>")
    for s in stages:
        note = html.escape(s.error or s.detail)
        parts.append(
            f"<tr><td>{html.escape(s.name)}</td><td>{pill(s.status)}</td>"
            f"<td>{s.elapsed:.1f} s</td><td>{note}</td></tr>"
        )
    parts.append("</table></div>")

    images = [(cap, p) for s in stages for cap, p in s.images] + extra_images
    if images:
        parts.append("<h2>Images</h2>")
        for caption, path in images:
            if not path.exists():
                continue
            parts.append(
                f'<figure><img src="data:image/png;base64,{_b64(path)}" alt="{html.escape(caption)}">'
                f"<figcaption>{html.escape(caption)} — <code>{html.escape(path.name)}</code></figcaption></figure>"
            )

    logs = [(s.name, s.log) for s in stages if s.log.strip()]
    if logs:
        parts.append("<h2>Logs</h2>")
        for name, text in logs:
            parts.append(f"<h3>{html.escape(name)}</h3><pre>{html.escape(text)}</pre>")

    report = out_dir / "report.html"
    report.write_text("\n".join(parts), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/e2e", help="output directory")
    ap.add_argument("--image", default=None, help="input image (default: upstream assets/demo.png)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--octree", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--views", type=int, default=4, help="preview turntable views")
    ap.add_argument("--resolution", type=int, default=448, help="preview render resolution")
    ap.add_argument("--tex-preset", default="safe", choices=("safe", "low", "normal"))
    ap.add_argument("--tex-views", type=int, default=6)
    ap.add_argument("--tex-resolution", type=int, default=512)
    ap.add_argument("--raster", default="auto", choices=("auto", "gpu", "cpu", "torch"))
    ap.add_argument("--skip-texture", action="store_true")
    ap.add_argument(
        "--mesh",
        default=None,
        help="texture an existing mesh instead of generating one "
             "(skips the shape stage, which costs minutes)",
    )
    ap.add_argument("--track-gpu", action="store_true", help="measure the GPU vs CPU op share")
    args = ap.parse_args()

    out_dir = PROJECT_DIR / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    image_path = Path(args.image) if args.image else (
        PROJECT_DIR / "Hunyuan3D-2.1" / "assets" / "demo.png"
    )
    if not image_path.exists():
        print(f"input image not found: {image_path}", file=sys.stderr)
        return 2

    import backend as backend_mod

    b = backend_mod.detect()
    env = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": b.kind,
        "device": b.device,
        "dtype": str(b.dtype).replace("torch.", ""),
        "rasterizer": b.raster,
        "gpu": b.gpu_name,
        "gpu_memory_gib": f"{backend_mod.gpu_memory_gb(b):.1f}" if backend_mod.gpu_memory_gb(b) else "?",
        "version": b.version,
        "input_image": str(image_path),
        "shape_params": f"steps={args.steps} guidance={args.guidance} octree={args.octree} seed={args.seed}",
    }
    print(json.dumps(env, indent=2, ensure_ascii=False))

    stages: list[Stage] = []
    extra_images: list[tuple[str, Path]] = []

    # Copy the input next to the report so the report is self-explanatory.
    src_png = out_dir / "input.png"
    Image.open(image_path).convert("RGBA").save(src_png)
    extra_images.append(("Input image", src_png))

    # Importing gradio_app builds the Blocks; that is intentional, it verifies the
    # UI graph constructs on this machine as a side effect of the test.
    st_import = Stage("import gradio_app (builds the UI graph)")
    stages.append(st_import)
    t0 = time.time()
    try:
        import gradio_app
        st_import.status = "pass"
        st_import.detail = "UI graph constructed"
    except Exception as e:
        st_import.status = "fail"
        st_import.error = f"{type(e).__name__}: {e}"
        st_import.log = traceback.format_exc()
        st_import.elapsed = time.time() - t0
        report = build_report(out_dir, stages, env, extra_images)
        print(f"\nreport: {report}")
        return 1
    st_import.elapsed = time.time() - t0

    image = Image.open(image_path).convert("RGBA")

    # ---- shape ------------------------------------------------------------
    mesh_state = None
    raw_glb = None

    if args.mesh:
        # Reuse an existing mesh so the texture stage can be iterated on without
        # repaying the multi-minute shape generation every time.
        import trimesh

        st_shape = Stage("reuse existing mesh (shape stage skipped)", args.mesh)
        stages.append(st_shape)
        t0 = time.time()
        try:
            loaded = trimesh.load(args.mesh, force="mesh", process=False)
            mesh_state = {
                "original": loaded,
                "mesh": loaded,
                "original_faces": int(len(loaded.faces)),
                "original_vertices": int(len(loaded.vertices)),
                "last_op": "reused",
                "image": image,
            }
            raw_glb = args.mesh
            st_shape.status = "pass"
            st_shape.detail = f"{len(loaded.vertices)} verts, {len(loaded.faces)} faces"
        except Exception as e:
            st_shape.status = "fail"
            st_shape.error = f"{type(e).__name__}: {e}"
            st_shape.log = traceback.format_exc()
        st_shape.elapsed = time.time() - t0
    else:
        st_shape = Stage("shape generation (DiT + VAE)", env["shape_params"])
        stages.append(st_shape)
        t0 = time.time()
        try:
            raw_glb, mesh_state, log_text, stats_md = gradio_app.generate_3d(
                image, True, args.steps, args.guidance, args.octree, args.seed,
                args.track_gpu, "en",
            )
            st_shape.log = log_text or ""
            if not raw_glb or not mesh_state:
                st_shape.status = "fail"
                st_shape.error = "no mesh returned"
            else:
                st_shape.status = "pass"
                st_shape.detail = (
                    f"{mesh_state['original_vertices']} verts, "
                    f"{mesh_state['original_faces']} faces"
                )
                if stats_md:
                    st_shape.log += "\n\n" + stats_md
        except Exception as e:
            st_shape.status = "fail"
            st_shape.error = f"{type(e).__name__}: {e}"
            st_shape.log = traceback.format_exc()
        st_shape.elapsed = time.time() - t0

    # ---- preview of the raw mesh -----------------------------------------
    if raw_glb:
        st_prev = Stage("render untextured mesh (torch_rasterizer)")
        stages.append(st_prev)
        t0 = time.time()
        try:
            sheet, _, info = render_views(raw_glb, out_dir, "shape", args.views, args.resolution)
            st_prev.status = "pass"
            st_prev.detail = f"{info['verts']} verts, {info['faces']} faces"
            st_prev.images.append(("Generated geometry — turntable, flat shaded", sheet))
        except Exception as e:
            st_prev.status = "fail"
            st_prev.error = f"{type(e).__name__}: {e}"
            st_prev.log = traceback.format_exc()
        st_prev.elapsed = time.time() - t0

    # ---- texture ----------------------------------------------------------
    if not args.skip_texture and mesh_state:
        import paint

        ok, reason = paint.availability()
        st_tex = Stage(
            "PBR texturing",
            f"preset={args.tex_preset} views={args.tex_views} res={args.tex_resolution} raster={args.raster}",
        )
        stages.append(st_tex)
        if not ok:
            st_tex.status = "skipped"
            st_tex.error = reason
        else:
            t0 = time.time()
            tex_path = None
            try:
                tex_path, gallery, tex_state, tex_log = gradio_app.texture_3d_handler(
                    mesh_state, image, args.tex_preset, args.tex_views,
                    args.tex_resolution, args.raster, True, "en", None,
                )
                st_tex.log = tex_log or ""
                if not tex_path:
                    st_tex.status = "fail"
                    st_tex.error = "no textured mesh returned (see log)"
                else:
                    st_tex.status = "pass"
                    st_tex.detail += f" -> {Path(tex_path).name}"
                    for item in gallery or []:
                        p = Path(item[0])
                        if p.exists() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                            png = out_dir / f"map_{item[1]}.png"
                            Image.open(p).convert("RGB").save(png)
                            st_tex.images.append((f"PBR map: {item[1]}", png))
            except Exception as e:
                st_tex.status = "fail"
                st_tex.error = f"{type(e).__name__}: {e}"
                st_tex.log = traceback.format_exc()
            st_tex.elapsed = time.time() - t0

            if tex_path:
                st_tp = Stage("render textured mesh (torch_rasterizer)")
                stages.append(st_tp)
                t0 = time.time()
                # Release the paint model before rendering the preview. Rendering
                # straight after texturing in the same process produced a corrupt
                # frame once (near-black, while the very same file rendered fine
                # in a fresh process), so do not read the GPU while it is still
                # holding the pipeline.
                try:
                    paint.unload()
                    backend_mod.empty_cache(b)
                except Exception:
                    pass
                try:
                    sheet, _, info = render_views(tex_path, out_dir, "textured",
                                                  args.views, args.resolution)
                    st_tp.status = "pass"
                    st_tp.detail = (
                        f"{info['verts']} verts, {info['faces']} faces, "
                        f"texture {info['texture_size']}"
                    )
                    st_tp.images.append(("Textured result — turntable, albedo sampled", sheet))
                except Exception as e:
                    st_tp.status = "fail"
                    st_tp.error = f"{type(e).__name__}: {e}"
                    st_tp.log = traceback.format_exc()
                st_tp.elapsed = time.time() - t0

    # ---- rasterizer conformance strips ------------------------------------
    vr_dir = PROJECT_DIR / "outputs" / "visual_raster"
    if vr_dir.is_dir():
        for p in sorted(vr_dir.glob("*.png")):
            extra_images.append((f"Rasterizer conformance: {p.stem}", p))

    report = build_report(out_dir, stages, env, extra_images)

    print()
    for s in stages:
        print(f"  [{s.status.upper():7s}] {s.name} ({s.elapsed:.1f} s) {s.error}")
    print(f"\nreport: {report}")
    (out_dir / "summary.json").write_text(
        json.dumps({"env": env, "stages": [s.to_dict() for s in stages]}, indent=2),
        encoding="utf-8",
    )
    return 0 if all(s.status in ("pass", "skipped") for s in stages) else 1


if __name__ == "__main__":
    raise SystemExit(main())
