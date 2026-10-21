#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exercise the paint pipeline's renderer and baker on the active backend.

The paint pipeline has two halves. The diffusion half needs ~6.5 GB of weights;
the other half — UV unwrap, rasterization, back-projection, cosine-weighted
baking, UV inpainting, OBJ/GLB export — needs none, and is where a new backend
actually breaks. `torch_rasterizer`'s own tests cover synthetic triangles;
this covers the real `MeshRender` on a real mesh, with `paint.apply_patches`
installed exactly as the app installs them.

So it is not a substitute for an end-to-end texturing run — it never loads the
paint UNet — but it is the part of that run you can check without the download,
and it is the part that is backend-specific.

The strongest check here is the last one: the whole render-and-bake is run twice,
once on CPU and once on the accelerator, and the two textures are compared. The
CPU result is the reference implementation by construction, so a device-specific
arithmetic fault shows up as a diverging bake rather than as a plausible-looking
texture nobody can falsify.

    python tests/test_paint_render.py
    python tests/test_paint_render.py --texture-size 2048 --skip-cpu
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import backend as backend_mod  # noqa: E402  (before torch — sets the MPS fallback)
import torch  # noqa: E402

# Six views around and above the object: enough that every triangle is seen by
# at least one of them, so an unbaked region means a real coverage bug rather
# than a view the object simply hides from.
VIEWS = [(0, 0), (0, 90), (0, 180), (0, 270), (90, 0), (-90, 0)]

_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:44s} {detail}")
    return ok


def build_mesh():
    """A UV-unwrapped test mesh with enough curvature to exercise the bake.

    An icosphere would be too kind: every face is the same size and faces the
    camera squarely. Scaling it into an ellipsoid and denting one pole gives
    grazing-angle faces, which is what the cosine weighting and the boundary
    detection exist to handle.
    """
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    v = np.asarray(mesh.vertices, dtype=np.float64).copy()
    v[:, 0] *= 1.35
    v[:, 2] *= 0.7
    dent = v[:, 1] > 0.35
    v[dent] *= 0.75
    return trimesh.Trimesh(vertices=v, faces=np.asarray(mesh.faces), process=False)


def run_pipeline(device: str, mesh, render_size: int, texture_size: int, out_dir: Path,
                 tag: str, save: bool):
    """Render, back-project, bake and inpaint on one device. Returns artifacts."""
    from DifferentiableRenderer.MeshRender import MeshRender
    from utils.uvwrap_utils import mesh_uv_wrap

    render = MeshRender(
        default_resolution=render_size,
        texture_size=texture_size,
        bake_mode="back_sample",
        raster_mode="cr",
        device=device,
    )
    render.load_mesh(mesh=mesh_uv_wrap(mesh.copy()))

    # Stand-in for the multiview diffusion output. Position renders are smooth,
    # view-dependent and strongly coloured, so a mis-projected view shows up as a
    # discontinuity in the baked atlas instead of blending invisibly away.
    views = [render.render_position(elev, azim, return_type="pl") for elev, azim in VIEWS]

    textures, cos_maps = [], []
    for image, (elev, azim) in zip(views, VIEWS):
        tex, cos_map, _boundary = render.back_project(image, elev, azim)
        textures.append(tex)
        cos_maps.append(cos_map * 1.0)
    texture, trust = render.fast_bake_texture(textures, cos_maps)

    mask = (trust.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
    inpainted = render.uv_inpaint(texture, mask)

    saved = None
    if save:
        render.set_texture(inpainted / 255.0)
        # A stand-in metallic-roughness map, so the export exercises the same
        # two-texture path a real bake takes rather than the albedo-only one.
        # Channel 0 is metallic, channel 1 roughness (see get_texture_mr).
        mr = np.zeros_like(inpainted, dtype=np.float32)
        mr[..., 0] = 0.1
        mr[..., 1] = 0.6
        render.set_texture_mr(mr)

        import mesh_io

        saved = mesh_io.save_textured_mesh(
            render, str(out_dir / f"{tag}.obj"), glb_path=str(out_dir / f"{tag}.glb"),
            downsample=False,
        )

    return dict(render=render, views=views, texture=texture, trust=trust,
                mask=mask, inpainted=inpainted, saved=saved)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/paint_render")
    ap.add_argument("--render-size", type=int, default=512)
    ap.add_argument("--texture-size", type=int, default=1024)
    ap.add_argument("--skip-cpu", action="store_true",
                    help="skip the CPU reference run (faster, but loses the cross-check)")
    args = ap.parse_args()

    out_dir = PROJECT_DIR / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    b = backend_mod.detect()
    print(backend_mod.describe(b))

    import paint

    ok, reason = paint.availability()
    # The paint *weights* are irrelevant here; only the code and the native
    # inpaint module are. Say so explicitly rather than skipping on a message
    # about a 6.5 GB download that this test never touches.
    if not ok and "mesh_inpaint_processor" in reason:
        print(f"cannot run: {reason}")
        return 2
    if not ok:
        print(f"note: {reason} (not needed by this test)")

    paint.apply_patches(b)
    check("paint.apply_patches on this backend", True, f"{b.kind}/{b.raster}")

    from PIL import Image

    mesh = build_mesh()
    print(f"test mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    t0 = time.time()
    try:
        got = run_pipeline(b.device, mesh, args.render_size, args.texture_size,
                           out_dir, "device", save=True)
    except Exception as e:
        check(f"render + bake on {b.device}", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    elapsed = time.time() - t0
    check(f"render + bake on {b.device}", True, f"{elapsed:.1f} s")

    # --- the renders themselves -------------------------------------------
    view_arrays = [np.asarray(v.convert("RGB"), dtype=np.float32) / 255.0 for v in got["views"]]
    for i, arr in enumerate(view_arrays):
        elev, azim = VIEWS[i]
        # A view that came back blank means the rasterizer produced no coverage,
        # which on a sphere at any angle can only be a bug.
        covered = float((arr.max(axis=-1) < 0.99).mean())
        if not check(f"view {i} (elev={elev}, azim={azim}) has coverage",
                     covered > 0.05, f"{covered * 100:.1f}% of pixels"):
            break
    for i, v in enumerate(got["views"]):
        v.save(out_dir / f"view{i}.png")

    # Opposite views must differ: identical ones mean the camera matrix ignored
    # the azimuth, which renders perfectly and bakes nonsense.
    d = float(np.abs(view_arrays[0] - view_arrays[2]).mean())
    check("front and back views differ", d > 0.02, f"mean abs diff {d:.3f}")

    # --- the bake ----------------------------------------------------------
    coverage = float((got["mask"] > 0).mean())
    check("baked UV coverage", coverage > 0.25, f"{coverage * 100:.1f}% of the atlas")

    tex = got["texture"].cpu().numpy()
    check("baked texture is finite", bool(np.isfinite(tex).all()))
    check("baked texture is not flat", float(tex.std()) > 0.02, f"std {tex.std():.3f}")

    inp = got["inpainted"]
    check("uv_inpaint returned an image", inp.dtype == np.uint8 and inp.ndim == 3,
          f"{inp.shape} {inp.dtype}")
    # Inpainting exists to fill the atlas gaps; if the output still has as many
    # empty texels as the mask had, mesh_inpaint_processor did nothing.
    holes_before = float((got["mask"] == 0).mean())
    holes_after = float((inp.max(axis=-1) == 0).mean())
    check("uv_inpaint filled the gaps", holes_after < holes_before * 0.5,
          f"{holes_before * 100:.1f}% -> {holes_after * 100:.1f}% empty")
    Image.fromarray(inp).save(out_dir / "texture_baked.png")

    # --- export ------------------------------------------------------------
    saved = got["saved"]
    check("OBJ written", bool(saved and Path(saved["obj"]).exists()))
    glb = saved.get("glb") if saved else None
    check("GLB written", bool(glb and Path(glb).exists()))
    if glb and Path(glb).exists():
        import trimesh

        reloaded = trimesh.load(glb, force="mesh", process=False)
        visual = getattr(reloaded, "visual", None)
        uv = getattr(visual, "uv", None)
        check("GLB round-trips with UVs", uv is not None and len(uv) == len(reloaded.vertices),
              f"{0 if uv is None else len(uv)} uvs / {len(reloaded.vertices)} verts")
        material = getattr(visual, "material", None)
        base = getattr(material, "baseColorTexture", None) if material else None
        check("GLB carries a base-colour texture", base is not None,
              "" if base is None else f"{base.size}")

    # --- CPU cross-check ---------------------------------------------------
    if args.skip_cpu:
        print("  [skip] CPU reference run (--skip-cpu)")
    elif b.device == "cpu":
        print("  [skip] CPU reference run (already the active device)")
    else:
        t0 = time.time()
        try:
            ref = run_pipeline("cpu", mesh, args.render_size, args.texture_size,
                               out_dir, "cpu", save=False)
        except Exception as e:
            check("render + bake on cpu (reference)", False, f"{type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
        check("render + bake on cpu (reference)", True, f"{time.time() - t0:.1f} s")

        m_dev, m_ref = got["mask"] > 0, ref["mask"] > 0
        agree = float((m_dev == m_ref).mean())
        check(f"{b.device} and cpu agree on which texels are covered", agree > 0.999,
              f"{agree * 100:.3f}% of texels")

        both = m_dev & m_ref
        diff = np.abs(got["texture"].cpu().numpy() - ref["texture"].cpu().numpy())[both]
        # fp32 throughout, so this is accumulation order, not precision: the bake
        # sums six weighted views. Anything past 1e-3 is a different computation.
        check(f"{b.device} and cpu bake the same texture",
              float(diff.max()) < 1e-3,
              f"max {diff.max():.2e}, mean {diff.mean():.2e}")

    # --- preview -----------------------------------------------------------
    if glb and Path(glb).exists():
        try:
            import render_preview as rp

            verts, faces, uv, albedo = rp.load_mesh(glb)
            normals = rp.vertex_normals(verts, faces)
            device = rp.preview_device()
            tiles = [rp.render(verts, faces, normals, 90.0 * i, 15.0, 384, device,
                               uv=uv, albedo=albedo) for i in range(4)]
            sheet = out_dir / "turntable.png"
            Image.fromarray(np.concatenate(tiles, axis=1)).save(sheet)
            check("textured turntable rendered", True, str(sheet.name))
        except Exception as e:
            check("textured turntable rendered", False, f"{type(e).__name__}: {e}")
            traceback.print_exc()

    failed = [n for n, ok, _ in _results if not ok]
    print()
    print(f"artifacts: {out_dir}")
    if failed:
        print(f"\n{len(failed)} CHECK(S) FAILED: {', '.join(failed)}")
        return 1
    print(f"\nALL {len(_results)} PAINT-RENDER CHECKS PASSED on {b.kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
