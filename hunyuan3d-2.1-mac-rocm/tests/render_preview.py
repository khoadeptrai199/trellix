#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render turntable previews of a mesh using this port's own rasterizer.

Two birds: it produces the images needed to actually look at a result, and it
does so through `torch_rasterizer`, so a preview that renders correctly is
direct evidence the rasterizer works on real generated geometry rather than only
on synthetic test triangles.

Deliberately no pyglet/OpenGL/pyrender: those need a display or EGL and are the
usual reason "just render a preview" fails on a headless or Windows box.

    python tests/render_preview.py outputs/mesh.glb --out outputs/preview --views 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# `backend` first, and before torch: it is what puts PYTORCH_ENABLE_MPS_FALLBACK
# into the environment, and torch only reads that while it initialises.
import backend as backend_mod  # noqa: E402
import torch  # noqa: E402
import torch_rasterizer as tr  # noqa: E402


def preview_device() -> torch.device:
    """The device this port would actually compute on.

    Not `cuda if available else cpu`: that spelling demotes every Mac to the CPU,
    since MPS is neither. Going through `backend.detect()` also makes HY3D_BACKEND
    work here, which is how the rasterizer gets bisected against a suspect GPU.
    """
    return torch.device(backend_mod.detect().device)


def load_mesh(path: str):
    """Return (vertices, faces, uv or None, albedo image or None)."""
    import trimesh

    loaded = trimesh.load(path, force="mesh", process=False)
    verts = np.asarray(loaded.vertices, dtype=np.float32)
    faces = np.asarray(loaded.faces, dtype=np.int32)

    uv, albedo = None, None
    visual = getattr(loaded, "visual", None)
    if visual is not None and getattr(visual, "uv", None) is not None:
        uv = np.asarray(visual.uv, dtype=np.float32)
        material = getattr(visual, "material", None)
        tex = getattr(material, "baseColorTexture", None) if material else None
        if tex is None and material is not None:
            tex = getattr(material, "image", None)
        if tex is not None:
            albedo = np.asarray(tex.convert("RGB"), dtype=np.float32) / 255.0
            # Untextured glTF files often carry a 1x1 or 2x2 placeholder colour.
            # Sampling it makes the whole render that flat colour — usually near
            # black — which reads as a broken renderer rather than a missing
            # texture. Treat anything that small as "no texture".
            if min(albedo.shape[:2]) < 8:
                albedo = None
    return verts, faces, uv, albedo


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals, computed here to avoid a trimesh round-trip."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(verts)
    for i in range(3):
        np.add.at(normals, faces[:, i], fn)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    length[length == 0] = 1.0
    return normals / length


def rotation(azim_deg: float, elev_deg: float) -> np.ndarray:
    """Yaw around Y, then pitch around X."""
    az, el = np.radians(azim_deg), np.radians(elev_deg)
    ry = np.array([
        [np.cos(az), 0.0, np.sin(az)],
        [0.0, 1.0, 0.0],
        [-np.sin(az), 0.0, np.cos(az)],
    ], dtype=np.float32)
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(el), -np.sin(el)],
        [0.0, np.sin(el), np.cos(el)],
    ], dtype=np.float32)
    return rx @ ry


def render(
    verts: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    azim: float,
    elev: float,
    resolution: int,
    device: torch.device,
    uv: np.ndarray | None = None,
    albedo: np.ndarray | None = None,
    background: tuple[float, float, float] = (0.09, 0.09, 0.10),
) -> np.ndarray:
    """Rasterize one view and shade it. Returns an HxWx3 uint8 image."""
    R = rotation(azim, elev)

    # Fit the rotated mesh into the frame with a small margin, so every view is
    # framed the same way regardless of the object's aspect.
    vr = verts @ R.T
    nr = normals @ R.T
    center = (vr.max(0) + vr.min(0)) * 0.5
    extent = float(np.abs(vr - center).max()) or 1.0
    vr = (vr - center) / extent * 0.85

    # Orthographic clip space: w = 1, depth increases away from the viewer, which
    # is what the rasterizer's min-token z-buffer expects (nearer = smaller).
    #
    # Both X and Y are negated, for different reasons:
    #
    #   Y — the rasterizer maps +Y to the *last* image row
    #       (sy = (0.5 + 0.5*y) * (H - 1)) while PNG row 0 is the top, so without
    #       this the render is upside down.
    #
    #   X — the camera looks along +Z, i.e. it sits at -Z looking back at the
    #       object. From there world +X is on the *left*. Mapping +X to the right
    #       column mirrors the image; text baked into geometry reads backwards.
    #       Numeric rasterizer tests cannot catch this — only a render can.
    clip = np.concatenate([vr, np.ones((vr.shape[0], 1), np.float32)], 1).astype(np.float32)
    clip[:, 0] = -clip[:, 0]
    clip[:, 1] = -clip[:, 1]
    nr = nr.copy()
    nr[:, 0] = -nr[:, 0]
    nr[:, 1] = -nr[:, 1]

    pos = torch.from_numpy(clip).unsqueeze(0).to(device)
    tri = torch.from_numpy(faces).to(device)
    findices, bary = tr.rasterize(pos, tri, (resolution, resolution))

    n_t = torch.from_numpy(np.ascontiguousarray(nr)).unsqueeze(0).to(device)
    normal_map = tr.interpolate(n_t, findices, bary, tri)[0]      # (H, W, 3)
    normal_map = torch.nn.functional.normalize(normal_map, dim=-1)

    # Key light plus a fill, so cavities do not go pure black.
    #
    # Directions are in the flipped clip space established above, where
    # +X = screen right, +Y = screen down, +Z = away from the viewer. So a light
    # shining from the upper left *towards* the camera has all three components
    # negative. Getting the Z sign wrong clamps every viewer-facing normal to
    # zero and renders the whole object black.
    key = torch.tensor([-0.35, -0.45, -0.82], device=device)
    key = key / key.norm()
    fill = torch.tensor([0.55, -0.25, -0.45], device=device)
    fill = fill / fill.norm()
    lambert = (normal_map @ key).clamp(min=0) * 0.8 \
        + (normal_map @ fill).clamp(min=0) * 0.35 + 0.16

    if uv is not None and albedo is not None:
        uv_t = torch.from_numpy(np.ascontiguousarray(uv)).unsqueeze(0).to(device)
        uv_map = tr.interpolate(uv_t, findices, bary, tri)[0]     # (H, W, 2)
        tex = torch.from_numpy(albedo).to(device)
        th, tw = tex.shape[0], tex.shape[1]
        # glTF UV origin is top-left, matching image row order directly.
        px = (uv_map[..., 0].clamp(0, 1) * (tw - 1)).long()
        py = (uv_map[..., 1].clamp(0, 1) * (th - 1)).long()
        base = tex[py, px]
    else:
        base = torch.full((resolution, resolution, 3), 0.72, device=device)

    shaded = (base * lambert.unsqueeze(-1)).clamp(0, 1)
    mask = (findices > 0).unsqueeze(-1)
    bg = torch.tensor(background, device=device).expand_as(shaded)
    image = torch.where(mask, shaded, bg)
    return (image.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mesh", help="path to a mesh (.glb / .obj / .ply)")
    ap.add_argument("--out", default="outputs/preview", help="output directory")
    ap.add_argument("--views", type=int, default=4, help="turntable views")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--elev", type=float, default=15.0)
    ap.add_argument("--name", default=None, help="output basename (default: mesh stem)")
    args = ap.parse_args()

    from PIL import Image

    device = preview_device()
    verts, faces, uv, albedo = load_mesh(args.mesh)
    normals = vertex_normals(verts, faces)
    print(
        f"{Path(args.mesh).name}: {len(verts)} verts, {len(faces)} faces, "
        f"uv={'yes' if uv is not None else 'no'}, texture="
        f"{'yes ' + str(albedo.shape[:2]) if albedo is not None else 'no'}, device={device}"
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or Path(args.mesh).stem

    tiles = []
    for i in range(args.views):
        azim = 360.0 * i / args.views
        img = render(verts, faces, normals, azim, args.elev, args.resolution,
                     device, uv=uv, albedo=albedo)
        tiles.append(img)
        p = out_dir / f"{stem}_view{i}.png"
        Image.fromarray(img).save(p)
        print(f"  view {i} azim={azim:5.1f} -> {p.name}")

    sheet = np.concatenate(tiles, axis=1)
    sheet_path = out_dir / f"{stem}_turntable.png"
    Image.fromarray(sheet).save(sheet_path)
    print(f"turntable sheet -> {sheet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
