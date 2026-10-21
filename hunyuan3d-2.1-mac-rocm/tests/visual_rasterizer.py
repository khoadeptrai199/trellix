#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visual conformance test: torch_rasterizer vs the C++-transcription oracle.

`test_torch_rasterizer.py` proves the numbers agree. This renders them, because
a rasterizer bug that survives numeric tolerance usually looks obvious as an
image — a one-pixel edge shift, a z-fighting seam, a flipped axis.

For each scene it writes a PNG strip:

    face-index map (oracle) | face-index map (torch) | difference | barycentric RGB

and prints the same pass/fail verdict as the numeric test.

    python tests/visual_rasterizer.py --out outputs/visual_raster
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import backend as backend_mod  # noqa: E402  (before torch — sets the MPS fallback)
import torch  # noqa: E402
import torch_rasterizer as tr  # noqa: E402
from test_torch_rasterizer import _make_scene, reference_rasterize  # noqa: E402


def colorize_faces(findices: np.ndarray, n_tri: int) -> np.ndarray:
    """Distinct colour per triangle id; black where nothing was hit.

    A fixed hue ramp rather than random colours, so two runs are comparable by
    eye and a shifted triangle id shows up as a colour shift, not noise.
    """
    h, w = findices.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    hit = findices > 0
    if not hit.any():
        return out
    ids = findices[hit].astype(np.int64)
    phase = (ids * 2.399963) % (2 * np.pi)  # golden-angle hue spread
    out[hit, 0] = (127 + 120 * np.sin(phase)).astype(np.uint8)
    out[hit, 1] = (127 + 120 * np.sin(phase + 2.094)).astype(np.uint8)
    out[hit, 2] = (127 + 120 * np.sin(phase + 4.188)).astype(np.uint8)
    return out


def bary_rgb(bary: np.ndarray) -> np.ndarray:
    return np.clip(bary * 255.0, 0, 255).astype(np.uint8)


def diff_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Green where both agree on coverage, red where they disagree."""
    h, w = a.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    same = a == b
    covered = (a > 0) | (b > 0)
    out[same & covered] = (40, 110, 40)
    out[~same] = (230, 40, 40)
    return out


def label_strip(images: list[np.ndarray], labels: list[str], scale: int) -> np.ndarray:
    """Upscale with nearest neighbour and join horizontally with separators.

    Nearest neighbour on purpose: these are index maps, and smoothing them would
    invent pixels that neither implementation produced.
    """
    from PIL import Image, ImageDraw

    tiles = []
    for img in images:
        pil = Image.fromarray(img).resize(
            (img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST
        )
        tiles.append(pil)

    pad, bar = 6, 18
    w = sum(t.width for t in tiles) + pad * (len(tiles) - 1)
    h = max(t.height for t in tiles) + bar
    canvas = Image.new("RGB", (w, h), (18, 18, 20))
    x = 0
    draw = ImageDraw.Draw(canvas)
    for tile, text in zip(tiles, labels):
        canvas.paste(tile, (x, bar))
        draw.text((x + 2, 3), text, fill=(210, 210, 215))
        x += tile.width + pad
    return np.array(canvas)


SCENES = [
    ("overlapping_8tri", dict(seed=0, n_tri=8, spread=0.9), 64, 64),
    ("clipped_40tri", dict(seed=1, n_tri=40, spread=1.4), 96, 64),
    ("nonsquare_25tri", dict(seed=2, n_tri=25, spread=0.7), 128, 48),
    ("degenerate", dict(seed=3, n_tri=4, spread=0.8), 64, 64),
    ("perspective", dict(seed=6, n_tri=10, spread=0.6), 64, 64),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="outputs/visual_raster", help="output directory")
    ap.add_argument("--scale", type=int, default=4, help="nearest-neighbour upscale")
    args = ap.parse_args()

    from PIL import Image

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(backend_mod.detect().device)
    print(f"device: {device}")

    all_ok = True
    written = []
    for name, kwargs, width, height in SCENES:
        V, F = _make_scene(**kwargs)
        if name == "degenerate":
            V[3] = V[4] = V[5] = V[3]
        if name == "perspective":
            V[:, 3] = np.linspace(1.0, 2.5, V.shape[0]).astype(np.float32)
            V[:, :3] *= V[:, 3:4]

        ref_f, ref_b = reference_rasterize(V, F, width, height)

        pos = torch.from_numpy(V).unsqueeze(0).to(device)
        tri = torch.from_numpy(F).to(device)
        got_f_t, got_b_t = tr.rasterize(pos, tri, (height, width))
        got_f = got_f_t.cpu().numpy().astype(np.int64)
        got_b = got_b_t.cpu().numpy().astype(np.float64)

        same = np.array_equal(got_f, ref_f)
        covered = ref_f > 0
        max_db = float(np.abs(got_b[covered] - ref_b[covered]).max()) if covered.any() else 0.0
        ok = same and max_db < 2e-4
        all_ok &= ok

        n_tri = F.shape[0]
        strip = label_strip(
            [
                colorize_faces(ref_f, n_tri),
                colorize_faces(got_f, n_tri),
                diff_map(ref_f, got_f),
                bary_rgb(got_b),
            ],
            ["oracle (C++ port)", "torch_rasterizer", "diff (red=mismatch)", "barycentric RGB"],
            args.scale,
        )
        path = out_dir / f"{name}.png"
        Image.fromarray(strip).save(path)
        written.append(path)

        print(
            f"  [{'PASS' if ok else 'FAIL'}] {name:18s} {width}x{height} "
            f"covered={int(covered.sum()):6d} identical={same} max_bary_diff={max_db:.2e} -> {path.name}"
        )

    print()
    print(f"{len(written)} images in {out_dir.resolve()}")
    print("ALL VISUAL CHECKS PASSED" if all_ok else "SOME VISUAL CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
