#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compose «input → result» strips for the README from finished example runs.

Each strip is one image: the source photo, then the textured turntable views, on
a common background with a hairline divider between input and output. One file
per example keeps the README readable and survives being viewed on a phone,
where a multi-column table of separate images does not.

Expects `tests/e2e_visual.py` to have already produced `outputs/ex_<name>/`.

    python tests/build_example_sheets.py
    python tests/build_example_sheets.py --only teapot robot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent

EXAMPLES = ("teapot", "robot", "owl", "sneaker", "armchair")

BG = (18, 18, 20)
DIVIDER = (70, 70, 82)


def build(name: str, height: int, gap: int) -> Path | None:
    from PIL import Image

    src = PROJECT_DIR / "assets" / "examples" / f"{name}.jpg"
    sheet = PROJECT_DIR / "outputs" / f"ex_{name}" / "textured_turntable.png"
    if not src.is_file():
        print(f"  skip {name}: no input image")
        return None
    if not sheet.is_file():
        print(f"  skip {name}: no textured turntable (run tests/e2e_visual.py first)")
        return None

    inp = Image.open(src).convert("RGB")
    out = Image.open(sheet).convert("RGB")

    # Scale both to a common height so the strip reads as one row.
    inp = inp.resize((round(inp.width * height / inp.height), height), Image.LANCZOS)
    out = out.resize((round(out.width * height / out.height), height), Image.LANCZOS)

    width = inp.width + gap + out.width
    canvas = Image.new("RGB", (width, height), BG)
    canvas.paste(inp, (0, 0))
    canvas.paste(out, (inp.width + gap, 0))

    # Hairline between the photo and the reconstruction, so it is obvious which
    # part is input and which is generated.
    arr = np.array(canvas)
    x = inp.width + gap // 2
    arr[:, x:x + 1] = DIVIDER
    canvas = Image.fromarray(arr)

    dst = PROJECT_DIR / "assets" / "examples" / f"{name}_result.jpg"
    canvas.save(dst, quality=88, optimize=True)
    print(f"  {name:9s} {canvas.width}x{canvas.height}  {dst.stat().st_size / 1024:6.0f} KB  {dst.name}")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None, help="build only these examples")
    ap.add_argument("--height", type=int, default=300)
    ap.add_argument("--gap", type=int, default=16)
    args = ap.parse_args()

    names = args.only or list(EXAMPLES)
    built = [p for n in names if (p := build(n, args.height, args.gap))]
    print(f"\n{len(built)}/{len(names)} strips written to assets/examples/")
    return 0 if built else 1


if __name__ == "__main__":
    raise SystemExit(main())
