"""Split a 2D image into transparent part crops using alpha or a color label mask.

This is image processing, not P3-SAM or X-Part mesh decomposition.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image
from scipy import ndimage


MAX_PIXELS = 16_000_000
MAX_PARTS = 64
# rembg leaves near-transparent halos at alpha 1-2 around the canvas. They are
# not visible image parts, even when a connected region exceeds min_pixels.
VISIBLE_ALPHA = 32


def split_parts(
    image: Image.Image,
    output_root: Path,
    *,
    color_mask: Image.Image | None = None,
    min_pixels: int = 100,
) -> tuple[list[Path], Path]:
    """Save cropped RGBA PNGs and a ZIP; return files in descending area order.

    Without a label mask, only disconnected alpha regions can be split. With a
    label mask, each distinct nonblack RGB color is one part, including regions
    that touch in the source. The mask must have the source image dimensions.
    """
    if image is None:
        raise ValueError("Upload an input image first.")
    if image.width * image.height > MAX_PIXELS:
        raise ValueError("Image exceeds 16 megapixels; resize it first.")
    if min_pixels < 1:
        raise ValueError("Minimum pixels must be at least 1.")

    rgba = np.asarray(image.convert("RGBA"))
    alpha = rgba[:, :, 3]
    regions: list[np.ndarray] = []
    if color_mask is None:
        labels, count = ndimage.label(alpha > VISIBLE_ALPHA)
        if count:
            sizes = np.bincount(labels.ravel(), minlength=count + 1)
            indexes = np.flatnonzero(sizes[1:] >= min_pixels) + 1
            indexes = sorted(indexes, key=lambda i: (-int(sizes[i]), int(i)))
            if len(indexes) > MAX_PARTS:
                raise ValueError(f"Found more than {MAX_PARTS} parts; raise the minimum pixel area.")
            regions = [labels == i for i in indexes]
    else:
        if color_mask.size != image.size:
            raise ValueError("Color mask and input image must have exactly the same width and height.")
        if color_mask.convert("RGB").getcolors(maxcolors=MAX_PARTS + 2) is None:
            raise ValueError(
                "The color label mask has too many colors. Upload the original photo "
                "as the image to split; a mask must use a solid color per part and black background."
            )
        mask_rgba = np.asarray(color_mask.convert("RGBA"))
        rgb = mask_rgba[:, :, :3]
        active = (mask_rgba[:, :, 3] > 0) & np.any(rgb != 0, axis=2) & (alpha > VISIBLE_ALPHA)
        colors, counts = np.unique(rgb[active], axis=0, return_counts=True)
        indexes = [i for i, n in enumerate(counts) if n >= min_pixels]
        indexes.sort(key=lambda i: (-int(counts[i]), tuple(int(x) for x in colors[i])))
        if len(indexes) > MAX_PARTS:
            raise ValueError(f"Found more than {MAX_PARTS} colors; simplify the mask.")
        regions = [active & np.all(rgb == colors[i], axis=2) for i in indexes]

    return save_regions(image, output_root, regions)


def save_regions(
    image: Image.Image, output_root: Path, regions: list[np.ndarray]
) -> tuple[list[Path], Path]:
    """Save binary regions as transparent crops with the source image colors."""
    if not regions:
        raise ValueError("No parts found. Try lowering the minimum pixel area or use a color label mask.")
    if len(regions) > MAX_PARTS:
        raise ValueError(f"Found more than {MAX_PARTS} parts; raise the minimum pixel area.")
    rgba = np.asarray(image.convert("RGBA"))
    destination = output_root / f"image_parts_{uuid4().hex[:12]}"
    destination.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    for number, region in enumerate(regions, start=1):
        if region.shape != rgba.shape[:2] or not np.any(region):
            raise ValueError("A part mask is empty or does not match the source image.")
        ys, xs = np.nonzero(region)
        left, right = int(xs.min()), int(xs.max()) + 1
        top, bottom = int(ys.min()), int(ys.max()) + 1
        part = rgba[top:bottom, left:right].copy()
        part[:, :, 3] = np.where(region[top:bottom, left:right], part[:, :, 3], 0)
        path = destination / f"part_{number:02d}.png"
        Image.fromarray(part).save(path)
        paths.append(path)

    archive = destination / "image_parts.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for path in paths:
            bundle.write(path, arcname=path.name)
    return paths, archive
