"""Optional SAM ViT-B proposals for 2D image parts on CUDA, ROCm, MPS or CPU.

SAM masks are visual candidates and may overlap. They are not semantic labels
or a disjoint decomposition of a 3D mesh.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
from PIL import Image

from image_parts import MAX_PIXELS, VISIBLE_ALPHA, save_regions


CHECKPOINT = Path(__file__).resolve().parent / "weights" / "sam_vit_b_01ec64.pth"
MAX_SIDE = 1024
MAX_CANDIDATES = 24
INSTALL_HELP = (
    "Install SAM: on Windows run powershell -ExecutionPolicy Bypass -File "
    "scripts/install_sam.ps1; on macOS/Linux run bash scripts/install_sam.sh."
)


def propose_parts(
    image: Image.Image, output_root: Path, *, min_pixels: int = 100
) -> tuple[list[Path], Path]:
    """Run SAM on the foreground and save filtered candidate masks as PNGs."""
    if not CHECKPOINT.is_file():
        raise RuntimeError(f"SAM weights are missing. {INSTALL_HELP}")
    if image.width * image.height > MAX_PIXELS:
        raise ValueError("Image exceeds 16 megapixels; resize it first.")
    try:
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise RuntimeError(f"SAM is not installed. {INSTALL_HELP}") from exc

    source = image.convert("RGBA")
    scale = min(1.0, MAX_SIDE / max(source.size))
    small = source.resize((round(source.width * scale), round(source.height * scale)), Image.Resampling.LANCZOS) if scale < 1 else source
    small_rgba = np.asarray(small)
    visible = small_rgba[:, :, 3] > VISIBLE_ALPHA
    visible_area = int(visible.sum())
    if visible_area < min_pixels:
        raise ValueError("No visible foreground found after background removal.")
    rgb = small_rgba[:, :, :3].copy()
    rgb[~visible] = 0

    device = ("cuda" if torch.cuda.is_available() else
              "mps" if torch.backends.mps.is_available() else "cpu")
    model = None
    generator = None
    try:
        model = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT)).to(device)
        generator = SamAutomaticMaskGenerator(
            model,
            points_per_side=12,
            points_per_batch=16,
            pred_iou_thresh=0.82,
            stability_score_thresh=0.86,
            crop_n_layers=0,
            min_mask_region_area=max(100, min_pixels),
        )
        # The upstream coordinate transform produces float64 for an integer
        # image size; MPS only accepts float32 here.
        original_apply_coords = generator.predictor.transform.apply_coords
        generator.predictor.transform.apply_coords = (
            lambda coords, shape: original_apply_coords(coords, shape).astype(np.float32)
        )
        with torch.inference_mode():
            raw_masks = generator.generate(rgb)
    finally:
        del generator, model
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()

    candidates: list[tuple[float, int, np.ndarray]] = []
    area_floor = max(min_pixels, round(visible_area * 0.002))
    for proposal in raw_masks:
        mask = proposal["segmentation"]
        intersection = mask & visible
        area = int(intersection.sum())
        if area < area_floor or area > 0.90 * visible_area:
            continue
        # Background proposals can trace a thin object outline but contain no
        # usable object pixels. Most of a valid proposal must be foreground.
        if area / max(int(mask.sum()), 1) < 0.55:
            continue
        quality = float(proposal["predicted_iou"]) * float(proposal["stability_score"])
        candidates.append((quality, area, intersection))
    candidates.sort(key=lambda item: (-item[0], -item[1]))

    selected: list[tuple[int, np.ndarray]] = []
    for _, area, mask in candidates:
        if any(int(np.count_nonzero(mask & kept)) / (area + kept_area - int(np.count_nonzero(mask & kept))) > 0.75
               for kept_area, kept in selected):
            continue
        selected.append((area, mask))
        if len(selected) >= MAX_CANDIDATES:
            break
    selected.sort(key=lambda item: -item[0])
    if not selected:
        raise ValueError("SAM found no usable part candidates. Try a clearer image or a color label mask.")

    regions = []
    for _, mask in selected:
        if scale < 1:
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
            mask = np.asarray(mask_image.resize(source.size, Image.Resampling.NEAREST)) > 0
        regions.append(mask)
    return save_regions(source, output_root, regions)
