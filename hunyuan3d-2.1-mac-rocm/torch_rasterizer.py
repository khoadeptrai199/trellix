#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure-PyTorch drop-in replacement for Hunyuan3D's `custom_rasterizer`.

Why this exists
---------------
`hy3dpaint` rasterizes through a compiled extension (`custom_rasterizer_kernel`,
a CUDA kernel plus a CPU fallback). Building it is the single most fragile step
of the paint pipeline:

  * Windows + ROCm: the `rocm-sdk` wheels ship the HIP runtime but no HIP
    headers and no offline compiler, so device code cannot be built. Worse, the
    Windows `torch 2.9.1+rocmsdk` wheel cannot link *any* C++ extension — even
    an empty `PYBIND11_MODULE` over `<torch/extension.h>` fails with an
    unresolved `c10::ValueError(SourceLocation, std::string)`, which is not
    exported from `c10.lib`. So neither the GPU nor the CPU variant is buildable.
  * macOS: no CUDA at all, and the upstream sources need patching to compile
    without a device compiler.

This module implements the same rasterization in plain PyTorch ops, so it needs
no compiler on any platform and runs on whatever device the tensors are on —
including the AMD GPU on ROCm and MPS on Apple Silicon.

Semantics are matched to upstream's `rasterize_image_cpu` in
`custom_rasterizer/lib/custom_rasterizer_kernel/rasterizer.cpp`, including the
quantized-depth z-buffer token trick, the truncating integer conversions, and
the perspective-correct barycentric second pass. `tests/test_torch_rasterizer.py`
checks this against a literal transcription of the C++ loops.

Register it under the name upstream imports:

    import sys, torch_rasterizer
    sys.modules["custom_rasterizer"] = torch_rasterizer
"""
from __future__ import annotations

import torch

# Upstream constants (rasterizer.h). MAXINT is deliberately 2^31-1 and is used
# as the radix that packs quantized depth and triangle index into one integer.
MAXINT = 2147483647
DEPTH_QUANT = 2 << 17  # 262144
EMPTY_MARKER = MAXINT - 1


def _screen_coords(V: torch.Tensor, width: int, height: int):
    """Clip space -> screen space, exactly as the C++ kernel does.

    Returns (sx, sy, sz, w) with the same half-pixel offsets and the 0.49999
    depth squeeze upstream uses to keep the quantized depth inside its range.
    """
    w = V[:, 3]
    sx = (V[:, 0] / w * 0.5 + 0.5) * (width - 1) + 0.5
    sy = (0.5 + 0.5 * V[:, 1] / w) * (height - 1) + 0.5
    sz = V[:, 2] / w * 0.49999 + 0.5
    return sx, sy, sz, w


def _signed_area2(ax, ay, bx, by, cx, cy):
    """calculateSignedArea2(a, b, c) from rasterizer.h."""
    return (cx - ax) * (by - ay) - (bx - ax) * (cy - ay)


def _barycentric(ax, ay, bx, by, cx, cy, px, py):
    """calculateBarycentricCoordinate, vectorised.

    Degenerate triangles (zero signed area) yield (-1, -1, -1), which fails the
    in-bounds test downstream — same as upstream.
    """
    area = _signed_area2(ax, ay, bx, by, cx, cy)
    beta_t = _signed_area2(ax, ay, px, py, cx, cy)
    gamma_t = _signed_area2(ax, ay, bx, by, px, py)

    degenerate = area == 0
    inv = torch.where(degenerate, torch.zeros_like(area), 1.0 / torch.where(degenerate, torch.ones_like(area), area))
    beta = beta_t * inv
    gamma = gamma_t * inv
    alpha = 1.0 - beta - gamma

    neg = torch.full_like(area, -1.0)
    alpha = torch.where(degenerate, neg, alpha)
    beta = torch.where(degenerate, neg, beta)
    gamma = torch.where(degenerate, neg, gamma)
    return alpha, beta, gamma


def _in_bounds(a, b, c):
    return (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1) & (c >= 0) & (c <= 1)


def _trunc_div(x: torch.Tensor, d: int) -> torch.Tensor:
    """C-style integer division (truncating toward zero), not Python's floor."""
    return torch.div(x, d, rounding_mode="trunc")


def rasterize(
    pos: torch.Tensor,
    tri: torch.Tensor,
    resolution,
    clamp_depth: torch.Tensor | None = None,
    use_depth_prior: int = 0,
):
    """Rasterize a mesh into a face-index map and barycentric coordinates.

    Signature matches `custom_rasterizer.render.rasterize`.

    Args:
        pos: (1, N, 4) clip-space homogeneous vertex positions.
        tri: (T, 3) triangle vertex indices.
        resolution: (height, width) — note the upstream call site passes
            `resolution[1]` as width and `resolution[0]` as height.
        clamp_depth: (H*W,) or (H, W) depth prior, used only when
            `use_depth_prior` is truthy.
        use_depth_prior: when set, pixels closer than the prior are discarded.

    Returns:
        findices: (H, W) int32 — triangle index + 1, 0 where nothing was hit.
        barycentric: (H, W, 3) float32 — perspective-correct, zero where empty.
    """
    if pos.dim() == 3:
        V = pos[0]
    else:
        V = pos
    V = V.float()
    device = V.device

    height, width = int(resolution[0]), int(resolution[1])
    n_pix = height * width

    faces = tri.long().to(device)
    n_tri = faces.shape[0]

    sx, sy, sz, w = _screen_coords(V, width, height)

    # Per-triangle vertex attributes.
    i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2]
    x0, y0, z0 = sx[i0], sy[i0], sz[i0]
    x1, y1, z1 = sx[i1], sy[i1], sz[i1]
    x2, y2, z2 = sx[i2], sy[i2], sz[i2]

    # Integer pixel bounding box. The C++ loop is
    #   for (int px = x_min; px < x_max + 1; ++px)
    # where `int px = x_min` truncates toward zero, so the low bound is trunc()
    # and the high bound is floor(). Out-of-frame pixels are skipped inside the
    # loop, which is equivalent to clamping the box to the image.
    xmin = torch.minimum(x0, torch.minimum(x1, x2)).trunc().clamp_(min=0)
    xmax = torch.maximum(x0, torch.maximum(x1, x2)).floor().clamp_(max=width - 1)
    ymin = torch.minimum(y0, torch.minimum(y1, y2)).trunc().clamp_(min=0)
    ymax = torch.maximum(y0, torch.maximum(y1, y2)).floor().clamp_(max=height - 1)

    bw = (xmax - xmin + 1).clamp_(min=0).long()
    bh = (ymax - ymin + 1).clamp_(min=0).long()
    area = bw * bh

    # int64 z-buffer holding packed (quantized depth, triangle index) tokens.
    # Initial value matches upstream's `maxint`, whose remainder mod MAXINT is
    # EMPTY_MARKER — that is how "no triangle here" is encoded.
    zbuf_init = MAXINT * MAXINT + EMPTY_MARKER
    zbuf = torch.full((n_pix,), zbuf_init, dtype=torch.int64, device=device)

    prior = None
    if use_depth_prior and clamp_depth is not None and clamp_depth.numel() >= n_pix:
        prior = clamp_depth.reshape(-1)[:n_pix].float().to(device)

    live = (area > 0).nonzero(as_tuple=True)[0]
    if live.numel() > 0:
        # Triangles vary enormously in footprint, so a single padded
        # (n_tri, max_area) grid would blow up memory on the few large ones.
        # Bucketing by power-of-two footprint keeps padding bounded while still
        # doing all the work in a handful of batched passes.
        for idx, kmax in _area_buckets(area[live], live):
            _accumulate_bucket(
                zbuf, idx, kmax,
                xmin, ymin, bw, area,
                x0, y0, z0, x1, y1, z1, x2, y2, z2,
                width, prior,
            )

    # --- decode the z-buffer -------------------------------------------------
    # C's `%` truncates toward zero; torch's `remainder` floors. Tokens are
    # non-negative in practice (negative depths are discarded above), but match
    # the C semantics anyway so the two implementations cannot diverge.
    f = zbuf - _trunc_div(zbuf, MAXINT) * MAXINT
    empty = f == EMPTY_MARKER
    findices = torch.where(empty, torch.zeros_like(f), f)

    barycentric = _second_pass_barycentric(
        findices, empty, faces, sx, sy, w, width, height, device
    )
    return findices.reshape(height, width).to(torch.int32), barycentric


def _area_buckets(areas: torch.Tensor, indices: torch.Tensor):
    """Group triangle indices into power-of-two footprint buckets.

    Yields (indices_in_bucket, padded_width) pairs. Padding waste per bucket is
    bounded at 2x, which is the point.
    """
    max_area = int(areas.max().item())
    lo = 1
    while lo <= max_area:
        hi = lo * 2
        sel = (areas >= lo) & (areas < hi)
        if sel.any():
            yield indices[sel], hi - 1
        lo = hi


def _accumulate_bucket(
    zbuf, idx, kmax,
    xmin, ymin, bw, area,
    x0, y0, z0, x1, y1, z1, x2, y2, z2,
    width, prior,
):
    """Rasterize one footprint bucket and min-reduce its tokens into zbuf."""
    device = zbuf.device
    bw_b = bw[idx].unsqueeze(1)           # (n, 1)
    area_b = area[idx].unsqueeze(1)       # (n, 1)
    off = torch.arange(kmax, device=device, dtype=torch.int64).unsqueeze(0)  # (1, K)

    valid = off < area_b
    dx = off % bw_b
    dy = _trunc_div(off, bw_b)

    px = xmin[idx].long().unsqueeze(1) + dx
    py = ymin[idx].long().unsqueeze(1) + dy
    pxf = px.float() + 0.5
    pyf = py.float() + 0.5

    def col(t):
        return t[idx].unsqueeze(1)

    alpha, beta, gamma = _barycentric(
        col(x0), col(y0), col(x1), col(y1), col(x2), col(y2), pxf, pyf
    )
    hit = valid & _in_bounds(alpha, beta, gamma)
    if not bool(hit.any()):
        return

    depth = alpha * col(z0) + beta * col(z1) + gamma * col(z2)

    if prior is not None:
        pixel = py * width + px
        thres = prior[pixel.clamp_(min=0, max=prior.numel() - 1)] * 0.49999 + 0.5 + 1e-6
    else:
        thres = torch.zeros_like(depth)
    hit = hit & (depth >= thres)
    if not bool(hit.any()):
        return

    # `int z_quantize = depth * (2<<17)` — truncating conversion.
    z_quant = (depth * DEPTH_QUANT).trunc().to(torch.int64)
    tri_id = (idx + 1).unsqueeze(1).expand_as(z_quant)
    token = z_quant * MAXINT + tri_id

    flat_pix = (py * width + px).reshape(-1)[hit.reshape(-1)]
    flat_tok = token.reshape(-1)[hit.reshape(-1)]
    zbuf.scatter_reduce_(0, flat_pix, flat_tok, reduce="amin")


def _second_pass_barycentric(findices, empty, faces, sx, sy, w, width, height, device):
    """Recompute perspective-correct barycentrics for the winning triangles.

    Upstream does exactly this in `barycentricFromImgcoordCPU`: the barycentrics
    from the coverage pass are screen-space, so they are recomputed and then
    divided by each vertex's w and renormalised.
    """
    n_pix = height * width
    barycentric = torch.zeros((n_pix, 3), dtype=torch.float32, device=device)

    t = findices - 1
    hit = (~empty) & (t >= 0)
    if not bool(hit.any()):
        return barycentric.reshape(height, width, 3)

    sel = hit.nonzero(as_tuple=True)[0]
    tsel = t[sel]

    pxf = (sel % width).float() + 0.5
    pyf = _trunc_div(sel, width).float() + 0.5

    i0, i1, i2 = faces[tsel, 0], faces[tsel, 1], faces[tsel, 2]
    alpha, beta, gamma = _barycentric(
        sx[i0], sy[i0], sx[i1], sy[i1], sx[i2], sy[i2], pxf, pyf
    )

    a = alpha / w[i0]
    b = beta / w[i1]
    c = gamma / w[i2]
    s = a + b + c
    # Guard the degenerate case; upstream would produce inf/nan here too, but a
    # nan in the texture bake is far more expensive to debug than a zero.
    inv = torch.where(s == 0, torch.zeros_like(s), 1.0 / torch.where(s == 0, torch.ones_like(s), s))

    barycentric[sel, 0] = a * inv
    barycentric[sel, 1] = b * inv
    barycentric[sel, 2] = c * inv
    return barycentric.reshape(height, width, 3)


def interpolate(col: torch.Tensor, findices: torch.Tensor, barycentric: torch.Tensor, tri: torch.Tensor):
    """Vertex-attribute interpolation, identical to upstream's `interpolate`."""
    f = findices - 1 + (findices == 0)
    vcol = col[0, tri.long()[f.long()]]
    result = barycentric.view(*barycentric.shape, 1) * vcol
    result = torch.sum(result, dim=-2)
    return result.view(1, *result.shape)
