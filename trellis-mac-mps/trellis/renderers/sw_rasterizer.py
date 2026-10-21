"""
Pure-PyTorch software triangle rasterizer, standing in for nvdiffrast.torch when a CUDA/OpenGL
context isn't available (e.g. on Apple Silicon). Implements just the subset of nvdiffrast's API
that trellis/renderers/mesh_renderer.py actually calls: rasterize / interpolate / antialias.

Scope and known differences from nvdiffrast:
- Batch size of 1 only (mesh_renderer.py never renders more than one mesh/camera per call).
- `antialias` is an identity passthrough here -- no analytic silhouette-edge blending, so edges
  are aliased at ssaa=1. Use MeshRenderer's `ssaa` option instead (renders at a higher internal
  resolution, then downsamples with F.interpolate(..., antialias=True), already wired up in
  mesh_renderer.py and unaffected by this limitation) to get anti-aliased output.
- Rasterization cost is O(H * W * num_triangles), chunked over triangles to bound memory, versus
  nvdiffrast's screen-space-binned CUDA rasterizer (roughly O(H*W + triangle pixel coverage)).
  Meaningfully slower for large meshes/resolutions, but the result is an exact standard
  edge-function rasterization (coverage test + nearest-wins z-buffer), not an approximation.
- Differentiability matches nvdiffrast's actual contract: `rasterize` is not differentiable w.r.t.
  vertex position (the pixel-to-triangle assignment is a discrete argmin), but `interpolate`'s
  barycentric-weighted gather is plain differentiable tensor arithmetic, same as nvdiffrast.

rast channel layout: [1, H, W, 4] = (barycentric_1, barycentric_2, ndc_z, triangle_id), where
triangle_id == 0 means background and triangle_id == k means face `tri[k-1]` (1-indexed, matching
nvdiffrast's convention -- this specifically matters because mesh_renderer.py reads
`rast[..., -1:] > 0` directly to build its mask output).
"""
from typing import Optional
import torch
import torch.nn.functional as F

__all__ = ['RasterizeContext', 'RasterizeCudaContext', 'RasterizeGLContext', 'rasterize', 'interpolate', 'antialias', 'texture']


class RasterizeContext:
    """Stand-in for nvdiffrast.torch.RasterizeCudaContext/RasterizeGLContext. This rasterizer
    needs no persistent GPU/GL context, so this just carries the device for API compatibility."""
    def __init__(self, device='cpu'):
        self.device = device


RasterizeCudaContext = RasterizeContext
RasterizeGLContext = RasterizeContext


def _default_chunk_size(H: int, W: int, target_elems: int = 32 * 1024 * 1024) -> int:
    return max(1, target_elems // max(1, H * W))


def rasterize(glctx, pos: torch.Tensor, tri: torch.Tensor, resolution, chunk_size: Optional[int] = None, **kwargs):
    """
    Args:
        glctx: unused (present for API compatibility with nvdiffrast).
        pos: [1, V, 4] clip-space homogeneous vertex positions.
        tri: [F, 3] int triangle vertex indices.
        resolution: (H, W).
        chunk_size: triangles processed per pass; auto-sized to bound memory if omitted.
        **kwargs: accepted and ignored for API compatibility (e.g. nvdiffrast's `grad_db`, which
            controls whether `rast_db` -- screen-space derivatives for texture mipmapping -- is
            computed; this rasterizer's `texture()` doesn't do mip selection, so it's unused here).
    Returns:
        (rast, rast_db): rast is [1, H, W, 4], see module docstring for channel layout. rast_db
        is always None -- nothing that calls this rasterizer uses it.
    """
    assert pos.dim() == 3 and pos.shape[0] == 1, "software rasterizer only supports batch size 1"
    H, W = resolution
    device = pos.device
    dtype = pos.dtype

    pos = pos[0]                          # [V, 4]
    tri = tri.long()
    w = pos[:, 3].clamp(min=1e-6)
    ndc = pos[:, :3] / w[:, None]          # [V, 3], NDC in [-1, 1]
    # matches intrinsics_to_projection's convention exactly (verified by hand against that
    # matrix): ndc_x maps linearly to [0,1] normalized column, ndc_y is already row-flipped.
    px = (ndc[:, 0] * 0.5 + 0.5) * W
    py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * H
    pz = ndc[:, 2]

    tx, ty, tz = px[tri], py[tri], pz[tri]  # [F, 3] each

    zbuf = torch.full((H, W), float('inf'), device=device, dtype=dtype)
    tri_id_buf = torch.zeros((H, W), dtype=torch.long, device=device)
    bary1_buf = torch.zeros((H, W), device=device, dtype=dtype)
    bary2_buf = torch.zeros((H, W), device=device, dtype=dtype)

    F_ = tri.shape[0]
    if F_ == 0:
        rast = torch.stack([bary1_buf, bary2_buf, zbuf.clamp(max=1e8), tri_id_buf.to(dtype)], dim=-1)
        return rast.unsqueeze(0), None

    if chunk_size is None:
        chunk_size = _default_chunk_size(H, W)

    xs = (torch.arange(W, device=device, dtype=dtype) + 0.5).view(1, 1, W)
    ys = (torch.arange(H, device=device, dtype=dtype) + 0.5).view(1, H, 1)

    for start in range(0, F_, chunk_size):
        end = min(start + chunk_size, F_)
        x0, x1, x2 = tx[start:end, 0], tx[start:end, 1], tx[start:end, 2]
        y0, y1, y2 = ty[start:end, 0], ty[start:end, 1], ty[start:end, 2]
        z0, z1, z2 = tz[start:end, 0], tz[start:end, 1], tz[start:end, 2]
        B = end - start

        x0_, y0_ = x0.view(B, 1, 1), y0.view(B, 1, 1)
        x1_, y1_ = x1.view(B, 1, 1), y1.view(B, 1, 1)
        x2_, y2_ = x2.view(B, 1, 1), y2.view(B, 1, 1)

        # edge function e(a, b; p) = (bx-ax)*(py-ay) - (by-ay)*(px-ax); wN is the edge opposite
        # vertex N, so barycentric weight for vertex N is wN / area (standard convention).
        w0 = (x2_ - x1_) * (ys - y1_) - (y2_ - y1_) * (xs - x1_)
        w1 = (x0_ - x2_) * (ys - y2_) - (y0_ - y2_) * (xs - x2_)
        w2 = (x1_ - x0_) * (ys - y0_) - (y1_ - y0_) * (xs - x0_)
        area = w0 + w1 + w2

        eps = 1e-9
        degenerate = area.abs() < eps
        safe_area = torch.where(degenerate, torch.full_like(area, eps), area)
        # winding-agnostic inside test: all-positive or all-negative edge functions
        inside = (~degenerate) & (((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0)))

        b0 = w0 / safe_area
        b1 = w1 / safe_area
        b2 = w2 / safe_area
        depth = b0 * z0.view(B, 1, 1) + b1 * z1.view(B, 1, 1) + b2 * z2.view(B, 1, 1)
        depth = torch.where(inside, depth, torch.full_like(depth, float('inf')))

        chunk_depth, chunk_idx = depth.min(dim=0)              # [H, W], nearest (smallest ndc_z) wins
        chunk_valid = torch.isfinite(chunk_depth)
        update = chunk_valid & (chunk_depth < zbuf)

        # Real MPS bug, confirmed via a minimal repro (all-+inf column -> argmin returns -1 on
        # MPS, 0 on CPU/CUDA): when an entire chunk's depth column is +inf (no triangle in this
        # chunk covers that pixel), torch.min's returned index can be -1 on this backend, which
        # crashes gather even though the gathered value is always discarded below (chunk_valid is
        # False for exactly these pixels). Clamping is safe -- it only changes which discarded
        # value gets computed, never which value is kept.
        gather_idx = chunk_idx.clamp(min=0)
        b1_win = torch.gather(b1, 0, gather_idx.unsqueeze(0)).squeeze(0)
        b2_win = torch.gather(b2, 0, gather_idx.unsqueeze(0)).squeeze(0)

        zbuf = torch.where(update, chunk_depth, zbuf)
        tri_id_buf = torch.where(update, start + chunk_idx + 1, tri_id_buf)
        bary1_buf = torch.where(update, b1_win, bary1_buf)
        bary2_buf = torch.where(update, b2_win, bary2_buf)

    rast = torch.stack([bary1_buf, bary2_buf, zbuf, tri_id_buf.to(dtype)], dim=-1)
    return rast.unsqueeze(0), None


def interpolate(attr: torch.Tensor, rast: torch.Tensor, tri: torch.Tensor, rast_db=None, diff_attrs=None):
    """
    Args:
        attr: [1, V, C] per-vertex (or, when `tri` indexes a duplicated-per-corner layout such as
              mesh_renderer.py's per-face normals, per-corner) attributes.
        rast: [1, H, W, 4] rasterizer output from `rasterize`.
        tri: [F, 3] int, the same triangle index tensor passed to `rasterize`.
        diff_attrs: if not None ('all' or a list of channel indices), also return a same-shaped
            zero placeholder for screen-space derivatives instead of None, so callers that
            unconditionally reshape/permute the second return value (e.g. utils3d's
            rasterize_triangle_faces) don't crash on a None. The values are always zero -- this
            rasterizer's `texture()` doesn't do derivative-based mip selection, so nothing
            downstream actually depends on these being real derivatives.
    Returns:
        (out, out_da): out is [1, H, W, C], barycentric-interpolated per pixel (0 at background
        pixels). out_da is None unless diff_attrs is given, in which case it's a zero tensor of
        shape [1, H, W, 2 * num_diff_channels].
    """
    assert attr.dim() == 3 and attr.shape[0] == 1
    tri = tri.long()
    b1 = rast[0, ..., 0]
    b2 = rast[0, ..., 1]
    tri_id = rast[0, ..., 3].long()
    valid = tri_id > 0
    safe_id = (tri_id - 1).clamp(min=0)

    corners = tri[safe_id]              # [H, W, 3]
    a0 = attr[0, corners[..., 0]]       # [H, W, C]
    a1 = attr[0, corners[..., 1]]
    a2 = attr[0, corners[..., 2]]
    b0 = 1.0 - b1 - b2

    out = b0.unsqueeze(-1) * a0 + b1.unsqueeze(-1) * a1 + b2.unsqueeze(-1) * a2
    out = torch.where(valid.unsqueeze(-1), out, torch.zeros_like(out))

    out_da = None
    if diff_attrs is not None:
        n_diff = attr.shape[-1] if diff_attrs == 'all' else len(diff_attrs)
        out_da = torch.zeros(*out.shape[:-1], 2 * n_diff, device=out.device, dtype=out.dtype)

    return out.unsqueeze(0), out_da


def antialias(color: torch.Tensor, rast: torch.Tensor, pos: torch.Tensor, tri: torch.Tensor, **kwargs):
    """
    Identity passthrough -- see module docstring "Scope and known differences". Use
    MeshRenderer's `ssaa` option for anti-aliased output instead of relying on this.
    """
    return color


def _bilinear_sample_manual(tex_nchw: torch.Tensor, grid_x: torch.Tensor, grid_y: torch.Tensor) -> torch.Tensor:
    """
    Differentiable bilinear sample, equivalent to
    F.grid_sample(tex_nchw, stack([grid_x,grid_y],-1), mode='bilinear', padding_mode='border',
    align_corners=False), reimplemented with gather/clamp/lerp because grid_sample's backward
    isn't implemented for the MPS device in this PyTorch version (grid_sampler_2d_backward) --
    this needs to be differentiable, since it backs bake_texture's gradient-descent optimization.

    Args:
        tex_nchw: [B, C, Ht, Wt].
        grid_x, grid_y: [B, H, W] in [-1, 1].
    Returns:
        [B, C, H, W].
    """
    B, C, Ht, Wt = tex_nchw.shape
    H, W = grid_x.shape[1], grid_x.shape[2]

    px = (grid_x + 1) * 0.5 * Wt - 0.5
    py = (grid_y + 1) * 0.5 * Ht - 0.5
    x0f = torch.floor(px)
    y0f = torch.floor(py)
    wx = (px - x0f).unsqueeze(1)   # [B,1,H,W]
    wy = (py - y0f).unsqueeze(1)

    x0 = x0f.long()
    y0 = y0f.long()
    x1 = x0 + 1
    y1 = y0 + 1
    x0c, x1c = x0.clamp(0, Wt - 1), x1.clamp(0, Wt - 1)
    y0c, y1c = y0.clamp(0, Ht - 1), y1.clamp(0, Ht - 1)

    tex_flat = tex_nchw.reshape(B, C, Ht * Wt)

    def gather(xc, yc):
        idx = (yc * Wt + xc).reshape(B, 1, H * W).expand(-1, C, -1)
        return torch.gather(tex_flat, 2, idx).reshape(B, C, H, W)

    v00, v01 = gather(x0c, y0c), gather(x1c, y0c)
    v10, v11 = gather(x0c, y1c), gather(x1c, y1c)
    top = v00 * (1 - wx) + v01 * wx
    bottom = v10 * (1 - wx) + v11 * wx
    return top * (1 - wy) + bottom * wy


def texture(tex: torch.Tensor, uv: torch.Tensor, uv_da=None, mip=None, filter_mode: str = 'auto', **kwargs):
    """
    Differentiable bilinear texture sample -- standing in for nvdiffrast's dr.texture. No
    mipmapping: `uv_da`/`mip` (screen-space UV derivatives, used by nvdiffrast to pick a mip
    level and avoid minification aliasing) are accepted but ignored. This is a real simplification
    (textures viewed at a strong minifying angle can alias here where nvdiffrast wouldn't), but it
    doesn't affect *correctness* of gradient-descent texture optimization (trellis's only user of
    this function, in postprocessing_utils.py's bake_texture) -- plain bilinear sampling is still
    exactly differentiable w.r.t. the texture.

    Args:
        tex: [B, H_tex, W_tex, C] texture image(s).
        uv: [B, H, W, 2] normalized UV coordinates in [0, 1], OpenGL convention (v=0 is the
            bottom row of `tex`, v=1 is the top row), matching xatlas's UV output convention.
    Returns:
        [B, H, W, C] sampled colors.
    """
    tex_nchw = tex.permute(0, 3, 1, 2)                      # [B, C, Ht, Wt]
    u = uv[..., 0]
    v = uv[..., 1]
    grid_x = u * 2.0 - 1.0
    grid_y = 1.0 - v * 2.0
    sampled = _bilinear_sample_manual(tex_nchw, grid_x, grid_y)
    return sampled.permute(0, 2, 3, 1)                      # [B, H, W, C]
