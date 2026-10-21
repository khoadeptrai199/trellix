"""
Pure-PyTorch 3D Gaussian Splatting rasterizer, standing in for `diff_gaussian_rasterization`
(the mip-splatting CUDA kernel used by trellis/renderers/gaussian_render.py) when it isn't
installed (e.g. on Apple Silicon). Implements the same forward algorithm as the CUDA kernel:

1. Project each Gaussian's 3D covariance to 2D screen space via the EWA splatting Jacobian
   (Zwicker et al.), the same formula the CUDA kernel and every other 3DGS reimplementation uses.
2. Add mip-splatting's low-pass filter (`kernel_size`, dilates the 2D covariance diagonal).
3. Evaluate per-Gaussian color from precomputed RGB or spherical harmonics (`sh_utils.eval_sh`,
   already pure math / device-portable, unchanged).
4. Sort Gaussians by camera-space depth once (global order), then composite front-to-back over
   the whole image in chunks, using `cumprod` within each chunk to get exact ordered
   alpha-compositing without a per-Gaussian Python loop (see `_composite_chunk`).

Differences from the CUDA kernel:
- No tile-based binning / per-tile culling -- this is an O(H * W * N) dense computation chunked
  over Gaussians (same "correctness first" structure as sw_rasterizer.py's triangle rasterizer),
  not the tile-sorted approach that makes the CUDA kernel fast. Meaningfully slower for large
  Gaussian counts; a fused Metal kernel (one thread per pixel, looping over depth-sorted Gaussians
  in registers, same pattern as metal_rasterizer.py) would be the natural next optimization if
  this becomes a measured bottleneck -- not attempted here.
- No backward pass / custom autograd needed: every operation here (matrix ops, exp, cumprod,
  gather) is plain differentiable PyTorch, so autograd just works, unlike the CUDA kernel which
  ships a hand-written backward.
- `radii` is a real (if only approximately not-tile-culled) per-Gaussian screen radius, returned
  only for API compatibility with the CUDA kernel's return signature; nothing in gaussian_render.py
  actually consumes it.
"""
from typing import Optional
import torch
import torch.nn as nn

from ..representations.gaussian.general_utils import build_scaling_rotation
from .sh_utils import eval_sh

__all__ = ['GaussianRasterizationSettings', 'GaussianRasterizer']


class GaussianRasterizationSettings:
    def __init__(self, image_height, image_width, tanfovx, tanfovy, kernel_size, subpixel_offset,
                 bg, scale_modifier, viewmatrix, projmatrix, sh_degree, campos, prefiltered, debug):
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.tanfovx = tanfovx
        self.tanfovy = tanfovy
        self.kernel_size = kernel_size
        self.bg = bg
        self.scale_modifier = scale_modifier
        self.viewmatrix = viewmatrix
        self.projmatrix = projmatrix
        self.sh_degree = sh_degree
        self.campos = campos


def _expand_cov3d(cov3D_precomp: torch.Tensor) -> torch.Tensor:
    """[N, 6] stripped-symmetric (xx,xy,xz,yy,yz,zz) -> [N, 3, 3] full matrix."""
    xx, xy, xz, yy, yz, zz = cov3D_precomp.unbind(-1)
    row0 = torch.stack([xx, xy, xz], dim=-1)
    row1 = torch.stack([xy, yy, yz], dim=-1)
    row2 = torch.stack([xz, yz, zz], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _composite_chunk(alpha: torch.Tensor, color: torch.Tensor):
    """
    Depth-ordered front-to-back alpha compositing over a chunk of Gaussians already sorted by
    depth, vectorized via cumprod instead of a sequential Python loop.

    Args:
        alpha: [B, H, W] per-Gaussian per-pixel opacity contribution, in depth order (index 0 nearest).
        color: [B, 1, 1, 3] per-Gaussian color (broadcastable over H, W).
    Returns:
        (chunk_color [H, W, 3], chunk_transmittance [H, W]): this chunk's contribution to the
        image, and the fraction of light that survives through the whole chunk (to be multiplied
        into the running global transmittance by the caller).
    """
    one_minus_alpha = (1.0 - alpha).clamp(min=1e-6, max=1.0)
    cum_after = torch.cumprod(one_minus_alpha, dim=0)                     # [B,H,W], transmittance AFTER gaussian k
    trans_before = torch.cat([torch.ones_like(cum_after[:1]), cum_after[:-1]], dim=0)  # transmittance BEFORE gaussian k
    contribution = (trans_before * alpha).unsqueeze(-1) * color           # [B,H,W,3]
    return contribution.sum(dim=0), cum_after[-1]


def project_and_sort_gaussians(
    means3D: torch.Tensor,
    opacities: torch.Tensor,
    settings: GaussianRasterizationSettings,
    shs: Optional[torch.Tensor] = None,
    colors_precomp: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    rotations: Optional[torch.Tensor] = None,
    cov3D_precomp: Optional[torch.Tensor] = None,
):
    """
    Shared EWA-splatting preprocessing: project every Gaussian's 3D covariance to a 2D
    screen-space covariance (inverted, ready for the density-function evaluation), its 2D mean,
    color, and opacity -- then sort everything by camera-space depth (nearest first). This is
    O(N), cheap, and identical whether the O(H*W*N) compositing step that follows runs as chunked
    PyTorch ops (`rasterize_gaussians` below) or a Metal kernel (`metal_gaussian_rasterizer.py`).

    Returns:
        (px, py, inv_a, inv_b, inv_c, opacity, color, bg, H, W, radii_out, order): per-Gaussian
        tensors are depth-sorted (index 0 = nearest); `radii_out`/`order` let the Metal path
        scatter its output radii back to original Gaussian order for API-compat, same as here.
    """
    device = means3D.device
    dtype = means3D.dtype
    N = means3D.shape[0]
    H, W = settings.image_height, settings.image_width
    view = settings.viewmatrix.to(device=device, dtype=dtype)     # world_view_transform: extrinsics.T, row-vector convention
    proj = settings.projmatrix.to(device=device, dtype=dtype)     # full_proj_transform: (perspective @ extrinsics).T

    bg = torch.zeros(H, W, 3, device=device, dtype=dtype) + settings.bg.to(device=device, dtype=dtype).view(1, 1, 3)

    if N == 0:
        empty = torch.zeros(0, device=device, dtype=dtype)
        return empty, empty, empty, empty, empty, empty, torch.zeros(0, 3, device=device, dtype=dtype), bg, H, W, torch.zeros(0, device=device, dtype=dtype), torch.zeros(0, dtype=torch.long, device=device)

    ones = torch.ones(N, 1, device=device, dtype=dtype)
    means_h = torch.cat([means3D, ones], dim=-1)                  # [N, 4]

    cam_xyz = (means_h @ view)[:, :3]                              # camera-space xyz, [N, 3]
    clip = means_h @ proj                                          # [N, 4]
    w = clip[:, 3].clamp(min=1e-6)
    ndc = clip[:, :3] / w[:, None]
    px = (ndc[:, 0] * 0.5 + 0.5) * W
    py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * H

    # 3D covariance
    if cov3D_precomp is not None:
        Sigma3D = _expand_cov3d(cov3D_precomp)
    else:
        L = build_scaling_rotation(scales * settings.scale_modifier, rotations)   # [N,3,3]
        Sigma3D = torch.bmm(L, L.transpose(1, 2))

    # EWA splatting: project 3D covariance to 2D screen-space covariance
    focal_x = W / (2.0 * settings.tanfovx)
    focal_y = H / (2.0 * settings.tanfovy)
    W_rot = view[:3, :3].transpose(0, 1)                            # actual world->camera rotation (view is already transposed)

    z = cam_xyz[:, 2].clamp(min=1e-6)
    x = cam_xyz[:, 0]
    y = cam_xyz[:, 1]
    zeros = torch.zeros_like(z)
    J = torch.stack([
        torch.stack([focal_x / z, zeros, -focal_x * x / (z * z)], dim=-1),
        torch.stack([zeros, focal_y / z, -focal_y * y / (z * z)], dim=-1),
        torch.stack([zeros, zeros, zeros], dim=-1),
    ], dim=-2)                                                       # [N, 3, 3]

    T = J @ W_rot.unsqueeze(0)                                       # [N, 3, 3]
    Sigma2D_3x3 = T @ Sigma3D @ T.transpose(1, 2)
    Sigma2D = Sigma2D_3x3[:, :2, :2].clone()
    Sigma2D[:, 0, 0] += settings.kernel_size
    Sigma2D[:, 1, 1] += settings.kernel_size

    a = Sigma2D[:, 0, 0]
    b = Sigma2D[:, 0, 1]
    c = Sigma2D[:, 1, 1]
    det = a * c - b * b
    valid = (det > 1e-12) & (z > 0.01)
    safe_det = torch.where(valid, det, torch.ones_like(det))
    inv_a = c / safe_det
    inv_b = -b / safe_det
    inv_c = a / safe_det

    radii = 3.0 * torch.sqrt(torch.clamp(torch.max(a, c), min=0.0))
    radii = torch.where(valid, radii, torch.zeros_like(radii))

    # color
    if colors_precomp is not None:
        color = colors_precomp
    else:
        assert shs is not None
        dir_pp = means3D - settings.campos.to(device=device, dtype=dtype).view(1, 3)
        dir_pp = dir_pp / dir_pp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        # shs: [N, coeff, 3] (Gaussian.get_features layout) -> eval_sh expects [..., C, coeff]
        sh_for_eval = shs.transpose(1, 2)   # [N, 3, coeff]
        color = torch.clamp_min(eval_sh(settings.sh_degree, sh_for_eval, dir_pp) + 0.5, 0.0)

    opacity = opacities.clamp(min=0.0, max=1.0).squeeze(-1) * valid.to(dtype)   # [N]
    opacity = torch.where(z > 0.01, opacity, torch.zeros_like(opacity))         # cull behind-camera Gaussians

    order = torch.argsort(z)   # nearest first
    px, py = px[order], py[order]
    inv_a, inv_b, inv_c = inv_a[order], inv_b[order], inv_c[order]
    opacity = opacity[order]
    color = color[order]

    return px, py, inv_a, inv_b, inv_c, opacity, color, bg, H, W, radii, order


def rasterize_gaussians(
    means3D: torch.Tensor,
    opacities: torch.Tensor,
    settings: GaussianRasterizationSettings,
    shs: Optional[torch.Tensor] = None,
    colors_precomp: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    rotations: Optional[torch.Tensor] = None,
    cov3D_precomp: Optional[torch.Tensor] = None,
    chunk_size: Optional[int] = None,
):
    device = means3D.device
    dtype = means3D.dtype
    N = means3D.shape[0]

    px, py, inv_a, inv_b, inv_c, opacity, color, bg, H, W, radii, order = project_and_sort_gaussians(
        means3D, opacities, settings, shs=shs, colors_precomp=colors_precomp,
        scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp,
    )

    if N == 0:
        image = bg.permute(2, 0, 1)
        return image, torch.zeros(0, device=device)

    if chunk_size is None:
        target_elems = 16 * 1024 * 1024
        chunk_size = max(1, target_elems // max(1, H * W))

    xs = (torch.arange(W, device=device, dtype=dtype) + 0.5).view(1, 1, W)
    ys = (torch.arange(H, device=device, dtype=dtype) + 0.5).view(1, H, 1)

    accum_color = torch.zeros(H, W, 3, device=device, dtype=dtype)
    trans = torch.ones(H, W, device=device, dtype=dtype)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        B = end - start
        dx = xs - px[start:end].view(B, 1, 1)
        dy = ys - py[start:end].view(B, 1, 1)
        maha2 = inv_a[start:end].view(B, 1, 1) * dx * dx \
              + 2 * inv_b[start:end].view(B, 1, 1) * dx * dy \
              + inv_c[start:end].view(B, 1, 1) * dy * dy
        alpha = opacity[start:end].view(B, 1, 1) * torch.exp(-0.5 * maha2.clamp(min=0.0))
        alpha = alpha.clamp(max=0.9999)

        chunk_color, chunk_trans_end = _composite_chunk(alpha, color[start:end].view(B, 1, 1, 3))
        accum_color += trans.unsqueeze(-1) * chunk_color
        trans = trans * chunk_trans_end

    image = accum_color + trans.unsqueeze(-1) * bg
    image = image.permute(2, 0, 1)   # [3, H, W]

    radii_out = torch.zeros(N, device=device, dtype=dtype)
    radii_out[order] = radii[order]
    return image, radii_out


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings: GaussianRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(self, means3D, means2D=None, opacities=None, shs=None, colors_precomp=None,
                scales=None, rotations=None, cov3D_precomp=None):
        return rasterize_gaussians(
            means3D, opacities, self.raster_settings,
            shs=shs, colors_precomp=colors_precomp, scales=scales, rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
