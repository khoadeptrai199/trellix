"""
Pure-PyTorch (MPS-compatible) forward-only renderer for the `Strivec` radiance-field
representation, replacing `diffoctreerast`'s CUDA-only `OctreeTrivecRasterizer`.

Ported from https://github.com/JeffreyXiang/diffoctreerast (src/octree_trivec_rasterizer/cuda/
forward.cu, auxiliary.h, config.h) by reading the CUDA source directly -- every math formula below
(ray generation, ray-AABB intersection, trivec trilinear sampling, softplus/sigmoid activations,
alpha-compositing accumulation) is a line-for-line port of that kernel, not a reimplementation from
a paper/description. See git history for the fetched source used as reference.

Scope reduction vs. the original library, both justified by how TRELLIS actually uses this:
- Only the "trivec" primitive is implemented (not voxel/gaussian/decoupoly) -- `Strivec` (the only
  representation class TRELLIS's pretrained decoder produces, trellis/models/structured_latent_vae/
  decoder_rf.py) hardcodes `primitive='trivec'`.
- Only sh_degree=0 (view-independent color) is implemented -- decoder_rf.py's `to_representation`
  hardcodes `sh_degree=0` on every Strivec it builds. Asserted at call time rather than silently
  producing wrong colors if that ever changes.
- Forward-only, no custom backward kernel -- unlike the hand-Metal mesh/Gaussian rasterizers (which
  needed a real backward pass for bake_texture's gradient-descent optimization), nothing in this
  port's actual pipeline differentiates through radiance-field rendering. Since this is plain
  PyTorch tensor ops (not a hand-written kernel), autograd works automatically if anyone does need
  gradients later -- no separate backward implementation required, unlike the CUDA original.

Algorithm: TRELLIS's decoder always produces *uniform-depth* leaves (decoder_rf.py sets every
node's depth to log2(resolution), regardless of what a general adaptive DfsOctree could represent)
-- i.e. this is really a sparse voxel grid at one fixed resolution, not a multi-depth tree. That
lets this port skip the CUDA original's tile-binning + per-tile depth-sort machinery entirely:
because octree leaves are non-overlapping axis-aligned cells that exactly tile the scene AABB, a
ray visits them in strictly monotonic distance order automatically. So instead of "for each node,
march within its local intersection interval" (which requires an explicit per-pixel sort across
nodes to get compositing order right), this does one *global* ray march through world space in
fixed steps, looking up "which leaf (if any) contains this sample point" via a dense grid at each
step. Front-to-back order falls out for free, no sort needed, and every step is a single vectorized
tensor op across all pixels at once (Python-level loop only over the marching step index, not over
pixels or nodes).
"""
import torch
import torch.nn.functional as F

SH_C0 = 0.28209479177387814


def _softplus(x):
    return F.softplus(x)


def _lerp(v0, v1, t):
    return v0 + t * (v1 - v0)


def render_trivec(
    octree,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    resolution: int,
    bg_color: torch.Tensor,
    colors_overwrite: torch.Tensor = None,
    near: float = 0.2,
):
    """
    Forward-render a Strivec (DfsOctree with primitive='trivec') via global dense-grid ray
    marching. See module docstring for the algorithm and what's intentionally not implemented.

    Args:
        octree: a DfsOctree/Strivec instance (primitive must be 'trivec', sh_degree must be 0).
        extrinsics: [4, 4] world-to-camera matrix.
        intrinsics: [3, 3] OpenCV-style camera intrinsics (normalized, as elsewhere in this repo).
        resolution: output image is resolution x resolution.
        bg_color: [3] background color tensor, same device/dtype as octree data.
        colors_overwrite: optional [N, 3] per-leaf color override (skips SH/color evaluation).
        near: near-plane distance (world units) for the global ray-AABB intersection.
    Returns:
        dict with 'color' [3,H,W], 'depth' [H,W], 'alpha' [H,W], 'percent_depth' [H,W].
    """
    assert octree.primitive == 'trivec', f"octree_renderer_mps only implements the 'trivec' primitive, got '{octree.primitive}'"
    assert octree.sh_degree == 0, f"octree_renderer_mps only implements sh_degree=0 (TRELLIS's decoder always uses this); got sh_degree={octree.sh_degree}"

    device = octree.position.device
    dtype = octree.position.dtype
    H = W = resolution
    dim = octree.primitive_config['dim']
    rank = octree.primitive_config['rank']
    res = octree.resolution
    N = octree.position.shape[0]

    aabb = octree.aabb  # [6]: xmin,ymin,zmin, xsize,ysize,zsize
    aabb_min = aabb[:3]
    aabb_size = aabb[3:]
    voxel_size = aabb_size / res  # [3], world-space size of one leaf

    extrinsics = extrinsics.to(device=device, dtype=dtype)
    intrinsics = intrinsics.to(device=device, dtype=dtype)

    # --- Camera rays (see module docstring: ray_dir = ndc_x*right + ndc_y*up + 1*forward, where
    # right/up/forward are ROWS of the world-to-camera rotation -- derived from diffoctreerast's
    # getRayDir, accounting for the raw-buffer transpose it operates on; see git history for the
    # derivation notes kept alongside the CUDA source this was ported from).
    focal_x, focal_y = intrinsics[0, 0], intrinsics[1, 1]
    tan_fovx = 0.5 / focal_x
    tan_fovy = 0.5 / focal_y
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype), torch.arange(W, device=device, dtype=dtype), indexing='ij')
    ndc_x = (2.0 * (xs + 0.5) / W - 1.0) * tan_fovx
    ndc_y = (2.0 * (ys + 0.5) / H - 1.0) * tan_fovy
    right, up, fwd = extrinsics[0, :3], extrinsics[1, :3], extrinsics[2, :3]
    ray_dir = ndc_x[..., None] * right + ndc_y[..., None] * up + fwd  # [H,W,3], un-normalized
    ray_len = ray_dir.norm(dim=-1, keepdim=True)
    ray_dir = ray_dir / ray_len
    depth_div = 1.0 / ray_len.squeeze(-1)  # [H,W] -- converts unit-ray distance back to CUDA's z-depth convention
    cam_pos = torch.inverse(extrinsics)[:3, 3].to(device=device, dtype=dtype)  # [3], same for every pixel

    # --- Global ray-AABB intersection (scene bounds, not per-leaf) ---
    aabb_max = aabb_min + aabb_size
    inv_dir = 1.0 / torch.where(ray_dir == 0, torch.full_like(ray_dir, 1e10), ray_dir)
    t0 = (aabb_min - cam_pos) * inv_dir
    t1 = (aabb_max - cam_pos) * inv_dir
    t_near = torch.maximum(torch.minimum(t0, t1).amax(dim=-1), torch.tensor(near, device=device, dtype=dtype))
    t_far = torch.maximum(t0, t1).amin(dim=-1)
    hit_scene = t_far > t_near  # [H,W]

    # --- Dense leaf-index grid: leaves are uniform-depth, so this is just a sparse voxel grid ---
    grid_idx = torch.full((res, res, res), -1, dtype=torch.long, device=device)
    voxel_coords = torch.clamp((octree.position * res - 0.5).round().long(), 0, res - 1)  # [N,3]
    grid_idx[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = torch.arange(N, device=device)

    trivec = octree.trivec  # [N, rank, 3, dim]
    densities_raw = octree.density  # [N, rank] -- raw, trivec's density_activation is identity (see octree_dfs.py)
    density_shift = getattr(octree, 'density_shift', 0.0)
    if colors_overwrite is None:
        colors_raw = SH_C0 * octree.features_dc.squeeze(2)  # [N, rank, 3], sh_degree=0 -> DC term only
    else:
        colors_overwrite = colors_overwrite.to(device=device, dtype=dtype)

    fine_step = (0.5 / dim) * voxel_size[0].item()
    # Empty-space skipping: leaves cover ~5-15% of the grid for typical TRELLIS output, so a global
    # march at the fine per-voxel step size (needed for correct trilinear sampling) across the
    # *entire* scene bounding box wastes the overwhelming majority of its steps in empty cells --
    # this was the initial version's actual bottleneck (measured ~80s/frame at 256x256, not
    # remotely usable). Instead: while the current sample point is in an unoccupied cell, jump
    # straight to the next cell boundary along the ray (cheap, one big step per empty cell,
    # DDA-style); only switch to the fine trivec-resolution step size once inside an occupied leaf.
    # This is still an *exact* re-derivation of the same marching integral, not an approximation --
    # empty cells contribute zero density regardless of how many samples land in them, so skipping
    # them changes nothing about the result, only how many (wasted) steps it takes to compute it.
    boundary_eps = float(voxel_size.min().item()) * 1e-3
    max_iters = int(3 * res + 4 * dim)  # generous bound: ~one coarse jump per grid cell crossed
                                          # along the longest axis, plus a few fine sub-loops' worth

    T = torch.ones((H, W), device=device, dtype=dtype)
    C = torch.zeros((H, W, 3), device=device, dtype=dtype)
    D = torch.zeros((H, W), device=device, dtype=dtype)
    PD = torch.zeros((H, W), device=device, dtype=dtype)
    t = t_near.clone()
    active = hit_scene.clone()

    def _axis_taps(node_trivec, axis, idx):
        idx_r = idx[..., axis:axis + 1].expand(-1, -1, rank)  # [H,W,rank]
        lo = torch.gather(node_trivec[..., axis, :], -1, idx_r.unsqueeze(-1)).squeeze(-1)
        hi = torch.gather(node_trivec[..., axis, :], -1, (idx_r + 1).unsqueeze(-1)).squeeze(-1)
        return lo, hi

    for _ in range(max_iters):
        if not active.any():
            break
        in_range = active & (t <= t_far)
        if not in_range.any():
            active = in_range
            continue

        p = cam_pos + t[..., None] * ray_dir  # [H,W,3]
        vc = torch.clamp(torch.floor((p - aabb_min) / voxel_size).long(), 0, res - 1)
        node_id = grid_idx[vc[..., 0], vc[..., 1], vc[..., 2]]  # [H,W]
        hit = in_range & (node_id >= 0)

        voxel_min = aabb_min + vc.to(dtype) * voxel_size
        voxel_max = voxel_min + voxel_size

        if hit.any():
            safe_id = node_id.clamp(min=0)
            # Local position within the voxel, in trivec-grid units minus a half-texel (matches
            # sample_trivec's `_p = (p - voxel_min)/(voxel_max - voxel_min) * dim - 0.5` exactly).
            local = (p - voxel_min) / voxel_size * dim - 0.5
            ip = torch.clamp(local.floor().long(), 0, dim - 2)
            w = local - ip.to(dtype)

            node_trivec = trivec[safe_id]  # [H,W,rank,3,dim]
            x_lo, x_hi = _axis_taps(node_trivec, 0, ip)
            y_lo, y_hi = _axis_taps(node_trivec, 1, ip)
            z_lo, z_hi = _axis_taps(node_trivec, 2, ip)
            tri = _lerp(x_lo, x_hi, w[..., 0:1]) * _lerp(y_lo, y_hi, w[..., 1:2]) * _lerp(z_lo, z_hi, w[..., 2:3])  # [H,W,rank]

            node_density = densities_raw[safe_id]  # [H,W,rank]
            density = (node_density * tri).sum(dim=-1)  # [H,W]
            density = _softplus(density - density_shift * 10) * min(1.0 / (1.0 - density_shift), 25.0)

            if colors_overwrite is None:
                node_color = colors_raw[safe_id]  # [H,W,rank,3]
                color = torch.sigmoid((node_color * tri.unsqueeze(-1)).sum(dim=-2))  # [H,W,3]
            else:
                color = colors_overwrite[safe_id]  # [H,W,3]

            alpha = torch.clamp(1.0 - torch.exp(-density * fine_step), max=0.999)
            weight = alpha * T
            hit_f = hit.to(dtype)
            C = C + color * (weight * hit_f).unsqueeze(-1)
            D = D + t * weight * hit_f
            newly_below_half = hit & (PD == 0) & ((T * (1 - alpha)) < 0.5)
            PD = torch.where(newly_below_half, t, PD)
            T = torch.where(hit, T * (1 - alpha), T)

        # Advance: fine step where currently inside an occupied leaf, else jump to the next cell
        # boundary along the ray (standard DDA next-boundary distance, per axis, take the nearest).
        t_to_bound = torch.where(
            ray_dir > 0, (voxel_max - p) / ray_dir,
            torch.where(ray_dir < 0, (voxel_min - p) / ray_dir, torch.full_like(ray_dir, float('inf'))),
        ).amin(dim=-1)
        step_amount = torch.where(hit, torch.full_like(t_to_bound, fine_step), t_to_bound + boundary_eps)
        t = torch.where(in_range, t + step_amount, t)
        active = in_range & (T >= 0.001)

    color_out = C + T.unsqueeze(-1) * bg_color
    return {
        'color': color_out.permute(2, 0, 1).contiguous(),  # [3,H,W]
        'depth': (D * depth_div),
        'alpha': (1.0 - T),
        'percent_depth': (PD * depth_div),
    }
