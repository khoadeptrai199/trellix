"""
Hand-written Metal compute kernel for the same trivec radiance-field marching algorithm as
octree_renderer_mps.py (see that module's docstring for the algorithm and its provenance -- ported
from diffoctreerast's forward.cu, empty-space-skipping dense-grid ray march). Same motivation and
design as metal_rasterizer.py/metal_gaussian_rasterizer.py: one GPU thread per output pixel.

Why this exists on top of the already-correct pure-PyTorch version: that version's Python-level
loop over marching steps pays MPS kernel-dispatch overhead on every single step (empty-space jump
or fine sub-voxel sample), which dominated cost even after fixing the empty-space-skipping
algorithm itself (18.1s/frame at 256x256 on a real model output -- see APPLE_SILICON.md). Here, the
*entire* per-pixel marching loop (empty-space DDA jumps and fine trivec sampling both) runs inside
a single compiled Metal kernel invocation, so there's exactly one dispatch per frame instead of one
per marching step. The math is unchanged -- this is the same algorithm as octree_renderer_mps.py,
translated to Metal Shading Language line-for-line, not a different approximation.

Falls back transparently to octree_renderer_mps.render_trivec if PyObjC / a Metal device / runtime
shader compilation aren't available for any reason.
"""
import numpy as np
import torch

from . import octree_renderer_mps as _pytorch_impl

__all__ = ['render_trivec', 'available']

_KERNEL_SRC = """
#include <metal_stdlib>
using namespace metal;

inline float lerp1(float a, float b, float t) { return a + t * (b - a); }

kernel void octree_trivec_march_kernel(
    device const float* ray_dir       [[buffer(0)]],   // [H*W*3]
    device const float* depth_div     [[buffer(1)]],   // [H*W]
    device const float* t_near        [[buffer(2)]],   // [H*W]
    device const float* t_far         [[buffer(3)]],   // [H*W]
    device const float* cam_pos_buf   [[buffer(4)]],   // [3]
    device const int*   grid_idx      [[buffer(5)]],   // [res^3]
    device const float* trivec        [[buffer(6)]],   // [N*rank*3*dim]
    device const float* densities     [[buffer(7)]],   // [N*rank]
    device const float* colors        [[buffer(8)]],   // [N*rank*3]
    device const float* aabb_min_buf  [[buffer(9)]],   // [3]
    device const float* voxel_sz_buf  [[buffer(10)]],  // [3]
    constant uint&       res          [[buffer(11)]],
    constant uint&       dim          [[buffer(12)]],
    constant uint&       rank         [[buffer(13)]],
    constant float&      density_shift [[buffer(14)]],
    constant float&      density_scale [[buffer(15)]],
    constant float&      fine_step     [[buffer(16)]],
    constant float&      boundary_eps  [[buffer(17)]],
    constant uint&       max_iters     [[buffer(18)]],
    device const float* bg_color_buf  [[buffer(19)]],  // [3]
    constant uint&       W             [[buffer(20)]],
    constant uint&       H             [[buffer(21)]],
    device float*        out_color        [[buffer(22)]],
    device float*        out_depth        [[buffer(23)]],
    device float*        out_alpha        [[buffer(24)]],
    device float*        out_percent_depth [[buffer(25)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= W || gid.y >= H) return;
    uint pix = gid.y * W + gid.x;

    float3 cam_pos = float3(cam_pos_buf[0], cam_pos_buf[1], cam_pos_buf[2]);
    float3 aabb_min = float3(aabb_min_buf[0], aabb_min_buf[1], aabb_min_buf[2]);
    float3 voxel_size = float3(voxel_sz_buf[0], voxel_sz_buf[1], voxel_sz_buf[2]);
    float3 bg_color = float3(bg_color_buf[0], bg_color_buf[1], bg_color_buf[2]);
    float3 dir = float3(ray_dir[pix * 3 + 0], ray_dir[pix * 3 + 1], ray_dir[pix * 3 + 2]);

    float tf = t_far[pix];
    float t = t_near[pix];
    float T = 1.0f;
    float3 C = float3(0.0f, 0.0f, 0.0f);
    float D = 0.0f, PD = 0.0f;
    int ires = int(res);

    for (uint it = 0; it < max_iters; it++) {
        if (t > tf || T < 0.001f) break;

        float3 p = cam_pos + t * dir;
        float3 fc = clamp(floor((p - aabb_min) / voxel_size), float3(0.0f), float3(float(ires) - 1.0f));
        int3 vc = int3(fc);
        int node_id = grid_idx[(vc.x * ires + vc.y) * ires + vc.z];

        float3 voxel_min = aabb_min + fc * voxel_size;
        float3 voxel_max = voxel_min + voxel_size;

        if (node_id >= 0) {
            float3 local = (p - voxel_min) / voxel_size * float(dim) - 0.5f;
            int3 ip = clamp(int3(floor(local)), int3(0), int3(int(dim) - 2));
            float3 w = local - float3(ip);

            float density = 0.0f;
            float3 color = float3(0.0f, 0.0f, 0.0f);
            uint node_base = uint(node_id) * rank * 3u * dim;
            for (uint r = 0; r < rank; r++) {
                uint rbase = node_base + r * 3u * dim;
                float tx = lerp1(trivec[rbase + uint(ip.x)], trivec[rbase + uint(ip.x) + 1u], w.x);
                float ty = lerp1(trivec[rbase + dim + uint(ip.y)], trivec[rbase + dim + uint(ip.y) + 1u], w.y);
                float tz = lerp1(trivec[rbase + 2u * dim + uint(ip.z)], trivec[rbase + 2u * dim + uint(ip.z) + 1u], w.z);
                float tri = tx * ty * tz;

                density += densities[uint(node_id) * rank + r] * tri;
                uint cbase = (uint(node_id) * rank + r) * 3u;
                color.x += colors[cbase + 0] * tri;
                color.y += colors[cbase + 1] * tri;
                color.z += colors[cbase + 2] * tri;
            }
            // softplus(x)=log(1+exp(x)); CUDA's x>=10 fast-path (returns x directly, avoiding
            // overflow in exp(x)) is unnecessary here in float32 for this port's density
            // magnitudes, so it's applied unconditionally rather than branched.
            float dens = log(1.0f + exp(density - density_shift * 10.0f)) * density_scale;
            float3 col = 1.0f / (1.0f + exp(-color));

            float alpha = min(1.0f - exp(-dens * fine_step), 0.999f);
            float weight = alpha * T;
            C += col * weight;
            D += t * weight;
            if (PD == 0.0f && T * (1.0f - alpha) < 0.5f) PD = t;
            T *= (1.0f - alpha);
            t += fine_step;
        } else {
            float3 t_bound = float3(
                dir.x > 0.0f ? (voxel_max.x - p.x) / dir.x : (dir.x < 0.0f ? (voxel_min.x - p.x) / dir.x : INFINITY),
                dir.y > 0.0f ? (voxel_max.y - p.y) / dir.y : (dir.y < 0.0f ? (voxel_min.y - p.y) / dir.y : INFINITY),
                dir.z > 0.0f ? (voxel_max.z - p.z) / dir.z : (dir.z < 0.0f ? (voxel_min.z - p.z) / dir.z : INFINITY)
            );
            t += min(t_bound.x, min(t_bound.y, t_bound.z)) + boundary_eps;
        }
    }

    float3 result = C + T * bg_color;
    out_color[pix * 3 + 0] = result.x;
    out_color[pix * 3 + 1] = result.y;
    out_color[pix * 3 + 2] = result.z;
    out_depth[pix] = D * depth_div[pix];
    out_alpha[pix] = 1.0f - T;
    out_percent_depth[pix] = PD * depth_div[pix];
}
"""

_state = {'checked': False, 'ok': False}


def _init():
    if _state['checked']:
        return _state['ok']
    _state['checked'] = True
    try:
        import Metal
        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            return False
        options = Metal.MTLCompileOptions.alloc().init()
        library, error = device.newLibraryWithSource_options_error_(_KERNEL_SRC, options, None)
        if library is None:
            raise RuntimeError(str(error))
        fn = library.newFunctionWithName_("octree_trivec_march_kernel")
        pipeline, error = device.newComputePipelineStateWithFunction_error_(fn, None)
        if pipeline is None:
            raise RuntimeError(str(error))
        _state['device'] = device
        _state['queue'] = device.newCommandQueue()
        _state['pipeline'] = pipeline
        _state['Metal'] = Metal
        _state['ok'] = True
    except Exception as e:
        import warnings
        warnings.warn(f"octree_renderer_metal: falling back to pure-PyTorch, Metal init failed: {e}")
        _state['ok'] = False
    return _state['ok']


def available() -> bool:
    return _init()


def _to_buffer(device, Metal, arr: np.ndarray):
    arr = np.ascontiguousarray(arr)
    return device.newBufferWithBytes_length_options_(arr.tobytes(), arr.nbytes, Metal.MTLResourceStorageModeShared)


def _readback(buf, shape, dtype):
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    raw = buf.contents().as_buffer(nbytes)
    return np.frombuffer(raw, dtype=dtype, count=int(np.prod(shape))).reshape(shape).copy()


def render_trivec(
    octree,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    resolution: int,
    bg_color: torch.Tensor,
    colors_overwrite: torch.Tensor = None,
    near: float = 0.2,
):
    """Drop-in replacement for octree_renderer_mps.render_trivec -- same signature, same
    algorithm, same output contract. See that module's docstring for what's implemented (trivec
    primitive, sh_degree=0 only) and why."""
    if not _init() or colors_overwrite is not None:
        # colors_overwrite (used by bake_texture-style color-only passes) isn't wired into the
        # kernel -- not needed by anything in this pipeline's actual radiance_field usage, which
        # only ever does full SH/color rendering. Falls back rather than silently ignoring it.
        return _pytorch_impl.render_trivec(octree, extrinsics, intrinsics, resolution, bg_color,
                                            colors_overwrite=colors_overwrite, near=near)

    assert octree.primitive == 'trivec', f"octree_renderer_metal only implements the 'trivec' primitive, got '{octree.primitive}'"
    assert octree.sh_degree == 0, f"octree_renderer_metal only implements sh_degree=0; got sh_degree={octree.sh_degree}"

    H = W = resolution
    dim = octree.primitive_config['dim']
    rank = octree.primitive_config['rank']
    res = octree.resolution
    N = octree.position.shape[0]

    if N == 0:
        color_out = bg_color.reshape(1, 1, 3).expand(H, W, 3).permute(2, 0, 1).contiguous()
        zeros = torch.zeros((H, W), device=octree.position.device, dtype=octree.position.dtype)
        return {'color': color_out, 'depth': zeros, 'alpha': zeros, 'percent_depth': zeros}

    device_t = octree.position.device
    dtype = octree.position.dtype
    extrinsics = extrinsics.to(device=device_t, dtype=dtype)
    intrinsics = intrinsics.to(device=device_t, dtype=dtype)

    aabb = octree.aabb
    aabb_min = aabb[:3]
    aabb_size = aabb[3:]
    voxel_size = aabb_size / res

    # --- Same camera-ray derivation as octree_renderer_mps.render_trivec -- see that module's
    # docstring for the getRayDir derivation this mirrors.
    focal_x, focal_y = intrinsics[0, 0], intrinsics[1, 1]
    tan_fovx = 0.5 / focal_x
    tan_fovy = 0.5 / focal_y
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device_t, dtype=dtype), torch.arange(W, device=device_t, dtype=dtype), indexing='ij')
    ndc_x = (2.0 * (xs + 0.5) / W - 1.0) * tan_fovx
    ndc_y = (2.0 * (ys + 0.5) / H - 1.0) * tan_fovy
    right, up, fwd = extrinsics[0, :3], extrinsics[1, :3], extrinsics[2, :3]
    ray_dir = ndc_x[..., None] * right + ndc_y[..., None] * up + fwd
    ray_len = ray_dir.norm(dim=-1, keepdim=True)
    ray_dir = ray_dir / ray_len
    depth_div = 1.0 / ray_len.squeeze(-1)
    cam_pos = torch.inverse(extrinsics)[:3, 3].to(device=device_t, dtype=dtype)

    aabb_max = aabb_min + aabb_size
    inv_dir = 1.0 / torch.where(ray_dir == 0, torch.full_like(ray_dir, 1e10), ray_dir)
    t0 = (aabb_min - cam_pos) * inv_dir
    t1 = (aabb_max - cam_pos) * inv_dir
    t_near = torch.maximum(torch.minimum(t0, t1).amax(dim=-1), torch.tensor(near, device=device_t, dtype=dtype))
    t_far = torch.maximum(t0, t1).amin(dim=-1)

    grid_idx = torch.full((res, res, res), -1, dtype=torch.long, device=device_t)
    voxel_coords = torch.clamp((octree.position * res - 0.5).round().long(), 0, res - 1)
    grid_idx[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = torch.arange(N, device=device_t)

    density_shift = getattr(octree, 'density_shift', 0.0)
    colors_raw = _pytorch_impl.SH_C0 * octree.features_dc.squeeze(2)  # [N, rank, 3]

    fine_step = (0.5 / dim) * voxel_size[0].item()
    boundary_eps = float(voxel_size.min().item()) * 1e-3
    max_iters = int(3 * res + 4 * dim)

    device = _state['device']
    Metal = _state['Metal']
    queue = _state['queue']
    pipeline = _state['pipeline']

    ray_dir_np = ray_dir.reshape(-1, 3).detach().to('cpu').to(torch.float32).numpy()
    depth_div_np = depth_div.reshape(-1).detach().to('cpu').to(torch.float32).numpy()
    t_near_np = t_near.reshape(-1).detach().to('cpu').to(torch.float32).numpy()
    t_far_np = t_far.reshape(-1).detach().to('cpu').to(torch.float32).numpy()
    cam_pos_np = cam_pos.detach().to('cpu').to(torch.float32).numpy()
    grid_idx_np = grid_idx.reshape(-1).detach().to('cpu').to(torch.int32).numpy()
    trivec_np = octree.trivec.detach().to('cpu').to(torch.float32).numpy()
    densities_np = octree.density.detach().to('cpu').to(torch.float32).numpy()
    colors_np = colors_raw.detach().to('cpu').to(torch.float32).numpy()
    aabb_min_np = aabb_min.detach().to('cpu').to(torch.float32).numpy()
    voxel_size_np = voxel_size.detach().to('cpu').to(torch.float32).numpy()
    bg_color_np = bg_color.detach().to('cpu').to(torch.float32).numpy()

    buffers = [
        _to_buffer(device, Metal, ray_dir_np),
        _to_buffer(device, Metal, depth_div_np),
        _to_buffer(device, Metal, t_near_np),
        _to_buffer(device, Metal, t_far_np),
        _to_buffer(device, Metal, cam_pos_np),
        _to_buffer(device, Metal, grid_idx_np),
        _to_buffer(device, Metal, trivec_np),
        _to_buffer(device, Metal, densities_np),
        _to_buffer(device, Metal, colors_np),
        _to_buffer(device, Metal, aabb_min_np),
        _to_buffer(device, Metal, voxel_size_np),
        _to_buffer(device, Metal, np.array([res], dtype=np.uint32)),
        _to_buffer(device, Metal, np.array([dim], dtype=np.uint32)),
        _to_buffer(device, Metal, np.array([rank], dtype=np.uint32)),
        _to_buffer(device, Metal, np.array([density_shift], dtype=np.float32)),
        _to_buffer(device, Metal, np.array([min(1.0 / (1.0 - density_shift), 25.0)], dtype=np.float32)),
        _to_buffer(device, Metal, np.array([fine_step], dtype=np.float32)),
        _to_buffer(device, Metal, np.array([boundary_eps], dtype=np.float32)),
        _to_buffer(device, Metal, np.array([max_iters], dtype=np.uint32)),
        _to_buffer(device, Metal, bg_color_np),
        _to_buffer(device, Metal, np.array([W], dtype=np.uint32)),
        _to_buffer(device, Metal, np.array([H], dtype=np.uint32)),
    ]
    n_pix = H * W
    buf_out_color = device.newBufferWithLength_options_(n_pix * 3 * 4, Metal.MTLResourceStorageModeShared)
    buf_out_depth = device.newBufferWithLength_options_(n_pix * 4, Metal.MTLResourceStorageModeShared)
    buf_out_alpha = device.newBufferWithLength_options_(n_pix * 4, Metal.MTLResourceStorageModeShared)
    buf_out_pdepth = device.newBufferWithLength_options_(n_pix * 4, Metal.MTLResourceStorageModeShared)
    buffers += [buf_out_color, buf_out_depth, buf_out_alpha, buf_out_pdepth]

    cmdbuf = queue.commandBuffer()
    encoder = cmdbuf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    for i, b in enumerate(buffers):
        encoder.setBuffer_offset_atIndex_(b, 0, i)
    tg_w = 16
    tg_h = pipeline.maxTotalThreadsPerThreadgroup() // tg_w
    grid = Metal.MTLSizeMake(W, H, 1)
    tg = Metal.MTLSizeMake(tg_w, max(1, tg_h), 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(grid, tg)
    encoder.endEncoding()
    cmdbuf.commit()
    cmdbuf.waitUntilCompleted()
    if cmdbuf.status() == 5:  # MTLCommandBufferStatusError
        err = cmdbuf.error()
        raise RuntimeError(f"octree_renderer_metal: Metal command buffer failed: {err}")

    color_np = _readback(buf_out_color, (H, W, 3), np.float32)
    depth_np = _readback(buf_out_depth, (H, W), np.float32)
    alpha_np = _readback(buf_out_alpha, (H, W), np.float32)
    pdepth_np = _readback(buf_out_pdepth, (H, W), np.float32)

    return {
        'color': torch.from_numpy(color_np).to(device_t, dtype).permute(2, 0, 1).contiguous(),
        'depth': torch.from_numpy(depth_np).to(device_t, dtype),
        'alpha': torch.from_numpy(alpha_np).to(device_t, dtype),
        'percent_depth': torch.from_numpy(pdepth_np).to(device_t, dtype),
    }
