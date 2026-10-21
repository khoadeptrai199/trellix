"""
Hand-written Metal compute kernel for 3D Gaussian splat compositing, standing in for the O(H*W*N)
chunked-PyTorch compositing loop in `gaussian_rasterizer_pytorch.py` when raw throughput matters
-- e.g. `postprocessing_utils.bake_texture`'s multi-view render, or any preview video with a
realistic (hundreds of thousands) splat count. Same motivation and same design as
`metal_rasterizer.py`'s triangle-rasterization kernel: one GPU thread per output pixel, looping
over all (already depth-sorted) Gaussians in registers, no atomics needed since each thread owns
exactly one pixel.

The EWA-splatting projection (3D covariance -> 2D, depth sort, color/opacity evaluation) is cheap
(O(N)) and shared with the pure-PyTorch path via `gaussian_rasterizer_pytorch.project_and_sort_gaussians`
-- only the expensive O(H*W*N) compositing loop is reimplemented here.

Compiled at *runtime* via MTLDevice.newLibraryWithSource_options_error_ (Metal.framework, present
on every Mac), same reasoning as metal_rasterizer.py: xcrun's offline shader compiler needs a full
Xcode.app install, which isn't necessarily present.

Falls back transparently to the pure-PyTorch path if PyObjC / a Metal device / runtime shader
compilation aren't available for any reason.
"""
from typing import Optional
import numpy as np
import torch

from .gaussian_rasterizer_pytorch import (
    GaussianRasterizationSettings, project_and_sort_gaussians, rasterize_gaussians as _rasterize_gaussians_pytorch,
)

__all__ = ['GaussianRasterizationSettings', 'GaussianRasterizer', 'available']

_KERNEL_SRC = """
#include <metal_stdlib>
using namespace metal;

kernel void gaussian_composite_kernel(
    device const float* px      [[buffer(0)]],
    device const float* py      [[buffer(1)]],
    device const float* inv_a   [[buffer(2)]],
    device const float* inv_b   [[buffer(3)]],
    device const float* inv_c   [[buffer(4)]],
    device const float* opacity [[buffer(5)]],
    device const float* color   [[buffer(6)]],
    constant uint&       N      [[buffer(7)]],
    constant uint&       W      [[buffer(8)]],
    constant uint&       H      [[buffer(9)]],
    device const float* bg      [[buffer(10)]],
    device float* out_color     [[buffer(11)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= W || gid.y >= H) return;

    float px_ = float(gid.x) + 0.5f;
    float py_ = float(gid.y) + 0.5f;

    float T = 1.0f;
    float3 C = float3(0.0f, 0.0f, 0.0f);

    for (uint i = 0; i < N; i++) {
        float dx = px_ - px[i];
        float dy = py_ - py[i];
        float maha2 = inv_a[i] * dx * dx + 2.0f * inv_b[i] * dx * dy + inv_c[i] * dy * dy;
        if (maha2 < 0.0f) continue;

        float alpha = opacity[i] * exp(-0.5f * maha2);
        alpha = min(alpha, 0.9999f);

        float3 col = float3(color[i * 3 + 0], color[i * 3 + 1], color[i * 3 + 2]);
        C += T * alpha * col;
        T *= (1.0f - alpha);

        if (T < 1e-4f) break;   // negligible remaining transmittance -- matches reference within fp noise
    }

    uint idx = gid.y * W + gid.x;
    float3 bgc = float3(bg[idx * 3 + 0], bg[idx * 3 + 1], bg[idx * 3 + 2]);
    float3 result = C + T * bgc;
    out_color[idx * 3 + 0] = result.x;
    out_color[idx * 3 + 1] = result.y;
    out_color[idx * 3 + 2] = result.z;
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
        fn = library.newFunctionWithName_("gaussian_composite_kernel")
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
        warnings.warn(f"metal_gaussian_rasterizer: falling back to pure-PyTorch, Metal init failed: {e}")
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


def rasterize_gaussians(
    means3D: torch.Tensor,
    opacities: torch.Tensor,
    settings: GaussianRasterizationSettings,
    shs: Optional[torch.Tensor] = None,
    colors_precomp: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    rotations: Optional[torch.Tensor] = None,
    cov3D_precomp: Optional[torch.Tensor] = None,
):
    if not _init():
        return _rasterize_gaussians_pytorch(
            means3D, opacities, settings, shs=shs, colors_precomp=colors_precomp,
            scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp,
        )

    orig_device = means3D.device
    dtype = means3D.dtype
    N = means3D.shape[0]

    px, py, inv_a, inv_b, inv_c, opacity, color, bg, H, W, radii, order = project_and_sort_gaussians(
        means3D, opacities, settings, shs=shs, colors_precomp=colors_precomp,
        scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp,
    )

    if N == 0:
        image = bg.permute(2, 0, 1)
        return image, torch.zeros(0, device=orig_device)

    device = _state['device']
    Metal = _state['Metal']
    queue = _state['queue']
    pipeline = _state['pipeline']

    # Batch every MPS->CPU transfer into a single sync point instead of one per array. Each
    # `.to('cpu')` call is a full MPSStream synchronize/copy barrier with real fixed overhead
    # beyond the raw bytes moved (confirmed via `sample`: with 8 separate calls, the process
    # spent most of its time inside at::mps::MPSStream::synchronize) -- concatenating everything
    # into one flat tensor and transferring once cuts that from 8 sync points to 1.
    scalars = torch.stack([px, py, inv_a, inv_b, inv_c, opacity], dim=0)   # [6, N]
    flat = torch.cat([scalars.reshape(-1), color.reshape(-1), bg.reshape(-1)])
    flat_np = flat.detach().to('cpu').to(torch.float32).numpy()

    n_scalars = 6 * N
    n_color = N * 3
    scalars_np = flat_np[:n_scalars].reshape(6, N)
    color_np = flat_np[n_scalars:n_scalars + n_color]
    bg_np = flat_np[n_scalars + n_color:]

    px_np, py_np, inv_a_np, inv_b_np, inv_c_np, opacity_np = scalars_np

    buf_px = _to_buffer(device, Metal, px_np)
    buf_py = _to_buffer(device, Metal, py_np)
    buf_ia = _to_buffer(device, Metal, inv_a_np)
    buf_ib = _to_buffer(device, Metal, inv_b_np)
    buf_ic = _to_buffer(device, Metal, inv_c_np)
    buf_op = _to_buffer(device, Metal, opacity_np)
    buf_col = _to_buffer(device, Metal, color_np)
    buf_bg = _to_buffer(device, Metal, bg_np)

    n_pix = H * W
    buf_out = device.newBufferWithLength_options_(n_pix * 3 * 4, Metal.MTLResourceStorageModeShared)

    buf_N = _to_buffer(device, Metal, np.array([N], dtype=np.uint32))
    buf_W = _to_buffer(device, Metal, np.array([W], dtype=np.uint32))
    buf_H = _to_buffer(device, Metal, np.array([H], dtype=np.uint32))

    cmdbuf = queue.commandBuffer()
    encoder = cmdbuf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    for i, b in enumerate([buf_px, buf_py, buf_ia, buf_ib, buf_ic, buf_op, buf_col, buf_N, buf_W, buf_H, buf_bg, buf_out]):
        encoder.setBuffer_offset_atIndex_(b, 0, i)
    tg_w = 16
    tg_h = pipeline.maxTotalThreadsPerThreadgroup() // tg_w
    grid = Metal.MTLSizeMake(W, H, 1)
    tg = Metal.MTLSizeMake(tg_w, max(1, tg_h), 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(grid, tg)
    encoder.endEncoding()
    cmdbuf.commit()
    cmdbuf.waitUntilCompleted()

    out_np = _readback(buf_out, (H, W, 3), np.float32)
    image = torch.from_numpy(out_np).to(orig_device, dtype).permute(2, 0, 1)   # [3, H, W]

    radii_out = torch.zeros(N, device=orig_device, dtype=dtype)
    radii_out[order] = radii[order]
    return image, radii_out


class GaussianRasterizer(torch.nn.Module):
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
