"""
Hand-written Metal compute kernel replacement for `sw_rasterizer.rasterize`, for when raw
throughput of the pure-PyTorch chunked rasterizer (trellis/renderers/sw_rasterizer.py) actually
matters. Same algorithm, same verified math (edge-function coverage test, nearest-wins z-buffer,
winding-agnostic inside test) -- just fused into a single Metal kernel dispatch with one GPU
thread per output pixel, looping over all triangles in registers, instead of ~num_triangles /
chunk_size sequential PyTorch op dispatches each materializing a [chunk, H, W] intermediate
tensor. No atomics are needed anywhere: each thread owns exactly one pixel, so there's no
cross-thread write contention.

Compiled at *runtime* via MTLDevice.newLibraryWithSource_options_error_ (Metal.framework, present
on every Mac) rather than the offline `xcrun metal` compiler, which requires a full Xcode.app
install that isn't necessarily present (Command Line Tools alone don't include it).

`interpolate` and `antialias` are unchanged from `sw_rasterizer` -- they're cheap, already-fast
gather/elementwise ops, not the bottleneck; only `rasterize` is reimplemented here.

Falls back to `sw_rasterizer.rasterize` transparently (see `rasterize()` below) if PyObjC / a
Metal device / runtime shader compilation aren't available for any reason -- this module is an
optional accelerator, not a hard dependency.
"""
from typing import Optional
import numpy as np
import torch

from . import sw_rasterizer
from .sw_rasterizer import RasterizeContext, RasterizeCudaContext, RasterizeGLContext, interpolate, antialias, texture  # noqa: F401

__all__ = ['RasterizeContext', 'RasterizeCudaContext', 'RasterizeGLContext', 'rasterize', 'interpolate', 'antialias', 'texture', 'available']

_KERNEL_SRC = """
#include <metal_stdlib>
using namespace metal;

kernel void rasterize_kernel(
    device const float* px      [[buffer(0)]],
    device const float* py      [[buffer(1)]],
    device const float* pz      [[buffer(2)]],
    device const int*   tri     [[buffer(3)]],
    constant uint&       F      [[buffer(4)]],
    constant uint&       W      [[buffer(5)]],
    constant uint&       H      [[buffer(6)]],
    device float* out_bary1     [[buffer(7)]],
    device float* out_bary2     [[buffer(8)]],
    device float* out_depth     [[buffer(9)]],
    device int*   out_triid     [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= W || gid.y >= H) return;

    float px_ = float(gid.x) + 0.5f;
    float py_ = float(gid.y) + 0.5f;

    float best_depth = INFINITY;
    int best_tri = 0;
    float best_b1 = 0.0f;
    float best_b2 = 0.0f;

    for (uint f = 0; f < F; f++) {
        int i0 = tri[f * 3 + 0];
        int i1 = tri[f * 3 + 1];
        int i2 = tri[f * 3 + 2];
        float x0 = px[i0], y0 = py[i0], z0 = pz[i0];
        float x1 = px[i1], y1 = py[i1], z1 = pz[i1];
        float x2 = px[i2], y2 = py[i2], z2 = pz[i2];

        float w0 = (x2 - x1) * (py_ - y1) - (y2 - y1) * (px_ - x1);
        float w1 = (x0 - x2) * (py_ - y2) - (y0 - y2) * (px_ - x2);
        float w2 = (x1 - x0) * (py_ - y0) - (y1 - y0) * (px_ - x0);
        float area = w0 + w1 + w2;

        bool degenerate = fabs(area) < 1e-9f;
        bool inside = (!degenerate) &&
            (((w0 >= 0.0f) && (w1 >= 0.0f) && (w2 >= 0.0f)) ||
             ((w0 <= 0.0f) && (w1 <= 0.0f) && (w2 <= 0.0f)));
        if (!inside) continue;

        float safe_area = degenerate ? 1e-9f : area;
        float b1 = w1 / safe_area;
        float b2 = w2 / safe_area;
        float b0 = w0 / safe_area;
        float depth = b0 * z0 + b1 * z1 + b2 * z2;

        if (depth < best_depth) {
            best_depth = depth;
            best_tri = int(f) + 1;
            best_b1 = b1;
            best_b2 = b2;
        }
    }

    uint idx = gid.y * W + gid.x;
    out_bary1[idx] = best_b1;
    out_bary2[idx] = best_b2;
    out_depth[idx] = best_depth;
    out_triid[idx] = best_tri;
}
"""

_state = {'checked': False, 'ok': False, 'device': None, 'queue': None, 'pipeline': None}


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
        fn = library.newFunctionWithName_("rasterize_kernel")
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
        warnings.warn(f"metal_rasterizer: falling back to sw_rasterizer, Metal init failed: {e}")
        _state['ok'] = False
    return _state['ok']


def available() -> bool:
    return _init()


def _to_metal_buffer(device, Metal, arr: np.ndarray):
    arr = np.ascontiguousarray(arr)
    buf = device.newBufferWithBytes_length_options_(arr.tobytes(), arr.nbytes, Metal.MTLResourceStorageModeShared)
    return buf


def _readback(buf, shape, dtype):
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    raw = buf.contents().as_buffer(nbytes)
    return np.frombuffer(raw, dtype=dtype, count=int(np.prod(shape))).reshape(shape).copy()


def rasterize(glctx, pos: torch.Tensor, tri: torch.Tensor, resolution, chunk_size: Optional[int] = None, **kwargs):
    """Drop-in replacement for sw_rasterizer.rasterize -- same signature, same rast layout.
    **kwargs accepted and ignored, same reasoning as sw_rasterizer.rasterize (e.g. nvdiffrast's
    `grad_db`, passed by utils3d's rasterize_triangle_faces)."""
    if not _init():
        return sw_rasterizer.rasterize(glctx, pos, tri, resolution, chunk_size=chunk_size)

    assert pos.dim() == 3 and pos.shape[0] == 1, "rasterizer only supports batch size 1"
    H, W = resolution
    orig_device = pos.device
    dtype = pos.dtype

    pos0 = pos[0]
    w = pos0[:, 3].clamp(min=1e-6)
    ndc = pos0[:, :3] / w[:, None]
    px = (ndc[:, 0] * 0.5 + 0.5) * W
    py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * H
    pz = ndc[:, 2]

    # NOTE: split device transfer and dtype cast into two separate .to() calls, each on its own
    # line. A combined `.to('cpu', dtype)` call on an MPS int64 tensor was found (empirically,
    # while debugging this module) to corrupt the *source* MPS tensor in place -- a real bug in
    # this PyTorch build's MPS backend, not something specific to this rasterizer. Two-step
    # conversion (device first, then dtype, or vice versa) avoids it entirely.
    px_np = px.detach().to('cpu').to(torch.float32).numpy()
    py_np = py.detach().to('cpu').to(torch.float32).numpy()
    pz_np = pz.detach().to('cpu').to(torch.float32).numpy()
    tri_np = tri.detach().to('cpu').to(torch.int32).numpy()
    F_ = tri_np.shape[0]

    device = _state['device']
    Metal = _state['Metal']
    queue = _state['queue']
    pipeline = _state['pipeline']

    if F_ == 0:
        rast = torch.zeros((1, H, W, 4), device=orig_device, dtype=dtype)
        return rast, None

    buf_px = _to_metal_buffer(device, Metal, px_np)
    buf_py = _to_metal_buffer(device, Metal, py_np)
    buf_pz = _to_metal_buffer(device, Metal, pz_np)
    buf_tri = _to_metal_buffer(device, Metal, tri_np.reshape(-1))

    n_pix = H * W
    buf_b1 = device.newBufferWithLength_options_(n_pix * 4, Metal.MTLResourceStorageModeShared)
    buf_b2 = device.newBufferWithLength_options_(n_pix * 4, Metal.MTLResourceStorageModeShared)
    buf_d = device.newBufferWithLength_options_(n_pix * 4, Metal.MTLResourceStorageModeShared)
    buf_id = device.newBufferWithLength_options_(n_pix * 4, Metal.MTLResourceStorageModeShared)

    F_arr = np.array([F_], dtype=np.uint32)
    W_arr = np.array([W], dtype=np.uint32)
    H_arr = np.array([H], dtype=np.uint32)
    buf_F = _to_metal_buffer(device, Metal, F_arr)
    buf_W = _to_metal_buffer(device, Metal, W_arr)
    buf_H = _to_metal_buffer(device, Metal, H_arr)

    cmdbuf = queue.commandBuffer()
    encoder = cmdbuf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    for i, b in enumerate([buf_px, buf_py, buf_pz, buf_tri, buf_F, buf_W, buf_H, buf_b1, buf_b2, buf_d, buf_id]):
        encoder.setBuffer_offset_atIndex_(b, 0, i)
    tg_w = 16
    tg_h = pipeline.maxTotalThreadsPerThreadgroup() // tg_w
    grid = Metal.MTLSizeMake(W, H, 1)
    tg = Metal.MTLSizeMake(tg_w, max(1, tg_h), 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(grid, tg)
    encoder.endEncoding()
    cmdbuf.commit()
    cmdbuf.waitUntilCompleted()

    b1 = _readback(buf_b1, (H, W), np.float32)
    b2 = _readback(buf_b2, (H, W), np.float32)
    d = _readback(buf_d, (H, W), np.float32)
    tid = _readback(buf_id, (H, W), np.int32)

    rast_np = np.stack([b1, b2, d, tid.astype(np.float32)], axis=-1)[None]  # [1,H,W,4]
    rast = torch.from_numpy(rast_np).to(orig_device, dtype)
    return rast, None
