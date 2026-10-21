"""
Tile-binned variant of metal_rasterizer.py's brute-force per-pixel Metal rasterizer.

Why: metal_rasterizer.rasterize()'s kernel is O(H*W*F) -- every pixel-thread loops over every
triangle. That's fine at small scale, but at postprocessing_utils._fill_holes's real workload
(1024x1024, ~35K faces, 1000 views) it dominates cost -- confirmed by benchmarking (see
APPLE_SILICON.md and scripts/bench_fill_holes_resolution.py): a sync/upload-count fix to that same
kernel produced no measurable speedup there, because the kernel's own compute, not the MPS->CPU
transfer, is the bottleneck.

This module bins triangles into screen-space tiles on the CPU (vectorized numpy, using the
already-CPU-resident px/py projected coordinates -- no extra sync beyond what the brute-force path
already does) and uploads a CSR-style (tile_offsets, bin_tri_ids) structure alongside the usual
per-vertex/per-triangle buffers. Each pixel-thread then only loops over the triangles actually
binned to its tile instead of the full triangle list, so cost scales with *local* triangle density
instead of global triangle count -- the real fix for the O(H*W*F) bottleneck (as opposed to the
sync-batching attempt in metal_rasterizer.py, which didn't touch this term at all and was
reverted).

Correctness: every triangle whose screen-space AABB overlaps a tile is included in that tile's
candidate list (not an approximation/culling heuristic) -- so this produces bit-identical output to
metal_rasterizer.rasterize() for the same inputs, just faster when triangles are small relative to
the screen (the common case for a decimated mesh at high resolution). A "big triangle" fallback
(AABB spanning more tiles than MAX_TILE_SPAN in either axis) adds that triangle to every tile
explicitly rather than approximating, so pathologically large triangles cost more but are never
silently dropped. See scripts/verify_metal_rasterizer_binned.py for the correctness suite this
claim is checked against (small scenes, random meshes, oversized/degenerate triangles, tile-boundary
straddling triangles).

Kept as a separate module from metal_rasterizer.py rather than replacing it in place: this is a
real algorithmic change (not a drop-in equivalent-output optimization like the reverted sync fix),
so it's opt-in until proven both correct and actually faster on the real _fill_holes workload.
"""
from typing import Optional
import numpy as np
import torch

from . import sw_rasterizer
from . import metal_rasterizer as _brute
from .sw_rasterizer import RasterizeContext, RasterizeCudaContext, RasterizeGLContext, interpolate, antialias, texture  # noqa: F401

__all__ = ['RasterizeContext', 'RasterizeCudaContext', 'RasterizeGLContext', 'rasterize', 'interpolate', 'antialias', 'texture', 'available']

_KERNEL_SRC = """
#include <metal_stdlib>
using namespace metal;

kernel void rasterize_kernel_binned(
    device const float* px           [[buffer(0)]],
    device const float* py           [[buffer(1)]],
    device const float* pz           [[buffer(2)]],
    device const int*   tri          [[buffer(3)]],
    constant uint&       W           [[buffer(4)]],
    constant uint&       H           [[buffer(5)]],
    constant uint&       TILE        [[buffer(6)]],
    constant uint&       NTX         [[buffer(7)]],
    device const uint*  tile_offsets [[buffer(8)]],
    device const int*   bin_tri_ids  [[buffer(9)]],
    device float* out_bary1          [[buffer(10)]],
    device float* out_bary2          [[buffer(11)]],
    device float* out_depth          [[buffer(12)]],
    device int*   out_triid          [[buffer(13)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= W || gid.y >= H) return;

    uint tile_x = gid.x / TILE;
    uint tile_y = gid.y / TILE;
    uint tile_id = tile_y * NTX + tile_x;
    uint start = tile_offsets[tile_id];
    uint end = tile_offsets[tile_id + 1];

    float px_ = float(gid.x) + 0.5f;
    float py_ = float(gid.y) + 0.5f;

    float best_depth = INFINITY;
    int best_tri = 0;
    float best_b1 = 0.0f;
    float best_b2 = 0.0f;

    for (uint k = start; k < end; k++) {
        int f = bin_tri_ids[k];
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
            best_tri = f + 1;
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

TILE = 32
# Triangles whose screen-space AABB spans more tiles than this in either axis are added to every
# tile explicitly instead of via the dense per-triangle candidate grid -- keeps the vectorized
# binning step's memory bounded (O(F * MAX_TILE_SPAN^2) worst case otherwise) without ever
# dropping a triangle from a tile it actually overlaps.
MAX_TILE_SPAN = 64


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
        fn = library.newFunctionWithName_("rasterize_kernel_binned")
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
        warnings.warn(f"metal_rasterizer_binned: falling back to metal_rasterizer, Metal init failed: {e}")
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


def _build_bins(px_np: np.ndarray, py_np: np.ndarray, tri_np: np.ndarray, W: int, H: int):
    """CSR-style (tile_offsets[ntiles+1], bin_tri_ids[nnz]) binning of triangles into TILE x TILE
    screen tiles by AABB overlap. Every triangle whose AABB overlaps a tile appears in that tile's
    list -- see module docstring for why this is exact, not an approximation."""
    ntx = (W + TILE - 1) // TILE
    nty = (H + TILE - 1) // TILE
    n_tiles = ntx * nty
    F_ = tri_np.shape[0]

    vx = px_np[tri_np]  # [F,3]
    vy = py_np[tri_np]  # [F,3]
    bbox_xmin = vx.min(axis=1)
    bbox_xmax = vx.max(axis=1)
    bbox_ymin = vy.min(axis=1)
    bbox_ymax = vy.max(axis=1)

    tx0 = np.clip(np.floor(bbox_xmin / TILE).astype(np.int64), 0, ntx - 1)
    tx1 = np.clip(np.floor(bbox_xmax / TILE).astype(np.int64), 0, ntx - 1)
    ty0 = np.clip(np.floor(bbox_ymin / TILE).astype(np.int64), 0, nty - 1)
    ty1 = np.clip(np.floor(bbox_ymax / TILE).astype(np.int64), 0, nty - 1)

    span_x = tx1 - tx0 + 1
    span_y = ty1 - ty0 + 1
    big = (span_x > MAX_TILE_SPAN) | (span_y > MAX_TILE_SPAN)
    small_idx = np.nonzero(~big)[0]
    big_idx = np.nonzero(big)[0]

    flat_tile_parts = []
    flat_tri_parts = []

    if len(small_idx) > 0:
        sx = span_x[small_idx]
        sy = span_y[small_idx]
        max_dx = int(sx.max())
        max_dy = int(sy.max())

        dxr = np.arange(max_dx)
        dyr = np.arange(max_dy)
        cand_tx = tx0[small_idx, None] + dxr[None, :]          # [n_small, max_dx]
        valid_x = dxr[None, :] < sx[:, None]                    # [n_small, max_dx]
        cand_ty = ty0[small_idx, None] + dyr[None, :]           # [n_small, max_dy]
        valid_y = dyr[None, :] < sy[:, None]                    # [n_small, max_dy]

        tile_id_full = cand_ty[:, None, :] * ntx + cand_tx[:, :, None]      # [n_small, max_dx, max_dy]
        valid_full = valid_x[:, :, None] & valid_y[:, None, :]              # [n_small, max_dx, max_dy]
        tri_id_full = np.broadcast_to(small_idx[:, None, None], tile_id_full.shape)

        flat_valid = valid_full.reshape(-1)
        flat_tile_parts.append(tile_id_full.reshape(-1)[flat_valid])
        flat_tri_parts.append(tri_id_full.reshape(-1)[flat_valid])

    if len(big_idx) > 0:
        all_tiles = np.arange(n_tiles)
        flat_tile_parts.append(np.tile(all_tiles, len(big_idx)))
        flat_tri_parts.append(np.repeat(big_idx, n_tiles))

    if flat_tile_parts:
        flat_tile = np.concatenate(flat_tile_parts)
        flat_tri = np.concatenate(flat_tri_parts).astype(np.int32)
    else:
        flat_tile = np.zeros((0,), dtype=np.int64)
        flat_tri = np.zeros((0,), dtype=np.int32)

    order = np.argsort(flat_tile, kind='stable')
    sorted_tile = flat_tile[order]
    bin_tri_ids = flat_tri[order]

    tile_counts = np.bincount(sorted_tile, minlength=n_tiles) if len(sorted_tile) else np.zeros(n_tiles, dtype=np.int64)
    tile_offsets = np.zeros(n_tiles + 1, dtype=np.uint32)
    np.cumsum(tile_counts, out=tile_offsets[1:])

    return tile_offsets, bin_tri_ids, ntx, nty


def rasterize(glctx, pos: torch.Tensor, tri: torch.Tensor, resolution, chunk_size: Optional[int] = None, **kwargs):
    """Drop-in replacement for metal_rasterizer.rasterize -- same signature, same rast layout,
    bit-identical output. Falls back to metal_rasterizer.rasterize (which itself falls back to
    sw_rasterizer.rasterize) if Metal init fails."""
    if not _init():
        return _brute.rasterize(glctx, pos, tri, resolution, chunk_size=chunk_size)

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

    # see metal_rasterizer.py's matching NOTE: two separate .to() calls avoid an MPS int64-cast
    # corruption bug in this PyTorch build.
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

    tile_offsets, bin_tri_ids, ntx, nty = _build_bins(px_np, py_np, tri_np, W, H)

    buf_px = _to_metal_buffer(device, Metal, px_np)
    buf_py = _to_metal_buffer(device, Metal, py_np)
    buf_pz = _to_metal_buffer(device, Metal, pz_np)
    buf_tri = _to_metal_buffer(device, Metal, tri_np.reshape(-1))
    buf_tile_offsets = _to_metal_buffer(device, Metal, tile_offsets)
    buf_bin_tri_ids = _to_metal_buffer(device, Metal, bin_tri_ids) if bin_tri_ids.size else _to_metal_buffer(device, Metal, np.zeros(1, dtype=np.int32))

    n_pix = H * W
    buf_b1 = device.newBufferWithLength_options_(n_pix * 4, Metal.MTLResourceStorageModeShared)
    buf_b2 = device.newBufferWithLength_options_(n_pix * 4, Metal.MTLResourceStorageModeShared)
    buf_d = device.newBufferWithLength_options_(n_pix * 4, Metal.MTLResourceStorageModeShared)
    buf_id = device.newBufferWithLength_options_(n_pix * 4, Metal.MTLResourceStorageModeShared)

    buf_W = _to_metal_buffer(device, Metal, np.array([W], dtype=np.uint32))
    buf_H = _to_metal_buffer(device, Metal, np.array([H], dtype=np.uint32))
    buf_TILE = _to_metal_buffer(device, Metal, np.array([TILE], dtype=np.uint32))
    buf_NTX = _to_metal_buffer(device, Metal, np.array([ntx], dtype=np.uint32))

    cmdbuf = queue.commandBuffer()
    encoder = cmdbuf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    buffers = [buf_px, buf_py, buf_pz, buf_tri, buf_W, buf_H, buf_TILE, buf_NTX,
               buf_tile_offsets, buf_bin_tri_ids, buf_b1, buf_b2, buf_d, buf_id]
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

    b1 = _readback(buf_b1, (H, W), np.float32)
    b2 = _readback(buf_b2, (H, W), np.float32)
    d = _readback(buf_d, (H, W), np.float32)
    tid = _readback(buf_id, (H, W), np.int32)

    rast_np = np.stack([b1, b2, d, tid.astype(np.float32)], axis=-1)[None]  # [1,H,W,4]
    rast = torch.from_numpy(rast_np).to(orig_device, dtype)
    return rast, None
