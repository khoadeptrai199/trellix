"""
Weight-only int8 quantized nn.Linear replacement with a hand-written Metal compute kernel, for
running TRELLIS's diffusion denoisers (sparse_structure_flow_model, slat_flow_model -- 555M and
600M params respectively) with a ~4x smaller weight footprint on Apple Silicon.

Scope, set deliberately: this is *weight-only* quantization (activations stay float32) -- the
standard, simpler, lower-accuracy-risk form (same family as bitsandbytes' LLM.int8() weight path,
without its activation-outlier handling, which this port doesn't need since it's not chasing
LLM-scale precision-sensitivity). Per-output-channel (row-wise) symmetric quantization: each output
neuron's weight row gets its own scale, computed once from the pretrained fp32 weights at
quantization time, not learned/calibrated -- a static post-training quantization, not
quantization-aware training or GPTQ-style calibration.

Why a hand kernel at all instead of dequantizing once and calling a normal matmul: the entire
point is to avoid ever materializing a full fp32 copy of a 600M-parameter model's weights, so
`torch.matmul` on a pre-dequantized weight would defeat the purpose. Standard int8 tooling doesn't
reach this platform: bitsandbytes doesn't support Apple Silicon at all, and PyTorch's `torch.ao`
quantization targets CPU (fbgemm/qnnpack) and CUDA, with no real MPS int8 op support.

Honest expectation-setting (see APPLE_SILICON.md for the profiling this is based on): an isolated
forward pass of these models already achieves a reasonable fraction of this hardware's peak
compute throughput (~14.8 TFLOPS measured), and fp16 (tested directly) gave no measurable speedup
over fp32 -- both signals that this workload isn't purely memory-bandwidth-starved the way weight
quantization most directly helps. This is being built for real, verified correctness and real
measured numbers either way, not a promised speedup -- see scripts/benchmark_quantized_linear.py
and scripts/verify_quantized_linear.py for what was actually measured.
"""
import numpy as np
import torch
import torch.nn as nn

__all__ = ['QuantizedLinearMetal', 'quantize_model_linears', 'available']

_KERNEL_SRC = """
#include <metal_stdlib>
using namespace metal;

// Multi-simdgroup simdgroup_matrix GEMM. History: v1 (one thread/output, scalar loop) was ~17x
// slower than PyTorch's tuned MPS matmul; v2 (16x16 shared-memory tiling) ~8.0x; v3 (4x4 register
// blocking on top of v2) ~6.4x -- the best pure-scalar result. v4 (single-simdgroup
// simdgroup_matrix, one 32x32 tile/threadgroup, 32 threads/threadgroup) landed at ~6.5x --
// essentially a wash with v3, diagnosed as too little parallelism per threadgroup dispatch (only
// one SIMD-group active) to benefit from the matrix-unit path. This version (v5) fixes that
// specific weakness: 4 simdgroups cooperate per threadgroup (128 threads total, arranged 2x2),
// each computing its own 32x32 sub-tile of a combined 64x64 threadgroup tile (matching v3's tile
// size, for a fair comparison) -- same hardware-matrix-multiply approach as v4, but with 4x the
// occupancy per dispatch (see quantized_linear_metal.py's module docstring for the measured
// result -- this is the "extreme new way" attempt, reported honestly whether or not it wins).
#define BM 64   // output rows per threadgroup
#define BN 64   // output cols per threadgroup
#define BK 8    // reduction chunk size -- simdgroup_float8x8 is fixed at 8x8
#define SUB 32  // each simdgroup's own sub-tile size (BM/2 x BN/2, 2x2 arrangement of simdgroups)
// threadgroup is 128 threads = 4 simdgroups (2x2 grid), each handling one SUBxSUB sub-tile

kernel void quantized_linear_kernel(
    device const float* x         [[buffer(0)]],  // [num_rows, in_features]
    device const char*  w_int8    [[buffer(1)]],  // [out_features, in_features]
    device const float* scale     [[buffer(2)]],  // [out_features]
    device const float* bias      [[buffer(3)]],  // [out_features], only read if has_bias != 0
    constant uint&       in_features  [[buffer(4)]],
    constant uint&       out_features [[buffer(5)]],
    constant uint&       num_rows     [[buffer(6)]],
    constant uint&       has_bias     [[buffer(7)]],
    device float*        y            [[buffer(8)]],
    uint  thread_idx     [[thread_index_in_threadgroup]],
    uint  simd_group_id  [[simdgroup_index_in_threadgroup]],
    uint2 group_id       [[threadgroup_position_in_grid]])
{
    threadgroup float x_tile[BM][BK];
    threadgroup float w_tile[BK][BN];

    uint row_base = group_id.y * BM;
    uint col_base = group_id.x * BN;

    uint sub_row_base = (simd_group_id / 2) * SUB;  // this simdgroup's offset within the 64x64 tile
    uint sub_col_base = (simd_group_id % 2) * SUB;

    simdgroup_float8x8 acc[4][4];  // SUB/8 x SUB/8 = 4x4 grid of 8x8 tiles per simdgroup's sub-tile
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            acc[i][j] = simdgroup_float8x8(0.0f);

    uint num_k_tiles = (in_features + BK - 1) / BK;
    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint k0 = kt * BK;

        // All 128 threads (all 4 simdgroups) cooperate on ONE shared load of the full 64x8 /
        // 8x64 tiles -- every simdgroup reuses the same load, not 4 redundant copies.
        for (uint e = thread_idx; e < BM * BK; e += 128) {
            uint r = e / BK, c = e % BK;
            uint gr = row_base + r, gk = k0 + c;
            x_tile[r][c] = (gr < num_rows && gk < in_features) ? x[gr * in_features + gk] : 0.0f;
        }
        for (uint e = thread_idx; e < BK * BN; e += 128) {
            uint r = e / BN, c = e % BN;
            uint gr = col_base + c, gk = k0 + r;
            w_tile[r][c] = (gr < out_features && gk < in_features) ? float(w_int8[gr * in_features + gk]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < 4; i++) {
            simdgroup_float8x8 xs;
            simdgroup_load(xs, &x_tile[sub_row_base + i * 8][0], BK);
            for (uint j = 0; j < 4; j++) {
                simdgroup_float8x8 ws;
                simdgroup_load(ws, &w_tile[0][sub_col_base + j * 8], BN);
                simdgroup_multiply_accumulate(acc[i][j], xs, ws, acc[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float out_tile[BM][BN];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            simdgroup_store(acc[i][j], &out_tile[sub_row_base + i * 8][sub_col_base + j * 8], BN);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = thread_idx; e < BM * BN; e += 128) {
        uint r = e / BN, c = e % BN;
        uint row = row_base + r, col = col_base + c;
        if (row < num_rows && col < out_features) {
            float v = out_tile[r][c] * scale[col];
            if (has_bias != 0) v += bias[col];
            y[row * out_features + col] = v;
        }
    }
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
        fn = library.newFunctionWithName_("quantized_linear_kernel")
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
        warnings.warn(f"quantized_linear_metal: falling back to PyTorch dequant matmul, Metal init failed: {e}")
        _state['ok'] = False
    return _state['ok']


def available() -> bool:
    return _init()


def _to_buffer(device, Metal, arr: np.ndarray):
    arr = np.ascontiguousarray(arr)
    return device.newBufferWithBytes_length_options_(arr.tobytes(), arr.nbytes, Metal.MTLResourceStorageModeShared)


def _write_into_buffer(buf, arr: np.ndarray):
    """Overwrites an existing MTLBuffer's contents in place via its CPU-visible pointer, instead
    of allocating a new buffer. This is the actual fix for the leak documented in
    _make_static_buffers' docstring: newBufferWithBytes_length_options_/newBufferWithLength_options_
    calls, in a tight loop, leak deterministically regardless of `del`/gc.collect()/autorelease
    pools/torch.mps.empty_cache() -- confirmed via an isolated repro (no PyTorch, no model, just
    Metal buffer alloc+del in a loop: ~linear, unbounded growth, exactly matching buffer size per
    iteration). Reusing one buffer and overwriting its contents (this function) showed zero growth
    over the same iteration count in that same repro."""
    arr = np.ascontiguousarray(arr)
    dst = buf.contents().as_buffer(arr.nbytes)
    np.frombuffer(dst, dtype=arr.dtype).reshape(arr.shape)[:] = arr


def _readback(buf, shape, dtype):
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    raw = buf.contents().as_buffer(nbytes)
    return np.frombuffer(raw, dtype=dtype, count=int(np.prod(shape))).reshape(shape).copy()


def _make_static_buffers(device, Metal, w_int8_np, scale_np, bias_np, out_features, in_features):
    """The weight/scale/bias/shape buffers for a given QuantizedLinearMetal never change across
    forward calls -- only the activations do. Building fresh copies of these on *every* forward
    call (the first version of this module did) both re-uploads immutable weight data pointlessly
    and, empirically, leaked: a 10-iteration benchmark loop over the real 555M-param model OOM'd
    ("other allocations" climbing to ~59GB for a model whose weights total 2.2GB) regardless of an
    explicit autorelease_pool(), explicit `del` + gc.collect() every call, or periodic
    torch.mps.empty_cache() -- none of which are supposed to be necessary if buffers are created
    once and reused, which is the actual fix applied here."""
    has_bias = 1 if bias_np is not None else 0
    bias_buf_np = bias_np if bias_np is not None else np.zeros(1, dtype=np.float32)
    return {
        'w': _to_buffer(device, Metal, w_int8_np),
        'scale': _to_buffer(device, Metal, scale_np),
        'bias': _to_buffer(device, Metal, bias_buf_np),
        'in_f': _to_buffer(device, Metal, np.array([in_features], dtype=np.uint32)),
        'out_f': _to_buffer(device, Metal, np.array([out_features], dtype=np.uint32)),
        'has_bias': _to_buffer(device, Metal, np.array([has_bias], dtype=np.uint32)),
    }


def _ensure_capacity(pool, key, device, Metal, nbytes):
    """Returns a buffer of at least `nbytes` from `pool[key]`, allocating (or growing) only when
    the cached one is too small -- the steady-state case (same activation shape call after call,
    typical across one sampling run) never allocates at all after the first call."""
    entry = pool.get(key)
    if entry is None or entry[1] < nbytes:
        buf = device.newBufferWithLength_options_(nbytes, Metal.MTLResourceStorageModeShared)
        pool[key] = (buf, nbytes)
        return buf
    return entry[0]


def _quantized_matmul_metal(static_bufs, dyn_pool, x_np, out_features, in_features):
    num_rows = x_np.shape[0]
    device = _state['device']
    Metal = _state['Metal']
    queue = _state['queue']
    pipeline = _state['pipeline']

    buf_x = _ensure_capacity(dyn_pool, 'x', device, Metal, x_np.nbytes)
    _write_into_buffer(buf_x, x_np)
    buf_rows = _ensure_capacity(dyn_pool, 'rows', device, Metal, 4)
    _write_into_buffer(buf_rows, np.array([num_rows], dtype=np.uint32))
    buf_y = _ensure_capacity(dyn_pool, 'y', device, Metal, num_rows * out_features * 4)

    cmdbuf = queue.commandBuffer()
    encoder = cmdbuf.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    buffers = [buf_x, static_bufs['w'], static_bufs['scale'], static_bufs['bias'],
               static_bufs['in_f'], static_bufs['out_f'], buf_rows, static_bufs['has_bias'], buf_y]
    for i, b in enumerate(buffers):
        encoder.setBuffer_offset_atIndex_(b, 0, i)

    # dispatchThreadgroups (not dispatchThreads): the kernel needs every threadgroup to have its
    # FULL complement of threads so cooperative tile-loading fills the whole shared tile --
    # dispatchThreads creates *non-uniform* (partial) threadgroups at grid edges when a dimension
    # isn't a multiple of the tile size, which silently leaves parts of the shared tile
    # uninitialized (confirmed: this broke exactly the test cases where num_rows < 16).
    # dispatchThreadgroups always launches complete threadgroups; the extra out-of-bounds threads
    # at the edges are handled by this kernel's existing (row<num_rows / col<out_features) guards.
    BM, BN = 64, 64  # must match the kernel's #define BM / BN
    num_threadgroups = Metal.MTLSizeMake((out_features + BN - 1) // BN, (num_rows + BM - 1) // BM, 1)
    tg = Metal.MTLSizeMake(128, 1, 1)  # 4 simdgroups (2x2) x 32 lanes, must match the kernel
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_threadgroups, tg)
    encoder.endEncoding()
    cmdbuf.commit()
    cmdbuf.waitUntilCompleted()
    if cmdbuf.status() == 5:
        raise RuntimeError(f"quantized_linear_metal: Metal command buffer failed: {cmdbuf.error()}")

    return _readback(buf_y, (num_rows, out_features), np.float32)


class QuantizedLinearMetal(nn.Module):
    """Drop-in replacement for a pretrained nn.Linear: int8 weights (per-output-channel scale),
    float32 bias and activations. Inference-only (no backward -- this port only ever runs these
    models for sampling, never trains them)."""

    def __init__(self, weight_int8: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor = None):
        super().__init__()
        self.register_buffer('weight_int8', weight_int8)  # [out, in], int8
        self.register_buffer('scale', scale)  # [out], float32
        self.register_buffer('bias', bias)  # [out], float32, or None
        self.out_features = weight_int8.shape[0]
        self.in_features = weight_int8.shape[1]
        self._metal_static_bufs = None  # built lazily on first forward(), see _static_buffers()
        self._metal_dyn_pool = {}  # activation/output buffers, grown (not recreated) as needed

    def _static_buffers(self):
        if self._metal_static_bufs is None:
            self._metal_static_bufs = _make_static_buffers(
                _state['device'], _state['Metal'],
                self.weight_int8.numpy(), self.scale.numpy(),
                self.bias.numpy() if self.bias is not None else None,
                self.out_features, self.in_features,
            )
        return self._metal_static_bufs

    @classmethod
    def from_float(cls, linear: nn.Linear) -> "QuantizedLinearMetal":
        w = linear.weight.detach().to('cpu', torch.float32)  # [out, in]
        row_absmax = w.abs().amax(dim=1).clamp(min=1e-8)
        scale = row_absmax / 127.0
        w_int8 = (w / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        bias = linear.bias.detach().to('cpu', torch.float32).clone() if linear.bias is not None else None
        return cls(w_int8, scale, bias)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        orig_device = x.device
        orig_dtype = x.dtype
        x2d = x.reshape(-1, self.in_features)

        if _init():
            x_np = x2d.detach().to('cpu', torch.float32).contiguous().numpy()
            y_np = _quantized_matmul_metal(self._static_buffers(), self._metal_dyn_pool, x_np, self.out_features, self.in_features)
            y = torch.from_numpy(y_np).to(orig_device, orig_dtype)
        else:
            w_dequant = (self.weight_int8.to(torch.float32) * self.scale[:, None]).to(orig_device, orig_dtype)
            y = x2d.to(orig_dtype) @ w_dequant.T
            if self.bias is not None:
                y = y + self.bias.to(orig_device, orig_dtype)

        return y.reshape(*orig_shape[:-1], self.out_features)


def quantize_model_linears(model: nn.Module, exclude_substrings=()) -> int:
    """Replaces every nn.Linear submodule in `model` with a QuantizedLinearMetal in place. Returns
    the count of layers quantized. `exclude_substrings`: skip any module whose dotted name contains
    one of these (e.g. a final output projection you want to keep at full precision)."""
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                full_name = f"{name}.{child_name}" if name else child_name
                if any(s in full_name for s in exclude_substrings):
                    continue
                setattr(module, child_name, QuantizedLinearMetal.from_float(child))
                count += 1
    return count
