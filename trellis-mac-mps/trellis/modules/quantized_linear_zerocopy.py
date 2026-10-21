"""
Same int8 weight-only quantized nn.Linear as quantized_linear_metal.py (identical v5
multi-simdgroup simdgroup_matrix kernel, byte-for-byte copied), but dispatched through
`torch.mps.compile_shader` instead of hand-written PyObjC/Metal buffer plumbing. This is an
ADDITIONAL, separate module -- quantized_linear_metal.py is untouched, still the file behind the
5.94x-slower result documented in APPLE_SILICON.md as of commit 40cce68.

Why this exists: quantized_linear_metal.py's forward() does `x2d.to('cpu').numpy()` into the
kernel and `torch.from_numpy(...).to(orig_device)` back out, on *every* call -- a real device
synchronization + copy each way, even though both PyTorch's MPS tensors and our Metal kernel's
buffers already live in the same unified memory. `torch.mps.compile_shader` (stable in this
PyTorch build, see `torch.mps.compile_shader.__doc__`) JIT-compiles a Metal source string and
returns kernel objects that are called *directly on torch MPS tensors* -- no CPU round trip, no
manual MTLBuffer allocation/readback, and no leak-prone buffer lifecycle (that whole class of bug
in quantized_linear_metal.py's docstring doesn't exist here: tensors are managed by PyTorch's own
MPS caching allocator, not manually alloc'd MTLBuffers).

One dispatch subtlety, confirmed empirically (not assumed) before writing this: `lib.kernel(...,
threads=X, group_size=G)` dispatches as *dispatchThreads*, which creates non-uniform (partial)
threadgroups at grid edges when X isn't an exact multiple of G -- e.g. threads=70, group_size=16
gives a last threadgroup where only thread_index_in_threadgroup 0..5 ever run, not 0..15. This
kernel's cooperative shared-memory tile loading (`for e = thread_idx; e < BM*BK; e += 128`) needs
every one of the 128 threads in every threadgroup to run, exactly the same requirement
quantized_linear_metal.py's own docstring already documents for `dispatchThreadgroups` vs
`dispatchThreads`. Fix here is the same in spirit: round `threads` up to an exact multiple of
`group_size` ourselves (see `_dispatch_dims`), so every threadgroup is always full; the kernel's
existing `row < num_rows` / `col < out_features` guards handle the resulting out-of-bounds threads
safely, same as they already do for the grid-edge case.
"""
import warnings

import torch
import torch.nn as nn

__all__ = ['QuantizedLinearZeroCopy', 'quantize_model_linears_zerocopy', 'available']

# Identical kernel source to quantized_linear_metal.py's v5 -- copied, not imported, so this
# module has no dependency on that one and neither can break the other.
_KERNEL_SRC = """
#include <metal_stdlib>
using namespace metal;

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

_GROUP_SIZE = (128, 1)  # must match the kernel: 128 threads/threadgroup, 4 simdgroups

_state = {'checked': False, 'ok': False}


def _init():
    if _state['checked']:
        return _state['ok']
    _state['checked'] = True
    try:
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS not available")
        _state['lib'] = torch.mps.compile_shader(_KERNEL_SRC)
        _state['ok'] = True
    except Exception as e:
        warnings.warn(f"quantized_linear_zerocopy: falling back to PyTorch dequant matmul, compile_shader failed: {e}")
        _state['ok'] = False
    return _state['ok']


def available() -> bool:
    return _init()


def _ceil_mul(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def _dispatch_dims(out_features: int, num_rows: int):
    """Total `threads` (per dispatchThreads-style grid) rounded up to a whole number of
    `group_size`-sized threadgroups, so every threadgroup this kernel sees is full -- see the
    module docstring for why partial edge threadgroups would silently corrupt this kernel's
    cooperative shared-memory tile loads."""
    num_tg_x = (out_features + 63) // 64  # BN = 64
    num_tg_y = (num_rows + 63) // 64      # BM = 64
    return (num_tg_x * _GROUP_SIZE[0], num_tg_y * _GROUP_SIZE[1])


class QuantizedLinearZeroCopy(nn.Module):
    """Drop-in replacement for a pretrained nn.Linear: int8 weights (per-output-channel scale),
    float32 bias and activations, same quantization scheme as QuantizedLinearMetal. Inference-only.
    Fast path requires weights and input both already resident on 'mps' (i.e. called after
    model.to('mps')); falls back to a plain PyTorch dequant matmul otherwise, same contract as
    quantized_linear_metal.QuantizedLinearMetal."""

    def __init__(self, weight_int8: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor = None):
        super().__init__()
        self.register_buffer('weight_int8', weight_int8)  # [out, in], int8
        self.register_buffer('scale', scale)  # [out], float32
        self.register_buffer('bias', bias)  # [out], float32, or None
        self.out_features = weight_int8.shape[0]
        self.in_features = weight_int8.shape[1]
        self._bias_placeholder = None  # lazily built on 'mps' only if bias is None but kernel needs a bound buffer

    @classmethod
    def from_float(cls, linear: nn.Linear) -> "QuantizedLinearZeroCopy":
        w = linear.weight.detach().to('cpu', torch.float32)  # [out, in]
        row_absmax = w.abs().amax(dim=1).clamp(min=1e-8)
        scale = row_absmax / 127.0
        w_int8 = (w / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        bias = linear.bias.detach().to('cpu', torch.float32).clone() if linear.bias is not None else None
        return cls(w_int8, scale, bias)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        orig_dtype = x.dtype
        x2d = x.reshape(-1, self.in_features)
        num_rows = x2d.shape[0]

        use_fast_path = (
            _init()
            and x2d.device.type == 'mps'
            and self.weight_int8.device.type == 'mps'
        )

        if use_fast_path:
            lib = _state['lib']
            x_f32 = x2d.contiguous().to(torch.float32)  # on-device cast/copy only, no host transfer
            has_bias = 1 if self.bias is not None else 0
            if self.bias is not None:
                bias_arg = self.bias
            else:
                if self._bias_placeholder is None:
                    self._bias_placeholder = torch.zeros(1, device=x2d.device, dtype=torch.float32)
                bias_arg = self._bias_placeholder
            y = torch.empty(num_rows, self.out_features, device=x2d.device, dtype=torch.float32)

            threads = _dispatch_dims(self.out_features, num_rows)
            lib.quantized_linear_kernel(
                x_f32, self.weight_int8, self.scale, bias_arg,
                self.in_features, self.out_features, num_rows, has_bias,
                y, threads=threads, group_size=_GROUP_SIZE,
            )
            # No explicit torch.mps.synchronize(): this kernel dispatch is a normal op on PyTorch's
            # MPS stream, so any later op consuming `y` (including the final .to(orig_dtype)/.cpu())
            # is ordered correctly by PyTorch itself -- forcing a sync here would reintroduce the
            # exact CPU/GPU stall this module exists to remove.
            y = y.to(orig_dtype)
        else:
            orig_device = x.device
            w_dequant = (self.weight_int8.to(torch.float32) * self.scale[:, None]).to(orig_device, orig_dtype)
            y = x2d.to(orig_dtype) @ w_dequant.T
            if self.bias is not None:
                y = y + self.bias.to(orig_device, orig_dtype)

        return y.reshape(*orig_shape[:-1], self.out_features)


def quantize_model_linears_zerocopy(model: nn.Module, exclude_substrings=()) -> int:
    """Same contract as quantized_linear_metal.quantize_model_linears, using this module's
    QuantizedLinearZeroCopy instead -- a separate opt-in path, not a replacement."""
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                full_name = f"{name}.{child_name}" if name else child_name
                if any(s in full_name for s in exclude_substrings):
                    continue
                setattr(module, child_name, QuantizedLinearZeroCopy.from_float(child))
                count += 1
    return count
