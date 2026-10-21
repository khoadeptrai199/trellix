"""
int8 weight-only quantized nn.Linear using Metal 4's `matmul2d` (Metal Performance Primitives) --
an ADDITIONAL, separate implementation alongside quantized_linear_metal.py's hand-rolled kernel
(v1-v5, best result: 5.94x slower than fp32 on the real model). This module does not modify or
replace that one; quantize_model_linears() and every other existing entry point are untouched.

Why this exists: reading PyTorch's actual MPS matmul source (see APPLE_SILICON.md's "Linear
algebra & parallel optimization" section) showed the real speed advantage of `torch.matmul` on
this platform comes from Apple's own MPSGraph/proprietary compiler, not a hand-writable trick.
Metal 4's `matmul2d` is Apple's own tuned primitive for exactly this (tiled GEMM with SIMD-group
cooperation configured via `execution_simdgroups<N>`), likely connecting to the same internals
MPSGraph uses. Verified end to end on this machine: max diff ~1.7e-4 vs an fp32 PyTorch reference
(pure fp16 accumulation noise, not a bug) and materially faster than the hand-rolled v5 kernel in
isolated per-call testing (see scripts/verify_quantized_linear_matmul2d.py and
scripts/benchmark_quantized_linear_matmul2d.py for what was actually measured).

Real infrastructure cost, unlike every other kernel in this port: `MetalPerformancePrimitives.h`
does not resolve under runtime JIT source compilation (`MTLDevice.newLibraryWithSource_options_error_`,
used everywhere else specifically to avoid requiring Xcode) -- confirmed empirically, not assumed.
This kernel needs Xcode's offline compiler (`xcrun metal` + `xcrun metallib`), built once via
scripts/build_matmul2d_kernel.sh into a checked-in-or-built .metallib, loaded at runtime via
`newLibraryWithURL_error_` instead. `available()` returns False (clean fallback, no crash) if the
prebuilt .metallib is missing and can't be built (e.g. no Xcode installed) -- this module is
strictly optional, matching every other Metal kernel's fallback contract in this port.

Layout convention (see the .metal source's own comment for the full story -- this took real trial
and error against the one public matmul2d example available, since Apple's own docs don't spell it
out): MTLTensor dextents are (width, height), the opposite of numpy/C (rows, cols) order. The
weight tensor is used in nn.Linear's *natural* [out_features, in_features] layout directly, with
`transposeRight=true` in the kernel -- no separate weight-transpose step needed.
"""
import os
import subprocess

import numpy as np
import torch
import torch.nn as nn

__all__ = ['QuantizedLinearMatmul2D', 'quantize_model_linears_matmul2d', 'available']

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_METALLIB_PATH = os.path.join(_REPO_ROOT, 'trellis', 'modules', 'metal_kernels', 'quantized_linear_matmul2d.metallib')
_BUILD_SCRIPT = os.path.join(_REPO_ROOT, 'scripts', 'build_matmul2d_kernel.sh')

_state = {'checked': False, 'ok': False}


def _try_build_metallib():
    if os.path.exists(_METALLIB_PATH):
        return True
    try:
        subprocess.run([_BUILD_SCRIPT], check=True, capture_output=True, timeout=120)
        return os.path.exists(_METALLIB_PATH)
    except Exception:
        return False


def _init():
    if _state['checked']:
        return _state['ok']
    _state['checked'] = True
    try:
        import Metal
        import Foundation
        if not _try_build_metallib():
            return False
        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            return False
        url = Foundation.NSURL.fileURLWithPath_(_METALLIB_PATH)
        library, error = device.newLibraryWithURL_error_(url, None)
        if library is None:
            raise RuntimeError(str(error))
        dequant_fn = library.newFunctionWithName_('dequantize_weights')
        dequant_pipeline, error = device.newComputePipelineStateWithFunction_error_(dequant_fn, None)
        if dequant_pipeline is None:
            raise RuntimeError(str(error))
        matmul_fn = library.newFunctionWithName_('matmul2d_linear')
        matmul_pipeline, error = device.newComputePipelineStateWithFunction_error_(matmul_fn, None)
        if matmul_pipeline is None:
            raise RuntimeError(str(error))
        _state['device'] = device
        _state['queue'] = device.newCommandQueue()
        _state['dequant_pipeline'] = dequant_pipeline
        _state['matmul_pipeline'] = matmul_pipeline
        _state['Metal'] = Metal
        _state['ok'] = True
    except Exception as e:
        import warnings
        warnings.warn(f"quantized_linear_matmul2d: unavailable, falling back elsewhere: {e}")
        _state['ok'] = False
    return _state['ok']


def available() -> bool:
    return _init()


def _to_buffer(device, Metal, arr: np.ndarray):
    arr = np.ascontiguousarray(arr)
    return device.newBufferWithBytes_length_options_(arr.tobytes(), arr.nbytes, Metal.MTLResourceStorageModeShared)


def _write_into_buffer(buf, arr: np.ndarray):
    arr = np.ascontiguousarray(arr)
    dst = buf.contents().as_buffer(arr.nbytes)
    np.frombuffer(dst, dtype=arr.dtype).reshape(arr.shape)[:] = arr


def _readback(buf, shape, dtype):
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    raw = buf.contents().as_buffer(nbytes)
    return np.frombuffer(raw, dtype=dtype, count=int(np.prod(shape))).reshape(shape).copy()


def _ensure_capacity(pool, key, device, Metal, nbytes):
    # Same buffer-pooling pattern as quantized_linear_metal.py -- repeatedly allocating fresh
    # MTLBuffers in a tight loop leaks deterministically on this PyObjC/Metal setup regardless of
    # del/gc.collect()/autorelease pools (see that module's docstring for the full diagnosis);
    # reusing buffers across calls, growing only when a larger size is actually needed, is the fix.
    entry = pool.get(key)
    if entry is None or entry[1] < nbytes:
        buf = device.newBufferWithLength_options_(nbytes, Metal.MTLResourceStorageModeShared)
        pool[key] = (buf, nbytes)
        return buf
    return entry[0]


class QuantizedLinearMatmul2D(nn.Module):
    """Drop-in replacement for a pretrained nn.Linear: int8 weights (per-output-channel scale,
    same quantization scheme as QuantizedLinearMetal), matmul via Metal 4's matmul2d. Inference
    only (no backward -- nothing in this pipeline differentiates through these models)."""

    def __init__(self, weight_int8: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor = None):
        super().__init__()
        self.register_buffer('weight_int8', weight_int8)  # [out, in], int8
        self.register_buffer('scale', scale)  # [out], float32
        self.register_buffer('bias', bias)  # [out], float32, or None -- matmul2d path ignores this
                                             # for now (see forward()); tracked for parity/fallback.
        self.out_features = weight_int8.shape[0]
        self.in_features = weight_int8.shape[1]
        self._dequant_buf = None  # built lazily, dequantized once (weights never change)
        self._dyn_pool = {}

    @classmethod
    def from_float(cls, linear: nn.Linear) -> "QuantizedLinearMatmul2D":
        w = linear.weight.detach().to('cpu', torch.float32)
        row_absmax = w.abs().amax(dim=1).clamp(min=1e-8)
        scale = row_absmax / 127.0
        w_int8 = (w / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        bias = linear.bias.detach().to('cpu', torch.float32).clone() if linear.bias is not None else None
        return cls(w_int8, scale, bias)

    def _dequantized_weight_buffer(self):
        if self._dequant_buf is None:
            device = _state['device']
            Metal = _state['Metal']
            queue = _state['queue']
            w_int8_np = self.weight_int8.numpy()
            scale_np = self.scale.numpy()
            buf_w_int8 = _to_buffer(device, Metal, w_int8_np.reshape(-1))
            buf_scale = _to_buffer(device, Metal, scale_np)
            buf_w_half = device.newBufferWithLength_options_(
                self.in_features * self.out_features * 2, Metal.MTLResourceStorageModeShared)
            buf_in_f = _to_buffer(device, Metal, np.array([self.in_features], dtype=np.uint32))
            buf_out_f = _to_buffer(device, Metal, np.array([self.out_features], dtype=np.uint32))

            cmdbuf = queue.commandBuffer()
            enc = cmdbuf.computeCommandEncoder()
            enc.setComputePipelineState_(_state['dequant_pipeline'])
            for i, b in enumerate([buf_w_int8, buf_scale, buf_w_half, buf_in_f, buf_out_f]):
                enc.setBuffer_offset_atIndex_(b, 0, i)
            grid = Metal.MTLSizeMake(self.in_features, self.out_features, 1)
            tg = Metal.MTLSizeMake(16, 16, 1)
            enc.dispatchThreads_threadsPerThreadgroup_(grid, tg)
            enc.endEncoding()
            cmdbuf.commit()
            cmdbuf.waitUntilCompleted()
            if cmdbuf.status() == 5:
                raise RuntimeError(f"quantized_linear_matmul2d: dequantize failed: {cmdbuf.error()}")
            self._dequant_buf = buf_w_half
        return self._dequant_buf

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        orig_device = x.device
        orig_dtype = x.dtype
        x2d = x.reshape(-1, self.in_features)
        num_rows = x2d.shape[0]

        if not _init():
            w_dequant = (self.weight_int8.to(torch.float32) * self.scale[:, None]).to(orig_device, orig_dtype)
            y = x2d.to(orig_dtype) @ w_dequant.T
            if self.bias is not None:
                y = y + self.bias.to(orig_device, orig_dtype)
            return y.reshape(*orig_shape[:-1], self.out_features)

        device = _state['device']
        Metal = _state['Metal']
        queue = _state['queue']

        buf_w = self._dequantized_weight_buffer()
        x_np = x2d.detach().to('cpu', torch.float32).contiguous().numpy().astype(np.float16)
        buf_x = _ensure_capacity(self._dyn_pool, 'x', device, Metal, x_np.nbytes)
        _write_into_buffer(buf_x, x_np)
        buf_y = _ensure_capacity(self._dyn_pool, 'y', device, Metal, num_rows * self.out_features * 4)
        buf_rows = _ensure_capacity(self._dyn_pool, 'rows', device, Metal, 4)
        _write_into_buffer(buf_rows, np.array([num_rows], dtype=np.uint32))
        buf_in_f = _ensure_capacity(self._dyn_pool, 'in_f', device, Metal, 4)
        _write_into_buffer(buf_in_f, np.array([self.in_features], dtype=np.uint32))
        buf_out_f = _ensure_capacity(self._dyn_pool, 'out_f', device, Metal, 4)
        _write_into_buffer(buf_out_f, np.array([self.out_features], dtype=np.uint32))

        cmdbuf = queue.commandBuffer()
        enc = cmdbuf.computeCommandEncoder()
        enc.setComputePipelineState_(_state['matmul_pipeline'])
        for i, b in enumerate([buf_x, buf_w, buf_y, buf_rows, buf_in_f, buf_out_f]):
            enc.setBuffer_offset_atIndex_(b, 0, i)
        num_tg = Metal.MTLSizeMake((self.out_features + 31) // 32, (num_rows + 63) // 64, 1)
        tg = Metal.MTLSizeMake(128, 1, 1)  # execution_simdgroups<4> = 4*32 lanes
        enc.dispatchThreadgroups_threadsPerThreadgroup_(num_tg, tg)
        enc.endEncoding()
        cmdbuf.commit()
        cmdbuf.waitUntilCompleted()
        if cmdbuf.status() == 5:
            raise RuntimeError(f"quantized_linear_matmul2d: matmul failed: {cmdbuf.error()}")

        y_np = _readback(buf_y, (num_rows, self.out_features), np.float32)
        y = torch.from_numpy(y_np).to(orig_device, orig_dtype)
        if self.bias is not None:
            y = y + self.bias.to(orig_device, orig_dtype)
        return y.reshape(*orig_shape[:-1], self.out_features)


def quantize_model_linears_matmul2d(model: nn.Module, exclude_substrings=()) -> int:
    """Same contract as quantized_linear_metal.quantize_model_linears, using this module's
    QuantizedLinearMatmul2D instead -- a separate opt-in path, not a replacement."""
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                full_name = f"{name}.{child_name}" if name else child_name
                if any(s in full_name for s in exclude_substrings):
                    continue
                setattr(module, child_name, QuantizedLinearMatmul2D.from_float(child))
                count += 1
    return count
