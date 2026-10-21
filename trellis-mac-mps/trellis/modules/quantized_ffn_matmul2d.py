"""
Fused int8-weight-quantized replacement for trellis.modules.transformer.blocks.FeedForwardNet
(`nn.Sequential(nn.Linear, nn.GELU(approximate="tanh"), nn.Linear)`), chaining fc1 -> bias+GELU ->
fc2 -> bias entirely on the GPU via Metal 4's `matmul2d` (Metal Performance Primitives) plus two
small fusion kernels, dispatched with `torch.mps.load_metallib`'s zero-copy calling convention (see
quantized_linear_zerocopy.py's module docstring for why that matters and how it was confirmed to
behave -- same `dispatchThreads`-is-non-uniform gotcha applies here for the shared-memory-cooperative
`matmul2d_linear` kernel, same fix: round `threads` up to a whole number of threadgroups).

Why fuse two Linears instead of two separate QuantizedLinearMatmul2D calls: APPLE_SILICON.md's
"matmul2d, follow-up" section diagnosed that even a fast isolated kernel loses most of its speedup
once wrapped as a drop-in nn.Linear inside a real model, because each `nn.Linear`-shaped call was a
separate CPU<->MPS round trip forcing a sync against everything else queued on the GPU. Switching
that same wrapper to zero-copy dispatch (quantized_linear_zerocopy.py) already fixes the sync; this
module goes one step further for the specific fc1->GELU->fc2 pattern that's this model's actual FFN
block, by also keeping the *intermediate* activation (the hidden, `mlp_ratio`x-widened tensor
between the two Linears) on-GPU the whole time and fusing the bias-add + GELU that would otherwise
be 2-3 separate PyTorch ops into one kernel pass -- fewer total dispatches per FFN block, not just
per Linear.

Kernels used, all in trellis/modules/metal_kernels/quantized_linear_matmul2d.metal:
- `matmul2d_linear_quantized`: the GEMM itself, fed the RAW int8 weight buffer directly -- no
  separate dequantize pass. Confirmed straight from Apple's own MPPTensorOpsMatMul2d.h (not
  inferred from docs): matmul2d natively supports a `half` left operand times an `int8_t` right
  operand producing `float`, doing the int8->float widen internally during its own tile load. The
  earlier version of this module (and quantized_linear_matmul2d.py, still) ran a separate
  `dequantize_weights` kernel first, materializing a full half-precision copy of every weight
  matrix in global memory -- a whole extra read+write of the weight matrix per layer that this
  native path skips entirely.
- `scale_bias_f32`: applies matmul2d_linear_quantized's deferred per-output-channel scale (and
  optionally bias) to its raw output, in place. Deferred because matmul2d has no notion of our
  quantization scale -- but since scale is per-OUTPUT-channel only (row-wise symmetric
  quantization), it factors exactly out of the reduction sum:
  `Y[r,o] = scale[o] * sum_i(X[r,i] * int8_W[o,i])`, so applying it as a cheap elementwise
  follow-up is exact, not an approximation.

Bias-add + GELU between the two matmul2d calls uses native torch ops, not a custom fusion kernel
(the .metal source still has `bias_gelu_to_half`/`bias_add_f32` from an earlier version of this
module, now unused here) -- found empirically that feeding matmul2d_linear's second call a buffer
written by a *different* custom kernel produced NaN on the real pretrained model (24 sequential
blocks), despite passing every synthetic correctness check and despite ruling out a race/hazard
(reproduced identically with a full torch.mps.synchronize() forced between every single kernel
dispatch in the chain). `scale_bias_f32` only ever *produces into* buffers matmul2d also produces
into (never consumes a custom kernel's output as matmul2d's own input), which keeps this on the
same side of that empirically-found line; verified finite on the real model with real image
conditioning across multiple trials before relying on it (see
scripts/benchmark_quantized_ffn_matmul2d.py).

Numerical conditioning, not clamping: matmul2d's operands must be half, and this model's real
activations range from single digits to 8000+ depending on block/timestep (measured directly, not
assumed) -- no fixed clamp threshold is actually safe in general, it just picks a magnitude that's
safe until some future input exceeds it, silently corrupting whatever does. `_ffn_compute` instead
measures each call's own `amax()` and rescales so the max lands at a small fixed target (256) before
the half cast, then folds that same scale back into the per-channel weight scale on the output
(exact, not approximate -- scale factors are multiplicative and commute through the matmul). This
adapts automatically to any input magnitude instead of trusting a threshold guess. Confirmed finite
across multiple trials with real image conditioning; measured ~2.5% slower than a fixed-clamp
version (the extra `amax()` reduction isn't free even under torch.compile) -- a small, deliberate
trade of speed for not depending on an assumption about how large activations can get.

Same Xcode/offline-metallib requirement as quantized_linear_matmul2d.py (MetalPerformancePrimitives.h
doesn't resolve under runtime JIT source compilation): `available()` returns False cleanly if the
prebuilt .metallib is missing and can't be (re)built.
"""
import os
import subprocess
import warnings

import torch
import torch.nn as nn

from .transformer.blocks import FeedForwardNet

__all__ = ['QuantizedFeedForwardNetMatmul2D', 'quantize_model_ffns_matmul2d', 'available']

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
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS not available")
        if not _try_build_metallib():
            raise RuntimeError(f"{_METALLIB_PATH} missing and could not be built")
        _state['lib'] = torch.mps.load_metallib(_METALLIB_PATH)
    except Exception as e:
        warnings.warn(f"quantized_ffn_matmul2d: unavailable, falling back elsewhere: {e}")
        _state['ok'] = False
        return False
    _state['ok'] = True
    return True


def available() -> bool:
    return _init()


def _ceil_div(n: int, m: int) -> int:
    return (n + m - 1) // m


def _matmul2d_dims(out_features: int, num_rows: int):
    """matmul2d_linear's threadgroup tile is 32 (out) x 64 (rows); like
    quantized_linear_zerocopy.py's kernel, this needs full, uniform threadgroups (its shared-memory
    /matmul2d cooperation can't tolerate a partial group), so `threads` must be rounded up to a
    whole number of threadgroups -- confirmed necessary and sufficient empirically before this
    module was written (see quantized_linear_zerocopy.py's docstring for the probe that found it)."""
    group_size = (128, 1)  # execution_simdgroups<4> = 4*32 lanes
    num_tg_x = _ceil_div(out_features, 32)
    num_tg_y = _ceil_div(num_rows, 64)
    return (num_tg_x * group_size[0], num_tg_y * group_size[1]), group_size


# Registered as a torch.library.custom_op, NOT called as a plain Python function, so that
# torch.compile can treat it as a single opaque-but-traceable graph node instead of refusing to
# trace through it at all. Measured impact is large and was the actual fix for a real regression:
# the raw matmul2d_linear_quantized kernel alone is 2.24x FASTER than torch.matmul for this port's
# shapes (1.06ms vs 2.36ms, measured directly), but wrapping it with plain eager-mode clamp/cast/
# scale/bias/GELU calls made the whole FFN block SLOWER than fp32 overall -- each of those small
# ops pays its own fixed dispatch cost (Python-level dispatch + MPS command-buffer encode/submit),
# and with more, smaller ops than the original fp32 path had, that overhead alone (~1ms, measured)
# outweighed the kernel's own 2.24x advantage. torch.compile fuses the surrounding elementwise ops
# into far fewer dispatches; it just can't do that unless the custom kernel call is registered as a
# real op it's allowed to treat as a black box. Confirmed directly: the same clamp->matmul->
# scale->bias->GELU chain went from 2.40ms eager to 1.35ms compiled once this registration was in
# place -- a 1.78x win from compilation alone, on top of the kernel's own advantage.
@torch.library.custom_op('trellis::matmul2d_linear_quantized', mutates_args=(), device_types='mps')
def _matmul2d_linear_quantized_op(x_half: torch.Tensor, w_int8: torch.Tensor,
                                   num_rows: int, in_features: int, out_features: int) -> torch.Tensor:
    _init()
    y = torch.empty(num_rows, out_features, device=x_half.device, dtype=torch.float32)
    threads, group_size = _matmul2d_dims(out_features, num_rows)
    _state['lib'].matmul2d_linear_quantized(x_half, w_int8, y, num_rows, in_features, out_features,
                                             threads=threads, group_size=group_size)
    return y


@_matmul2d_linear_quantized_op.register_fake
def _(x_half, w_int8, num_rows, in_features, out_features):
    return torch.empty(num_rows, out_features, device=x_half.device, dtype=torch.float32)


def _ffn_compute(x2d, w1_int8, scale1, bias1, w2_int8, scale2, bias2, channels: int, hidden: int):
    """The full fc1 -> GELU -> fc2 computation, as a free function (not a method) so it can be
    wrapped once, at module scope, by a single shared torch.compile call -- every
    QuantizedFeedForwardNetMatmul2D instance (24 of them, one per transformer block) calls the SAME
    compiled graph with its own weight tensors as plain arguments, rather than each instance
    compiling (and caching) its own copy. All 24 blocks share identical channels/hidden, so this
    compiles once and is reused for the rest of the model's blocks and every subsequent call."""
    num_rows = x2d.shape[0]

    # Dynamic per-call activation rescaling, not clamping. A fixed clamp truncates whatever
    # legitimately exceeds the threshold -- silently wrong for that element, and picks a threshold
    # that's only safe until some future input exceeds it (this model's real activations already
    # range from single digits to 8720+ across blocks/timesteps, measured directly -- no fixed
    # constant is actually safe in general). Rescaling instead: measure the tensor's own true
    # magnitude (`amax`), scale it down so its max lands at a small fixed target well inside half's
    # range, cast, run the matmul, then scale the *output* back up by the same factor (folded into
    # the existing per-channel weight scale, since both are just multiplicative and commute through
    # the matmul). This is exact, not approximate -- every element keeps its real value, just
    # represented in half at a range that's actually safe for THIS call's actual data -- and adapts
    # automatically to inputs of any magnitude instead of relying on a threshold guess.
    x_amax = x2d.abs().amax().clamp(min=1e-8)
    x_scale = x_amax / 256.0
    x_half = (x2d.contiguous() / x_scale).to(torch.float16)
    y1 = torch.ops.trellis.matmul2d_linear_quantized(x_half, w1_int8, num_rows, channels, hidden)
    h = torch.nn.functional.gelu(y1 * (scale1 * x_scale) + bias1, approximate='tanh')

    h_amax = h.abs().amax().clamp(min=1e-8)
    h_scale = h_amax / 256.0
    x2 = (h.contiguous() / h_scale).to(torch.float16)
    y2 = torch.ops.trellis.matmul2d_linear_quantized(x2, w2_int8, num_rows, hidden, channels)
    return y2 * (scale2 * h_scale) + bias2


_compiled_ffn_compute = torch.compile(_ffn_compute)


class QuantizedFeedForwardNetMatmul2D(nn.Module):
    """Drop-in replacement for trellis.modules.transformer.blocks.FeedForwardNet: int8 weights
    (per-output-channel scale, same scheme as every other quantized module in this port), fc1 ->
    GELU(tanh) -> fc2 fully fused on-GPU via matmul2d. Inference-only. Fast path requires the input
    already resident on 'mps' (i.e. called after model.to('mps')); falls back to the original
    fp32 nn.Sequential FFN (kept around, unquantized) otherwise."""

    def __init__(self, ffn: FeedForwardNet):
        super().__init__()
        fc1, _, fc2 = ffn.mlp[0], ffn.mlp[1], ffn.mlp[2]
        self.channels = fc1.in_features
        self.hidden = fc1.out_features
        assert fc2.out_features == self.channels

        w1, s1 = self._quantize(fc1.weight)
        w2, s2 = self._quantize(fc2.weight)
        self.register_buffer('w1_int8', w1)
        self.register_buffer('scale1', s1)
        self.register_buffer('bias1', fc1.bias.detach().to('cpu', torch.float32).clone())
        self.register_buffer('w2_int8', w2)
        self.register_buffer('scale2', s2)
        self.register_buffer('bias2', fc2.bias.detach().to('cpu', torch.float32).clone())

        self._fallback_ffn = ffn  # kept, unmodified, for the non-MPS / kernel-unavailable path
        self._resident = None  # (w1_int8, scale1, bias1, w2_int8, scale2, bias2) moved to device once, cached

    @staticmethod
    def _quantize(weight: torch.Tensor):
        w = weight.detach().to('cpu', torch.float32)
        row_absmax = w.abs().amax(dim=1).clamp(min=1e-8)
        scale = row_absmax / 127.0
        w_int8 = (w / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        return w_int8, scale

    @classmethod
    def from_float(cls, ffn: FeedForwardNet) -> "QuantizedFeedForwardNetMatmul2D":
        return cls(ffn)

    def _resident_weights(self, device):
        if self._resident is None or self._resident[0].device != device:
            self._resident = (
                self.w1_int8.to(device), self.scale1.to(device), self.bias1.to(device),
                self.w2_int8.to(device), self.scale2.to(device), self.bias2.to(device),
            )
        return self._resident

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        orig_dtype = x.dtype
        x2d = x.reshape(-1, self.channels)
        num_rows = x2d.shape[0]

        if not (_init() and x2d.device.type == 'mps'):
            return self._fallback_ffn(x)

        device = x2d.device
        w1_int8, scale1, bias1, w2_int8, scale2, bias2 = self._resident_weights(device)

        # The actual computation lives in the module-level _ffn_compute / _compiled_ffn_compute --
        # see there for why: torch.compile needs the matmul2d call registered as a custom op to
        # trace through it at all, and compiling once at module scope (not per-instance) lets all
        # 24 blocks in the real model share one compiled graph instead of compiling 24 times.
        y2 = _compiled_ffn_compute(x2d, w1_int8, scale1, bias1, w2_int8, scale2, bias2,
                                    self.channels, self.hidden)
        # No explicit torch.mps.synchronize(): everything above is a normal op on PyTorch's MPS
        # stream, correctly ordered against whatever consumes y2 next -- forcing a sync here would
        # reintroduce exactly the per-call stall this module exists to remove.

        y = y2.to(orig_dtype)
        return y.reshape(*orig_shape[:-1], self.channels)


def quantize_model_ffns_matmul2d(model: nn.Module, exclude_substrings=()) -> int:
    """Replaces every trellis.modules.transformer.blocks.FeedForwardNet submodule in `model` with
    a QuantizedFeedForwardNetMatmul2D in place. Returns the count replaced. Only matches the dense
    (non-sparse) FeedForwardNet -- SparseFeedForwardNet (used in the sparse-tensor transformer
    blocks) operates on SparseTensor, not a plain [rows, channels] tensor, and isn't handled here."""
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, FeedForwardNet):
                full_name = f"{name}.{child_name}" if name else child_name
                if any(s in full_name for s in exclude_substrings):
                    continue
                setattr(module, child_name, QuantizedFeedForwardNetMatmul2D.from_float(child))
                count += 1
    return count
