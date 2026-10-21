"""
Int8-weight-quantized replacement for trellis.modules.attention.modules.MultiHeadAttention's
Linear projections (to_qkv / to_q+to_kv / to_out), applying the exact pattern proven in
quantized_ffn_matmul2d.py: matmul2d's native `half x int8_t -> float` GEMM (no separate dequantize
pass), dispatched zero-copy via torch.mps.load_metallib, with the surrounding elementwise ops
(dynamic rescale, RMSNorm, reshape) wrapped in one torch.compile'd function per attention config so
Dynamo can fuse them around the opaque matmul2d custom_op instead of paying per-op eager dispatch
cost. `scaled_dot_product_attention` itself is left completely untouched, native PyTorch/MPS -- per
APPLE_SILICON.md, quantizing SDPA itself would mean writing a flash-attention-class kernel from
scratch, well beyond this port's kernel-writing scope, and PyTorch's own tuned SDPA path is already
believed close to optimal for this hardware.

Reuses `torch.ops.trellis.matmul2d_linear_quantized`, the custom_op already registered by
quantized_ffn_matmul2d.py, rather than re-registering an identical op -- same kernel
(`matmul2d_linear_quantized` in metal_kernels/quantized_linear_matmul2d.metal), same metallib, same
zero-copy dispatch convention. quantized_ffn_matmul2d.py is untouched by this module.

Scope, deliberately narrow, per CLAUDE.md: this model (`sparse_structure_flow_model`, checked
directly) only ever instantiates MultiHeadAttention in two configs -- self-attention with
qk_rms_norm=True, use_rope=False, and cross-attention with qk_rms_norm=False, use_rope=False (RoPE
is never enabled for cross-attention in the base class either). This wrapper only implements those
two cases and raises NotImplementedError for anything else (windowed attn_mode, use_rope=True, or
qk_rms_norm in the "wrong" direction for a given type) rather than silently mishandling an
unsupported config.

Same dynamic per-call activation rescaling as quantized_ffn_matmul2d.py (measure each tensor's own
amax(), rescale so it lands at a fixed target of 256 before the half cast, fold the same scale back
into the per-channel weight scale on the output -- exact, not approximate) instead of a fixed clamp,
for the same reason: this model's real activations span a wide, input-dependent magnitude range and
no fixed threshold is safe in general.

Before writing this module, the "still open" cache-effect question in APPLE_SILICON.md's FFN
section (isolated benchmark predicted ~1.35ms/block, real model measured ~2.50ms/block, hypothesized
as a GPU-cache/distinct-weight-set effect) was investigated directly
(scripts/investigate_ffn_cache_gap.py) and refuted -- it was a benchmark-shape bug (the isolated
test used a guessed num_rows, not the real model's actual 4096), not a cache effect. See
APPLE_SILICON.md's "Resolved: the cache-effect hypothesis above is wrong" paragraph. Practical
consequence for this module: no GPU-cache-driven slowdown is expected purely from cycling through
many distinct attention blocks' weights, but any new isolated benchmark for this module must still
capture real shapes from an instrumented forward pass rather than guessing them.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention.modules import MultiHeadAttention
from .attention.full_attn import scaled_dot_product_attention
from .quantized_ffn_matmul2d import available

__all__ = ['QuantizedMultiHeadAttentionMatmul2D', 'quantize_model_attention_matmul2d', 'available']


def _quantize_weight(weight: torch.Tensor):
    """Per-output-channel (row-wise) symmetric int8 quantization -- identical scheme to every
    other quantized module in this port (see QuantizedFeedForwardNetMatmul2D._quantize)."""
    w = weight.detach().to('cpu', torch.float32)
    row_absmax = w.abs().amax(dim=1).clamp(min=1e-8)
    scale = row_absmax / 127.0
    w_int8 = (w / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return w_int8, scale


def _self_attn_compute(x2d, w_qkv_int8, scale_qkv, bias_qkv, w_out_int8, scale_out, bias_out,
                        q_gamma, k_gamma, B: int, L: int, channels: int, num_heads: int, head_dim: int,
                        compute_dtype: torch.dtype):
    """Self-attention, qk_rms_norm=True, use_rope=False -- the only self-attention config this
    model uses. to_qkv -> unbind -> RMSNorm(q,k) -> native SDPA -> to_out, matmul2d for both
    Linears, everything else plain torch ops so Dynamo can fuse them around the two opaque
    matmul2d calls.

    `compute_dtype` (the caller's orig_dtype, i.e. this model's real fp16) matters here, not just
    cosmetically at the final return: matmul2d_linear_quantized's custom_op always returns float32
    (see quantized_ffn_matmul2d.py), so without an explicit downcast SDPA would run on float32
    q/k/v -- measured directly (scripts/investigate_attention_dispatch_overhead.py) to cost 5.63ms
    vs. 5.02ms for the same call on float16 q/k/v, ~0.61ms of the ~0.58-0.68ms gap this module had
    against native fp16 attention, i.e. nearly the entire regression. FFN's _ffn_compute doesn't
    need this (GELU is cheap and dtype-insensitive; there's no big fused matmul-heavy native op
    sandwiched between its two matmul2d calls the way SDPA sits here)."""
    num_rows = x2d.shape[0]

    x_amax = x2d.abs().amax().clamp(min=1e-8)
    x_scale = x_amax / 256.0
    x_half = (x2d.contiguous() / x_scale).to(torch.float16)
    qkv2d = torch.ops.trellis.matmul2d_linear_quantized(x_half, w_qkv_int8, num_rows, channels, channels * 3)
    qkv2d = (qkv2d * (scale_qkv * x_scale) + bias_qkv).to(compute_dtype)
    qkv = qkv2d.reshape(B, L, 3, num_heads, head_dim)
    q, k, v = qkv.unbind(dim=2)

    rms_scale = head_dim ** 0.5
    q = (F.normalize(q.float(), dim=-1) * q_gamma * rms_scale).to(compute_dtype)
    k = (F.normalize(k.float(), dim=-1) * k_gamma * rms_scale).to(compute_dtype)

    h = scaled_dot_product_attention(q, k, v)  # native SDPA, untouched -- now runs in compute_dtype
    h2d = h.reshape(num_rows, channels)

    h_amax = h2d.abs().amax().clamp(min=1e-8)
    h_scale = h_amax / 256.0
    h_half = (h2d.contiguous() / h_scale).to(torch.float16)
    out2d = torch.ops.trellis.matmul2d_linear_quantized(h_half, w_out_int8, num_rows, channels, channels)
    return (out2d * (scale_out * h_scale) + bias_out).to(compute_dtype)


def _cross_attn_compute(x2d, ctx2d, w_q_int8, scale_q, bias_q, w_kv_int8, scale_kv, bias_kv,
                         w_out_int8, scale_out, bias_out, B: int, L: int, Lkv: int,
                         channels: int, ctx_channels: int, num_heads: int, head_dim: int,
                         compute_dtype: torch.dtype):
    """Cross-attention, no RMSNorm, use_rope=False -- the only cross-attention config this model
    uses. to_q(x) / to_kv(context) -> native SDPA -> to_out, matmul2d for the three Linears. Same
    compute_dtype downcast before SDPA as _self_attn_compute, for the same measured reason."""
    num_rows = x2d.shape[0]
    num_ctx_rows = ctx2d.shape[0]

    x_amax = x2d.abs().amax().clamp(min=1e-8)
    x_scale = x_amax / 256.0
    x_half = (x2d.contiguous() / x_scale).to(torch.float16)
    q2d = torch.ops.trellis.matmul2d_linear_quantized(x_half, w_q_int8, num_rows, channels, channels)
    q2d = (q2d * (scale_q * x_scale) + bias_q).to(compute_dtype)
    q = q2d.reshape(B, L, num_heads, head_dim)

    ctx_amax = ctx2d.abs().amax().clamp(min=1e-8)
    ctx_scale = ctx_amax / 256.0
    ctx_half = (ctx2d.contiguous() / ctx_scale).to(torch.float16)
    kv2d = torch.ops.trellis.matmul2d_linear_quantized(ctx_half, w_kv_int8, num_ctx_rows, ctx_channels, channels * 2)
    kv2d = (kv2d * (scale_kv * ctx_scale) + bias_kv).to(compute_dtype)
    kv = kv2d.reshape(B, Lkv, 2, num_heads, head_dim)
    k, v = kv.unbind(dim=2)

    h = scaled_dot_product_attention(q, k, v)  # native SDPA, untouched -- now runs in compute_dtype
    h2d = h.reshape(num_rows, channels)

    h_amax = h2d.abs().amax().clamp(min=1e-8)
    h_scale = h_amax / 256.0
    h_half = (h2d.contiguous() / h_scale).to(torch.float16)
    out2d = torch.ops.trellis.matmul2d_linear_quantized(h_half, w_out_int8, num_rows, channels, channels)
    return (out2d * (scale_out * h_scale) + bias_out).to(compute_dtype)


# Compiled once at module scope, not per-instance -- same reasoning as
# quantized_ffn_matmul2d.py's _compiled_ffn_compute: every self-attention block in the real model
# shares identical channels/num_heads/head_dim, so this compiles once and is reused by all of them
# (verify with torch._dynamo.utils.counters['stats']['unique_graphs'] before trusting speed numbers,
# same as the FFN module -- don't assume reuse, confirm it).
_compiled_self_attn_compute = torch.compile(_self_attn_compute)
_compiled_cross_attn_compute = torch.compile(_cross_attn_compute)


class QuantizedMultiHeadAttentionMatmul2D(nn.Module):
    """Drop-in replacement for trellis.modules.attention.modules.MultiHeadAttention, restricted to
    the two configs this model actually uses (see module docstring). Inference-only. Fast path
    requires the input already resident on 'mps'; falls back to the original fp32 MultiHeadAttention
    (kept around, unmodified) otherwise."""

    def __init__(self, mha: MultiHeadAttention):
        super().__init__()
        if mha.attn_mode != 'full':
            raise NotImplementedError(f"quantized attention only supports attn_mode='full', got {mha.attn_mode!r}")
        if mha.use_rope:
            raise NotImplementedError("quantized attention does not support RoPE (use_rope=True)")

        self._type = mha._type
        self.channels = mha.channels
        self.ctx_channels = mha.ctx_channels
        self.num_heads = mha.num_heads
        self.head_dim = mha.head_dim

        if self._type == 'self':
            if not mha.qk_rms_norm:
                raise NotImplementedError(
                    "quantized self-attention only supports qk_rms_norm=True (the only self-attention "
                    "config this model uses); got qk_rms_norm=False")
            w_qkv, s_qkv = _quantize_weight(mha.to_qkv.weight)
            self.register_buffer('w_qkv_int8', w_qkv)
            self.register_buffer('scale_qkv', s_qkv)
            self.register_buffer('bias_qkv', mha.to_qkv.bias.detach().to('cpu', torch.float32).clone())
            self.register_buffer('q_gamma', mha.q_rms_norm.gamma.detach().to('cpu', torch.float32).clone())
            self.register_buffer('k_gamma', mha.k_rms_norm.gamma.detach().to('cpu', torch.float32).clone())
        else:
            if mha.qk_rms_norm:
                raise NotImplementedError(
                    "quantized cross-attention only supports qk_rms_norm=False (the only cross-attention "
                    "config this model uses); got qk_rms_norm=True")
            w_q, s_q = _quantize_weight(mha.to_q.weight)
            w_kv, s_kv = _quantize_weight(mha.to_kv.weight)
            self.register_buffer('w_q_int8', w_q)
            self.register_buffer('scale_q', s_q)
            self.register_buffer('bias_q', mha.to_q.bias.detach().to('cpu', torch.float32).clone())
            self.register_buffer('w_kv_int8', w_kv)
            self.register_buffer('scale_kv', s_kv)
            self.register_buffer('bias_kv', mha.to_kv.bias.detach().to('cpu', torch.float32).clone())

        w_out, s_out = _quantize_weight(mha.to_out.weight)
        self.register_buffer('w_out_int8', w_out)
        self.register_buffer('scale_out', s_out)
        self.register_buffer('bias_out', mha.to_out.bias.detach().to('cpu', torch.float32).clone())

        self._fallback_mha = mha  # kept, unmodified, for the non-MPS / kernel-unavailable path
        self._resident = None  # weights moved to device once, cached (same pattern as the FFN module)

    @classmethod
    def from_float(cls, mha: MultiHeadAttention) -> "QuantizedMultiHeadAttentionMatmul2D":
        return cls(mha)

    def _resident_weights(self, device):
        if self._resident is not None and self._resident[0].device == device:
            return self._resident
        if self._type == 'self':
            self._resident = (
                self.w_qkv_int8.to(device), self.scale_qkv.to(device), self.bias_qkv.to(device),
                self.w_out_int8.to(device), self.scale_out.to(device), self.bias_out.to(device),
                self.q_gamma.to(device), self.k_gamma.to(device),
            )
        else:
            self._resident = (
                self.w_q_int8.to(device), self.scale_q.to(device), self.bias_q.to(device),
                self.w_kv_int8.to(device), self.scale_kv.to(device), self.bias_kv.to(device),
                self.w_out_int8.to(device), self.scale_out.to(device), self.bias_out.to(device),
            )
        return self._resident

    @torch.no_grad()
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not (available() and x.device.type == 'mps'):
            return self._fallback_mha(x, context, indices)

        B, L, C = x.shape
        orig_dtype = x.dtype
        device = x.device
        x2d = x.reshape(-1, self.channels)

        if self._type == 'self':
            w_qkv_int8, scale_qkv, bias_qkv, w_out_int8, scale_out, bias_out, q_gamma, k_gamma = \
                self._resident_weights(device)
            out2d = _compiled_self_attn_compute(
                x2d, w_qkv_int8, scale_qkv, bias_qkv, w_out_int8, scale_out, bias_out,
                q_gamma, k_gamma, B, L, self.channels, self.num_heads, self.head_dim, orig_dtype)
        else:
            Lkv = context.shape[1]
            ctx2d = context.reshape(-1, self.ctx_channels)
            w_q_int8, scale_q, bias_q, w_kv_int8, scale_kv, bias_kv, w_out_int8, scale_out, bias_out = \
                self._resident_weights(device)
            out2d = _compiled_cross_attn_compute(
                x2d, ctx2d, w_q_int8, scale_q, bias_q, w_kv_int8, scale_kv, bias_kv,
                w_out_int8, scale_out, bias_out, B, L, Lkv, self.channels, self.ctx_channels,
                self.num_heads, self.head_dim, orig_dtype)
        # No explicit torch.mps.synchronize() -- same reasoning as the FFN module: everything above
        # is a normal op on PyTorch's MPS stream, correctly ordered against whatever consumes the
        # output next.
        #
        # out2d is already `orig_dtype` (the compute functions downcast internally, before SDPA --
        # see their docstrings for why that's a real speed issue, not just a correctness one). The
        # explicit .to(orig_dtype) here stays anyway as a cheap, always-correct guarantee rather
        # than trusting every path through the compute functions to have gotten it right --
        # matmul2d's custom_op always returns float32 (quantized_ffn_matmul2d.py's
        # _matmul2d_linear_quantized_op), and feeding a float32 tensor into this model's fp16
        # residual stream was a real, confirmed bug found while first verifying this module on the
        # real model: it crashed a downstream native fp16 nn.Linear with a hard, uncatchable MPS
        # assertion ("Destination NDArray and Accumulator NDArray cannot have different datatype
        # in MPSNDArrayMatrixMultiplication") rather than a Python exception.
        return out2d.to(orig_dtype).reshape(B, L, self.channels)


def quantize_model_attention_matmul2d(model: nn.Module, exclude_substrings=()) -> int:
    """Replaces every trellis.modules.attention.modules.MultiHeadAttention submodule in `model`
    with a QuantizedMultiHeadAttentionMatmul2D in place. Returns the count replaced. Raises
    NotImplementedError (not silently skips) if any matched module is in a config this wrapper
    doesn't support -- see the class/module docstrings for the two supported configs."""
    count = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, MultiHeadAttention):
                full_name = f"{name}.{child_name}" if name else child_name
                if any(s in full_name for s in exclude_substrings):
                    continue
                setattr(module, child_name, QuantizedMultiHeadAttentionMatmul2D.from_float(child))
                count += 1
    return count
