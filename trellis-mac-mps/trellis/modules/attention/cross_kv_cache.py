"""
Cross-step cross-attention K/V caching for TRELLIS's flow denoisers.

Both SLatFlowModel and SparseStructureFlowModel run model.forward(x, t, cond) once per sampler
step (~25 steps per generation, ~50 real calls once CFG's cond/uncond pair is counted), and every
one of those calls recomputes `to_kv(context)` fresh in every one of the 24 blocks' cross_attn
modules -- even though `context` (the DINOv2 image conditioning) is the exact same tensor object
for the entire sample() call. `to_kv`'s output depends only on `context` and the frozen weights,
never on `x`/`t`, so recomputing it every step is pure, deterministic redundant compute. Confirmed
directly in trellis/models/structured_latent_flow.py (`cond` passed unchanged into every block on
every call) and trellis/models/sparse_structure_flow.py (same pattern); see research.md Part II.

Patches at the granularity of `cross_attn.to_kv` (an nn.Linear), not the model/block forward
signature -- neither trellis/models/*.py nor trellis/modules/*/transformer/modulated.py needs to
change at all, so the cache is invisible to every caller that doesn't opt in, per this project's
non-destructive versioning convention (new file, no edits to any existing file). Works identically
for both the dense (trellis/modules/attention) and sparse (trellis/modules/sparse/attention)
MultiHeadAttention/SparseMultiHeadAttention classes, because in both, `context` passed to
cross_attn is always a plain torch.Tensor (never a SparseTensor -- confirmed: `cond` is typed
`torch.Tensor` in both models' forward signatures) and `to_kv` is always a plain nn.Linear called
directly on it, so patching nn.Linear.forward is sufficient without needing separate dense/sparse
implementations.

Cache key is primarily Python object identity (`is`), not a hash/id() int -- safe here
specifically because the cache holds a strong reference to the cached input tensor itself, so
there's no risk of a freed tensor's id() being reused for an unrelated tensor (the classic
id()-as-dict-key bug). Two entries (cond, neg_cond) is the expected steady state per generation;
max_entries bounds memory if more distinct context tensors flow through (e.g. batched multi-image
generation).

Real-model finding (not hypothetical -- measured via scripts/verify_cross_attn_kv_cache.py Stage
2): identity alone gets ZERO hits on the real pretrained models, because both
SparseStructureFlowModel.forward and SLatFlowModel.forward do `cond = cond.type(self.dtype)`
before handing `cond` down to blocks as `context`. When an actual cast is needed (real
conditioning is fp32, model runs fp16), `.type()` allocates a brand-new tensor every single call
-- even though the value is identical every sampling step -- so pure identity never matches across
steps. (It matched in synthetic testing only because the synthetic model was constructed with
`use_fp16=False` and synthetic `cond` was already fp32, making `.type()` a genuine no-op that
returns the same object.) Fix: fall back to a value-equality check (`torch.equal`, after a cheap
shape/dtype/device guard) when identity fails. Still a large net win -- comparing ~1.4M elements
is far cheaper than the Linear matmul it replaces -- and correct, since `.type()`'s cast of a given
source tensor is deterministic, so equal values at matching shape/dtype/device only arise from the
same conceptual input (cond vs. neg_cond, being semantically different conditioning, essentially
never collide).
"""
from typing import List, Tuple
import types
import torch
import torch.nn as nn


def _make_cached_forward(max_entries: int):
    def cached_forward(self: nn.Linear, input: torch.Tensor) -> torch.Tensor:
        for cached_input, cached_output in self._kv_cache_entries:
            if input is cached_input:
                self._kv_cache_hits += 1
                return cached_output
        for cached_input, cached_output in self._kv_cache_entries:
            if (input.shape == cached_input.shape and input.dtype == cached_input.dtype
                    and input.device == cached_input.device and torch.equal(input, cached_input)):
                self._kv_cache_hits += 1
                return cached_output
        output = self._kv_cache_original_forward(input)
        self._kv_cache_entries.append((input, output))
        if len(self._kv_cache_entries) > max_entries:
            self._kv_cache_entries.pop(0)
        self._kv_cache_misses += 1
        return output
    return cached_forward


def _iter_cross_attn_to_kv_linears(model: nn.Module) -> List[nn.Linear]:
    linears = []
    for block in model.blocks:
        cross_attn = getattr(block, "cross_attn", None)
        if cross_attn is None:
            continue
        to_kv = getattr(cross_attn, "to_kv", None)
        if isinstance(to_kv, nn.Linear):
            linears.append(to_kv)
    return linears


def enable_cross_attention_kv_cache(model: nn.Module, max_entries: int = 2) -> int:
    """
    Patches every transformer block's cross_attn.to_kv (an nn.Linear) in `model` to cache its
    output keyed by input-tensor identity. Returns the number of Linears patched (== num_blocks
    on both TRELLIS flow models, one cross_attn per block). Idempotent: calling twice on an
    already-patched model is a no-op for the already-patched Linears, not a double-patch.
    """
    patched = 0
    for linear in _iter_cross_attn_to_kv_linears(model):
        if getattr(linear, "_kv_cache_enabled", False):
            continue
        linear._kv_cache_original_forward = linear.forward
        linear._kv_cache_entries = []
        linear._kv_cache_hits = 0
        linear._kv_cache_misses = 0
        linear._kv_cache_enabled = True
        linear.forward = types.MethodType(_make_cached_forward(max_entries), linear)
        patched += 1
    return patched


def disable_cross_attention_kv_cache(model: nn.Module) -> int:
    """Restores every patched cross_attn.to_kv to its original, uncached forward."""
    restored = 0
    for linear in _iter_cross_attn_to_kv_linears(model):
        if not getattr(linear, "_kv_cache_enabled", False):
            continue
        linear.forward = linear._kv_cache_original_forward
        del linear._kv_cache_original_forward
        del linear._kv_cache_entries
        del linear._kv_cache_hits
        del linear._kv_cache_misses
        linear._kv_cache_enabled = False
        restored += 1
    return restored


def clear_cross_attention_kv_cache(model: nn.Module) -> None:
    """
    Drops all cached entries without disabling the cache -- call at the start of each sample()
    call for hygiene (bounds memory retention of large cond tensors across generations, though
    the max_entries eviction in enable_cross_attention_kv_cache already bounds it either way).
    """
    for linear in _iter_cross_attn_to_kv_linears(model):
        if getattr(linear, "_kv_cache_enabled", False):
            linear._kv_cache_entries = []


def cross_attention_kv_cache_stats(model: nn.Module) -> Tuple[int, int]:
    """Returns (total_hits, total_misses) across all patched cross_attn.to_kv Linears."""
    hits = misses = 0
    for linear in _iter_cross_attn_to_kv_linears(model):
        hits += getattr(linear, "_kv_cache_hits", 0)
        misses += getattr(linear, "_kv_cache_misses", 0)
    return hits, misses
