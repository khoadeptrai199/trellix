#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Torch-level compatibility patches for non-CUDA backends.

Upstream Hunyuan3D 2.1 hard-codes CUDA in three ways that break on a machine
without an NVIDIA GPU:

  1. `torch.autocast(device_type="cuda", ...)`      → "Torch not compiled with CUDA enabled"
  2. `torch.cuda.synchronize` / `Event` / `empty_cache`
  3. `torch.backends.cuda.sdp_kernel(...)` around attention

All three are patched here — but ONLY when the active device is not literally
called `cuda`. On ROCm the device *is* `cuda` (PyTorch/ROCm masquerades as
CUDA), so applying these patches there would replace working fp16 autocast
with a no-op and slow the model down for no reason.

Call `apply(backend)` once, right after `backend.detect()`.
"""
from __future__ import annotations

import contextlib
import os

import torch

_APPLIED = False


def apply(b) -> bool:
    """Apply the compatibility patches this backend needs. Idempotent.

    Two independent groups, because they are needed in different cases:

      * device redirection (autocast, torch.cuda.*) — only off a CUDA-like
        device. On ROCm the device really is called "cuda", so redirecting would
        replace working fp16 autocast with a no-op for nothing.

      * SDPA backend relaxation — also needed on ROCm. hy3dshape's DiT wraps its
        attention in `sdp_kernel(enable_flash=True, enable_math=False,
        enable_mem_efficient=True)`. Forbidding the math fallback is a sound
        optimisation on NVIDIA, but on ROCm neither flash nor mem-efficient SDPA
        covers these shapes, so PyTorch has no candidate left and raises
        "No available kernel. Aborting execution." Dropping the restriction lets
        it fall back to math and the model runs.

    Returns True if anything was patched.
    """
    global _APPLIED
    if _APPLIED:
        return True

    applied: list[str] = []

    if b.needs_device_patches:
        _patch_autocast(b)
        _patch_torch_cuda_namespace(b)
        applied += ["torch.autocast", "torch.cuda.*"]
    else:
        print(
            f"[compat] {b.kind}: device is reported as `{b.device}`, "
            f"no CUDA redirection needed",
            flush=True,
        )

    # CUDA is left alone: there, flash attention both exists and is faster.
    if b.kind != "cuda":
        if _patch_sdp_kernel():
            applied.append("torch.backends.cuda.sdp_kernel")
        if _patch_sdpa_chunking(sdpa_budget(b)):
            applied.append("F.scaled_dot_product_attention (chunked)")

    if b.kind == "mps" and _patch_isin():
        applied.append("torch.isin (sorted)")

    if applied:
        _APPLIED = True
        print(f"[compat] {b.kind}: patched {', '.join(applied)}", flush=True)
    return bool(applied)


def _patch_autocast(b) -> None:
    """Turn `autocast(device_type="cuda")` into something harmless.

    MPS autocast is unreliable and CPU fp16 autocast is pointless, so for a
    `cuda` request we return a null context and let the model run in whatever
    dtype its weights already are.
    """
    orig_autocast = torch.autocast

    class SafeAutocast:
        def __init__(self, device_type, *args, **kwargs):
            self.device_type = device_type
            self.args = args
            self.kwargs = kwargs
            self._ctx = None

        def __enter__(self):
            if self.device_type == "cuda" or not b.autocast_ok:
                self._ctx = contextlib.nullcontext()
            else:
                self._ctx = orig_autocast(self.device_type, *self.args, **self.kwargs)
            return self._ctx.__enter__()

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self._ctx.__exit__(exc_type, exc_val, exc_tb)

        # Upstream never uses autocast as a decorator, but diffusers does.
        def __call__(self, func):
            import functools

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.__class__(self.device_type, *self.args, **self.kwargs):
                    return func(*args, **kwargs)

            return wrapper

    torch.autocast = SafeAutocast
    if hasattr(torch, "amp"):
        torch.amp.autocast = SafeAutocast


def _patch_torch_cuda_namespace(b) -> None:
    """Stub the `torch.cuda` calls upstream sprinkles through the pipelines."""
    if getattr(torch.cuda, "_hy3d_patched", False):
        return

    class DummyEvent:
        def __init__(self, *a, **k):
            pass

        def record(self, *a, **k):
            pass

        def elapsed_time(self, other):
            return 0.0

        def synchronize(self):
            pass

    def empty_cache():
        # Route to the real backend cache instead of doing nothing, so that
        # upstream's `torch.cuda.empty_cache()` between pipeline stages still
        # frees memory on MPS.
        try:
            if b.kind == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass

    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.Event = DummyEvent
    torch.cuda.empty_cache = empty_cache
    torch.cuda._hy3d_patched = True


def patch_scheduler_timestep_lookup() -> bool:
    """Make the shape scheduler tolerate a timestep it cannot match exactly.

    `hy3dshape/schedulers.py::index_for_timestep` locates the current step with

        indices = (schedule_timesteps == timestep).nonzero()
        return indices[pos].item()

    which is exact float equality. When nothing matches, `indices` is empty and
    the lookup raises `IndexError: index 0 is out of bounds for dimension 0 with
    size 0`, killing a generation that is already minutes in. It was observed
    intermittently — identical settings and seed, same image succeeding on a
    retry — so it is a robustness hole rather than a deterministic bug.

    Falling back to the nearest timestep is what current diffusers does, and it
    is exactly right here: the value is only used to index the sigma schedule,
    and the nearest entry is the one the exact match would have found anyway.

    Call after `hy3dshape` is importable. Returns True if the patch went in.
    """
    try:
        from hy3dshape import schedulers
    except Exception as e:
        print(f"[compat] scheduler patch skipped ({type(e).__name__}: {e})", flush=True)
        return False

    target = getattr(schedulers, "FlowMatchEulerDiscreteScheduler", None)
    if target is None or not hasattr(target, "index_for_timestep"):
        print("[compat] scheduler patch skipped: unexpected upstream layout", flush=True)
        return False
    if getattr(target.index_for_timestep, "_hy3d_patched", False):
        return True

    original = target.index_for_timestep

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        try:
            return original(self, timestep, schedule_timesteps)
        except IndexError:
            steps = self.timesteps if schedule_timesteps is None else schedule_timesteps
            if steps is None or len(steps) == 0:
                raise
            t = timestep
            if isinstance(t, torch.Tensor):
                t = t.to(steps.device)
            idx = int(torch.argmin((steps - t).abs()).item())
            print(
                f"[compat] scheduler: timestep {float(t):g} had no exact match; "
                f"using nearest index {idx} ({float(steps[idx]):g})",
                flush=True,
            )
            return idx

    index_for_timestep._hy3d_patched = True
    target.index_for_timestep = index_for_timestep
    return True


def _patch_isin() -> bool:
    """Give `torch.isin` a memory-safe implementation on MPS.

    `MeshRender.back_project` (MeshRender.py:1275) ends its `back_sample` bake
    with `torch.isin(valid_idx, inner_valid_idx)`. Both sides are index vectors
    over the UV atlas, so at a 1024x1024 texture they are hundreds of thousands
    of elements each. ATen picks between a broadcast comparison and a
    sort-based algorithm; on MPS it takes the broadcast, which means an
    `(len(elements), len(test_elements))` intermediate. Measured here: a request
    for a single 111 GB buffer, and because it is Metal that refuses it the
    process does not raise — it dies on
    `failed assertion 'Failed to allocate private MTLBuffer'`, which no
    try/except can catch.

    Sorting `test_elements` once and binary-searching gives exactly the same
    answer in O(n log n) with no large intermediate. It stays on the GPU, unlike
    bouncing the call to CPU, and at 200k x 700k it costs ~90 ms.

    Left alone on CUDA and ROCm: there ATen already chooses the sorting path for
    inputs this size.
    """
    original = torch.isin
    if getattr(original, "_hy3d_patched", False):
        return True

    def isin(elements, test_elements, *, assume_unique=False, invert=False):
        e, t = elements, test_elements
        # Only take over when both sides are real tensors of the same dtype on
        # MPS and the broadcast would actually be large. Scalars, mixed dtypes
        # and small calls keep upstream's exact semantics.
        if (
            isinstance(e, torch.Tensor) and isinstance(t, torch.Tensor)
            and e.device.type == "mps" and t.device.type == "mps"
            and e.dtype == t.dtype and e.numel() * t.numel() > (1 << 24)
        ):
            flat_e = e.reshape(-1).contiguous()
            flat_t = torch.sort(t.reshape(-1))[0]
            if flat_t.numel() == 0:
                found = torch.zeros_like(flat_e, dtype=torch.bool)
            else:
                # searchsorted returns the insertion point; clamping it keeps the
                # gather in range for elements past the largest test value, which
                # then simply compare unequal.
                idx = torch.searchsorted(flat_t, flat_e).clamp_(max=flat_t.numel() - 1)
                found = flat_t[idx] == flat_e
            found = found.view(e.shape)
            return ~found if invert else found
        return original(e, t, assume_unique=assume_unique, invert=invert)

    isin._hy3d_patched = True
    torch.isin = isin
    return True


def _patch_sdp_kernel() -> bool:
    """Neutralise `torch.backends.cuda.sdp_kernel(...)` so SDPA can pick freely.

    Replacing it with a null context leaves every backend enabled, which is what
    we want: PyTorch then chooses flash or mem-efficient where they apply and
    falls back to math where they do not, instead of erroring out because the
    only permitted backends cannot handle the shapes.

    Also silences the deprecation warning recent torch emits for this API.
    """
    backends_cuda = getattr(torch.backends, "cuda", None)
    if backends_cuda is None or not hasattr(backends_cuda, "sdp_kernel"):
        return False
    if getattr(backends_cuda, "_hy3d_patched", False):
        return True

    def sdp_kernel(*args, **kwargs):
        return contextlib.nullcontext()

    backends_cuda.sdp_kernel = sdp_kernel
    backends_cuda._hy3d_patched = True
    return True


# Cap on the attention score matrix a single call may materialise, in bytes.
#
# This is a speed/memory dial, and the ends are far apart: the paint pipeline's
# multiview attention wants ~67 GiB in one tensor, so a 1 GiB budget means ~68
# sequential chunks per attention call and makes texturing extremely slow, while
# too large a budget reproduces the allocation failure it exists to avoid.
# 4 GiB keeps the chunk count in single digits and still allocates reliably on an
# APU carve-out. Override with HY3D_SDPA_BUDGET_GIB when tuning.
# `sdpa_budget()` below scales this down on machines with a smaller pool.
_SDPA_DEFAULT_BUDGET = 4 << 30


def sdpa_budget(b=None) -> int:
    """Chunk budget for `scaled_dot_product_attention`, in bytes.

    4 GiB is the measured ROCm figure. It is clamped to a quarter of the
    accelerator pool when that is smaller, because the budget is a *peak*: the
    score matrix has to coexist with the UNet weights and the rest of the
    activations. On a 128 GB APU the clamp never binds; on a 16 GB Mac it drops
    the budget to ~3 GiB, which is the difference between chunking that helps and
    chunking that just relocates the allocation failure.
    """
    raw = os.environ.get("HY3D_SDPA_BUDGET_GIB")
    if raw:
        try:
            return max(1, int(float(raw) * (1 << 30)))
        except ValueError:
            print(f"[compat] ignoring invalid HY3D_SDPA_BUDGET_GIB={raw!r}", flush=True)

    budget = _SDPA_DEFAULT_BUDGET
    if b is not None:
        try:
            import backend as backend_mod

            pool = backend_mod.gpu_memory_gb(b)
        except Exception:
            pool = None
        if pool:
            budget = min(budget, max(1 << 30, int(0.25 * pool * (1 << 30))))
    return budget


def _patch_sdpa_chunking(budget: int = _SDPA_DEFAULT_BUDGET) -> bool:
    """Make `scaled_dot_product_attention` compute in query chunks when huge.

    Why this is necessary rather than a nicety: on consumer and APU AMD GPUs
    PyTorch disables the flash and mem-efficient SDPA backends ("still
    experimental"), so attention falls back to the math backend, which
    materialises the full (B*H, Lq, Lk) score matrix. MPS has no flash kernel to
    disable in the first place — its SDPA is an MPSGraph matmul + softmax — so it
    is in the same position, with the extra constraint that Metal rejects a
    single buffer past `maxBufferLength` outright. The paint pipeline's
    multiview reference attention concatenates all views into one sequence, so
    at 6 views and 512px that single tensor is ~67 GiB — it fails to allocate
    even with ~100 GiB free, because no allocator will hand out one block that
    size. Lowering the view resolution shrinks it but the cumulative pressure
    across layers still exhausts memory.

    Slicing the query dimension bounds the peak at `budget` per call while
    computing exactly the same result — softmax is applied per query row, so
    rows are independent and the outputs simply concatenate. This is the same
    idea as diffusers' attention slicing, applied one level lower so it also
    covers upstream's custom attention processors, which bypass diffusers'
    slicing API entirely.

    AOTriton's experimental kernels are the other way out of this, but on
    gfx1151 enabling them segfaults, so this path is what actually works.
    """
    import torch.nn.functional as F

    original = F.scaled_dot_product_attention
    if getattr(original, "_hy3d_patched", False):
        return True

    def scaled_dot_product_attention(query, key, value, attn_mask=None,
                                     dropout_p=0.0, is_causal=False, **kwargs):
        # A causal mask is defined relative to absolute positions, so slicing
        # queries would silently change which keys each row may attend to.
        # Not worth handling: nothing in this pipeline uses it.
        if is_causal or query.dim() < 3:
            return original(query, key, value, attn_mask, dropout_p, is_causal, **kwargs)

        heads = 1
        for d in query.shape[:-2]:
            heads *= d
        len_q, len_k = query.shape[-2], key.shape[-2]
        needed = heads * len_q * len_k * query.element_size()
        if needed <= budget:
            return original(query, key, value, attn_mask, dropout_p, is_causal, **kwargs)

        per_row = max(1, heads * len_k * query.element_size())
        step = max(1, min(len_q, budget // per_row))

        outputs = []
        for start in range(0, len_q, step):
            stop = min(start + step, len_q)
            q_chunk = query[..., start:stop, :]
            mask_chunk = attn_mask
            # Masks may be broadcast over the query axis; only slice when this
            # one actually enumerates queries.
            if attn_mask is not None and attn_mask.dim() >= 2 and attn_mask.shape[-2] == len_q:
                mask_chunk = attn_mask[..., start:stop, :]
            outputs.append(
                original(q_chunk, key, value, mask_chunk, dropout_p, False, **kwargs)
            )
        return torch.cat(outputs, dim=-2)

    scaled_dot_product_attention._hy3d_patched = True
    F.scaled_dot_product_attention = scaled_dot_product_attention
    # Some callers reach it through torch.nn.functional, others through the
    # torch namespace; keep both pointing at the wrapper.
    try:
        torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention
    except Exception:
        pass
    print(
        f"[compat] SDPA will chunk queries when the score matrix would exceed "
        f"{budget / (1 << 30):.1f} GiB",
        flush=True,
    )
    return True
