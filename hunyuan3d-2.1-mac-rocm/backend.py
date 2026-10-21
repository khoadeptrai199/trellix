#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compute-backend detection and policy for the Hunyuan3D 2.1 port.

One module answers every "where does this run and in what precision" question
for all four supported backends:

    rocm  — AMD GPU via PyTorch/ROCm (HIP). Reports itself as `cuda`.
    cuda  — NVIDIA GPU.
    mps   — Apple Silicon via Metal Performance Shaders.
    cpu   — fallback.

Import this BEFORE `torch` anywhere else: the MPS fallback flag has to be in
the environment before the torch extension module initialises.

Why ROCm needs its own `kind` even though its device string is `cuda`:
PyTorch/ROCm makes `torch.cuda.is_available()` return True and names the
device `cuda`, so all of upstream's hard-coded `"cuda"` strings just work.
The compatibility patches in `compat_patches.py` must therefore NOT be applied
on ROCm — patching `torch.autocast(device_type="cuda")` into a no-op there
would silently disable fp16 autocast on a perfectly capable GPU.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ---- Environment must be set before torch loads ----------------------------
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# MPS: lift the soft allocation cap. PyTorch aborts an allocation once the
# process passes `recommended_max_memory()` (about 75 % of unified memory),
# even though macOS would happily page the rest — and the paint pipeline's
# multiview attention crosses that line on any Mac under ~32 GB. The failure is
# an exception mid-bake, minutes in, not a graceful degradation, so the cap buys
# nothing here.
#
# This lives next to the ROCm allocator settings for the same reason they do:
# launch.sh sets it, but tests, `scripts/doctor.py` and anything importing this
# module directly bypass the launcher and would otherwise run under a different
# memory policy than the app.
if sys.platform == "darwin":
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# Allocator and kernel-cache settings for ROCm. These live here, not only in the
# launchers, because anything that imports this module — the app, the tests, the
# doctor — needs them. Test runs that bypassed launch.ps1 were missing
# expandable_segments and hit allocation failures the app would not have.
#
# expandable_segments lets a large tensor grow across several smaller physical
# segments instead of demanding one contiguous block. That matters a great deal
# here: the paint UNet's multiview attention asks for tens of GiB in a single
# allocation, which fails on an APU carve-out even with far more memory free.
#
# setdefault throughout, so the launchers and the user keep the final say.
# PYTORCH_ALLOC_CONF is the current name; PYTORCH_HIP_ALLOC_CONF is the older
# one, still what torch 2.x reads on ROCm. Setting both keeps either version
# happy without a version check.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

# MIOPEN_FIND_MODE is deliberately NOT set. Setting it to FAST looks like a free
# win — it skips convolution autotuning — but it routes MIOpen through
# GetSolutionsFallback, which on this stack hands the solver a null workspace:
#     Warning [IsEnoughWorkspace] Solver <GemmFwdRest>, workspace required:
#     150994944, provided ptr: 0000000000000000 size: 0
# and then dereferences it, crashing the process with an access violation. The
# paint pipeline (which is convolution-heavy) died this way in under two minutes;
# with FAST removed the warnings disappear. MIOpen's own default is NORMAL, so
# the correct action is to leave it alone.
_miopen_cache = Path(__file__).resolve().parent / ".cache" / "miopen"
try:
    _miopen_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MIOPEN_USER_DB_PATH", str(_miopen_cache))
    os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(_miopen_cache))
except OSError:
    pass

if sys.platform == "win32":
    # The Hugging Face cache materialises snapshots as symlinks into blobs/.
    # Creating a symlink on Windows needs SeCreateSymbolicLinkPrivilege, which an
    # ordinary account does not have unless Developer Mode is on — the download
    # then completes and *then* dies with
    # "OSError: [WinError 1314] A required privilege is not held by the client",
    # leaving an unusable half-populated snapshot. Copying instead costs disk but
    # always works. Every entry point imports this module, so setting it here
    # covers the app, the tests and the installers alike.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import torch  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent
BUILD_INFO_PATH = PROJECT_DIR / ".build" / "extensions.json"

KINDS = ("rocm", "cuda", "mps", "cpu")

# Rasterizer backends for the paint pipeline, in descending preference:
#   "gpu"   native custom_rasterizer extension, CUDA/HIP kernel
#   "cpu"   native custom_rasterizer extension, its built-in CPU path
#   "torch" torch_rasterizer.py — pure PyTorch, no compiler needed, runs on
#           whatever device the tensors are on
# "torch" is not a degraded mode. It needs no toolchain, it runs on the GPU, and
# it doubles as the clean-room comparison for ROCm/ROCm#5981 (corrupted textures
# on ROCm) because it shares no code with the HIP kernel.
RASTER_MODES = ("gpu", "cpu", "torch")


@dataclass(frozen=True)
class Backend:
    """Immutable description of the active compute backend."""

    kind: str            # one of KINDS
    device: str          # torch device string: "cuda" | "mps" | "cpu"
    dtype: torch.dtype   # working dtype for model weights
    raster: str          # custom_rasterizer backend: "gpu" | "cpu"
    gpu_name: str        # human-readable accelerator name
    version: str         # HIP / CUDA / torch version string, for the UI

    # --- capability flags used by the rest of the app ---
    @property
    def is_gpu(self) -> bool:
        return self.kind in ("rocm", "cuda", "mps")

    @property
    def is_cuda_like(self) -> bool:
        """True when the device string is literally `cuda` (NVIDIA or ROCm).

        This is the flag that decides whether upstream's hard-coded
        `device="cuda"` needs patching at all.
        """
        return self.kind in ("rocm", "cuda")

    @property
    def needs_device_patches(self) -> bool:
        """True when upstream's `cuda` strings must be redirected."""
        return not self.is_cuda_like

    @property
    def autocast_ok(self) -> bool:
        """MPS autocast is unreliable; CPU fp16 autocast is pointless."""
        return self.is_cuda_like

    @property
    def label(self) -> str:
        """Short backend label for the UI, e.g. `ROCm (HIP 6.4)`."""
        pretty = {"rocm": "ROCm", "cuda": "CUDA", "mps": "MPS", "cpu": "CPU"}[self.kind]
        return f"{pretty} ({self.version})" if self.version else pretty


# ---- Native-extension capability ------------------------------------------
def _extension_raster_capability() -> Optional[str]:
    """Read what `scripts/build_extensions.py` actually built.

    Returns "gpu", "cpu", "torch" (nothing native could be built) or None when
    the builder has never run here. A GPU build of custom_rasterizer still
    contains the CPU path, so a "gpu" build can serve both native modes; a "cpu"
    build cannot serve "gpu".
    """
    try:
        info = json.loads(BUILD_INFO_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = info.get("custom_rasterizer", {}).get("backend")
    return value if value in RASTER_MODES else None


def _default_raster(kind: str) -> str:
    """Pick the rasterizer backend for a freshly detected device.

    Native modes are used only when the builder confirmed it produced them.
    Anything else falls back to the pure-PyTorch rasterizer, which always works
    — notably on Windows, where the ROCm torch wheel cannot link any C++
    extension at all, so neither native mode is reachable.

    MPS deliberately prefers `torch` even when the native CPU path did build.
    The native path is a *CPU* path, so using it on a GPU backend means copying
    positions down and the index map back on every view. Measured on an M4 Pro
    with 79k triangles at 2048x2048: native CPU 165 ms, torch on MPS 152 ms —
    the same work without leaving the device. `HY3D_RASTER=cpu` still selects it,
    which is what makes it useful as a clean-room comparison.
    """
    built = _extension_raster_capability()
    if built == "gpu" and kind in ("rocm", "cuda"):
        return "gpu"
    if built == "cpu" and kind != "mps":
        return "cpu"
    return "torch"


# ---- Detection -------------------------------------------------------------
def detect(
    force_kind: Optional[str] = None,
    force_raster: Optional[str] = None,
    force_dtype: Optional[str] = None,
) -> Backend:
    """Detect the active backend.

    Overridable through the environment, which is what makes the ROCm texture
    corruption bisect (ROCm/ROCm#5981) practical without code edits:

        HY3D_BACKEND=cpu|mps|cuda|rocm
        HY3D_RASTER=gpu|cpu
        HY3D_DTYPE=float16|bfloat16|float32
    """
    force_kind = force_kind or os.environ.get("HY3D_BACKEND") or None
    force_raster = force_raster or os.environ.get("HY3D_RASTER") or None
    force_dtype = force_dtype or os.environ.get("HY3D_DTYPE") or None

    if force_kind is not None and force_kind not in KINDS:
        raise ValueError(f"HY3D_BACKEND must be one of {KINDS}, got {force_kind!r}")
    if force_raster is not None and force_raster not in RASTER_MODES:
        raise ValueError(f"HY3D_RASTER must be one of {RASTER_MODES}, got {force_raster!r}")

    kind = force_kind or _detect_kind()

    device = {"rocm": "cuda", "cuda": "cuda", "mps": "mps", "cpu": "cpu"}[kind]
    dtype = _resolve_dtype(kind, force_dtype)
    raster = force_raster or _default_raster(kind)

    if raster == "gpu" and kind not in ("rocm", "cuda"):
        # The native GPU kernel only exists for CUDA/HIP.
        print(f"[backend] raster=gpu is not available on {kind}; using torch", flush=True)
        raster = "torch"

    return Backend(
        kind=kind,
        device=device,
        dtype=dtype,
        raster=raster,
        gpu_name=_gpu_name(kind),
        version=_version_string(kind),
    )


def _detect_kind() -> str:
    if torch.cuda.is_available():
        return "rocm" if getattr(torch.version, "hip", None) else "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(kind: str, force: Optional[str]) -> torch.dtype:
    if force:
        try:
            dtype = getattr(torch, force)
        except AttributeError:
            raise ValueError(f"HY3D_DTYPE={force!r} is not a torch dtype")
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"HY3D_DTYPE={force!r} is not a torch dtype")
        return dtype
    # The shape checkpoint ships as fp16 (`model.fp16.ckpt`) and the paint
    # pipeline is loaded with torch_dtype=float16, so fp16 is the native
    # choice on every real GPU. CPU has no usable fp16 kernels.
    return torch.float16 if kind in ("rocm", "cuda", "mps") else torch.float32


def _gpu_name(kind: str) -> str:
    try:
        if kind in ("rocm", "cuda"):
            return torch.cuda.get_device_name(0)
        if kind == "mps":
            return f"Apple Silicon ({platform.machine()})"
    except Exception:
        pass
    return platform.processor() or platform.machine() or "CPU"


def _version_string(kind: str) -> str:
    if kind == "rocm":
        return f"HIP {getattr(torch.version, 'hip', '?')}"
    if kind == "cuda":
        return f"CUDA {getattr(torch.version, 'cuda', '?')}"
    return f"torch {torch.__version__}"


# ---- Runtime helpers -------------------------------------------------------
def empty_cache(b: Backend) -> None:
    """Release cached device memory, whatever the backend is."""
    try:
        if b.is_cuda_like:
            torch.cuda.empty_cache()
        elif b.kind == "mps":
            torch.mps.empty_cache()
    except Exception as e:
        print(f"[backend] empty_cache failed: {e}", flush=True)


def gpu_memory_gb(b: Backend) -> Optional[float]:
    """Total accelerator memory in GiB, or None when it cannot be determined.

    On an APU with a small carve-out (Strix Halo defaults to 4 GB) this
    reports the carve-out, not the whole unified pool — which is exactly the
    number the memory preset should key off, since exceeding it spills to GTT.
    """
    try:
        if b.is_cuda_like:
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if b.kind == "mps":
            return torch.mps.recommended_max_memory() / (1024 ** 3)
    except Exception:
        pass
    try:
        import psutil

        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return None


# ---- Memory presets for the paint pipeline --------------------------------
# Upstream defaults (render_size=2048, texture_size=4096, 8 views @768) need
# roughly 21 GB of VRAM. These presets trade texture detail for headroom.
# The "safe" preset is not a guess — 6 views at 256px is the configuration that
# actually completed on a Radeon 8060S (gfx1151, Windows, ROCm 7.2), in 136 s.
# Its attention matrix is 4.2 GiB, against 67.5 GiB for "low" and 607 GiB for
# "normal", which is why it fits where the others cannot. See
# `paint.attention_preflight`.
PAINT_PRESETS: dict[str, dict[str, int]] = {
    "safe": dict(max_num_view=6, resolution=256, render_size=1024, texture_size=2048),
    "low": dict(max_num_view=6, resolution=512, render_size=1024, texture_size=2048),
    "normal": dict(max_num_view=8, resolution=768, render_size=2048, texture_size=4096),
}
PAINT_PRESET_ORDER = ("safe", "low", "normal")


def suggest_paint_preset(b: Backend) -> str:
    """Pick a paint preset for this backend.

    Total memory only decides this on CUDA. Everywhere else the binding
    constraint is the attention matrix, not the pool: without flash attention the
    paint UNet materialises it in full, and it grows with the square of
    views x tokens — so a machine with 100 GiB of GPU-visible memory still cannot
    run the "normal" preset. Suggesting it there would be actively misleading.
    See `paint.attention_preflight`.

    MPS is in the same position as ROCm and for the same reason. PyTorch's MPS
    SDPA builds the score matrix through MPSGraph rather than a flash kernel, so
    "low" already asks for a 67.5 GiB tensor on a machine whose largest possible
    Metal buffer is a fraction of that. `safe` is the honest default.
    """
    if b.kind != "cuda":
        return "safe"
    mem = gpu_memory_gb(b)
    if mem is None:
        return "low"
    return "normal" if mem >= 24.0 else "low"


def describe(b: Backend) -> str:
    """One-line summary for the console banner."""
    mem = gpu_memory_gb(b)
    mem_str = f", {mem:.1f} GiB" if mem else ""
    return (
        f"backend={b.kind} device={b.device} dtype={str(b.dtype).replace('torch.', '')} "
        f"raster={b.raster} gpu={b.gpu_name}{mem_str} [{b.version}]"
    )
