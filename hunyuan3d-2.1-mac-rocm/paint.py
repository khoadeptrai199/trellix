#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PBR texturing (the `hy3dpaint` pipeline) on ROCm, CUDA, MPS or CPU.

Upstream's paint pipeline assumes an NVIDIA GPU, a Blender install and a
compiled rasterizer. This module supplies what is missing, as monkey patches
over the imported upstream modules rather than edits to the clone — the clone
stays a pristine checkout that `git pull` still works on.

What gets patched, and why each one is needed:

  device strings
      `Hunyuan3DPaintConfig.device` is the literal "cuda"; `MeshRender` defaults
      to `device="cuda"` and the pipeline constructs it without passing one;
      `hunyuanpaintpbr/pipeline.py` and `unet/attn_processor.py` contain
      unconditional `.to("cuda")` / `.to("cuda:0")` on the inference path.
      On ROCm and CUDA all of these are already correct — PyTorch/ROCm calls its
      device "cuda" — so they are only rewritten for MPS and CPU.

  rasterizer
      `MeshRender.__init__` does `import custom_rasterizer as cr`. When the
      native extension is unavailable (always on Windows/ROCm, see
      scripts/build_extensions.py) we register `torch_rasterizer` under that
      name instead. When the native *CPU* build is in use on a GPU device, only
      the rasterize call is bounced to CPU while the rest of the render stays on
      the GPU.

  weights location
      `multiviewDiffusionNet` calls `snapshot_download(repo_id=...)`, which
      ignores an already-downloaded local `weights/` tree and fetches ~10 GB
      again. We make an existing local directory win.

  Blender
      `MeshRender.save_mesh` and the pipeline's GLB export go through `bpy`.
      Replaced by `mesh_io`, which needs no Blender.

  super-resolution
      `imageSuperNet` builds `RealESRGANer(half=True, gpu_id=None)`, which picks
      `cuda if available else cpu`. Correct on ROCm; on MPS that lands on CPU
      where fp16 is pathologically slow, so precision and device are chosen per
      backend.
"""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import backend as backend_mod
import mesh_io

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR / "Hunyuan3D-2.1"
PAINT_DIR = REPO_DIR / "hy3dpaint"
WEIGHTS_DIR = PROJECT_DIR / "weights" / "Hunyuan3D-2.1"
PAINT_WEIGHTS = WEIGHTS_DIR / "hunyuan3d-paintpbr-v2-1"
ESRGAN_CKPT = PAINT_DIR / "ckpt" / "RealESRGAN_x4plus.pth"

_pipeline = None
_patched = False
_patched_for: Optional[str] = None
# The pristine upstream MeshRender.raster_rasterize, captured before we wrap it.
_ORIGINAL_RASTER_RASTERIZE = None


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------
def availability() -> tuple[bool, str]:
    """Can texturing run here? Returns (ok, human-readable reason)."""
    if not PAINT_DIR.is_dir():
        return False, f"upstream hy3dpaint not found at {PAINT_DIR}"
    if not (PAINT_DIR / "textureGenPipeline.py").is_file():
        return False, "upstream hy3dpaint is incomplete"
    if not PAINT_WEIGHTS.is_dir():
        return False, (
            "paint weights are missing — download hunyuan3d-paintpbr-v2-1 "
            "(the installer does this unless you passed --shape-only)"
        )

    dr = PAINT_DIR / "DifferentiableRenderer"
    if not (list(dr.glob("mesh_inpaint_processor*.pyd")) + list(dr.glob("mesh_inpaint_processor*.so"))):
        return False, (
            "mesh_inpaint_processor is not built — run scripts/build_extensions.py "
            "(it is a plain pybind11 module and builds on every platform)"
        )
    if not ESRGAN_CKPT.is_file():
        return False, f"RealESRGAN weights are missing at {ESRGAN_CKPT}"
    # The real checkpoint is ~64 MiB. A truncated download is common (the GitHub
    # release asset is flaky) and would otherwise surface much later as an opaque
    # unpickling error from inside RealESRGANer.
    esrgan_mib = ESRGAN_CKPT.stat().st_size / (1024 ** 2)
    if esrgan_mib < 20:
        return False, (
            f"RealESRGAN weights look truncated ({esrgan_mib:.1f} MiB, expected ~64 MiB) "
            f"at {ESRGAN_CKPT} — delete the file and download it again"
        )
    return True, "ready"


# ---------------------------------------------------------------------------
# Attention memory preflight
# ---------------------------------------------------------------------------
# Calibrated, not guessed. Measured on a Radeon 8060S: 6 views at resolution 512
# made the paint UNet request a single 67.50 GiB attention tensor. With
#   N = views * (resolution / 8)**2 = 6 * 64**2 = 24576
# and fp16, one (N, N) matrix is 1.125 GiB, so the effective batch*heads factor
# is 67.50 / 1.125 = 60.
_ATTN_BATCH_HEADS = 60
# Refuse above this. Set from measurement, not taste:
#   6 views @ 256px = 4.22 GiB — completed in 136 s
#   6 views @ 320px = 10.30 GiB — see HY3D_ATTN_LIMIT_GIB if you want to try it
#   6 views @ 512px = 67.50 GiB — no allocator will serve it; the attempt reset
#                                 the GPU driver and rebooted the machine
# An APU shares its memory with the OS, so the failure mode above the limit is a
# system reset rather than an exception. Hence a conservative default that admits
# the configuration known to work, with an explicit env override for exploring.
def _attn_limit_override() -> Optional[int]:
    raw = os.environ.get("HY3D_ATTN_LIMIT_GIB")
    if not raw:
        return None
    try:
        return max(1, int(float(raw) * (1 << 30)))
    except ValueError:
        print(f"[paint] ignoring invalid HY3D_ATTN_LIMIT_GIB={raw!r}", flush=True)
        return None


_ATTN_SAFE_LIMIT = 5 << 30


def attention_limit(b) -> int:
    """Largest attention score matrix this backend may be asked for, in bytes.

    ROCm and CPU keep the measured 5 GiB (see above). MPS gets its own number
    because its constraint is different in kind: unified memory means the pool
    is large, but a single Metal buffer cannot exceed the device's
    `maxBufferLength`, and PyTorch's MPS SDPA materialises the score matrix as
    one buffer. Scaling with the pool is therefore the right shape — 40 % of the
    recommended working set leaves room for the UNet weights and activations
    that have to coexist with it — with the ROCm figure as the floor so a small
    Mac is never *stricter* than the configuration already proven to work.
    """
    override = _attn_limit_override()
    if override is not None:
        return override
    if b is not None and getattr(b, "kind", None) == "mps":
        try:
            pool = backend_mod.gpu_memory_gb(b)
        except Exception:
            pool = None
        if pool:
            return max(_ATTN_SAFE_LIMIT, int(0.40 * pool * (1 << 30)))
    return _ATTN_SAFE_LIMIT


def estimate_attention_bytes(views: int, resolution: int) -> int:
    """Bytes in the largest single attention score matrix the paint UNet will ask for.

    The multiview reference attention concatenates every view into one sequence,
    so this grows with the *square* of views x tokens — which is why lowering
    either setting helps so much more than it looks like it should.
    """
    tokens = views * (max(64, int(resolution)) // 8) ** 2
    return _ATTN_BATCH_HEADS * tokens * tokens * 2


def attention_preflight(b, views: int, resolution: int) -> tuple[bool, str]:
    """Decide whether texturing may proceed on this backend at these settings.

    Without flash or mem-efficient SDPA, PyTorch materialises the full attention
    matrix. On a discrete NVIDIA card that is merely wasteful; on an AMD APU,
    where GPU memory is system memory, pushing tens of GiB through it crashed the
    display driver and rebooted the machine during testing. So this refuses by
    default instead of trying and hoping.

    MPS lands between the two and gets its own message: Apple Silicon is also
    unified memory, but an over-large tensor there raises
    `RuntimeError: Invalid buffer size` — measured on an M4 Pro, where 12 GiB in
    one buffer allocated and 24 GiB did not — so the cost of guessing wrong is an
    exception and a very slow run, not a reboot. The limit still exists because
    `compat_patches` has to slice the matrix into ever more chunks to stay under
    it, and past a point that is hours of work rather than minutes.

    Set HY3D_ALLOW_RISKY_TEXTURING=1 to override.
    """
    if b.kind == "cuda":
        return True, "flash attention available"
    if os.environ.get("HY3D_ALLOW_RISKY_TEXTURING") == "1":
        return True, "risky texturing explicitly allowed"

    limit = attention_limit(b)
    need = estimate_attention_bytes(views, resolution)
    if need <= limit:
        return True, (
            f"estimated attention matrix {need / (1 << 30):.1f} GiB "
            f"(limit {limit / (1 << 30):.1f} GiB) — within limits"
        )

    if b.kind == "mps":
        import compat_patches

        chunks = max(1, -(-need // compat_patches.sdpa_budget(b)))
        return False, (
            f"texturing at {views} views / {resolution}px would need a single "
            f"{need / (1 << 30):.1f} GiB attention matrix, against a limit of "
            f"{limit / (1 << 30):.1f} GiB on this Mac.\n\n"
            f"PyTorch's MPS backend has no flash attention: `scaled_dot_product_"
            f"attention` builds the score matrix through MPSGraph, and Metal "
            f"refuses a single buffer that large. The chunked SDPA path in "
            f"`compat_patches` can still compute it exactly, but it would take "
            f"about {chunks} sequential passes per attention call, which turns a "
            f"few-minute bake into a very long one and pushes the machine into "
            f"swap.\n\n"
            f"Options: use the «Safe» preset (6 views / 256 px), which fits; or "
            f"set HY3D_ALLOW_RISKY_TEXTURING=1 and accept the wait. Shape "
            f"generation is unaffected."
        )

    if os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") == "1":
        return False, (
            f"estimated attention matrix is {need / (1 << 30):.1f} GiB and AOTriton "
            f"experimental kernels are enabled. On gfx1151 that combination "
            f"segfaulted during testing. Unset "
            f"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL and lower the view count or "
            f"resolution."
        )

    return False, (
        f"texturing at {views} views / {resolution}px would need a single "
        f"{need / (1 << 30):.1f} GiB attention matrix on this backend.\n\n"
        f"PyTorch disables flash and mem-efficient attention on this AMD GPU "
        f"(\"still experimental\"), so attention falls back to the math backend, "
        f"which materialises that matrix in full. During testing this reset the "
        f"GPU driver and rebooted the machine, so it is refused rather than "
        f"attempted.\n\n"
        f"Options: run texturing on Linux/ROCm or an NVIDIA GPU, where flash "
        f"attention works; or set HY3D_ALLOW_RISKY_TEXTURING=1 if you accept the "
        f"risk of a machine reset. Shape generation is unaffected and works fine."
    )


def _ensure_paths() -> None:
    """Put the upstream packages on sys.path the way upstream expects.

    `textureGenPipeline` imports `DifferentiableRenderer.MeshRender` and
    `utils.multiview_utils` as top-level packages, so `hy3dpaint` itself must be
    on the path, not just the repo root.
    """
    for p in (str(PAINT_DIR), str(REPO_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Patches
# ---------------------------------------------------------------------------
def apply_patches(b) -> None:
    """Install every compatibility patch the paint pipeline needs.

    Idempotent per backend configuration; re-applies if the rasterizer mode
    changed (which is how the UI's rasterizer selector takes effect).
    """
    global _patched, _patched_for
    signature = f"{b.kind}:{b.device}:{b.raster}"
    if _patched and _patched_for == signature:
        return

    _ensure_paths()

    # The torch-level patches are not optional for this pipeline, and asking
    # every caller to remember them is how `scripts/doctor.py` and the tests
    # ended up running the bake under a different torch than the app does. It is
    # idempotent, so calling it again when gradio_app already did costs nothing.
    # On MPS the `torch.isin` patch in particular is load-bearing: without it
    # `MeshRender.back_project` kills the process outright.
    import compat_patches

    compat_patches.apply(b)

    # Order matters: the bpy stub has to exist before anything imports
    # mesh_utils, which MeshRender does at import time.
    mesh_io.install_bpy_stub()
    _patch_rasterizer_module(b)
    _patch_snapshot_download()

    _patch_config(b)
    _patch_remesh()
    _patch_mesh_render(b)
    _patch_super_resolution(b)
    if b.kind == "mps":
        _patch_texture_conversion()
    if b.needs_device_patches:
        _patch_hardcoded_cuda(b)

    _patched = True
    _patched_for = signature
    print(f"[paint] patches applied for {signature}", flush=True)


def _patch_rasterizer_module(b) -> None:
    """Decide what `import custom_rasterizer` resolves to."""
    if b.raster == "torch":
        import torch_rasterizer

        sys.modules["custom_rasterizer"] = torch_rasterizer
        print("[paint] rasterizer: torch_rasterizer (pure PyTorch, no extension)", flush=True)
        return

    try:
        import custom_rasterizer  # noqa: F401
    except ImportError as e:
        import torch_rasterizer

        sys.modules["custom_rasterizer"] = torch_rasterizer
        print(
            f"[paint] rasterizer: native extension unavailable ({e}); "
            f"falling back to torch_rasterizer",
            flush=True,
        )
        return
    print(f"[paint] rasterizer: native custom_rasterizer ({b.raster})", flush=True)


def _patch_snapshot_download() -> None:
    """Let an existing local weights directory satisfy `snapshot_download`.

    `multiviewDiffusionNet` hard-codes
    `snapshot_download(repo_id=config.multiview_pretrained_path, ...)`. Pointing
    `multiview_pretrained_path` at our local `weights/` tree is not enough on its
    own, because snapshot_download would treat it as a repo id and re-fetch.
    """
    import huggingface_hub

    if getattr(huggingface_hub.snapshot_download, "_hy3d_patched", False):
        return
    original = huggingface_hub.snapshot_download

    def snapshot_download(repo_id=None, *args, **kwargs):
        if repo_id and os.path.isdir(str(repo_id)):
            print(f"[paint] using local weights: {repo_id}", flush=True)
            return str(repo_id)
        return original(repo_id, *args, **kwargs)

    snapshot_download._hy3d_patched = True
    huggingface_hub.snapshot_download = snapshot_download
    # multiview_utils does `import huggingface_hub` and calls through the module,
    # so patching the attribute is enough — but utils may already be imported.
    try:
        from utils import multiview_utils

        multiview_utils.huggingface_hub.snapshot_download = snapshot_download
    except Exception:
        pass


def _patch_config(b) -> None:
    """Point the config at this machine: device, absolute paths, local weights."""
    import textureGenPipeline as tgp

    original_init = tgp.Hunyuan3DPaintConfig.__init__
    if getattr(original_init, "_hy3d_patched", False):
        return

    def __init__(self, max_num_view, resolution):
        original_init(self, max_num_view, resolution)
        self.device = b.device
        # Upstream's defaults are relative to the repo root, which only works if
        # the process happens to be started from there.
        self.multiview_cfg_path = str(PAINT_DIR / "cfgs" / "hunyuan-paint-pbr.yaml")
        self.custom_pipeline = str(PAINT_DIR / "hunyuanpaintpbr")
        self.realesrgan_ckpt_path = str(ESRGAN_CKPT)
        if WEIGHTS_DIR.is_dir():
            self.multiview_pretrained_path = str(WEIGHTS_DIR)

    __init__._hy3d_patched = True
    tgp.Hunyuan3DPaintConfig.__init__ = __init__


def _patch_remesh() -> None:
    """Fix the decimation target in upstream's remesh step.

    `mesh_simplify_trimesh` calls `simplify_quadric_decimation(target_count)`
    positionally. Upstream pins trimesh 4.4.7; in current trimesh the first
    positional parameter is `percent`, so 40000 is read as "4,000,000 percent"
    and the mesh is not decimated at all. The paint pipeline then bakes against
    a mesh of the original density, which both wastes time and degrades the bake.

    Passing the count by keyword is version-proof: `face_count` exists in both
    the old and new signatures.
    """
    import inspect

    import trimesh
    from utils import simplify_mesh_utils as smu

    if getattr(smu.mesh_simplify_trimesh, "_hy3d_patched", False):
        return

    try:
        params = inspect.signature(trimesh.Trimesh.simplify_quadric_decimation).parameters
    except (TypeError, ValueError):
        params = {}
    use_kw = "face_count" in params

    def mesh_simplify_trimesh(inputpath, outputpath, target_count=40000):
        import pymeshlab

        ms = pymeshlab.MeshSet()
        if str(inputpath).endswith(".glb"):
            ms.load_new_mesh(str(inputpath), load_in_a_single_layer=True)
        else:
            ms.load_new_mesh(str(inputpath))
        obj_out = str(outputpath).replace(".glb", ".obj")
        ms.save_current_mesh(obj_out, save_textures=False)

        mesh = trimesh.load(obj_out, force="mesh")
        if mesh.faces.shape[0] > target_count:
            if use_kw:
                mesh = mesh.simplify_quadric_decimation(face_count=target_count)
            else:
                mesh = mesh.simplify_quadric_decimation(target_count)
        mesh.export(str(outputpath))

    def remesh_mesh(mesh_path, remesh_path):
        mesh_simplify_trimesh(mesh_path, remesh_path)

    mesh_simplify_trimesh._hy3d_patched = True
    smu.mesh_simplify_trimesh = mesh_simplify_trimesh
    smu.remesh_mesh = remesh_mesh
    # textureGenPipeline imported remesh_mesh by value at module import time, so
    # rebinding it on the source module alone would have no effect.
    try:
        import textureGenPipeline as tgp

        tgp.remesh_mesh = remesh_mesh
    except Exception:
        pass
    print(
        f"[paint] remesh decimation target passed by "
        f"{'keyword' if use_kw else 'position'} (trimesh compatibility)",
        flush=True,
    )


def _patch_mesh_render(b) -> None:
    """Device injection, the CPU rasterize bounce, and Blender-free saving."""
    import torch
    import textureGenPipeline as tgp
    from DifferentiableRenderer.MeshRender import MeshRender

    # 1) The pipeline builds MeshRender without a device, so it defaults to
    #    "cuda". Inject the real one.
    original_pipe_init = tgp.Hunyuan3DPaintPipeline.__init__
    if not getattr(original_pipe_init, "_hy3d_patched", False):

        def pipe_init(self, config=None):
            cfg = config if config is not None else tgp.Hunyuan3DPaintConfig(6, 512)
            self.config = cfg
            self.models = {}
            self.stats_logs = {}
            self.render = MeshRender(
                default_resolution=cfg.render_size,
                texture_size=cfg.texture_size,
                bake_mode=cfg.bake_mode,
                raster_mode=cfg.raster_mode,
                device=cfg.device,
            )
            from utils.pipeline_utils import ViewProcessor

            self.view_processor = ViewProcessor(cfg, self.render)
            self.load_models()

        pipe_init._hy3d_patched = True
        tgp.Hunyuan3DPaintPipeline.__init__ = pipe_init

    # 2+3) Rebuild raster_rasterize from the pristine upstream method every time,
    #      so switching rasterizer mode at runtime replaces the wrapper instead of
    #      stacking another one on top of it.
    global _ORIGINAL_RASTER_RASTERIZE
    if _ORIGINAL_RASTER_RASTERIZE is None:
        _ORIGINAL_RASTER_RASTERIZE = MeshRender.raster_rasterize
    pristine = _ORIGINAL_RASTER_RASTERIZE

    # Native CPU rasterizer on a GPU device: bounce only the rasterize call.
    # Everything else in the render — UV interpolation at render_size, baking at
    # texture_size — stays on the GPU.
    bounce = b.raster == "cpu" and b.device != "cpu"

    def raster_rasterize(self, pos, tri, resolution, ranges=None, grad_db=True):
        # Upstream normalises float64 and int64 but not half, and both the native
        # CPU path and torch_rasterizer want fp32 vertices.
        if pos.dtype in (torch.float16, torch.bfloat16):
            pos = pos.float()
        if not bounce:
            return pristine(self, pos, tri, resolution, ranges, grad_db)
        device = pos.device
        out, db = pristine(self, pos.detach().cpu(), tri.detach().cpu(), resolution, ranges, grad_db)
        return out.to(device), (db.to(device) if db is not None else None)

    MeshRender.raster_rasterize = raster_rasterize
    if bounce:
        print("[paint] rasterize bounced to CPU; the rest of the render stays on GPU", flush=True)

    # 4) Blender-free save. Upstream's MeshRender.save_mesh calls mesh_utils'
    #    save_mesh, and the pipeline then calls bpy-based convert_obj_to_glb.
    #    We write both OBJ and GLB here and stash the paths for the caller.
    if not getattr(MeshRender.save_mesh, "_hy3d_patched", False):

        def save_mesh(self, mesh_path, downsample=False):
            obj_path = os.path.splitext(str(mesh_path))[0] + ".obj"
            glb_path = os.path.splitext(str(mesh_path))[0] + ".glb"
            self._hy3d_saved = mesh_io.save_textured_mesh(
                self, obj_path, glb_path=glb_path, downsample=downsample
            )
            return self._hy3d_saved

        save_mesh._hy3d_patched = True
        MeshRender.save_mesh = save_mesh


def _patch_texture_conversion() -> None:
    """Stop `_convert_texture_format` from putting a float64 tensor on MPS.

    `MeshRender._convert_texture_format` (MeshRender.py:277) ends its non-forced
    branch with

        tex = np.array(tex) / 255.0          # float64
        return torch.from_numpy(tex).to(device).float()

    — it moves to the device *first* and narrows afterwards. Metal has no
    float64 at all, so on MPS that raises

        TypeError: Cannot convert a MPS Tensor to float64 dtype

    and it is not a corner case: the paint pipeline reaches it on every run
    through `set_texture_mr(texture_mr)` (textureGenPipeline.py:184), right at
    the end, after the whole bake has been paid for.

    Narrowing before the transfer is the same computation in the other order,
    so the result is identical on every backend; only MPS cares. ROCm and CUDA
    keep upstream's version, per this port's rule of not patching what works.
    """
    import numpy as np
    import torch
    from PIL import Image
    import DifferentiableRenderer.MeshRender as mr_mod

    if getattr(mr_mod._convert_texture_format, "_hy3d_patched", False):
        return

    def _convert_texture_format(tex, texture_size, device, force_set: bool = False):
        if not force_set:
            if isinstance(tex, np.ndarray):
                tex = Image.fromarray((tex * 255).astype(np.uint8))
            elif isinstance(tex, torch.Tensor):
                tex = Image.fromarray((tex.detach().cpu().numpy() * 255).astype(np.uint8))
            tex = tex.resize(texture_size).convert("RGB")
            return torch.from_numpy(np.array(tex, dtype=np.float32) / 255.0).to(device)
        if isinstance(tex, np.ndarray):
            tex = torch.from_numpy(tex)
        return tex.float().to(device)

    _convert_texture_format._hy3d_patched = True
    mr_mod._convert_texture_format = _convert_texture_format
    print("[paint] texture conversion narrows to fp32 before the device copy (MPS has no fp64)",
          flush=True)


def _patch_super_resolution(b) -> None:
    """Build RealESRGAN with a device and precision that suit this backend."""
    import torch
    import utils.image_super_utils as isu

    if getattr(isu.imageSuperNet, "_hy3d_patched", False):
        return

    # basicsr imports torchvision.transforms.functional_tensor, removed in
    # modern torchvision. Upstream ships a shim; apply it before the import.
    try:
        from torchvision_fix import apply_fix

        apply_fix()
    except Exception as e:
        print(f"[paint] torchvision_fix unavailable ({e}); basicsr may fail to import", flush=True)

    def __init__(self, config):
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                        num_grow_ch=32, scale=4)
        if b.is_cuda_like:
            device, half = None, True          # let RealESRGANer pick cuda; fp16 is fine
        elif b.kind == "mps":
            # RealESRGANer would fall back to CPU here, where fp16 is far slower
            # than fp32. Run it on MPS in fp32 instead.
            device, half = torch.device("mps"), False
        else:
            device, half = torch.device("cpu"), False

        self.upsampler = RealESRGANer(
            scale=4,
            model_path=config.realesrgan_ckpt_path,
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=half,
            device=device,
            gpu_id=None,
        )

    __init__._hy3d_patched = True
    isu.imageSuperNet.__init__ = __init__
    isu.imageSuperNet._hy3d_patched = True


def _patch_hardcoded_cuda(b) -> None:
    """Redirect the literal `.to("cuda")` calls on the inference path. MPS/CPU only.

    Two call sites need it:
      * `hunyuanpaintpbr/pipeline.py:245` — `.to("cuda")` inside
        `convert_pil_list_to_tensor`, a closure defined *inside* a 300-line
        `__call__`, so it cannot be replaced without vendoring the whole method.
      * `unet/attn_processor.py:750` — `.to("cuda:0")`, unconditional in
        `SelfAttnProcessor2_0.__call__`.

    Because neither can be reached individually, the redirect is installed on
    `Tensor.to` / `Module.to` and narrowed to the string forms "cuda"/"cuda:N".
    That adds a Python-level wrapper to a hot method, which is a real cost — so
    it is only installed when the device is genuinely not CUDA-like. On ROCm and
    CUDA this function never runs, and those paths keep the native `.to`.
    """
    import torch
    import torch.nn as nn

    def _redirect(original):
        def wrapper(self, *args, **kwargs):
            if args and isinstance(args[0], str) and args[0].startswith("cuda"):
                args = (b.device,) + args[1:]
            elif isinstance(kwargs.get("device"), str) and kwargs["device"].startswith("cuda"):
                kwargs = dict(kwargs, device=b.device)
            return original(self, *args, **kwargs)

        wrapper._hy3d_patched = True
        return wrapper

    if not getattr(torch.Tensor.to, "_hy3d_patched", False):
        torch.Tensor.to = _redirect(torch.Tensor.to)
        print(f"[paint] Tensor.to('cuda') redirected to '{b.device}'", flush=True)
    if not getattr(nn.Module.to, "_hy3d_patched", False):
        nn.Module.to = _redirect(nn.Module.to)
        print(f"[paint] Module.to('cuda') redirected to '{b.device}'", flush=True)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def resolve_params(preset: str = "low", overrides: Optional[dict] = None) -> dict:
    """Memory preset plus any explicit per-run overrides."""
    params = dict(backend_mod.PAINT_PRESETS.get(preset, backend_mod.PAINT_PRESETS["low"]))
    for key, value in (overrides or {}).items():
        if key in params and value:
            params[key] = int(value)
    return params


def load_pipeline(b, preset: str = "low", overrides: Optional[dict] = None):
    """Lazily build the paint pipeline for the given memory preset."""
    global _pipeline
    params = resolve_params(preset, overrides)
    key = (b.kind, b.raster, tuple(sorted(params.items())))

    if _pipeline is not None and getattr(_pipeline, "_hy3d_key", None) == key:
        return _pipeline
    if _pipeline is not None:
        # Any change to view count or resolution changes the model's input
        # shapes, so the pipeline is rebuilt rather than reconfigured.
        unload()

    apply_patches(b)
    import textureGenPipeline as tgp

    cfg = tgp.Hunyuan3DPaintConfig(params["max_num_view"], params["resolution"])
    cfg.render_size = params["render_size"]
    cfg.texture_size = params["texture_size"]

    print(
        f"[paint] loading pipeline: preset={preset} views={params['max_num_view']} "
        f"resolution={params['resolution']} render={cfg.render_size} texture={cfg.texture_size}",
        flush=True,
    )
    pipeline = tgp.Hunyuan3DPaintPipeline(cfg)
    pipeline._hy3d_key = key
    _pipeline = pipeline
    return _pipeline


def unload() -> None:
    """Drop the paint pipeline and release device memory."""
    global _pipeline
    if _pipeline is None:
        return
    _pipeline = None
    gc.collect()
    try:
        import backend as bm

        bm.empty_cache(bm.detect())
    except Exception:
        pass


def texture_mesh(
    mesh_path: str,
    image,
    b,
    preset: str = "low",
    overrides: Optional[dict] = None,
    use_remesh: bool = True,
    output_dir: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    """Texture a mesh on disk. Returns paths and the generated PBR maps.

    Args:
        mesh_path: geometry to texture (.obj or .glb).
        image: the conditioning image (PIL Image or path).
        b: the active Backend.
        preset: memory preset name from backend.PAINT_PRESETS.
        use_remesh: let the paint pipeline re-mesh to ~40k triangles. Leave on
            unless the input is already in that range — the bake quality depends
            on reasonably uniform triangles.
        output_dir: where to write; defaults to the mesh's directory.

    Returns:
        {"obj": path, "glb": path|None, "maps": {...}, "elapsed": seconds}
    """
    say = log or (lambda m: print(f"[paint] {m}", flush=True))

    ok, reason = availability()
    if not ok:
        raise RuntimeError(reason)

    params = resolve_params(preset, overrides)
    safe, why = attention_preflight(b, params["max_num_view"], params["resolution"])
    say(f"attention preflight: {why}")
    if not safe:
        raise RuntimeError(why)

    t0 = time.time()
    pipe = load_pipeline(b, preset, overrides)

    out_dir = Path(output_dir) if output_dir else Path(mesh_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_obj = out_dir / (Path(mesh_path).stem + "_textured.obj")

    say(f"texturing {mesh_path} -> {out_obj}")
    pipe(
        mesh_path=str(mesh_path),
        image_path=image,
        output_mesh_path=str(out_obj),
        use_remesh=use_remesh,
        # Upstream's GLB export goes through Blender; ours runs inside the
        # patched MeshRender.save_mesh instead.
        save_glb=False,
    )

    saved = getattr(pipe.render, "_hy3d_saved", None)
    if not saved:
        raise RuntimeError(
            "the paint pipeline finished but produced no files — "
            "MeshRender.save_mesh patch did not run"
        )
    saved = dict(saved)
    saved["elapsed"] = time.time() - t0
    say(f"done in {saved['elapsed']:.1f} s: {saved.get('glb') or saved['obj']}")
    return saved
