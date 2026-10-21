#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hunyuan3D 2.1 — web interface for Apple Silicon.

Run:
    ./launch.sh   (or)   python gradio_app.py

This UI wraps the Hunyuan3D 2.1 shape pipeline (DiT + VAE) and the PBR paint
pipeline, on whichever backend is available: ROCm (AMD), CUDA (NVIDIA), MPS
(Apple Silicon) or CPU. See `backend.py` for the detection and policy rules.

Console logs are always in English. UI is localised to English (default),
Chinese and Russian via the language radio at the top of the page.
"""
from __future__ import annotations

import os
import sys
import gc
import time
from pathlib import Path
from typing import Optional

# ==== Backend detection =====================================================
# `backend` has to be imported before torch: it puts PYTORCH_ENABLE_MPS_FALLBACK
# and TOKENIZERS_PARALLELISM into the environment, which only takes effect if
# it happens before the torch extension module initialises.
import backend as backend_mod  # noqa: E402
import compat_patches  # noqa: E402

import contextlib
import traceback
import numpy as np
import torch
import trimesh
import gradio as gr
from PIL import Image

# Lowpoly optimization algorithms (quadric / iso / planar / hybrid / auto)
# live in a standalone module that can be used from the debug CLI too.
import lowpoly  # noqa: E402
from image_parts import split_parts

# One detection, one set of patches, for the whole process.
# On ROCm/CUDA `compat_patches.apply` is a deliberate no-op — see its docstring.
BACKEND = backend_mod.detect()
compat_patches.apply(BACKEND)
print(f"[init] {backend_mod.describe(BACKEND)}", flush=True)

# ---- Device usage tracker (GPU vs CPU) ------------------------------------
# Wraps torch operations through __torch_function__, counts which device
# each resulting tensor ends up on. Answers the question "how much actually
# ran on the GPU vs how much fell back to CPU".
try:
    from torch.overrides import TorchFunctionMode as _TFMode
    _HAS_TFMODE = True
except Exception:
    _HAS_TFMODE = False

if _HAS_TFMODE:
    class DeviceUsageTracker(_TFMode):
        """Counts torch operations by output tensor device."""
        def __init__(self) -> None:
            super().__init__()
            self.counts: dict[str, int] = {}
            # per-op per-device counts — used to find the ops that leak to CPU
            self.op_per_dev: dict[str, dict[str, int]] = {}

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            result = func(*args, **kwargs)
            devices: set[str] = set()
            def _walk(x):
                if isinstance(x, torch.Tensor):
                    try:
                        devices.add(str(x.device).split(":")[0])
                    except Exception:
                        pass
                elif isinstance(x, (list, tuple)):
                    for y in x:
                        _walk(y)
            _walk(result)
            op_name = getattr(func, "__name__", None) or str(func)
            for d in devices:
                self.counts[d] = self.counts.get(d, 0) + 1
                bucket = self.op_per_dev.setdefault(op_name, {})
                bucket[d] = bucket.get(d, 0) + 1
            return result

        def summary_text(self, lang: str = "en") -> str:
            total = sum(self.counts.values())
            if total == 0:
                return t(lang, "stats_none")
            lines = [t(lang, "stats_total", total=f"{total:,}".replace(",", " "))]
            for dev in sorted(self.counts, key=lambda d: -self.counts[d]):
                cnt = self.counts[dev]
                pct = 100.0 * cnt / total
                lines.append(t(
                    lang, "stats_device_line",
                    dev=dev,
                    cnt=f"{cnt:,}".replace(",", " "),
                    pct=pct,
                ))
            # Top-10 ops that bleed to CPU
            cpu_ops = []
            for op, dc in self.op_per_dev.items():
                cpu_cnt = dc.get("cpu", 0)
                if cpu_cnt > 0:
                    total_op = sum(dc.values())
                    cpu_ops.append((op, cpu_cnt, total_op))
            cpu_ops.sort(key=lambda x: -x[1])
            if cpu_ops:
                lines.append("")
                lines.append(t(lang, "stats_cpu_header"))
                for op, cpu_cnt, total_op in cpu_ops[:10]:
                    pct_op = 100.0 * cpu_cnt / total_op if total_op else 0.0
                    lines.append(t(
                        lang, "stats_cpu_line",
                        op=op,
                        cnt=f"{cpu_cnt:,}".replace(",", " "),
                        pct=pct_op,
                    ))
            return "\n".join(lines)
else:
    class DeviceUsageTracker:  # type: ignore
        """Fallback when torch.overrides.TorchFunctionMode is unavailable."""
        def __init__(self) -> None:
            self.counts = {}
            self.op_per_dev = {}
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def summary_text(self, lang: str = "en") -> str:
            return t(lang, "stats_disabled")

# ---- Device selection ------------------------------------------------------
# Kept as module-level names because the rest of this file (and the i18n
# substitutions) read them directly. The single source of truth is BACKEND.
DEVICE = BACKEND.device
DTYPE = BACKEND.dtype

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR    = PROJECT_DIR / "Hunyuan3D-2.1"
WEIGHTS_DIR = PROJECT_DIR / "weights" / "Hunyuan3D-2.1"
OUTPUT_DIR  = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add Hunyuan3D code to PYTHONPATH.
# Real layout: Hunyuan3D-2.1/hy3dshape/hy3dshape/pipelines.py (double-nested).
# We add both the root (for top-level utilities) and the subpackage dirs.
if REPO_DIR.exists():
    sys.path.insert(0, str(REPO_DIR))
    for sub in ("hy3dshape", "hy3dpaint"):
        sub_path = REPO_DIR / sub
        if sub_path.exists():
            sys.path.insert(0, str(sub_path))

# ==== Internationalization ==================================================
# English is the default and the source-of-truth keyset. Zh/Ru are overlays;
# if a key is missing in zh/ru we silently fall back to English.
# All console prints (stdout/stderr) stay in English regardless of UI language.
LANG_CODES = ("en", "zh", "ru")
LANG_DISPLAY = {"en": "English", "zh": "中文", "ru": "Русский"}
LANG_BY_DISPLAY = {v: k for k, v in LANG_DISPLAY.items()}

I18N: dict[str, dict[str, str]] = {
    "en": {
        # ----- static UI strings -----
        "parts_header": "Split 2D image into parts",
        "parts_help": ("Upload the image below, or use the main Input image further down the page. "
                       "Choose fast connected regions or SAM AI candidate regions for touching pieces. "
                       "SAM candidates may overlap and need review. "
                       "For touching parts, open Advanced color mask and provide a same-size PNG "
                       "with a solid color for each part and black background. "
                       "Outputs are 2D crops, not separated 3D meshes."),
        "parts_source": "Image to split (upload here)",
        "parts_method": "Split method",
        "parts_mask_advanced": "Advanced: color mask for touching parts",
        "parts_mask": "Color label mask (not the original photo)",
        "parts_min": "Minimum area per part (pixels)",
        "parts_btn": "Split image",
        "parts_gallery": "2D parts",
        "parts_zip": "Download all parts (ZIP)",
        "parts_choice": "Part to use for 3D generation",
        "parts_use": "Use selected part as input image",
        "parts_status": "Part splitting status",
        "lang_label": "Interface language",
        "description": (
            "# Hunyuan3D 2.1 — local interface\n\n"
            "Generate a 3D mesh from a single image using Tencent Hunyuan3D 2.1.\n\n"
            "**Backend:** `{backend}` • **GPU:** {gpu} • **dtype:** `{dtype}` • "
            "**rasterizer:** `{raster}`\n\n"
            "**Project:** `{project}`\n\n"
            "> Shape generation runs on the GPU. Texturing (PBR) is available "
            "below once the mesh is ready — it needs the native extensions to "
            "be built and the paint weights downloaded. Some operations may "
            "fall back to CPU automatically (tick «Measure GPU vs CPU op share» "
            "to see the exact breakdown)."
        ),
        "input_image":    "Input image (object on a plain background)",
        "remove_bg":      "Automatically remove background (rembg)",
        "track_gpu":      "Measure GPU vs CPU op share",
        "gen_params":     "Generation parameters",
        "num_steps":      "Diffusion steps (more = higher quality, slower)",
        "guidance":       "Guidance scale (classifier strength)",
        "octree":         "Octree resolution (mesh detail)",
        "seed":           "Seed (random seed)",
        "generate_btn":   "🚀 Generate 3D object",
        "log_label":      "Execution log",
        "preview_label":  "Result (preview)",
        "gpu_stats_initial": "_Op tracker is off. Tick the checkbox on the left to measure the GPU vs CPU share._",
        "gpu_stats_header":  "### GPU vs CPU (operations during generation)\n",
        "gpu_stats_off_note": "_Tracking was requested but `torch.overrides.TorchFunctionMode` is unavailable in this PyTorch version._",
        "postproc_header":  "### Post-processing (mesh decimation)",
        "opt_level_label":  "Optimization level (always applied to the original mesh)",
        "opt_mode_label":   "Algorithm",
        "opt_mode_help": (
            "- **Quadric** — universal, uniform reduction; best overall quality/size ratio.\n"
            "- **Iso** — uniform-edge remeshing with feature-edge preservation (CAD-like).\n"
            "- **Planar** — consolidates large flat regions into few big triangles "
            "(door panels, walls, boxes). Keeps curved regions as-is.\n"
            "- **Hybrid** — planar first, then quadric. Fewer triangles, keeps flat look.\n"
            "- **Auto** — planar for hard-surface meshes, quadric for organic ones.\n"
        ),
        "opt_btn":          "🛠️ Apply optimization",
        "opt_log":          "Post-processing log",
        "download_header":  "### Download mesh",
        "dl_format":        "Output format",
        "dl_btn":           "💾 Save and download",
        "dl_file":          "File to download",
        "tips": (
            "### Tips\n"
            "- Quick test: `octree=128`, `steps=30`\n"
            "- Maximum quality: `octree=320+`, `steps=80+` (slow!)\n"
            "- If the GPU runs out of memory — close other apps or reduce octree\n"
            "- «Measure GPU vs CPU op share» routes every tensor op through Python, "
            "which slows generation down by roughly an order of magnitude. Use it to "
            "diagnose, not for normal runs\n"
            "- For CAD use: start with «Medium» optimization, check preview, tighten if needed\n"
            "- Texturing needs much more memory than shape generation. Start with "
            "the «Safe» preset; raise it only if it fits\n"
            "- Output files are saved to `outputs/` next to this script\n"
        ),
        # ----- optimization level labels -----
        "opt_none":   "No optimization",
        "opt_weak":   "Light (≈ 30% of original)",
        "opt_medium": "Medium (≈ 8% of original)",
        "opt_strong": "Heavy (≈ 2% of original)",
        # ----- optimization mode labels -----
        "opt_mode_quadric": "Quadric (universal)",
        "opt_mode_iso":     "Iso (uniform triangles)",
        "opt_mode_planar":  "Planar (big flat triangles)",
        "opt_mode_hybrid":  "Hybrid (planar + quadric)",
        "opt_mode_auto":    "Auto (detect & pick)",
        # ----- dynamic log messages -----
        "log_device":          "Device: {device}, dtype: {dtype}",
        "log_preprocess":      "Preprocessing image (remove_bg={remove_bg})...",
        "log_after_size":      "Size after preprocessing: {size}",
        "log_loading_pipeline":"Loading pipeline (first run may take a few minutes)...",
        "log_pipeline_comps":  "Pipeline components:\n{diag}",
        "log_generation":      "Mesh generation: steps={steps}, guidance={guidance}, octree={octree}, seed={seed}, track_gpu={track_gpu}",
        "log_mesh_ready":      "Mesh ready: {verts} vertices, {faces} faces",
        "log_done":            "✅ Done in {elapsed:.1f} s",
        "log_preview_glb":     "Preview (GLB): {path}",
        "log_next_step":       "Pick a format on the right to download. Post-processing is also on the right.",
        "log_warn_no_image":   "⚠️ Please upload an input image",
        "log_err_model":       "❌ Model load error: {e}",
        "log_err_oom":         "❌ Out of memory or GPU failure: {msg}\n\nHint: reduce octree_resolution to 128 or num_steps to 30.\n\nTraceback:\n{tb}",
        "log_err_inference":   "❌ Inference error: {msg}\n\nTraceback:\n{tb}",
        "log_err_unknown":     "❌ Unknown error: {name}: {e}\n\nTraceback:\n{tb}",
        "log_no_mesh_yet":     "⚠️ Generate a mesh first, then optimize.",
        "log_shown_original":  "Showing original without optimization: {verts} vertices, {faces} faces.",
        "log_already_small":   "Mesh is already smaller than target ({faces} ≤ {target}). Optimization not needed.",
        "log_decimating":      "Decimating original ({faces} faces) → ~{target} faces (level: «{level}»)...",
        "log_optimizing":      "Optimizing: algorithm «{mode}», level «{level}», input {faces} faces...",
        "log_auto_pick":       "Auto classifier: large-planar area = {ratio:.0f}% → chose «{chosen}».",
        "log_decimation_done": "✅ Done in {elapsed:.1f} s: {v0}→{v1} vertices, {f0}→{f1} faces (reduced by {pct:.1f}%)",
        "log_preview":         "Preview: {path}",
        # ----- op stats block -----
        "stats_total":       "**Total tensor operations:** {total}",
        "stats_device_line": "- `{dev}`: {cnt} ({pct:.1f}%)",
        "stats_cpu_header":  "**Top CPU operations** (these are what hurts performance):",
        "stats_cpu_line":    "- `{op}`: {cnt} on CPU ({pct:.0f}% of this operation)",
        "stats_none":        "The op tracker recorded no operations (probably disabled).",
        "stats_disabled":    "_torch.overrides.TorchFunctionMode is unavailable in this PyTorch version — tracker is off._",
        # ----- texturing (PBR) -----
        "tex_header":      "### Texturing (PBR)",
        "tex_help": (
            "Generates albedo and metallic-roughness maps with the Hunyuan3D paint "
            "model, then bakes them into the mesh UVs.\n\n"
            "- Needs `mesh_inpaint_processor` built and the `hunyuan3d-paintpbr-v2-1` "
            "weights. No compiler is needed for the rasterizer: `torch_rasterizer` "
            "runs it on whatever device the tensors are on, including MPS.\n"
            "- Far more memory-hungry than shape generation. «Safe» is the safe start.\n"
            "- The paint model re-meshes to ~40k triangles internally, so heavy "
            "decimation before texturing is wasted work.\n"
        ),
        "tex_views":       "Views (more = better coverage, more memory)",
        "tex_resolution":  "Multiview resolution",
        "tex_preset":      "Memory preset",
        "tex_preset_safe":   "Safe (6 views, 256, texture 2048) — fits without flash attention",
        "tex_preset_low":    "Low (6 views, 512, texture 2048) — needs flash attention",
        "tex_preset_normal": "Normal (8 views, 768, texture 4096) — needs flash attention",
        "tex_raster":      "Rasterizer backend",
        "tex_raster_auto": "Auto (from the detected backend)",
        "tex_raster_gpu":  "GPU (native HIP/CUDA kernel — AMD/NVIDIA only)",
        "tex_raster_cpu":  "CPU (native C++ path; also the fallback for bad GPU output)",
        "tex_raster_torch": "Torch (pure PyTorch, no compiler — runs on MPS/GPU)",
        "tex_btn":         "🎨 Generate texture",
        "tex_log":         "Texturing log",
        "tex_preview":     "Textured result",
        "tex_maps":        "PBR maps",
        "tex_dl_btn":      "💾 Download textured model",
        "tex_dl_file":     "Textured file",
        "tex_use_remesh":  "Let the paint model re-mesh (recommended)",
        # ----- texturing log messages -----
        "log_tex_no_mesh":   "⚠️ Generate a mesh first, then texture it.",
        "log_tex_disabled":  "Texturing is not available: {reason}",
        "log_tex_start":     "Texturing: preset «{preset}», views={views}, resolution={resolution}, rasterizer={raster}",
        "log_tex_unload":    "Freeing the shape pipeline before loading the paint model...",
        "log_tex_loading":   "Loading the paint model (first run downloads several GB)...",
        "log_tex_input":     "Input mesh: {faces} faces, saved to {path}",
        "log_tex_running":   "Running multiview diffusion, super-resolution and baking. This is the slow part.",
        "log_tex_done":      "✅ Texture ready in {elapsed:.1f} s: {path}",
        "log_tex_err_oom":   "❌ Out of memory while texturing: {msg}\n\nHint: switch to the «Safe» preset, lower the view count or the multiview resolution, or close other apps.\n\nTraceback:\n{tb}",
        "log_tex_err":       "❌ Texturing error: {name}: {e}\n\nTraceback:\n{tb}",
    },
    "zh": {
        "lang_label": "界面语言",
        "description": (
            "# 混元3D 2.1 — 本地界面\n\n"
            "使用腾讯 Hunyuan3D 2.1 从单张图像生成3D网格。\n\n"
            "**后端：** `{backend}` • **GPU：** {gpu} • **精度：** `{dtype}` • "
            "**光栅化器：** `{raster}`\n\n"
            "**项目：** `{project}`\n\n"
            "> 形状生成在 GPU 上运行。网格生成后可在下方进行 PBR 纹理生成 — "
            "需要先编译本地扩展并下载 paint 权重。"
            "部分算子会自动回退到 CPU（勾选「统计 GPU 与 CPU 操作占比」"
            "可查看具体分布）。"
        ),
        "input_image":    "输入图像（物体置于纯色背景）",
        "remove_bg":      "自动移除背景（rembg）",
        "track_gpu":      "统计 GPU 与 CPU 操作占比",
        "gen_params":     "生成参数",
        "num_steps":      "扩散步数（越多质量越高，也越慢)",
        "guidance":       "引导强度（classifier-free guidance)",
        "octree":         "八叉树分辨率（网格细节)",
        "seed":           "随机种子",
        "generate_btn":   "🚀 生成 3D 对象",
        "log_label":      "运行日志",
        "preview_label":  "结果（预览）",
        "gpu_stats_initial": "_算子跟踪已关闭。勾选左侧的复选框以测量 GPU 与 CPU 占比。_",
        "gpu_stats_header":  "### GPU 与 CPU（生成期间的算子）\n",
        "gpu_stats_off_note": "_已请求跟踪，但当前 PyTorch 版本不支持 `torch.overrides.TorchFunctionMode`。_",
        "postproc_header":  "### 后处理（网格简化）",
        "opt_level_label":  "优化级别（始终基于原始网格应用）",
        "opt_btn":          "🛠️ 应用优化",
        "opt_log":          "后处理日志",
        "download_header":  "### 下载网格",
        "dl_format":        "输出格式",
        "dl_btn":           "💾 保存并下载",
        "dl_file":          "待下载文件",
        "tips": (
            "### 提示\n"
            "- 快速测试：`octree=128`，`steps=30`\n"
            "- 最高质量：`octree=320+`，`steps=80+`（较慢！）\n"
            "- 如果 GPU 内存不足 — 关闭其他应用或降低 octree\n"
            "- 「统计 GPU 与 CPU 操作占比」会让每个算子都经过 Python，"
            "生成速度大约慢一个数量级。仅用于诊断，日常生成请关闭\n"
            "- CAD 用途：从「中度」优化开始，查看预览后再决定是否加强\n"
            "- 纹理生成比形状生成消耗更多内存。请先使用「安全」预设\n"
            "- 输出文件保存在脚本旁的 `outputs/` 目录\n"
        ),
        "opt_none":   "不优化",
        "opt_weak":   "轻度（约原始的 30%）",
        "opt_medium": "中度（约原始的 8%）",
        "opt_strong": "高度（约原始的 2%）",
        "opt_mode_label":   "算法",
        "opt_mode_help": (
            "- **Quadric** — 通用，均匀简化；综合质量/体积最佳。\n"
            "- **Iso** — 各向同性重新网格化，保留特征边（CAD 风格）。\n"
            "- **Planar** — 把大型平面区域合并成少量大三角形"
            "（门板、墙面、盒子）。曲面区域保持原样。\n"
            "- **Hybrid** — 先 Planar，再 Quadric。三角形更少，保留平面外观。\n"
            "- **Auto** — 硬表面网格用 Planar，有机造型用 Quadric。\n"
        ),
        "opt_mode_quadric": "Quadric（通用）",
        "opt_mode_iso":     "Iso（均匀三角形）",
        "opt_mode_planar":  "Planar（大平面三角形）",
        "opt_mode_hybrid":  "Hybrid（Planar + Quadric）",
        "opt_mode_auto":    "Auto（自动检测）",
        "log_device":          "设备：{device}，精度：{dtype}",
        "log_preprocess":      "预处理图像（remove_bg={remove_bg}）……",
        "log_after_size":      "预处理后尺寸：{size}",
        "log_loading_pipeline":"加载流水线（首次运行可能需要几分钟）……",
        "log_pipeline_comps":  "流水线组件：\n{diag}",
        "log_generation":      "网格生成：steps={steps}，guidance={guidance}，octree={octree}，seed={seed}，track_gpu={track_gpu}",
        "log_mesh_ready":      "网格已生成：{verts} 个顶点，{faces} 个面",
        "log_done":            "✅ 完成，用时 {elapsed:.1f} 秒",
        "log_preview_glb":     "预览（GLB）：{path}",
        "log_next_step":       "在右侧选择格式下载。后处理也在右侧。",
        "log_warn_no_image":   "⚠️ 请先上传输入图像",
        "log_err_model":       "❌ 模型加载错误：{e}",
        "log_err_oom":         "❌ 内存不足或 GPU 失败：{msg}\n\n提示：将 octree_resolution 降至 128 或 num_steps 降至 30。\n\nTraceback：\n{tb}",
        "log_err_inference":   "❌ 推理错误：{msg}\n\nTraceback：\n{tb}",
        "log_err_unknown":     "❌ 未知错误：{name}: {e}\n\nTraceback：\n{tb}",
        "log_no_mesh_yet":     "⚠️ 请先生成网格，然后再优化。",
        "log_shown_original":  "显示未优化的原始网格：{verts} 个顶点，{faces} 个面。",
        "log_already_small":   "网格已小于目标面数（{faces} ≤ {target}）。无需优化。",
        "log_decimating":      "对原始网格进行简化（{faces} 个面）→ 约 {target} 个面（级别：「{level}」）……",
        "log_optimizing":      "优化中：算法「{mode}」，级别「{level}」，输入 {faces} 个面……",
        "log_auto_pick":       "Auto 分类器：大平面面积占比 {ratio:.0f}% → 选择「{chosen}」。",
        "log_decimation_done": "✅ 完成，用时 {elapsed:.1f} 秒：{v0}→{v1} 顶点，{f0}→{f1} 面（减少 {pct:.1f}%)",
        "log_preview":         "预览：{path}",
        "stats_total":       "**张量操作总数：** {total}",
        "stats_device_line": "- `{dev}`: {cnt} ({pct:.1f}%)",
        "stats_cpu_header":  "**CPU 算子 Top 列表**（正是它们拖慢速度）：",
        "stats_cpu_line":    "- `{op}`: {cnt} 次在 CPU 上执行（占该算子的 {pct:.0f}%)",
        "stats_none":        "算子跟踪器未记录任何操作（可能已禁用）。",
        "stats_disabled":    "_当前 PyTorch 版本不支持 torch.overrides.TorchFunctionMode — 跟踪器已禁用。_",
        # ----- 纹理生成 (PBR) -----
        "tex_header":      "### 纹理生成（PBR）",
        "tex_help": (
            "使用 Hunyuan3D paint 模型生成 albedo 与 metallic-roughness 贴图，"
            "然后烘焙到网格 UV 上。\n\n"
            "- 需要编译 `mesh_inpaint_processor` 以及 `hunyuan3d-paintpbr-v2-1` 权重。"
            "光栅化器无需编译器：`torch_rasterizer` 会在张量所在的设备上运行，"
            "包括 MPS。\n"
            "- 比形状生成消耗更多内存。建议从「安全」开始。\n"
            "- paint 模型内部会重新网格化到约 4 万三角面，因此纹理化之前的"
            "强力简化是徒劳的。\n"
        ),
        "tex_views":       "视角数量（越多覆盖越好，也更耗内存）",
        "tex_resolution":  "多视角分辨率",
        "tex_preset":      "内存预设",
        "tex_preset_safe":   "安全（6 视角，256，贴图 2048）— 无需 flash attention",
        "tex_preset_low":    "低（6 视角，512，贴图 2048）— 需要 flash attention",
        "tex_preset_normal": "常规（8 视角，768，贴图 4096）— 需要 flash attention",
        "tex_raster":      "光栅化后端",
        "tex_raster_auto": "自动（依据检测到的后端）",
        "tex_raster_gpu":  "GPU（原生 HIP/CUDA 内核 — 仅 AMD/NVIDIA）",
        "tex_raster_cpu":  "CPU（原生 C++ 路径；GPU 输出异常时的回退方案）",
        "tex_raster_torch": "Torch（纯 PyTorch，无需编译器 — 可在 MPS/GPU 上运行）",
        "tex_btn":         "🎨 生成纹理",
        "tex_log":         "纹理生成日志",
        "tex_preview":     "带纹理的结果",
        "tex_maps":        "PBR 贴图",
        "tex_dl_btn":      "💾 下载带纹理的模型",
        "tex_dl_file":     "带纹理的文件",
        "tex_use_remesh":  "允许 paint 模型重新网格化（推荐）",
        "log_tex_no_mesh":   "⚠️ 请先生成网格，然后再进行纹理生成。",
        "log_tex_disabled":  "纹理生成不可用：{reason}",
        "log_tex_start":     "纹理生成：预设「{preset}」，views={views}，resolution={resolution}，光栅化器={raster}",
        "log_tex_unload":    "在加载 paint 模型之前释放 shape 流水线……",
        "log_tex_loading":   "加载 paint 模型（首次运行需下载数 GB）……",
        "log_tex_input":     "输入网格：{faces} 个面，已保存到 {path}",
        "log_tex_running":   "正在运行多视角扩散、超分辨率与烘焙。这是最慢的阶段。",
        "log_tex_done":      "✅ 纹理完成，用时 {elapsed:.1f} 秒：{path}",
        "log_tex_err_oom":   "❌ 纹理生成时内存不足：{msg}\n\n提示：切换到「安全」预设、减少视角数量或多视角分辨率，或关闭其他应用。\n\nTraceback：\n{tb}",
        "log_tex_err":       "❌ 纹理生成错误：{name}: {e}\n\nTraceback：\n{tb}",
    },
    "ru": {
        "lang_label": "Язык интерфейса",
        "description": (
            "# Hunyuan3D 2.1 — локальный интерфейс\n\n"
            "Генерация 3D-меша из одного изображения с помощью Tencent Hunyuan3D 2.1.\n\n"
            "**Бэкенд:** `{backend}` • **GPU:** {gpu} • **dtype:** `{dtype}` • "
            "**растеризатор:** `{raster}`\n\n"
            "**Проект:** `{project}`\n\n"
            "> Генерация формы идёт на GPU. Текстурирование (PBR) доступно ниже, "
            "когда меш готов — нужны собранные нативные расширения и скачанные "
            "paint-веса. Часть операций может автоматически уходить на CPU "
            "(включи «Считать долю операций GPU vs CPU», чтобы увидеть точную "
            "раскладку)."
        ),
        "input_image":    "Входное изображение (объект на простом фоне)",
        "remove_bg":      "Автоматически удалять фон (rembg)",
        "track_gpu":      "Считать долю операций GPU vs CPU",
        "gen_params":     "Параметры генерации",
        "num_steps":      "Шагов диффузии (больше = качественнее, дольше)",
        "guidance":       "Guidance scale (сила классификатора)",
        "octree":         "Octree resolution (детализация меша)",
        "seed":           "Seed (случайное зерно)",
        "generate_btn":   "🚀 Сгенерировать 3D-объект",
        "log_label":      "Лог выполнения",
        "preview_label":  "Результат (превью)",
        "gpu_stats_initial": "_Трекер операций выключен. Включи слева чек-бокс, чтобы увидеть долю GPU vs CPU._",
        "gpu_stats_header":  "### GPU vs CPU (операции во время генерации)\n",
        "gpu_stats_off_note": "_Трекинг запрошен, но `torch.overrides.TorchFunctionMode` недоступен в этой версии PyTorch._",
        "postproc_header":  "### Пост-обработка (декимация меша)",
        "opt_level_label":  "Уровень оптимизации (всегда применяется к исходному мешу)",
        "opt_mode_label":   "Алгоритм",
        "opt_mode_help": (
            "- **Quadric** — универсальный, равномерная децимация; "
            "лучшее соотношение качества и размера.\n"
            "- **Iso** — равномерные треугольники с сохранением характерных "
            "рёбер (CAD-вид).\n"
            "- **Planar** — сливает большие плоские регионы в несколько "
            "крупных треугольников (дверные панели, стены, коробки). "
            "Изогнутые участки остаются как есть.\n"
            "- **Hybrid** — сначала planar, затем quadric. Меньше треугольников, "
            "сохраняется «плоский» вид.\n"
            "- **Auto** — planar для hard-surface, quadric для органики.\n"
        ),
        "opt_btn":          "🛠️ Применить оптимизацию",
        "opt_log":          "Лог пост-обработки",
        "download_header":  "### Скачать меш",
        "dl_format":        "Формат выгрузки",
        "dl_btn":           "💾 Сохранить и скачать",
        "dl_file":          "Файл для скачивания",
        "tips": (
            "### Советы\n"
            "- Быстрый тест: `octree=128`, `steps=30`\n"
            "- Максимум качества: `octree=320+`, `steps=80+` (долго!)\n"
            "- Если GPU падает по памяти — закрой другие приложения или уменьши octree\n"
            "- «Считать долю операций GPU vs CPU» прогоняет каждую тензорную операцию "
            "через Python и замедляет генерацию примерно на порядок. Включай для "
            "диагностики, а не для обычных прогонов\n"
            "- Для CAD: начни со «Средней» оптимизации, глянь превью, при необходимости усиль\n"
            "- Текстурирование требует существенно больше памяти, чем генерация формы. "
            "Начинай с пресета «Безопасный»\n"
            "- Выходные файлы сохраняются в `outputs/` рядом со скриптом\n"
        ),
        "opt_none":   "Без оптимизации",
        "opt_weak":   "Слабая (≈ 30% от исходного)",
        "opt_medium": "Средняя (≈ 8% от исходного)",
        "opt_strong": "Сильная (≈ 2% от исходного)",
        "opt_mode_quadric": "Quadric (универсальный)",
        "opt_mode_iso":     "Iso (равномерные треугольники)",
        "opt_mode_planar":  "Planar (крупные плоские треугольники)",
        "opt_mode_hybrid":  "Hybrid (planar + quadric)",
        "opt_mode_auto":    "Auto (автоподбор)",
        "log_device":          "Устройство: {device}, dtype: {dtype}",
        "log_preprocess":      "Предобработка изображения (remove_bg={remove_bg})...",
        "log_after_size":      "Размер после предобработки: {size}",
        "log_loading_pipeline":"Загружаю пайплайн (первый запуск может занять минуты)...",
        "log_pipeline_comps":  "Компоненты пайплайна:\n{diag}",
        "log_generation":      "Генерация меша: steps={steps}, guidance={guidance}, octree={octree}, seed={seed}, track_gpu={track_gpu}",
        "log_mesh_ready":      "Меш готов: {verts} вершин, {faces} граней",
        "log_done":            "✅ Готово за {elapsed:.1f} с",
        "log_preview_glb":     "Превью (GLB): {path}",
        "log_next_step":       "Выбери формат справа для скачивания. Пост-обработка — там же.",
        "log_warn_no_image":   "⚠️ Загрузите входное изображение",
        "log_err_model":       "❌ Ошибка загрузки модели: {e}",
        "log_err_oom":         "❌ Нехватка памяти или сбой GPU: {msg}\n\nСовет: уменьши octree_resolution до 128 или num_steps до 30.\n\nTraceback:\n{tb}",
        "log_err_inference":   "❌ Ошибка inference: {msg}\n\nTraceback:\n{tb}",
        "log_err_unknown":     "❌ Неизвестная ошибка: {name}: {e}\n\nTraceback:\n{tb}",
        "log_no_mesh_yet":     "⚠️ Сначала сгенерируй меш, потом оптимизируй.",
        "log_shown_original":  "Показан исходник без оптимизации: {verts} вершин, {faces} граней.",
        "log_already_small":   "Меш уже меньше целевого ({faces} ≤ {target}). Оптимизация не нужна.",
        "log_decimating":      "Декимация исходника ({faces} граней) → ~{target} граней (уровень: «{level}»)...",
        "log_optimizing":      "Оптимизация: алгоритм «{mode}», уровень «{level}», на входе {faces} граней...",
        "log_auto_pick":       "Auto-классификатор: площадь крупных плоских регионов {ratio:.0f}% → выбран «{chosen}».",
        "log_decimation_done": "✅ Готово за {elapsed:.1f} с: {v0}→{v1} вершин, {f0}→{f1} граней (уменьшено на {pct:.1f}%)",
        "log_preview":         "Превью: {path}",
        "stats_total":       "**Всего операций с тензорами:** {total}",
        "stats_device_line": "- `{dev}`: {cnt} ({pct:.1f}%)",
        "stats_cpu_header":  "**Топ CPU-операций** (именно эти ломают скорость):",
        "stats_cpu_line":    "- `{op}`: {cnt} на CPU ({pct:.0f}% от этой операции)",
        "stats_none":        "Трекер операций не зафиксировал ни одной операции (возможно, отключён).",
        "stats_disabled":    "_torch.overrides.TorchFunctionMode недоступен в этой версии PyTorch — трекер отключён._",
        # ----- текстурирование (PBR) -----
        "tex_header":      "### Текстурирование (PBR)",
        "tex_help": (
            "Генерирует карты albedo и metallic-roughness моделью Hunyuan3D paint, "
            "затем запекает их в UV меша.\n\n"
            "- Нужны собранный `mesh_inpaint_processor` и веса "
            "`hunyuan3d-paintpbr-v2-1`. Для растеризатора компилятор не нужен: "
            "`torch_rasterizer` работает на том устройстве, где лежат тензоры, "
            "включая MPS.\n"
            "- Памяти требует значительно больше, чем генерация формы. Безопасный "
            "старт — «Безопасный».\n"
            "- Paint-модель внутри сама ремешит до ~40k треугольников, поэтому "
            "сильная декимация перед текстурированием — потерянная работа.\n"
        ),
        "tex_views":       "Ракурсов (больше = лучше покрытие, больше памяти)",
        "tex_resolution":  "Разрешение multiview",
        "tex_preset":      "Пресет памяти",
        "tex_preset_safe":   "Безопасный (6 ракурсов, 256, текстура 2048) — работает без flash attention",
        "tex_preset_low":    "Низкий (6 ракурсов, 512, текстура 2048) — нужен flash attention",
        "tex_preset_normal": "Обычный (8 ракурсов, 768, текстура 4096) — нужен flash attention",
        "tex_raster":      "Бэкенд растеризатора",
        "tex_raster_auto": "Авто (по обнаруженному бэкенду)",
        "tex_raster_gpu":  "GPU (нативное ядро HIP/CUDA — только AMD/NVIDIA)",
        "tex_raster_cpu":  "CPU (нативный C++-путь; он же обходной путь при битом выводе GPU)",
        "tex_raster_torch": "Torch (чистый PyTorch, без компилятора — работает на MPS/GPU)",
        "tex_btn":         "🎨 Сгенерировать текстуру",
        "tex_log":         "Лог текстурирования",
        "tex_preview":     "Результат с текстурой",
        "tex_maps":        "PBR-карты",
        "tex_dl_btn":      "💾 Скачать модель с текстурой",
        "tex_dl_file":     "Файл с текстурой",
        "tex_use_remesh":  "Разрешить paint-модели ремешить (рекомендуется)",
        "log_tex_no_mesh":   "⚠️ Сначала сгенерируй меш, потом текстурируй.",
        "log_tex_disabled":  "Текстурирование недоступно: {reason}",
        "log_tex_start":     "Текстурирование: пресет «{preset}», ракурсов={views}, разрешение={resolution}, растеризатор={raster}",
        "log_tex_unload":    "Освобождаю shape-пайплайн перед загрузкой paint-модели...",
        "log_tex_loading":   "Загружаю paint-модель (первый запуск скачает несколько ГБ)...",
        "log_tex_input":     "Входной меш: {faces} граней, сохранён в {path}",
        "log_tex_running":   "Запускаю multiview-диффузию, апскейл и запекание. Это самая долгая часть.",
        "log_tex_done":      "✅ Текстура готова за {elapsed:.1f} с: {path}",
        "log_tex_err_oom":   "❌ Нехватка памяти при текстурировании: {msg}\n\nСовет: переключись на пресет «Безопасный», уменьши число ракурсов или разрешение multiview, либо закрой другие приложения.\n\nTraceback:\n{tb}",
        "log_tex_err":       "❌ Ошибка текстурирования: {name}: {e}\n\nTraceback:\n{tb}",
    },
}


def t(lang: str, key: str, **kw) -> str:
    """Translate a key into the target language, falling back to English."""
    text = I18N.get(lang, {}).get(key)
    if text is None:
        text = I18N["en"].get(key, key)
    try:
        return text.format(**kw) if kw else text
    except Exception:
        return text


# ==== Lazy pipeline loading =================================================
_pipeline = None   # shape generation
_rembg_session = None

def _load_pipeline():
    """Lazy initialization of the Hunyuan3D shape pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    print(f"[init] Loading Hunyuan3D shape pipeline on device: {DEVICE}, dtype={DTYPE}")

    # Hunyuan3D 2.1 ships Hunyuan3DDiTFlowMatchingPipeline in hy3dshape/pipelines.py.
    try:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    except Exception as e:
        raise RuntimeError(
            "Cannot import hy3dshape from the Hunyuan3D-2.1 repo. "
            f"Make sure it is cloned into {REPO_DIR}. Error: {e}"
        )

    # Only possible once hy3dshape is importable, hence here rather than at
    # startup. Guards against an intermittent IndexError that otherwise throws
    # away a generation several minutes in — see the function's docstring.
    compat_patches.patch_scheduler_timestep_lookup()

    model_path = str(WEIGHTS_DIR) if WEIGHTS_DIR.exists() else "tencent/Hunyuan3D-2.1"
    # Notes on the flags below:
    #   - weights ship as .ckpt (not .safetensors), file name is model.fp16.ckpt
    #     → use_safetensors=False  keeps the original extension
    #   - variant='fp16'           picks the fp16 checkpoint
    #   - device=DEVICE must be passed explicitly: Tencent default is
    #     device='cuda' which crashes with "Torch not compiled with CUDA enabled"
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path,
        subfolder="hunyuan3d-dit-v2-1",
        use_safetensors=False,
        variant="fp16",
        device=DEVICE,
        dtype=DTYPE,
    )
    # Safety net — move the pipeline again if something stayed on CPU.
    # IMPORTANT: Tencent's pipeline.to() does NOT return self (unlike standard
    # PyTorch). Never do `pipeline = pipeline.to(...)` — you'd get None.
    # We call .to() only for the side effect.
    if hasattr(pipeline, "to"):
        try:
            pipeline.to(DEVICE)
        except Exception as e:
            print(f"[init] pipeline.to({DEVICE}) failed: {e}. "
                  f"Moving components manually.", flush=True)
            for attr in ("model", "vae", "conditioner"):
                obj = getattr(pipeline, attr, None)
                if obj is not None and hasattr(obj, "to"):
                    obj.to(DEVICE)
    else:
        for attr in ("model", "vae", "conditioner"):
            obj = getattr(pipeline, attr, None)
            if obj is not None and hasattr(obj, "to"):
                obj.to(DEVICE)
    _pipeline = pipeline
    print("[init] Pipeline loaded")
    return _pipeline

def _load_rembg():
    """Lazy rembg session for background removal."""
    global _rembg_session
    if _rembg_session is not None:
        return _rembg_session
    try:
        from rembg import new_session
        _rembg_session = new_session("u2net")
        return _rembg_session
    except Exception as e:
        print(f"[warn] rembg unavailable: {e}")
        return None

# ==== Helpers ===============================================================
def preprocess_image(img: Image.Image, remove_bg: bool) -> Image.Image:
    """Return an RGBA 512x512 canvas with the object centered and bg removed."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    if remove_bg:
        sess = _load_rembg()
        if sess is not None:
            from rembg import remove
            img = remove(img, session=sess)
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2), img)
    return canvas.resize((512, 512), Image.LANCZOS)


def split_image_handler(source_image, main_image, method, color_mask, min_pixels):
    """Split the current image and expose each crop to the existing shape UI."""
    try:
        image = source_image if source_image is not None else main_image
        if image is None:
            if color_mask is not None:
                raise ValueError("The photo is in the color mask field. Move it to 'Image to split' above, then clear the mask field.")
            raise ValueError("Upload an image into 'Image to split' above, or into the main Input image below.")
        if color_mask is None and image.convert("RGBA").getextrema()[3][0] == 255:
            session = _load_rembg()
            if session is None:
                raise RuntimeError("Background remover is unavailable. Upload a transparent PNG or a color label mask.")
            from rembg import remove
            image = remove(image.convert("RGBA"), session=session)
        if method == "sam" and color_mask is None:
            unload_shape_pipeline()
            from sam_parts import propose_parts
            paths, archive = propose_parts(image, OUTPUT_DIR, min_pixels=int(min_pixels))
        else:
            paths, archive = split_parts(
                image, OUTPUT_DIR, color_mask=color_mask, min_pixels=int(min_pixels)
            )
        sam_mode = method == "sam" and color_mask is None
        label = "Candidate" if sam_mode else "Part"
        choices = [(f"{label} {i:02d}", str(path)) for i, path in enumerate(paths, 1)]
        if sam_mode:
            note = " SAM candidates can overlap; review each crop before 3D generation."
        elif color_mask is None and len(paths) == 1:
            note = (" Only one connected foreground region was found. Try SAM "
                    "candidate regions or provide a color label mask to split touching parts.")
        else:
            note = ""
        return (
            [(str(path), f"{label} {i:02d}") for i, path in enumerate(paths, 1)],
            str(archive),
            gr.update(choices=choices, value=str(paths[0])),
            f"Found {len(paths)} {('SAM candidates' if sam_mode else 'visible parts')}. Download the ZIP below.{note}",
        )
    except Exception as exc:
        return [], None, gr.update(choices=[], value=None), f"Split failed: {exc}"


def use_part_handler(path):
    if not path or not Path(path).is_file() or Path(path).parent.parent != OUTPUT_DIR:
        raise gr.Error("Split an image and select a part first.")
    return Image.open(path).convert("RGBA")

def mesh_to_file(mesh: trimesh.Trimesh, fmt: str, suffix: str = "") -> str:
    """Save a trimesh to the requested format. Returns an absolute path."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    ext = fmt.lower()
    allowed = ("obj", "glb", "stl", "ply", "dae", "off", "3mf")
    if ext not in allowed:
        ext = "glb"
    name = f"hunyuan3d_{ts}{('_' + suffix) if suffix else ''}.{ext}"
    path = OUTPUT_DIR / name
    mesh.export(str(path))
    return str(path)

# ==== Post-processing: decimation ===========================================
# Optimization levels are language-agnostic: internally we use "none"/"weak"/
# "medium"/"strong". Display labels come from I18N.
OPT_PARAMS: dict[str, Optional[dict]] = {
    "none":   None,
    "weak":   dict(ratio=0.30, min_faces=3000),
    "medium": dict(ratio=0.08, min_faces=1500),
    "strong": dict(ratio=0.02, min_faces=400),
}
OPT_LEVEL_ORDER = ("none", "weak", "medium", "strong")


def _opt_level_choices(lang: str) -> list[tuple[str, str]]:
    """Return [(display_label, internal_key), ...] for a gr.Radio."""
    return [(t(lang, f"opt_{k}"), k) for k in OPT_LEVEL_ORDER]


# Order matters: the first entry is the default displayed at startup, and
# `quadric` matches the pre-existing behavior so opening the UI in a new
# session is indistinguishable from before.
OPT_MODE_ORDER = ("quadric", "iso", "planar", "hybrid", "auto")


def _opt_mode_choices(lang: str) -> list[tuple[str, str]]:
    return [(t(lang, f"opt_mode_{k}"), k) for k in OPT_MODE_ORDER]


def _preclean_mesh_pymeshlab(ms) -> None:
    """Clean-up series before decimation: duplicates, non-manifold, tiny components.

    A clean mesh decimates a lot better — the dangling triangles and T-vertices
    left by marching cubes confuse the quadric solver badly.
    """
    for step in (
        "meshing_remove_duplicate_vertices",
        "meshing_remove_duplicate_faces",
        "meshing_remove_unreferenced_vertices",
        "meshing_remove_null_faces",
        "meshing_remove_t_vertices",
        "meshing_repair_non_manifold_edges",
        "meshing_repair_non_manifold_vertices",
    ):
        if hasattr(ms, step):
            try:
                getattr(ms, step)()
            except Exception:
                pass


def _taubin_smooth(ms, steps: int = 2) -> None:
    """Gentle Taubin smoothing WITHOUT shape shrinkage (unlike plain Laplacian).

    Good after decimation: removes stair-step noise on flat regions without
    eating sharp edges, provided you only run 1-2 iterations.
    """
    if hasattr(ms, "apply_coord_taubin_smoothing"):
        try:
            ms.apply_coord_taubin_smoothing(stepsmoothnum=int(steps))
            return
        except Exception:
            pass
    try:
        ms.apply_filter("apply_coord_taubin_smoothing", stepsmoothnum=int(steps))
    except Exception:
        pass


def decimate_mesh(
    mesh: trimesh.Trimesh,
    target_faces: int,
    preserve_boundary: bool = True,
    feature_smooth: bool = True,
) -> trimesh.Trimesh:
    """Improved decimation: pre-clean → planar quadric → light Taubin smoothing.

    Uses pymeshlab with CAD-friendly parameters. Falls back to trimesh if
    pymeshlab is not available.
    """
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        pm = pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        )
        ms.add_mesh(pm, "orig")

        # 1) Pre-clean — critical for marching-cubes meshes.
        _preclean_mesh_pymeshlab(ms)

        # 2) Decimation proper.
        # Key params for CAD quality:
        #   planarquadric=True + planarweight small → preserves flat faces
        #   qualitythr=0.3 → forbids creating slivers that cause shading artefacts
        #   preservenormal=True → keeps shading smooth on existing faces
        #   preservetopology=True → won't create holes
        #   boundaryweight>1.0 → keeps silhouette sharp
        decimate_kwargs = dict(
            targetfacenum=int(target_faces),
            preserveboundary=preserve_boundary,
            preservenormal=True,
            preservetopology=True,
            optimalplacement=True,
            planarquadric=True,
            planarweight=0.002,
            qualitythr=0.3,
            qualityweight=False,
            autoclean=True,
        )
        if preserve_boundary:
            decimate_kwargs["boundaryweight"] = 1.5

        filter_tried = False
        for filt_name in (
            "meshing_decimation_quadric_edge_collapse",
            "simplification_quadric_edge_collapse_decimation",
        ):
            if hasattr(ms, filt_name):
                try:
                    getattr(ms, filt_name)(**decimate_kwargs)
                    filter_tried = True
                    break
                except TypeError:
                    # Older pymeshlab: smaller accepted param set — fall back.
                    try:
                        getattr(ms, filt_name)(
                            targetfacenum=int(target_faces),
                            preserveboundary=preserve_boundary,
                            preservenormal=True,
                            preservetopology=True,
                            optimalplacement=True,
                            planarquadric=True,
                            autoclean=True,
                        )
                        filter_tried = True
                        break
                    except Exception:
                        continue
        if not filter_tried:
            ms.apply_filter(
                "meshing_decimation_quadric_edge_collapse",
                **decimate_kwargs,
            )

        # 3) Light smoothing — flattens marching-cubes stair-steps but keeps
        # sharp corners if we use few iterations.
        if feature_smooth:
            _taubin_smooth(ms, steps=2)

        # 4) Final clean — duplicates can appear after decimation too.
        _preclean_mesh_pymeshlab(ms)

        out = ms.current_mesh()
        new_mesh = trimesh.Trimesh(
            vertices=out.vertex_matrix(),
            faces=out.face_matrix(),
            process=False,
        )
        if len(new_mesh.faces) < max(50, int(target_faces * 0.1)):
            print(
                f"[decimate] pymeshlab produced too few faces "
                f"({len(new_mesh.faces)} vs target {target_faces}) — "
                f"falling back to trimesh",
                flush=True,
            )
            raise RuntimeError("too few faces after decimation")
        return new_mesh
    except Exception as e:
        print(f"[decimate] pymeshlab failed ({e}); falling back to trimesh", flush=True)
        try:
            return mesh.simplify_quadric_decimation(int(target_faces))
        except Exception as e2:
            print(f"[decimate] trimesh fallback also failed: {e2}", flush=True)
            return mesh  # better than nothing


# ==== Main generation =======================================================
DOWNLOAD_FORMATS = ["glb", "obj", "stl", "ply", "dae", "off", "3mf"]


def generate_3d(
    image: Optional[Image.Image],
    remove_bg: bool,
    num_steps: int,
    guidance_scale: float,
    octree_resolution: int,
    seed: int,
    track_gpu: bool,
    lang: str,
):
    """Generate a mesh from an image.

    Returns the tuple Gradio expects:
      (preview_glb_path, mesh_state, log_text, gpu_stats_markdown)
    """
    if image is None:
        return None, None, t(lang, "log_warn_no_image"), ""

    log_lines: list[str] = []
    t0 = time.time()
    log_lines.append(t(lang, "log_device", device=DEVICE, dtype=DTYPE))
    log_lines.append(t(lang, "log_preprocess", remove_bg=remove_bg))
    image = preprocess_image(image, remove_bg)
    log_lines.append(t(lang, "log_after_size", size=image.size))

    log_lines.append(t(lang, "log_loading_pipeline"))
    try:
        pipe = _load_pipeline()
    except Exception as e:
        return None, None, t(lang, "log_err_model", e=e), ""

    # Diagnostics — catch None components BEFORE the call.
    diag = []
    for attr in ("vae", "model", "scheduler", "conditioner", "image_processor"):
        obj = getattr(pipe, attr, "__MISSING__")
        if obj == "__MISSING__":
            diag.append(f"  {attr}: <no attribute>")
        elif obj is None:
            diag.append(f"  {attr}: None  ← cause is here!")
        else:
            diag.append(f"  {attr}: {type(obj).__name__}")
    diag_str = "\n".join(diag)
    print(f"[diag] Pipeline state:\n{diag_str}", flush=True)
    log_lines.append(t(lang, "log_pipeline_comps", diag=diag_str))

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    log_lines.append(t(
        lang, "log_generation",
        steps=num_steps, guidance=guidance_scale,
        octree=octree_resolution, seed=seed, track_gpu=track_gpu,
    ))

    tracker: Optional[DeviceUsageTracker] = None
    gpu_stats_md = ""
    try:
        if track_gpu and _HAS_TFMODE:
            tracker = DeviceUsageTracker()
            tracker_ctx = tracker
        else:
            tracker_ctx = contextlib.nullcontext()

        with torch.inference_mode(), tracker_ctx:
            result = pipe(
                image=image,
                num_inference_steps=int(num_steps),
                guidance_scale=float(guidance_scale),
                octree_resolution=int(octree_resolution),
                generator=gen,
            )
    except RuntimeError as e:
        msg = str(e)
        tb = traceback.format_exc()
        print(tb, flush=True)
        if "out of memory" in msg.lower() or "hip" in msg.lower() or "mps" in msg.lower():
            return None, None, t(lang, "log_err_oom", msg=msg, tb=tb), ""
        return None, None, t(lang, "log_err_inference", msg=msg, tb=tb), ""
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, flush=True)
        return None, None, t(lang, "log_err_unknown", name=type(e).__name__, e=e, tb=tb), ""
    finally:
        backend_mod.empty_cache(BACKEND)
        gc.collect()

    mesh = result[0] if isinstance(result, (list, tuple)) else result
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = getattr(mesh, "mesh", mesh)

    n_v, n_f = len(mesh.vertices), len(mesh.faces)
    log_lines.append(t(lang, "log_mesh_ready", verts=n_v, faces=n_f))

    glb_path = mesh_to_file(mesh, "glb", suffix="raw")

    elapsed = time.time() - t0
    log_lines.append(t(lang, "log_done", elapsed=elapsed))
    log_lines.append(t(lang, "log_preview_glb", path=glb_path))
    log_lines.append(t(lang, "log_next_step"))

    if tracker is not None:
        gpu_stats_md = t(lang, "gpu_stats_header") + "\n" + tracker.summary_text(lang=lang)
    elif track_gpu:
        gpu_stats_md = t(lang, "gpu_stats_off_note")

    # mesh_state keeps two meshes:
    #   original — raw output of the model, IMMUTABLE
    #   mesh     — current preview (= original or an optimized version)
    # This matters so that switching «Strong → Medium» re-runs Medium on the
    # original, not on the already-shrunk mesh.
    mesh_state = {
        "original": mesh,
        "mesh": mesh,
        "original_faces": n_f,
        "original_vertices": n_v,
        "last_op": "original",
        # The preprocessed image is kept so texturing can reuse the exact
        # conditioning the geometry was generated from.
        "image": image,
    }
    return glb_path, mesh_state, "\n".join(log_lines), gpu_stats_md


def unload_shape_pipeline() -> None:
    """Free the shape pipeline so the paint model has room.

    Shape and paint must never be resident at the same time: the paint pipeline
    alone wants ~21 GB at upstream's defaults, and holding the DiT as well is
    what makes texturing OOM on machines that could otherwise manage it.
    """
    global _pipeline
    if _pipeline is None:
        return
    print("[init] unloading the shape pipeline to free memory for texturing", flush=True)
    _pipeline = None
    gc.collect()
    backend_mod.empty_cache(BACKEND)


# ==== Post-processing and download handlers ================================
def _recover_mesh_state(
    mesh_state: Optional[dict],
    preview_path: Optional[str],
) -> Optional[dict]:
    """Return a usable mesh_state, rebuilding from the preview file if the
    session-scoped `gr.State` has expired.

    Gradio's `gr.State` is tied to the browser session. If the user leaves
    the tab idle for long enough the state is garbage-collected even though
    the <model-viewer> still shows the generated file (that file lives in
    `outputs/` and is served statically). In that case we reload the file
    from disk and treat it as a fresh "original" so optimization and
    download keep working.
    """
    if mesh_state and "original" in mesh_state:
        return mesh_state
    # Gradio may send the state as an empty dict or None.
    if not preview_path:
        return mesh_state
    try:
        # Model3D passes either a local path or a dict with {"path": ...}.
        if isinstance(preview_path, dict):
            preview_path = preview_path.get("path") or preview_path.get("name")
        if not preview_path or not os.path.exists(preview_path):
            return mesh_state
        loaded = trimesh.load(preview_path, force="mesh")
        if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
            return mesh_state
        return {
            "original": loaded,
            "mesh": loaded,
            "original_faces": int(len(loaded.faces)),
            "original_vertices": int(len(loaded.vertices)),
            "last_op": "recovered",
        }
    except Exception as e:
        print(f"[recover] failed to reload preview {preview_path!r}: {e}",
              flush=True)
        return mesh_state


def optimize_3d_handler(
    mesh_state: Optional[dict],
    mode_key: str,
    level_key: str,
    lang: str,
    preview_path: Optional[str] = None,
):
    """Apply the chosen algorithm + level to the ORIGINAL mesh.

    Returns (preview_glb, new_state, log).
    """
    mesh_state = _recover_mesh_state(mesh_state, preview_path)
    if not mesh_state or "original" not in mesh_state:
        if mesh_state and "mesh" in mesh_state:
            mesh_state = dict(mesh_state)
            mesh_state["original"] = mesh_state["mesh"]
        else:
            return None, mesh_state, t(lang, "log_no_mesh_yet")

    original: trimesh.Trimesh = mesh_state["original"]
    n_v0, n_f0 = len(original.vertices), len(original.faces)

    # «None» is independent of algorithm — just surface the original mesh.
    if level_key == "none":
        preview = mesh_to_file(original, "glb", suffix="raw")
        new_state = dict(mesh_state)
        new_state["mesh"] = original
        new_state["last_op"] = "original"
        return preview, new_state, t(
            lang, "log_shown_original", verts=n_v0, faces=n_f0,
        )

    t0 = time.time()
    mode_label = t(lang, f"opt_mode_{mode_key}")
    level_label = t(lang, f"opt_{level_key}")
    log_lines = [t(
        lang, "log_optimizing",
        mode=mode_label, level=level_label, faces=n_f0,
    )]

    try:
        result = lowpoly.optimize(original, mode=mode_key, level=level_key)
    except Exception as e:
        tb = traceback.format_exc()
        log_lines.append(f"❌ {type(e).__name__}: {e}\n{tb}")
        return None, mesh_state, "\n".join(log_lines)

    new_mesh = result.mesh
    n_v1, n_f1 = len(new_mesh.vertices), len(new_mesh.faces)

    if mode_key == "auto" and result.classifier_ratio is not None:
        chosen_label = t(lang, f"opt_mode_{result.mode_used}")
        log_lines.append(t(
            lang, "log_auto_pick",
            ratio=result.classifier_ratio * 100.0, chosen=chosen_label,
        ))

    suffix = f"{mode_key}_{level_key}_{n_f1}"
    preview = mesh_to_file(new_mesh, "glb", suffix=suffix)
    elapsed = time.time() - t0
    saved_pct = 100.0 * (1.0 - n_f1 / max(n_f0, 1))
    log_lines.append(t(
        lang, "log_decimation_done",
        elapsed=elapsed, v0=n_v0, v1=n_v1, f0=n_f0, f1=n_f1, pct=saved_pct,
    ))
    log_lines.append(t(lang, "log_preview", path=preview))

    new_state = {
        "original": original,
        "mesh": new_mesh,
        "original_faces": n_f0,
        "original_vertices": n_v0,
        "last_op": f"{mode_key}_{level_key}",
    }
    return preview, new_state, "\n".join(log_lines)


def download_3d_handler(
    mesh_state: Optional[dict],
    fmt: str,
    preview_path: Optional[str] = None,
):
    """Export the current mesh in the chosen format. Returns the file path."""
    mesh_state = _recover_mesh_state(mesh_state, preview_path)
    if not mesh_state or "mesh" not in mesh_state:
        return None
    mesh: trimesh.Trimesh = mesh_state["mesh"]
    suffix = mesh_state.get("last_op", "")
    safe_suffix = "".join(c if c.isalnum() else "_" for c in str(suffix))[:32]
    path = mesh_to_file(mesh, fmt, suffix=safe_suffix)
    return path


# ==== Texturing (PBR) =======================================================
TEX_PRESET_ORDER = backend_mod.PAINT_PRESET_ORDER
TEX_RASTER_ORDER = ("auto", "gpu", "cpu", "torch")


def _texture_availability() -> tuple[bool, str]:
    """Check texturing readiness without letting a broken install kill the UI.

    Called while the Blocks are being built, so it must never raise: a missing
    upstream clone or unbuilt extension should grey the button out and explain
    itself, not prevent the app from starting.
    """
    try:
        import paint

        return paint.availability()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _tex_preset_choices(lang: str) -> list[tuple[str, str]]:
    return [(t(lang, f"tex_preset_{k}"), k) for k in TEX_PRESET_ORDER]


def _tex_raster_choices(lang: str) -> list[tuple[str, str]]:
    return [(t(lang, f"tex_raster_{k}"), k) for k in TEX_RASTER_ORDER]


def _zip_textured(saved: dict) -> Optional[str]:
    """Bundle the OBJ, its MTL and every generated map into one archive.

    A bare .obj is useless to a user — the maps and the .mtl that references
    them have to travel with it.
    """
    import zipfile

    obj = saved.get("obj")
    if not obj or not os.path.exists(obj):
        return None
    base = os.path.splitext(obj)[0]
    zip_path = base + ".zip"
    members = [obj]
    mtl = base + ".mtl"
    if os.path.exists(mtl):
        members.append(mtl)
    members.extend(p for p in saved.get("maps", {}).values() if os.path.exists(p))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for m in members:
            zf.write(m, arcname=os.path.basename(m))
    return zip_path


def texture_3d_handler(
    mesh_state: Optional[dict],
    image: Optional[Image.Image],
    preset_key: str,
    views: int,
    resolution: int,
    raster_key: str,
    use_remesh: bool,
    lang: str,
    preview_path: Optional[str] = None,
):
    """Generate PBR textures for the current mesh.

    Returns (textured_glb_path, gallery_items, tex_state, log_text).
    """
    import paint

    mesh_state = _recover_mesh_state(mesh_state, preview_path)
    if not mesh_state or "mesh" not in mesh_state:
        return None, None, None, t(lang, "log_tex_no_mesh")

    ok, reason = paint.availability()
    if not ok:
        return None, None, None, t(lang, "log_tex_disabled", reason=reason)

    cond = image or mesh_state.get("image")
    if cond is None:
        return None, None, None, t(lang, "log_tex_no_mesh")

    overrides = {"max_num_view": views, "resolution": resolution}
    params = paint.resolve_params(preset_key, overrides)
    tex_backend = (
        BACKEND if raster_key == "auto"
        else backend_mod.detect(force_kind=BACKEND.kind, force_raster=raster_key)
    )

    # Check the attention memory budget before touching anything: on an AMD APU
    # an over-large attention matrix does not raise, it resets the GPU driver and
    # takes the machine with it. Failing here also avoids writing a mesh to disk
    # and unloading the shape pipeline for nothing.
    safe, why = paint.attention_preflight(
        tex_backend, params["max_num_view"], params["resolution"]
    )
    if not safe:
        return None, None, None, t(lang, "log_tex_disabled", reason=why)

    log_lines = [t(
        lang, "log_tex_start",
        preset=preset_key, views=params["max_num_view"],
        resolution=params["resolution"], raster=tex_backend.raster,
    )]

    mesh: trimesh.Trimesh = mesh_state["mesh"]
    # The paint pipeline reads its input from disk, so the current mesh has to be
    # materialised first. OBJ, because the pipeline's own remesh step round-trips
    # through pymeshlab and handles it most predictably.
    src_path = mesh_to_file(mesh, "obj", suffix="for_texture")
    log_lines.append(t(lang, "log_tex_input", faces=len(mesh.faces), path=src_path))

    log_lines.append(t(lang, "log_tex_unload"))
    unload_shape_pipeline()

    log_lines.append(t(lang, "log_tex_loading"))
    log_lines.append(t(lang, "log_tex_running"))

    try:
        saved = paint.texture_mesh(
            mesh_path=src_path,
            image=cond,
            b=tex_backend,
            preset=preset_key,
            overrides=overrides,
            use_remesh=bool(use_remesh),
            output_dir=str(OUTPUT_DIR),
            log=lambda m: print(f"[paint] {m}", flush=True),
        )
    except RuntimeError as e:
        msg = str(e)
        tb = traceback.format_exc()
        print(tb, flush=True)
        if "out of memory" in msg.lower() or "hip" in msg.lower() or "mps" in msg.lower():
            log_lines.append(t(lang, "log_tex_err_oom", msg=msg, tb=tb))
        else:
            log_lines.append(t(lang, "log_tex_err", name=type(e).__name__, e=e, tb=tb))
        return None, None, None, "\n".join(log_lines)
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, flush=True)
        log_lines.append(t(lang, "log_tex_err", name=type(e).__name__, e=e, tb=tb))
        return None, None, None, "\n".join(log_lines)
    finally:
        backend_mod.empty_cache(BACKEND)
        gc.collect()

    preview = saved.get("glb") or saved.get("obj")
    log_lines.append(t(lang, "log_tex_done", elapsed=saved.get("elapsed", 0.0), path=preview))

    gallery = [(p, k) for k, p in saved.get("maps", {}).items() if os.path.exists(p)]
    return preview, gallery, saved, "\n".join(log_lines)


def texture_download_handler(tex_state: Optional[dict], want_zip: bool):
    """Hand back the textured GLB, or a zip of OBJ + MTL + maps."""
    if not tex_state:
        return None
    if want_zip:
        return _zip_textured(tex_state)
    return tex_state.get("glb") or tex_state.get("obj")


# ==== Gradio UI =============================================================
DEFAULT_LANG = "en"


def _desc_kwargs() -> dict:
    """Substitutions for the `description` header — one definition, two uses
    (initial render and language switch)."""
    return dict(
        backend=BACKEND.label,
        gpu=BACKEND.gpu_name,
        device=BACKEND.device,
        dtype=str(BACKEND.dtype).replace("torch.", ""),
        raster=BACKEND.raster,
        project=PROJECT_DIR,
    )

with gr.Blocks(title="Hunyuan3D 2.1", theme=gr.themes.Soft()) as demo:
    # Language state — stored as an internal code ("en"/"zh"/"ru").
    lang_state = gr.State(value=DEFAULT_LANG)
    # Mesh state — a Python dict held in memory between calls.
    mesh_state = gr.State(value=None)

    # Language picker sits at the very top.
    lang_radio = gr.Radio(
        choices=[LANG_DISPLAY[k] for k in LANG_CODES],
        value=LANG_DISPLAY[DEFAULT_LANG],
        label=t(DEFAULT_LANG, "lang_label"),
    )

    desc_md = gr.Markdown(t(DEFAULT_LANG, "description", **_desc_kwargs()))

    with gr.Accordion(t(DEFAULT_LANG, "parts_header"), open=False) as parts_acc:
        parts_help = gr.Markdown(t(DEFAULT_LANG, "parts_help"))
        with gr.Row():
            parts_source = gr.Image(label=t(DEFAULT_LANG, "parts_source"), type="pil",
                                    sources=["upload", "clipboard"], height=300)
            with gr.Column():
                parts_method = gr.Radio(
                    choices=[("Connected regions (fast)", "connected"),
                             ("SAM AI candidates (GPU/CPU)", "sam")],
                    value="connected", label=t(DEFAULT_LANG, "parts_method"),
                )
                parts_min = gr.Number(label=t(DEFAULT_LANG, "parts_min"), value=100,
                                      precision=0, minimum=1)
                parts_btn = gr.Button(t(DEFAULT_LANG, "parts_btn"), variant="secondary")
                parts_status = gr.Textbox(label=t(DEFAULT_LANG, "parts_status"),
                                          interactive=False, lines=2)
                parts_choice = gr.Dropdown(label=t(DEFAULT_LANG, "parts_choice"), choices=[])
                parts_use = gr.Button(t(DEFAULT_LANG, "parts_use"), variant="secondary")
                parts_zip = gr.File(label=t(DEFAULT_LANG, "parts_zip"))
        with gr.Accordion(t(DEFAULT_LANG, "parts_mask_advanced"), open=False) as parts_mask_acc:
            parts_mask = gr.Image(label=t(DEFAULT_LANG, "parts_mask"), type="pil",
                                  sources=["upload", "clipboard"], image_mode="RGBA")
        parts_gallery = gr.Gallery(label=t(DEFAULT_LANG, "parts_gallery"), columns=4,
                                   height=260)

    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(
                label=t(DEFAULT_LANG, "input_image"),
                type="pil",
                sources=["upload", "clipboard"],
                height=360,
            )
            remove_bg = gr.Checkbox(
                label=t(DEFAULT_LANG, "remove_bg"),
                value=True,
            )
            track_gpu = gr.Checkbox(
                label=t(DEFAULT_LANG, "track_gpu"),
                value=False,
            )
            with gr.Accordion(t(DEFAULT_LANG, "gen_params"), open=False) as gen_acc:
                num_steps = gr.Slider(
                    minimum=10, maximum=100, value=50, step=5,
                    label=t(DEFAULT_LANG, "num_steps"),
                )
                guidance_scale = gr.Slider(
                    minimum=1.0, maximum=15.0, value=5.0, step=0.5,
                    label=t(DEFAULT_LANG, "guidance"),
                )
                octree_resolution = gr.Dropdown(
                    choices=[128, 192, 256, 320, 384],
                    value=256,
                    label=t(DEFAULT_LANG, "octree"),
                )
                seed = gr.Number(
                    value=42, precision=0, label=t(DEFAULT_LANG, "seed"),
                )
            btn = gr.Button(t(DEFAULT_LANG, "generate_btn"), variant="primary")
            log = gr.Textbox(label=t(DEFAULT_LANG, "log_label"), lines=14, max_lines=30)

        with gr.Column(scale=1):
            model3d = gr.Model3D(
                label=t(DEFAULT_LANG, "preview_label"),
                clear_color=[0.0, 0.0, 0.0, 0.0],
                height=420,
            )

            gpu_stats = gr.Markdown(value=t(DEFAULT_LANG, "gpu_stats_initial"))

            with gr.Group():
                postproc_md = gr.Markdown(t(DEFAULT_LANG, "postproc_header"))
                opt_mode = gr.Radio(
                    choices=_opt_mode_choices(DEFAULT_LANG),
                    value="quadric",
                    label=t(DEFAULT_LANG, "opt_mode_label"),
                )
                opt_mode_help = gr.Markdown(t(DEFAULT_LANG, "opt_mode_help"))
                opt_level = gr.Radio(
                    choices=_opt_level_choices(DEFAULT_LANG),
                    value="medium",
                    label=t(DEFAULT_LANG, "opt_level_label"),
                )
                opt_btn = gr.Button(t(DEFAULT_LANG, "opt_btn"), variant="secondary")
                opt_log = gr.Textbox(
                    label=t(DEFAULT_LANG, "opt_log"),
                    lines=4,
                    max_lines=10,
                    interactive=False,
                )

            with gr.Group():
                download_md = gr.Markdown(t(DEFAULT_LANG, "download_header"))
                dl_format = gr.Radio(
                    choices=DOWNLOAD_FORMATS,
                    value="glb",
                    label=t(DEFAULT_LANG, "dl_format"),
                )
                dl_btn = gr.Button(t(DEFAULT_LANG, "dl_btn"), variant="secondary")
                dl_file = gr.File(label=t(DEFAULT_LANG, "dl_file"))

    # --- texturing (PBR) ---------------------------------------------------
    _tex_ok, _tex_reason = _texture_availability()
    tex_state = gr.State(value=None)

    with gr.Group():
        tex_md = gr.Markdown(t(DEFAULT_LANG, "tex_header"))
        tex_status = gr.Markdown(
            value="" if _tex_ok else t(DEFAULT_LANG, "log_tex_disabled", reason=_tex_reason)
        )
        tex_help = gr.Markdown(t(DEFAULT_LANG, "tex_help"))

        with gr.Row():
            with gr.Column(scale=1):
                tex_preset = gr.Radio(
                    choices=_tex_preset_choices(DEFAULT_LANG),
                    value=backend_mod.suggest_paint_preset(BACKEND),
                    label=t(DEFAULT_LANG, "tex_preset"),
                )
                _suggested = backend_mod.PAINT_PRESETS[backend_mod.suggest_paint_preset(BACKEND)]
                tex_views = gr.Slider(
                    minimum=6, maximum=9, step=1,
                    value=_suggested["max_num_view"],
                    label=t(DEFAULT_LANG, "tex_views"),
                )
                # 256 and 320 are here because they are what fits on a GPU without
                # flash attention: attention memory grows with the square of
                # views x tokens, so 256px needs 4.2 GiB where 512px needs 67.5.
                tex_resolution = gr.Dropdown(
                    choices=[256, 320, 512, 768],
                    value=_suggested["resolution"],
                    label=t(DEFAULT_LANG, "tex_resolution"),
                )
                tex_raster = gr.Radio(
                    choices=_tex_raster_choices(DEFAULT_LANG),
                    value="auto",
                    label=t(DEFAULT_LANG, "tex_raster"),
                )
                tex_remesh = gr.Checkbox(
                    label=t(DEFAULT_LANG, "tex_use_remesh"), value=True
                )
                tex_btn = gr.Button(
                    t(DEFAULT_LANG, "tex_btn"), variant="primary", interactive=_tex_ok
                )
                tex_log = gr.Textbox(
                    label=t(DEFAULT_LANG, "tex_log"), lines=10, max_lines=25, interactive=False
                )

            with gr.Column(scale=1):
                tex_model3d = gr.Model3D(
                    label=t(DEFAULT_LANG, "tex_preview"),
                    clear_color=[0.0, 0.0, 0.0, 0.0],
                    height=420,
                )
                tex_gallery = gr.Gallery(
                    label=t(DEFAULT_LANG, "tex_maps"), columns=3, height=200, show_label=True
                )
                tex_zip = gr.Checkbox(label="OBJ + MTL + maps (.zip)", value=False)
                tex_dl_btn = gr.Button(t(DEFAULT_LANG, "tex_dl_btn"), variant="secondary")
                tex_dl_file = gr.File(label=t(DEFAULT_LANG, "tex_dl_file"))

    tips_md = gr.Markdown(t(DEFAULT_LANG, "tips"))

    # --- language switching logic ------------------------------------------
    def apply_language(lang_display: str):
        """Return a dict of component updates for the chosen language."""
        lang = LANG_BY_DISPLAY.get(lang_display, "en")
        return (
            lang,  # lang_state
            gr.update(label=t(lang, "lang_label")),                          # lang_radio
            gr.update(value=t(lang, "description", **_desc_kwargs())),       # desc_md
            gr.update(label=t(lang, "input_image")),                         # img_in
            gr.update(label=t(lang, "remove_bg")),                           # remove_bg
            gr.update(label=t(lang, "track_gpu")),                           # track_gpu
            gr.update(label=t(lang, "gen_params")),                          # gen_acc
            gr.update(label=t(lang, "num_steps")),                           # num_steps
            gr.update(label=t(lang, "guidance")),                            # guidance_scale
            gr.update(label=t(lang, "octree")),                              # octree_resolution
            gr.update(label=t(lang, "seed")),                                # seed
            gr.update(value=t(lang, "generate_btn")),                        # btn
            gr.update(label=t(lang, "log_label")),                           # log
            gr.update(label=t(lang, "preview_label")),                       # model3d
            gr.update(value=t(lang, "gpu_stats_initial")),                   # gpu_stats
            gr.update(value=t(lang, "postproc_header")),                     # postproc_md
            gr.update(
                choices=_opt_mode_choices(lang),
                label=t(lang, "opt_mode_label"),
            ),                                                               # opt_mode
            gr.update(value=t(lang, "opt_mode_help")),                       # opt_mode_help
            gr.update(
                choices=_opt_level_choices(lang),
                label=t(lang, "opt_level_label"),
            ),                                                               # opt_level
            gr.update(value=t(lang, "opt_btn")),                             # opt_btn
            gr.update(label=t(lang, "opt_log")),                             # opt_log
            gr.update(value=t(lang, "download_header")),                     # download_md
            gr.update(label=t(lang, "dl_format")),                           # dl_format
            gr.update(value=t(lang, "dl_btn")),                              # dl_btn
            gr.update(label=t(lang, "dl_file")),                             # dl_file
            gr.update(label=t(lang, "parts_header")),                         # parts_acc
            gr.update(value=t(lang, "parts_help")),                           # parts_help
            gr.update(label=t(lang, "parts_source")),                         # parts_source
            gr.update(label=t(lang, "parts_method")),                         # parts_method
            gr.update(label=t(lang, "parts_mask_advanced")),                  # parts_mask_acc
            gr.update(label=t(lang, "parts_mask")),                           # parts_mask
            gr.update(label=t(lang, "parts_min")),                            # parts_min
            gr.update(value=t(lang, "parts_btn")),                            # parts_btn
            gr.update(label=t(lang, "parts_choice")),                         # parts_choice
            gr.update(value=t(lang, "parts_use")),                            # parts_use
            gr.update(label=t(lang, "parts_zip")),                            # parts_zip
            gr.update(label=t(lang, "parts_gallery")),                        # parts_gallery
            gr.update(label=t(lang, "parts_status")),                         # parts_status
            # --- texturing ---
            gr.update(value=t(lang, "tex_header")),                          # tex_md
            gr.update(value="" if _tex_ok
                      else t(lang, "log_tex_disabled", reason=_tex_reason)),  # tex_status
            gr.update(value=t(lang, "tex_help")),                            # tex_help
            gr.update(choices=_tex_preset_choices(lang),
                      label=t(lang, "tex_preset")),                          # tex_preset
            gr.update(label=t(lang, "tex_views")),                           # tex_views
            gr.update(label=t(lang, "tex_resolution")),                      # tex_resolution
            gr.update(choices=_tex_raster_choices(lang),
                      label=t(lang, "tex_raster")),                          # tex_raster
            gr.update(label=t(lang, "tex_use_remesh")),                      # tex_remesh
            gr.update(value=t(lang, "tex_btn")),                             # tex_btn
            gr.update(label=t(lang, "tex_log")),                             # tex_log
            gr.update(label=t(lang, "tex_preview")),                         # tex_model3d
            gr.update(label=t(lang, "tex_maps")),                            # tex_gallery
            gr.update(value=t(lang, "tex_dl_btn")),                          # tex_dl_btn
            gr.update(label=t(lang, "tex_dl_file")),                         # tex_dl_file
            gr.update(value=t(lang, "tips")),                                # tips_md
        )

    lang_radio.change(
        fn=apply_language,
        inputs=[lang_radio],
        outputs=[
            lang_state,
            lang_radio, desc_md,
            img_in, remove_bg, track_gpu,
            gen_acc, num_steps, guidance_scale, octree_resolution, seed,
            btn, log, model3d, gpu_stats,
            postproc_md, opt_mode, opt_mode_help, opt_level, opt_btn, opt_log,
            download_md, dl_format, dl_btn, dl_file,
            parts_acc, parts_help, parts_source, parts_method,
            parts_mask_acc, parts_mask,
            parts_min, parts_btn,
            parts_choice, parts_use, parts_zip, parts_gallery, parts_status,
            tex_md, tex_status, tex_help, tex_preset, tex_views, tex_resolution,
            tex_raster, tex_remesh, tex_btn, tex_log, tex_model3d, tex_gallery,
            tex_dl_btn, tex_dl_file,
            tips_md,
        ],
    )

    # Selecting a memory preset moves the view/resolution sliders to its values;
    # they stay editable so a preset is a starting point, not a cage.
    def apply_tex_preset(preset_key: str):
        params = backend_mod.PAINT_PRESETS.get(preset_key, backend_mod.PAINT_PRESETS["low"])
        return (
            gr.update(value=params["max_num_view"]),
            gr.update(value=params["resolution"]),
        )

    tex_preset.change(
        fn=apply_tex_preset,
        inputs=[tex_preset],
        outputs=[tex_views, tex_resolution],
    )

    tex_btn.click(
        fn=texture_3d_handler,
        inputs=[mesh_state, img_in, tex_preset, tex_views, tex_resolution,
                tex_raster, tex_remesh, lang_state, model3d],
        outputs=[tex_model3d, tex_gallery, tex_state, tex_log],
        concurrency_id="gpu_heavy", concurrency_limit=1,
    )

    tex_dl_btn.click(
        fn=texture_download_handler,
        inputs=[tex_state, tex_zip],
        outputs=[tex_dl_file],
    )

    btn.click(
        fn=generate_3d,
        inputs=[img_in, remove_bg, num_steps, guidance_scale,
                octree_resolution, seed, track_gpu, lang_state],
        outputs=[model3d, mesh_state, log, gpu_stats],
        concurrency_id="gpu_heavy", concurrency_limit=1,
    )

    parts_btn.click(
        fn=split_image_handler,
        inputs=[parts_source, img_in, parts_method, parts_mask, parts_min],
        outputs=[parts_gallery, parts_zip, parts_choice, parts_status],
        concurrency_id="gpu_heavy", concurrency_limit=1,
    )

    parts_use.click(fn=use_part_handler, inputs=[parts_choice], outputs=[img_in])

    opt_btn.click(
        fn=optimize_3d_handler,
        inputs=[mesh_state, opt_mode, opt_level, lang_state, model3d],
        outputs=[model3d, mesh_state, opt_log],
    )

    dl_btn.click(
        fn=download_3d_handler,
        inputs=[mesh_state, dl_format, model3d],
        outputs=[dl_file],
    )

# ==== Entry point ===========================================================
if __name__ == "__main__":
    print(f"[launch] {backend_mod.describe(BACKEND)}")
    print(f"[launch] Weights: {WEIGHTS_DIR}")
    port = int(os.environ.get("GRADIO_PORT", "7860"))
    demo.queue(max_size=4).launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=True,
        show_error=True,
    )
