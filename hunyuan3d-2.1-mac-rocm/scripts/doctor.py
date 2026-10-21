#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose a Hunyuan3D 2.1 install: backend, extensions, weights, imports.

Prints one line per check with OK / WARN / FAIL, then a summary that says
plainly which of the two pipelines (shape, texturing) is actually usable.

    python scripts/doctor.py
    python scripts/doctor.py --json     # machine-readable
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

REPO_DIR = PROJECT_DIR / "Hunyuan3D-2.1"
WEIGHTS_DIR = PROJECT_DIR / "weights" / "Hunyuan3D-2.1"

OK, WARN, FAIL = "OK", "WARN", "FAIL"
_SYMBOL = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]"}


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> str:
        self.rows.append((status, name, detail))
        return status

    def worst(self, names: list[str]) -> str:
        got = [s for s, n, _ in self.rows if n in names]
        if FAIL in got:
            return FAIL
        if WARN in got:
            return WARN
        return OK

    def print(self) -> None:
        width = max((len(n) for _, n, _ in self.rows), default=10)
        for status, name, detail in self.rows:
            line = f"{_SYMBOL[status]}  {name.ljust(width)}"
            if detail:
                line += f"  {detail}"
            print(line)


def check_backend(rep: Report) -> object | None:
    try:
        import backend as backend_mod
    except Exception as e:
        rep.add(FAIL, "backend module", f"{type(e).__name__}: {e}")
        return None
    try:
        b = backend_mod.detect()
    except Exception as e:
        rep.add(FAIL, "backend detect", f"{type(e).__name__}: {e}")
        return None

    mem = backend_mod.gpu_memory_gb(b)
    mem_s = f", {mem:.1f} GiB" if mem else ""
    status = OK if b.is_gpu else WARN
    rep.add(status, "compute backend", f"{b.kind} -> device={b.device} [{b.version}]")
    rep.add(status, "accelerator", f"{b.gpu_name}{mem_s}")
    rep.add(OK, "dtype", str(b.dtype).replace("torch.", ""))
    rep.add(OK, "rasterizer", b.raster)
    if not b.is_gpu:
        rep.add(WARN, "performance", "no GPU detected — generation will be very slow on CPU")
    return b


def check_extensions(rep: Report) -> None:
    info_path = PROJECT_DIR / ".build" / "extensions.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
            cr = info.get("custom_rasterizer", {}).get("backend", "?")
            rep.add(OK, "build record", f"rasterizer={cr}, built {info.get('platform', {}).get('built_at', '?')}")
        except Exception as e:
            rep.add(WARN, "build record", f"unreadable: {e}")
    else:
        rep.add(WARN, "build record", "scripts/build_extensions.py has not run here")

    # The pure-PyTorch rasterizer needs nothing built, so it is checked first:
    # if it imports, texturing has a working rasterizer regardless of the rest.
    try:
        importlib.import_module("torch_rasterizer")
        rep.add(OK, "torch_rasterizer", "pure-PyTorch rasterizer available")
    except Exception as e:
        rep.add(FAIL, "torch_rasterizer", f"{type(e).__name__}: {e}")

    try:
        m = importlib.import_module("custom_rasterizer_kernel")
        rep.add(OK, "custom_rasterizer_kernel", getattr(m, "__file__", "builtin"))
    except Exception:
        rep.add(WARN, "custom_rasterizer_kernel", "not built (fine — torch_rasterizer covers it)")

    dr = REPO_DIR / "hy3dpaint" / "DifferentiableRenderer"
    built = list(dr.glob("mesh_inpaint_processor*.pyd")) + list(dr.glob("mesh_inpaint_processor*.so"))
    if built:
        rep.add(OK, "mesh_inpaint_processor", built[0].name)
    else:
        rep.add(
            FAIL,
            "mesh_inpaint_processor",
            "missing — texturing cannot bake; run scripts/build_extensions.py",
        )


def check_repo(rep: Report) -> None:
    if not REPO_DIR.is_dir():
        rep.add(FAIL, "upstream repo", f"missing: {REPO_DIR}")
        return
    rep.add(OK, "upstream repo", str(REPO_DIR))

    shape = REPO_DIR / "hy3dshape" / "hy3dshape" / "pipelines.py"
    rep.add(OK if shape.is_file() else FAIL, "hy3dshape sources", str(shape.relative_to(PROJECT_DIR)))
    paint = REPO_DIR / "hy3dpaint" / "textureGenPipeline.py"
    rep.add(OK if paint.is_file() else FAIL, "hy3dpaint sources", str(paint.relative_to(PROJECT_DIR)))


def _hf_cache_dir(repo_folder: str) -> Path | None:
    """Locate a repo in the Hugging Face cache, honouring the usual overrides."""
    roots = []
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(var):
            roots.append(Path(os.environ[var]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    for root in roots:
        candidate = root / repo_folder
        # A directory alone is not enough: an interrupted download leaves the
        # folder behind with only blobs/ half-written and no snapshot.
        if (candidate / "snapshots").is_dir() and any((candidate / "snapshots").iterdir()):
            return candidate
    return None


def check_weights(rep: Report) -> None:
    if not WEIGHTS_DIR.is_dir():
        rep.add(FAIL, "weights dir", f"missing: {WEIGHTS_DIR}")
        return

    def size_gb(p: Path) -> float:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 ** 3)

    for sub, required in (
        ("hunyuan3d-dit-v2-1", True),
        ("hunyuan3d-vae-v2-1", True),
        ("hunyuan3d-paintpbr-v2-1", False),
    ):
        p = WEIGHTS_DIR / sub
        if p.is_dir():
            rep.add(OK, f"weights/{sub}", f"{size_gb(p):.1f} GiB")
        else:
            rep.add(
                FAIL if required else WARN,
                f"weights/{sub}",
                "missing" + ("" if required else " — texturing unavailable"),
            )

    esrgan = REPO_DIR / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
    if esrgan.is_file():
        rep.add(OK, "RealESRGAN weights", f"{esrgan.stat().st_size / 1024**2:.0f} MiB")
    else:
        rep.add(WARN, "RealESRGAN weights", "missing — texture super-resolution unavailable")

    # DINOv2 conditions the paint UNet on the reference image and is pulled from
    # the HF cache, not weights/. Worth reporting because upstream fetches it
    # lazily: an uncached run stalls for ~4.5 GB in the middle of texturing,
    # after the shape stage has already been paid for, and prints nothing while
    # it does.
    dino = _hf_cache_dir("models--facebook--dinov2-giant")
    if dino is not None:
        rep.add(OK, "dinov2-giant (HF cache)", str(dino))
    else:
        rep.add(
            WARN,
            "dinov2-giant (HF cache)",
            "not cached — the first texturing run will pause to download ~4.5 GB",
        )


def check_gpu_sanity(rep: Report, b) -> None:
    """Verify the GPU still returns correct arithmetic.

    Not paranoia. After a driver reset — which on an APU can follow an
    over-large allocation and take the desktop with it — this GPU kept running
    and kept returning *wrong* results: a 1024x1024 matmul of ones gave 947.3
    instead of 1024.0, while reductions stayed correct. Nothing raises in that
    state, so a corrupted mesh or texture looks like a bug in the pipeline.

    A matmul is the right probe: it exercises the tensor path everything else
    depends on, and unlike a reduction it caught the fault.
    """
    if b is None or not b.is_gpu:
        return
    try:
        import torch

        n = 1024
        ones = torch.ones(n, n, device=b.device, dtype=torch.float32)
        got = float((ones @ ones).mean())
        drift = abs(got - n) / n
        total = float(torch.arange(100, device=b.device, dtype=torch.float32).sum())
    except Exception as e:
        rep.add(WARN, "gpu arithmetic", f"probe failed: {type(e).__name__}: {e}")
        return

    if drift < 1e-4 and abs(total - 4950.0) < 1e-3:
        rep.add(OK, "gpu arithmetic", f"matmul and reduction exact (matmul mean {got:.1f})")
    else:
        rep.add(
            FAIL,
            "gpu arithmetic",
            f"WRONG RESULTS: matmul mean {got:.1f}, expected {n}. The GPU is in a bad "
            f"state (usually after a driver reset) — reboot before trusting any output",
        )


def check_texture_feasibility(rep: Report, b) -> None:
    """Report whether texturing can actually run, not just whether files exist.

    Every component can be present and texturing still be impossible: without
    flash attention the paint UNet's multiview attention needs a single
    allocation of tens of GiB. Reporting "OK" on file presence alone would send
    the user straight into a GPU driver reset.
    """
    if b is None:
        return
    try:
        import backend as backend_mod
        import paint
    except Exception as e:
        rep.add(WARN, "texture feasibility", f"could not evaluate: {type(e).__name__}: {e}")
        return

    preset = backend_mod.suggest_paint_preset(b)
    params = backend_mod.PAINT_PRESETS[preset]
    views, res = params["max_num_view"], params["resolution"]
    gib = paint.estimate_attention_bytes(views, res) / (1024 ** 3)
    ok, why = paint.attention_preflight(b, views, res)
    if ok:
        rep.add(OK, "texture feasibility", f"preset «{preset}» ({views}x{res}px), attention {gib:.1f} GiB")
    else:
        rep.add(
            FAIL,
            "texture feasibility",
            f"preset «{preset}» ({views}x{res}px) needs a {gib:.1f} GiB attention "
            f"matrix — refused on this backend (see README: Attention speed on AMD)",
        )


def check_imports(rep: Report) -> None:
    for sub in ("hy3dshape", "hy3dpaint"):
        p = REPO_DIR / sub
        if p.is_dir():
            sys.path.insert(0, str(p))
    if REPO_DIR.is_dir():
        sys.path.insert(0, str(REPO_DIR))

    try:
        importlib.import_module("hy3dshape.pipelines")
        rep.add(OK, "import hy3dshape.pipelines", "")
    except Exception as e:
        rep.add(FAIL, "import hy3dshape.pipelines", f"{type(e).__name__}: {e}")

    for mod, label in (
        ("trimesh", "trimesh"),
        ("pymeshlab", "pymeshlab"),
        ("xatlas", "xatlas"),
        ("pygltflib", "pygltflib"),
        ("gradio", "gradio"),
    ):
        try:
            m = importlib.import_module(mod)
            rep.add(OK, f"import {label}", getattr(m, "__version__", ""))
        except Exception as e:
            rep.add(WARN, f"import {label}", f"{type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    rep = Report()
    b = check_backend(rep)
    check_gpu_sanity(rep, b)
    check_repo(rep)
    check_extensions(rep)
    check_weights(rep)
    check_imports(rep)
    check_texture_feasibility(rep, b)

    shape_ready = rep.worst(
        ["compute backend", "gpu arithmetic", "hy3dshape sources",
         "import hy3dshape.pipelines",
         "weights/hunyuan3d-dit-v2-1", "weights/hunyuan3d-vae-v2-1"]
    )
    texture_ready = rep.worst(
        ["gpu arithmetic", "hy3dpaint sources", "mesh_inpaint_processor",
         "torch_rasterizer", "weights/hunyuan3d-paintpbr-v2-1", "texture feasibility"]
    )

    if args.json:
        print(json.dumps({
            "checks": [{"status": s, "name": n, "detail": d} for s, n, d in rep.rows],
            "shape_pipeline": shape_ready,
            "texture_pipeline": texture_ready,
        }, indent=2))
        return 0 if shape_ready != FAIL else 1

    print()
    rep.print()
    print()
    print(f"  shape generation : {_SYMBOL[shape_ready]}")
    print(f"  texturing (PBR)  : {_SYMBOL[texture_ready]}")
    if texture_ready == FAIL:
        feasibility = next(
            (s for s, n, _ in rep.rows if n == "texture feasibility"), None
        )
        if feasibility == FAIL:
            print("    -> this backend cannot fit the paint model's attention; see the")
            print("       «Attention speed on AMD» section of the README for the options")
        else:
            print("    -> fix the FAIL rows above; scripts/build_extensions.py covers the extensions")
    print()
    return 0 if shape_ready != FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
