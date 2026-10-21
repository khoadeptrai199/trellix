#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the two native extensions the Hunyuan3D paint pipeline needs.

    custom_rasterizer_kernel   triangle rasterization (GPU kernel + CPU path)
    mesh_inpaint_processor     UV texture inpainting (pure C++/pybind11)

Neither is shipped pre-built by upstream, and neither builds the same way on
every platform, so the logic lives here instead of in the shell installers.

custom_rasterizer
-----------------
Upstream's `setup.py` always uses `CUDAExtension`, which on a ROCm PyTorch
auto-hipifies `rasterizer_gpu.cu`. That works on Linux — hipify reports zero
unsupported CUDA calls on these sources. On Windows it does not work at all,
for two independent reasons:

  1. The `rocm-sdk` wheels ship the HIP runtime but no HIP headers and no
     offline compiler — `include/`, `lib/` and `cmake/` under `_rocm_sdk_devel`
     are empty — so there is nothing to compile device code with.
  2. More fundamentally, the Windows `torch 2.9.1+rocmsdk` wheel cannot link
     *any* C++ extension: `c10.lib` does not export
     `c10::ValueError(SourceLocation, std::string)`, and even an empty
     `PYBIND11_MODULE` over `<torch/extension.h>` fails with LNK2001. So the
     CPU variant is unbuildable there too.

Hence the three-way fallback: native GPU kernel, native CPU path, and finally
`torch_rasterizer.py` — a pure-PyTorch implementation that needs no compiler,
runs on whatever device the tensors are on, and is checked against a literal
transcription of the C++ loops in `tests/test_torch_rasterizer.py`. At 80k
triangles and 2048x2048 it costs ~80 ms per call on an AMD 8060S, against
minutes for the multiview diffusion, so it is not a hot spot.

Source-variant selection for the native builds matters and is not cosmetic:

  * `custom_rasterizer_kernel`             uses `INT64 = unsigned long long` and
    `data_ptr<long>()`. Correct on Linux, where `int64_t` *is* `long`. Wrong on
    MSVC (`long` is 32-bit) and wrong on macOS (`int64_t` is `long long`, so
    `data_ptr<long>` is not an instantiable overload).
  * `custom_rasterizer_kernel_for_windows` uses `int64_t`/`uint64_t` throughout.
    Portable to MSVC and macOS. Its `.cu` uses `atomicMin(uint64_t*)`, which on
    64-bit Linux resolves to `unsigned long*` and does NOT match CUDA/HIP's
    `unsigned long long*` overload — so it must not be used for GPU builds.

Hence: GPU builds take the default sources, CPU builds take the portable ones.

Usage
-----
    python scripts/build_extensions.py                 # auto
    python scripts/build_extensions.py --raster cpu    # force the CPU variant
    python scripts/build_extensions.py --force         # rebuild from scratch
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Optional

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = PROJECT_DIR / "Hunyuan3D-2.1"
PAINT_DIR = REPO_DIR / "hy3dpaint"
CR_DIR = PAINT_DIR / "custom_rasterizer"
DR_DIR = PAINT_DIR / "DifferentiableRenderer"
BUILD_DIR = PROJECT_DIR / ".build"
INFO_PATH = BUILD_DIR / "extensions.json"

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"

# Sources whose integer types are portable to MSVC and macOS.
PORTABLE_SRC = "lib/custom_rasterizer_kernel_for_windows"
# Sources whose integer types match the CUDA/HIP atomic overloads on Linux.
NATIVE_SRC = "lib/custom_rasterizer_kernel"


class BuildError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[build] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Source patching
# ---------------------------------------------------------------------------
def _patch_header(text: str) -> str:
    """Make `rasterizer.h` compilable by a plain host C++ compiler.

    Two problems for a CPU-only build:

    1. `#include <ATen/cuda/CUDAContext.h>` is unconditional, but only the
       `.cu` needs it. On ROCm it hipifies to a header that pulls in
       `hip/hip_runtime_api.h`, which MSVC cannot find (and should not need)
       when compiling the plain `.cpp` translation units.
    2. `__host__ __device__` is meaningless outside a device compiler.
    """
    # Collapse the device annotations first. Doing this before inserting the
    # guarded include matters: the guard itself contains `__host__ __device__`
    # in its HOSTDEV definition, so a blanket check afterwards would trip on
    # our own text.
    text, n_annot = re.subn(r"__host__\s+__device__\s+inline\b", "HOSTDEV inline", text)
    if n_annot == 0:
        raise BuildError(
            "rasterizer.h has no `__host__ __device__ inline` helpers — "
            "upstream layout changed, patch needs updating"
        )
    if re.search(r"__host__|__device__", text):
        raise BuildError("rasterizer.h still contains device annotations after patching")

    guarded_include = (
        "#if defined(__CUDACC__) || defined(__HIPCC__)\n"
        "#include <ATen/cuda/CUDAContext.h> // For CUDA/HIP context\n"
        "#define HOSTDEV __host__ __device__\n"
        "#else\n"
        "#define HOSTDEV\n"
        "#endif\n"
    )
    text, n_inc = re.subn(
        r"#include\s*<ATen/cuda/CUDAContext\.h>[^\n]*\n", guarded_include, text, count=1
    )
    if n_inc == 0:
        raise BuildError(
            "rasterizer.h does not include ATen/cuda/CUDAContext.h — "
            "upstream layout changed, patch needs updating"
        )
    log(f"patched rasterizer.h ({n_annot} device annotations, guarded CUDA include)")
    return text


def _patch_dispatch(text: str) -> str:
    """Route `rasterize_image` to the CPU path when there is no GPU kernel.

    Without this the CPU-only build still references `rasterize_image_gpu`,
    which is defined in the `.cu` we are deliberately not compiling, and the
    link fails.
    """
    needle = "    int device_id = V.get_device();\n"
    if needle not in text:
        raise BuildError(
            "rasterizer.cpp does not contain the expected device dispatch — "
            "upstream layout changed, patch needs updating"
        )
    guard = (
        "#ifdef CR_CPU_ONLY\n"
        "    // CPU-only build: the .cu translation unit was not compiled, so\n"
        "    // rasterize_image_gpu does not exist to link against.\n"
        "    return rasterize_image_cpu(V, F, D, width, height, occlusion_truncation, use_depth_prior);\n"
        "#else\n"
        "    int device_id = V.get_device();\n"
    )
    text = text.replace(needle, guard, 1)

    # Close the #ifdef after the original if/else dispatch.
    tail = (
        "        return rasterize_image_gpu(V, F, D, width, height, occlusion_truncation, use_depth_prior);\n"
        "}\n"
    )
    if tail not in text:
        raise BuildError("rasterizer.cpp dispatch tail not found — patch needs updating")
    text = text.replace(
        tail,
        "        return rasterize_image_gpu(V, F, D, width, height, occlusion_truncation, use_depth_prior);\n"
        "#endif\n"
        "}\n",
        1,
    )

    # Drop the build_hierarchy bindings in the CPU build.
    #
    # They live in grid_neighbor.cpp, which we do not compile here: on Windows
    # it fails to link because it instantiates c10::ValueError's
    # (SourceLocation, std::string) constructor, which the Windows torch build
    # does not export from c10.dll. That is acceptable because nothing calls
    # them — `build_hierarchy` and `build_hierarchy_with_feat` are bound but
    # never referenced from any Python code in the upstream repo; the only
    # entry point the paint pipeline uses is `rasterize_image`, via
    # custom_rasterizer/render.py.
    text, n_defs = re.subn(
        r'(\n\s*m\.def\("build_hierarchy(?:_with_feat)?".*?\);)',
        r"\n#ifndef CR_CPU_ONLY\1\n#endif",
        text,
    )
    if n_defs != 2:
        raise BuildError(
            f"expected 2 build_hierarchy bindings in rasterizer.cpp, guarded {n_defs} — "
            "upstream layout changed, patch needs updating"
        )
    log(f"patched rasterizer.cpp (CPU dispatch, {n_defs} hierarchy bindings excluded)")
    return text


def _patch_narrowing(text: str) -> str:
    """Fix the one narrowing conversion clang rejects.

        float vt[2] = {px + 0.5, py + 0.5};   // px, py are int

    `int + 0.5` is a double, and narrowing a double to float inside a braced
    initialiser is ill-formed since C++11. MSVC and GCC let it through with a
    warning; clang makes it an error, so this single line is the reason the
    "portable" sources do not actually build on macOS. `0.5f` keeps the
    expression float throughout and changes no arithmetic — the value was always
    going to be stored as float.
    """
    text, n = re.subn(r"\{px \+ 0\.5, py \+ 0\.5\}", "{px + 0.5f, py + 0.5f}", text)
    if n:
        log(f"patched rasterizer.cpp ({n} narrowing conversion — clang rejects it)")
    return text


def _stage_cpu_sources() -> Path:
    """Copy the portable rasterizer sources into .build/ and patch them.

    Upstream's clone is left untouched — it stays a pristine checkout that can
    be `git pull`ed, which is the same policy the rest of this port follows.
    """
    stage = BUILD_DIR / "custom_rasterizer_cpu"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    src_root = CR_DIR / PORTABLE_SRC
    if not src_root.is_dir():
        raise BuildError(f"portable rasterizer sources not found: {src_root}")

    # Python package part, verbatim.
    shutil.copytree(CR_DIR / "custom_rasterizer", stage / "custom_rasterizer")

    kernel_dir = stage / "kernel"
    kernel_dir.mkdir()
    # grid_neighbor.cpp is intentionally absent — see _patch_dispatch.
    for name in ("rasterizer.h", "rasterizer.cpp"):
        text = (src_root / name).read_text(encoding="utf-8")
        if name == "rasterizer.h":
            text = _patch_header(text)
        else:
            text = _patch_narrowing(_patch_dispatch(text))
        (kernel_dir / name).write_text(text, encoding="utf-8")

    (stage / "setup.py").write_text(_CPU_SETUP_PY, encoding="utf-8")
    log(f"staged patched CPU sources in {stage}")
    return stage


_CPU_SETUP_PY = '''\
"""CPU-only build of custom_rasterizer (generated by scripts/build_extensions.py).

`rasterizer_gpu.cu` is excluded because there is no device compiler here;
`rasterizer.cpp` carries a full CPU implementation and is patched to dispatch
to it unconditionally. `grid_neighbor.cpp` is excluded because the functions it
defines are never called from Python and it does not link on Windows.
"""
import sys

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, library_paths

# The extension links libc10/libtorch_cpu, but nothing records where they live:
# on macOS the resulting .so carries `@rpath/libc10.dylib` with no matching
# rpath, so it compiles, installs, and then fails at import with
# "Library not loaded: @rpath/libc10.dylib". Baking torch's lib directory in as
# an rpath is what makes the built module actually loadable. Windows resolves
# DLLs by search path instead and has no -rpath.
link_args = [] if sys.platform == "win32" else [
    f"-Wl,-rpath,{p}" for p in library_paths()
]

setup(
    name="custom_rasterizer",
    version="0.1",
    packages=find_packages(),
    package_dir={"": "."},
    ext_modules=[
        CppExtension(
            "custom_rasterizer_kernel",
            ["kernel/rasterizer.cpp"],
            define_macros=[("CR_CPU_ONLY", "1")],
            extra_link_args=link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
'''


# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------
def _find_vcvars() -> Optional[str]:
    """Locate vcvars64.bat, so a Windows build does not need a developer shell.

    setuptools can usually find MSVC on its own, but not reliably: whether it
    does depends on which shell the build was launched from, and Visual Studio
    *Build Tools* installs (no full VS) are the common miss. Rather than tell the
    user to open a "x64 Native Tools" prompt, find the batch file and run inside it.
    """
    if not IS_WINDOWS:
        return None
    program_files = os.environ.get("ProgramFiles(x86)") or r"C:\Program Files (x86)"
    vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    candidates: list[Path] = []
    if vswhere.is_file():
        try:
            proc = subprocess.run(
                [str(vswhere), "-latest", "-products", "*",
                 "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                 "-property", "installationPath"],
                capture_output=True, text=True, timeout=60,
            )
            root = proc.stdout.strip().splitlines()
            if root:
                candidates.append(Path(root[0]) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat")
        except Exception:
            pass
    # vswhere can come back empty for Build Tools depending on the flags, so keep
    # a couple of conventional locations as a backstop.
    for base in (r"C:\Program Files (x86)\Microsoft Visual Studio",
                 r"C:\Program Files\Microsoft Visual Studio"):
        for year in ("2022", "2019"):
            for edition in ("BuildTools", "Community", "Professional", "Enterprise"):
                candidates.append(
                    Path(base) / year / edition / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                )
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def _run_in_msvc_env(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Re-run a build command inside the MSVC environment.

    DISTUTILS_USE_SDK=1 is required once vcvars has run, otherwise setuptools
    refuses with "the VC environment is activated but DISTUTILS_USE_SDK is not set".
    """
    vcvars = _find_vcvars()
    if vcvars is None:
        raise BuildError(
            "no MSVC toolchain found. Install Visual Studio Build Tools with the "
            "'Desktop development with C++' workload."
        )
    log(f"retrying inside the MSVC environment: {vcvars}")
    quoted = " ".join(f'"{c}"' for c in cmd)
    # A temporary .bat rather than `cmd /c "... && ..."`: cmd strips the outer
    # quotes when the argument itself begins with one, which silently mangles a
    # vcvars path containing spaces and yields an empty, unexplained failure.
    script = cwd / "_msvc_build.bat"
    script.write_text(
        "@echo off\r\n"
        f'call "{vcvars}" >nul 2>&1\r\n'
        "set DISTUTILS_USE_SDK=1\r\n"
        f"{quoted}\r\n"
        "exit /b %ERRORLEVEL%\r\n",
        encoding="utf-8",
    )
    try:
        return subprocess.run(
            ["cmd", "/c", str(script)],
            cwd=str(cwd), capture_output=True, text=True, errors="replace",
        )
    finally:
        script.unlink(missing_ok=True)


def _needs_msvc_env(output: str) -> bool:
    return "Unable to find a compatible Visual Studio installation" in output or (
        "Microsoft Visual C++" in output and "required" in output
    )


def _pip_install(directory: Path) -> str:
    """Install a staged extension. Returns the combined output on success.

    Output is captured rather than streamed so the caller can recognise known
    toolchain defects and explain them instead of dumping a linker wall of text.
    """
    cmd = [sys.executable, "-m", "pip", "install", "--no-build-isolation", str(directory)]
    log(" ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 and IS_WINDOWS and _needs_msvc_env(output):
        proc = _run_in_msvc_env(cmd, PROJECT_DIR)
        output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise ExtensionBuildFailed(output)
    return output


class ExtensionBuildFailed(BuildError):
    """A native build failed; carries the compiler/linker output."""

    def __init__(self, output: str) -> None:
        super().__init__("native build failed")
        self.output = output

    def diagnosis(self) -> str:
        """Turn known toolchain defects into a sentence a human can act on."""
        out = self.output
        if "c10::ValueError" in out and "LNK2001" in out:
            return (
                "this PyTorch wheel cannot link ANY C++ extension: c10.lib does not "
                "export c10::ValueError(SourceLocation, std::string). Even an empty "
                "PYBIND11_MODULE over <torch/extension.h> fails the same way. This is "
                "a defect in the wheel, not in the extension sources — a different "
                "torch build would be needed. Using the pure-PyTorch rasterizer instead."
            )
        if "hip/hip_runtime_api.h" in out or "No such file or directory" in out and "hip" in out:
            return (
                "the ROCm install has no HIP headers (rocm-sdk ships an empty "
                "include/ on Windows), so device code cannot be compiled."
            )
        if "Microsoft Visual" in out and "not found" in out.lower():
            return "no MSVC toolchain found — install Visual Studio Build Tools."
        tail = "\n".join(line for line in out.splitlines() if line.strip())[-1500:]
        return f"see the build output below.\n{tail}"


def _detect_kind() -> str:
    import torch

    if torch.cuda.is_available():
        return "rocm" if getattr(torch.version, "hip", None) else "cuda"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def build_custom_rasterizer(mode: str) -> dict:
    """Build custom_rasterizer. Returns a record for extensions.json.

    Tries, in order: the native GPU kernel, the native CPU path, and finally the
    pure-PyTorch rasterizer, which needs no build at all. `--raster torch` skips
    straight to the last one.
    """
    kind = _detect_kind()

    if mode == "torch":
        log("raster=torch requested: no native build, using torch_rasterizer.py")
        return dict(backend="torch", kind=kind, sources="torch_rasterizer.py")

    if not CR_DIR.is_dir():
        raise BuildError(f"custom_rasterizer not found: {CR_DIR} (clone the upstream repo first)")

    want_gpu = mode == "gpu" or (mode == "auto" and kind in ("rocm", "cuda"))

    if want_gpu and IS_WINDOWS and kind == "rocm":
        # Explain instead of emitting a confusing compiler error: the rocm-sdk
        # wheels populate only bin/, so there are no HIP headers to compile with.
        log(
            "GPU rasterizer on Windows/ROCm: the rocm-sdk wheels ship the HIP runtime "
            "but no HIP headers or offline compiler, so device code cannot be built. "
            "Skipping the GPU variant."
        )
        if mode == "gpu":
            raise BuildError("--raster gpu is not achievable on Windows/ROCm")
        want_gpu = False

    if want_gpu:
        log(f"attempting native GPU rasterizer build ({kind}) from {NATIVE_SRC}")
        try:
            _pip_install(CR_DIR)
            _verify_import("custom_rasterizer_kernel")
            return dict(backend="gpu", kind=kind, sources=NATIVE_SRC)
        except ExtensionBuildFailed as e:
            if mode == "gpu":
                raise BuildError(f"GPU rasterizer build failed: {e.diagnosis()}")
            log(f"native GPU build failed: {e.diagnosis()}")
        except BuildError as e:
            if mode == "gpu":
                raise
            log(f"native GPU build failed: {e}")

    log(f"attempting native CPU rasterizer build from {PORTABLE_SRC}")
    try:
        stage = _stage_cpu_sources()
        _pip_install(stage)
        _verify_import("custom_rasterizer_kernel")
        return dict(backend="cpu", kind=kind, sources=PORTABLE_SRC)
    except ExtensionBuildFailed as e:
        diag = e.diagnosis()
        if mode == "cpu":
            raise BuildError(f"CPU rasterizer build failed: {diag}")
        log(f"native CPU build failed: {diag}")
    except BuildError as e:
        if mode == "cpu":
            raise
        log(f"native CPU build failed: {e}")

    log("falling back to the pure-PyTorch rasterizer (torch_rasterizer.py)")
    return dict(backend="torch", kind=kind, sources="torch_rasterizer.py")


def build_mesh_inpaint_processor() -> dict:
    """Compile mesh_inpaint_processor next to MeshRender.

    `MeshRender` imports it as `from .mesh_inpaint_processor import ...`, so the
    built module has to land inside the upstream DifferentiableRenderer package
    rather than site-packages.

    Upstream ships `compile_mesh_painter.sh`, which is a bare `c++` invocation:
    it has no MSVC equivalent and is missing `-undefined dynamic_lookup` that
    macOS needs to leave Python symbols unresolved in a bundle. Driving it
    through setuptools instead handles all three platforms, including locating
    MSVC without a pre-activated vcvars shell.
    """
    src = DR_DIR / "mesh_inpaint_processor.cpp"
    if not src.is_file():
        raise BuildError(f"mesh_inpaint_processor.cpp not found: {src}")

    try:
        import pybind11  # noqa: F401
    except ImportError:
        raise BuildError("pybind11 is required: pip install pybind11")

    stage = BUILD_DIR / "mesh_inpaint_processor"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copy2(src, stage / "mesh_inpaint_processor.cpp")
    (stage / "setup.py").write_text(_INPAINT_SETUP_PY, encoding="utf-8")

    cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
    log(" ".join(cmd) + f"  (cwd={stage})")
    proc = subprocess.run(cmd, cwd=str(stage), capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        output = (proc.stdout or "") + (proc.stderr or "")
        if IS_WINDOWS and _needs_msvc_env(output):
            proc = _run_in_msvc_env(cmd, stage)
        if proc.returncode != 0:
            raise ExtensionBuildFailed((proc.stdout or "") + (proc.stderr or ""))

    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    built = list(stage.glob(f"mesh_inpaint_processor*{suffix}")) or list(
        stage.glob("mesh_inpaint_processor*")
    )
    built = [p for p in built if p.suffix in (".pyd", ".so", ".dylib")]
    if not built:
        raise BuildError(f"no compiled module produced in {stage}")

    dest = DR_DIR / built[0].name
    shutil.copy2(built[0], dest)
    log(f"installed {dest}")
    return dict(path=str(dest.relative_to(PROJECT_DIR)), suffix=suffix)


_INPAINT_SETUP_PY = '''\
"""Build of mesh_inpaint_processor (generated by scripts/build_extensions.py).

Pure pybind11 + STL: no torch, no CUDA, no OpenMP. Builds identically on
Linux, macOS and Windows. Pybind11Extension supplies the macOS
`-undefined dynamic_lookup` that upstream's compile_mesh_painter.sh omits.
"""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="mesh_inpaint_processor",
    version="0.1",
    ext_modules=[Pybind11Extension("mesh_inpaint_processor", ["mesh_inpaint_processor.cpp"], cxx_std=11)],
    cmdclass={"build_ext": build_ext},
)
'''


def _verify_import(module: str) -> None:
    """Import the freshly built module in a clean interpreter.

    A subprocess is used on purpose: importing into the build process can pick
    up a stale copy from the source tree instead of the installed one.
    """
    code = f"import {module}; print(getattr({module}, '__file__', 'builtin'))"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_DIR),
    )
    if proc.returncode != 0:
        raise BuildError(f"{module} failed to import after build:\n{proc.stderr.strip()}")
    log(f"{module} imports OK: {proc.stdout.strip()}")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--raster",
        choices=("auto", "gpu", "cpu", "torch"),
        default="auto",
        help=(
            "rasterizer backend: auto tries native GPU, then native CPU, then the "
            "pure-PyTorch implementation; torch skips the native build entirely"
        ),
    )
    ap.add_argument("--force", action="store_true", help="rebuild even if already recorded")
    ap.add_argument(
        "--skip-inpaint", action="store_true", help="only build custom_rasterizer"
    )
    args = ap.parse_args()

    needs_repo = args.raster != "torch" or not args.skip_inpaint
    if needs_repo and not REPO_DIR.is_dir():
        print(
            f"[build] upstream repo not found at {REPO_DIR}\n"
            f"        run the installer for your platform first",
            file=sys.stderr,
        )
        return 2

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    info: dict = {}
    if INFO_PATH.exists() and not args.force:
        try:
            info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
        except Exception:
            info = {}

    try:
        info["custom_rasterizer"] = build_custom_rasterizer(args.raster)
        if not args.skip_inpaint:
            info["mesh_inpaint_processor"] = build_mesh_inpaint_processor()
    except ExtensionBuildFailed as e:
        print(f"[build] FAILED: {e.diagnosis()}", file=sys.stderr)
        return 1
    except BuildError as e:
        print(f"[build] FAILED: {e}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:
        print(f"[build] FAILED: command exited with {e.returncode}", file=sys.stderr)
        return 1

    info["platform"] = dict(
        system=platform.system(),
        machine=platform.machine(),
        python=platform.python_version(),
        built_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
    log(f"wrote {INFO_PATH}")
    log(f"rasterizer backend: {info['custom_rasterizer']['backend']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
