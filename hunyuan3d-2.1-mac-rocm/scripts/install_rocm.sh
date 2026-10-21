#!/usr/bin/env bash
# =============================================================================
# Hunyuan3D 2.1 — installation for AMD ROCm on Linux
#
# Linux is the full-capability target: unlike Windows, the HIP toolchain is
# available here, so the native custom_rasterizer kernel can actually be built
# (torch's CUDAExtension hipifies rasterizer_gpu.cu with zero unsupported CUDA
# calls). The pure-PyTorch rasterizer remains available as a fallback and as the
# clean-room comparison for ROCm/ROCm#5981.
#
# The wheel index is chosen from the detected GPU architecture, because this is
# where people lose days:
#   * pytorch.org ROCm wheels are built against ROCm 6.x. On a ROCm 7.x runtime
#     they segfault on GPU memory access.
#   * gfx1151 (Strix Halo / Radeon 8060S) is not in AMD's support matrix and
#     needs the TheRock nightly index, which ships native gfx1151 kernels — so
#     HSA_OVERRIDE_GFX_VERSION is not needed with those wheels.
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${PROJECT_DIR}/Hunyuan3D-2.1"
VENV_DIR="${PROJECT_DIR}/venv"
MODELS_DIR="${PROJECT_DIR}/weights/Hunyuan3D-2.1"
REPO_URL="https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git"
HF_MODEL_ID="tencent/Hunyuan3D-2.1"

SKIP_WEIGHTS=0
SHAPE_ONLY=0
INSTALL_TORCH=0
RASTER="auto"

usage() {
    cat <<EOF
Usage: $0 [options]

  --skip-weights     do not download model weights
  --shape-only       download the shape model but not the paint (texturing) weights
  --install-torch    (re)install torch from the ROCm wheel index
  --raster MODE      rasterizer to build: auto (default), gpu, cpu, torch
  -h, --help         this text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-weights)  SKIP_WEIGHTS=1; shift ;;
        --shape-only)    SHAPE_ONLY=1; shift ;;
        --install-torch) INSTALL_TORCH=1; shift ;;
        --raster)        RASTER="$2"; shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info() { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()   { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()  { echo -e "${RED}[ERR ]${NC}  $*" >&2; }

echo "============================================================"
echo "  Hunyuan3D 2.1 — ROCm install (Linux)"
echo "  Project: ${PROJECT_DIR}"
echo "============================================================"

[[ "$(uname)" == "Linux" ]] || { err "this script is for Linux; on macOS use install.sh, on Windows scripts/install_rocm_windows.ps1"; exit 1; }

command -v git >/dev/null 2>&1 || { err "git not found"; exit 1; }
ok "git: $(git --version)"

# ---- Python ---------------------------------------------------------------
find_python() {
    for v in 3.12 3.11 3.10; do
        command -v "python${v}" >/dev/null 2>&1 && { echo "python${v}"; return 0; }
    done
    command -v python3 >/dev/null 2>&1 && { echo python3; return 0; }
    return 1
}
PYTHON_BIN="$(find_python)" || { err "python 3.10+ not found"; exit 1; }
ok "python: $(${PYTHON_BIN} --version)"

if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating venv in ${VENV_DIR}"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
ok "venv active: $(python --version)"
python -m pip install --upgrade pip setuptools wheel

# ---- GPU architecture -----------------------------------------------------
GFX_ARCH=""
if command -v rocminfo >/dev/null 2>&1; then
    GFX_ARCH="$(rocminfo 2>/dev/null | grep -om1 'gfx[0-9a-f]*' || true)"
fi
if [[ -z "$GFX_ARCH" ]] && command -v rocm-smi >/dev/null 2>&1; then
    GFX_ARCH="$(rocm-smi --showhw 2>/dev/null | grep -om1 'gfx[0-9a-f]*' || true)"
fi
if [[ -n "$GFX_ARCH" ]]; then
    ok "detected GPU architecture: ${GFX_ARCH}"
else
    warn "could not detect the GPU architecture (rocminfo/rocm-smi missing?)"
fi

# ---- torch ----------------------------------------------------------------
torch_state() {
    python - <<'PY' 2>/dev/null || echo "missing"
import torch
hip = getattr(torch.version, "hip", None)
print("rocm" if hip and torch.cuda.is_available() else ("nohip" if not hip else "nogpu"))
PY
}
STATE="$(torch_state)"

if [[ "$STATE" == "rocm" && "$INSTALL_TORCH" -eq 0 ]]; then
    python - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print(f"[ OK ]  torch {torch.__version__}, HIP {torch.version.hip}")
print(f"[ OK ]  GPU: {p.name} [{getattr(p, 'gcnArchName', '?')}], {p.total_memory/1024**3:.1f} GiB")
PY
    info "Working ROCm torch found — leaving it alone (pass --install-torch to override)"
else
    case "$GFX_ARCH" in
        gfx1151|gfx1150|gfx1200|gfx1201)
            INDEX="https://rocm.nightlies.amd.com/v2/${GFX_ARCH}/"
            info "Architecture ${GFX_ARCH} is not in AMD's ROCm support matrix."
            info "Using the TheRock nightly index, which ships native ${GFX_ARCH} kernels: ${INDEX}"
            ;;
        gfx1100|gfx1101|gfx1102|gfx90a|gfx942)
            INDEX="https://download.pytorch.org/whl/rocm6.4"
            info "Architecture ${GFX_ARCH} is officially supported; using ${INDEX}"
            warn "If your runtime is ROCm 7.x, these wheels will segfault — use the nightly index instead."
            ;;
        *)
            INDEX="https://download.pytorch.org/whl/rocm6.4"
            warn "Unknown architecture '${GFX_ARCH:-?}'. Defaulting to ${INDEX}; verify afterwards."
            ;;
    esac
    info "Installing torch from ${INDEX}"
    python -m pip install --index-url "$INDEX" torch torchvision torchaudio

    STATE="$(torch_state)"
    if [[ "$STATE" != "rocm" ]]; then
        err "torch still does not see a ROCm GPU (state=${STATE})."
        err "Check that your user is in the 'render' and 'video' groups and that the amdgpu driver is loaded."
        exit 1
    fi
    ok "ROCm torch installed and functional"
fi

# rocm-sdk (TheRock wheels) keeps its devel payload packed until initialised.
if python -c "import rocm_sdk" >/dev/null 2>&1; then
    info "Expanding rocm-sdk devel contents"
    python -m rocm_sdk init >/dev/null 2>&1 || warn "rocm-sdk init failed; native builds may not find hipcc"
    if ROCM_BIN="$(python -m rocm_sdk path --bin 2>/dev/null)"; then
        # torch looks for $ROCM_HOME/bin/hipcc, which is not where the wheel puts it.
        export ROCM_HOME="$(dirname "$ROCM_BIN")"
        export ROCM_PATH="$ROCM_HOME"
        export PATH="${ROCM_BIN}:${PATH}"
        ok "ROCM_HOME set to ${ROCM_HOME}"
    fi
fi

# ---- upstream repo --------------------------------------------------------
if [[ ! -d "$REPO_DIR" ]]; then
    info "Cloning ${REPO_URL}"
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
    ok "cloned into ${REPO_DIR}"
else
    info "Upstream repo present; trying a fast-forward pull"
    ( cd "$REPO_DIR" && git pull --ff-only ) || warn "pull failed, continuing with the current checkout"
fi

# ---- dependencies ---------------------------------------------------------
info "Installing dependencies from requirements_rocm.txt"
python -m pip install -r "${PROJECT_DIR}/requirements_rocm.txt" || warn "some packages failed; see above"
ok "dependencies installed"

# ---- native extensions ----------------------------------------------------
info "Building native extensions (rasterizer mode: ${RASTER})"
python "${PROJECT_DIR}/scripts/build_extensions.py" --raster "$RASTER" \
    || warn "extension build reported problems; texturing may be limited"

# ---- weights --------------------------------------------------------------
if [[ "$SKIP_WEIGHTS" -eq 1 ]]; then
    info "Skipping weight downloads (--skip-weights)"
else
    python -c "import huggingface_hub" >/dev/null 2>&1 || python -m pip install --upgrade "huggingface_hub[cli]"
    HF_CMD="hf"
    command -v hf >/dev/null 2>&1 || HF_CMD="huggingface-cli"

    mkdir -p "$MODELS_DIR"
    info "Downloading the shape model (DiT + VAE, ~15 GB). Resumable."
    "$HF_CMD" download "$HF_MODEL_ID" --local-dir "$MODELS_DIR" \
        --include "hunyuan3d-dit-v2-1/*" --include "hunyuan3d-vae-v2-1/*" --include "README*" \
        || { err "shape weights failed. 401/403 means you need: ${HF_CMD} auth login"; exit 1; }
    ok "shape model in ${MODELS_DIR}"

    if [[ "$SHAPE_ONLY" -eq 0 ]]; then
        info "Downloading the paint model for texturing (~10 GB)"
        "$HF_CMD" download "$HF_MODEL_ID" --local-dir "$MODELS_DIR" \
            --include "hunyuan3d-paintpbr-v2-1/*" \
            || warn "paint weights failed; texturing will be unavailable"

        CKPT_DIR="${REPO_DIR}/hy3dpaint/ckpt"
        mkdir -p "$CKPT_DIR"
        if [[ ! -f "${CKPT_DIR}/RealESRGAN_x4plus.pth" ]]; then
            info "Downloading RealESRGAN_x4plus.pth (~64 MB)"
            URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            if command -v curl >/dev/null 2>&1; then
                curl -fL "$URL" -o "${CKPT_DIR}/RealESRGAN_x4plus.pth" || warn "RealESRGAN download failed"
            else
                wget -q "$URL" -O "${CKPT_DIR}/RealESRGAN_x4plus.pth" || warn "RealESRGAN download failed"
            fi
        fi
        [[ -f "${CKPT_DIR}/RealESRGAN_x4plus.pth" ]] && ok "RealESRGAN weights present"
    fi
fi

# ---- verify ---------------------------------------------------------------
info "Verifying the installation"
python "${PROJECT_DIR}/scripts/doctor.py" || true

echo ""
echo "============================================================"
echo "  Done. Start the UI with: ./launch.sh"
echo "  Then open http://localhost:7860"
echo "============================================================"
