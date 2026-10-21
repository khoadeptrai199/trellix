#!/usr/bin/env bash
# =============================================================================
# Launching the Hunyuan3D 2.1 web interface (macOS/MPS or Linux/ROCm).
#
# The environment a run needs differs per backend, and both sets are harmless on
# the other platform, so this picks by OS rather than asking.
#
#   MPS   PYTORCH_ENABLE_MPS_FALLBACK lets unimplemented ops drop to CPU instead
#         of raising; the high-watermark ratio of 0 disables the soft allocation
#         cap that otherwise aborts long runs on smaller Macs.
#
#   ROCm  expandable_segments stops the caching allocator from fragmenting
#         across the paint pipeline's mix of 2048² renders and 4096² textures.
#         The MIOpen cache dirs persist convolution autotuning between runs —
#         without them every launch repays the warm-up.
#
# HSA_OVERRIDE_GFX_VERSION is intentionally not set: it is only needed when the
# torch build lacks kernels for your GPU, and forcing it when native kernels do
# exist is a pessimisation. Export it yourself if you know you need it.
# =============================================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"

usage() {
    cat <<EOF
Usage: $0 [--raster gpu|cpu|torch] [--backend rocm|cuda|mps|cpu] [--port N]
          [--fast-attention]

  --raster           override the rasterizer backend
  --backend          override device detection (debugging; also used to bisect
                     the ROCm texture-corruption issue)
  --port             Gradio port (default 7860)
  --fast-attention   set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1.
                     On consumer/APU AMD GPUs PyTorch disables the flash and
                     mem-efficient SDPA backends ("still experimental") and runs
                     attention on the math backend instead. This enables the
                     AOTriton kernels, which can be substantially faster.
                     Opt-in on purpose: it is an experimental path, so check the
                     output if you turn it on.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --raster)  export HY3D_RASTER="$2"; shift 2 ;;
        --backend) export HY3D_BACKEND="$2"; shift 2 ;;
        --port)    export GRADIO_PORT="$2"; shift 2 ;;
        --fast-attention) export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -d "$VENV_DIR" ]]; then
    # shellcheck disable=SC1090
    source "${VENV_DIR}/bin/activate"
else
    echo "[WARN] venv not found at $VENV_DIR — using the current interpreter"
fi

export TOKENIZERS_PARALLELISM=false
export GRADIO_PORT="${GRADIO_PORT:-7860}"

if [[ "$(uname)" == "Darwin" ]]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO="${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.0}"
else
    export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:True}"
    # MIOPEN_FIND_MODE is intentionally left unset — see the comment in backend.py.
    # FAST routes MIOpen through a fallback path that passes solvers a null
    # workspace and crashes the process with an access violation.
    MIOPEN_CACHE="${PROJECT_DIR}/.cache/miopen"
    mkdir -p "$MIOPEN_CACHE"
    export MIOPEN_USER_DB_PATH="$MIOPEN_CACHE"
    export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_CACHE"
fi

echo "=============================================="
echo "  Hunyuan3D 2.1 — Web UI"
echo "  Project: $PROJECT_DIR"
echo "  URL:     http://localhost:${GRADIO_PORT}"
[[ -n "${HY3D_RASTER:-}" ]]  && echo "  Raster:  ${HY3D_RASTER} (forced)"
[[ -n "${HY3D_BACKEND:-}" ]] && echo "  Backend: ${HY3D_BACKEND} (forced)"
[[ -n "${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-}" ]] && echo "  Attention: AOTriton experimental kernels ENABLED"
echo "=============================================="

cd "$PROJECT_DIR"
exec python gradio_app.py
