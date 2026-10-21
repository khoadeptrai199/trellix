#!/usr/bin/env bash
# =============================================================================
# Hunyuan3D 2.1 – Automatic Installation for macOS (Apple Silicon)
# Adapted for Macs with M1/M2/M3/M4 processors (>= 16 GB unified memory recommended)
#
# Installs BOTH pipelines: shape generation and PBR texturing. This script used
# to say texturing needs CUDA — it does not, see docs/ROCM_TEXTURING_PLAN.md.
# The rasterizer runs through torch_rasterizer.py (pure PyTorch, on MPS),
# Blender is not needed, and the only native code required is
# mesh_inpaint_processor, a plain pybind11 module with no CUDA in it.
#
#   ./install.sh                 # everything (~30 GB: shape + paint + dinov2)
#   ./install.sh --shape-only    # shape generation only (~15 GB)
#   ./install.sh --skip-weights  # no weight downloads
# =============================================================================
set -euo pipefail

SKIP_WEIGHTS=0
SHAPE_ONLY=0

usage() {
    cat <<EOF
Использование: $0 [опции]

  --skip-weights     не скачивать веса
  --shape-only       скачать только shape-модель, без paint-весов
                     (текстурирование останется недоступным)
  -h, --help         эта справка
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-weights) SKIP_WEIGHTS=1; shift ;;
        --shape-only)   SHAPE_ONLY=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "неизвестная опция: $1" >&2; usage; exit 1 ;;
    esac
done

# ---- Конфигурация ----------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${PROJECT_DIR}/Hunyuan3D-2.1"
VENV_DIR="${PROJECT_DIR}/venv"
MODELS_DIR="${PROJECT_DIR}/weights"
REPO_URL="https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git"
HF_MODEL_ID="tencent/Hunyuan3D-2.1"
PYTHON_MIN_VERSION="3.10"

# ---- Цветной вывод ---------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERR ]${NC}  $*" >&2; }

# ---- Проверки окружения ----------------------------------------------------
info "Проверка окружения macOS..."

if [[ "$(uname)" != "Darwin" ]]; then
    err "Скрипт предназначен для macOS. Обнаружена система: $(uname)"
    exit 1
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
    warn "Обнаружена архитектура $ARCH. Apple Silicon (arm64) работает лучше всего."
fi
ok "macOS $(sw_vers -productVersion), архитектура: $ARCH"

# Xcode Command Line Tools
if ! xcode-select -p >/dev/null 2>&1; then
    warn "Xcode Command Line Tools не установлены. Устанавливаю..."
    xcode-select --install || true
    err "После завершения установки Xcode CLT запусти скрипт заново."
    exit 1
fi
ok "Xcode Command Line Tools: $(xcode-select -p)"

# Homebrew
if ! command -v brew >/dev/null 2>&1; then
    err "Homebrew не найден. Установи: https://brew.sh"
    exit 1
fi
ok "Homebrew: $(brew --version | head -1)"

# Python 3.10+
find_python() {
    for v in 3.12 3.11 3.10; do
        if command -v "python${v}" >/dev/null 2>&1; then
            echo "python${v}"; return 0
        fi
    done
    return 1
}

PYTHON_BIN="$(find_python || true)"
if [[ -z "$PYTHON_BIN" ]]; then
    info "Python 3.10-3.12 не найден. Устанавливаю python@3.11 через Homebrew..."
    brew install python@3.11
    PYTHON_BIN="python3.11"
fi
ok "Python: $(${PYTHON_BIN} --version)"

# Git
command -v git >/dev/null 2>&1 || { err "git не найден. Установи: brew install git"; exit 1; }
ok "Git: $(git --version)"

# git-lfs (для больших весов модели)
if ! command -v git-lfs >/dev/null 2>&1; then
    info "Устанавливаю git-lfs..."
    brew install git-lfs
fi
git lfs install >/dev/null 2>&1 || true
ok "git-lfs готов"

# ---- Системная память ------------------------------------------------------
MEM_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
info "Системная память: ${MEM_GB} GB"
if (( MEM_GB < 16 )); then
    warn "Рекомендуется минимум 16 GB unified memory. Возможны проблемы с запуском."
elif (( MEM_GB < 24 )); then
    warn "У тебя ${MEM_GB} GB. Возможно потребуется использовать fp16/низкое разрешение."
else
    ok "${MEM_GB} GB unified memory — достаточно для shape-генерации"
fi

# ---- Клонирование репозитория ---------------------------------------------
if [[ ! -d "$REPO_DIR" ]]; then
    info "Клонирую Hunyuan3D 2.1 из $REPO_URL ..."
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
    ok "Репозиторий склонирован в $REPO_DIR"
else
    info "Репозиторий уже существует, пропускаю клонирование."
    ( cd "$REPO_DIR" && git pull --ff-only ) || warn "Не удалось обновить, продолжаю с текущей версией."
fi

# ---- Виртуальное окружение -------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    info "Создаю виртуальное окружение: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
ok "venv активирован: $(python --version)"

python -m pip install --upgrade pip setuptools wheel

# ---- Установка PyTorch с MPS -----------------------------------------------
info "Устанавливаю PyTorch (nightly с поддержкой MPS для Apple Silicon)..."
# Используем stable для arm64 — поддержка MPS уже в 2.1+
python -m pip install --upgrade torch torchvision torchaudio

# Проверка MPS
python - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built:     {torch.backends.mps.is_built()}")
PY
ok "PyTorch установлен"

# ---- Остальные зависимости -------------------------------------------------
info "Устанавливаю Python-зависимости из requirements_mac.txt ..."
python -m pip install -r "${PROJECT_DIR}/requirements_mac.txt"
ok "Зависимости установлены"

# ---- Нативные расширения ---------------------------------------------------
# Никаких mac_compat.py в клоне апстрима больше нет: определение устройства и
# все патчи совместимости живут в backend.py / compat_patches.py / paint.py,
# а клон остаётся нетронутым — на него по-прежнему можно делать git pull.
#
# Обязателен здесь ровно один модуль — mesh_inpaint_processor (pybind11, без
# CUDA): без него paint-пайплайн не может выполнить UV-инпейнт.
#
# Растеризатор собирается «до кучи» в CPU-варианте. На MPS он не станет
# основным — backend.py выбирает torch_rasterizer, который считает там же, где
# лежат тензоры, и на M4 Pro оказался не медленнее (152 мс против 165 мс на
# 79k треугольников при 2048²). Но собранный нативный путь даёт `HY3D_RASTER=cpu`
# — независимую реализацию для сверки, если результат запекания вызовет сомнения.
# Если компилятора нет, сборка деградирует до torch-варианта сама.
info "Собираю нативные расширения..."
python "${PROJECT_DIR}/scripts/build_extensions.py" --raster auto \
    || warn "сборка расширений завершилась с проблемами — текстурирование может быть недоступно"

# ---- Загрузка весов --------------------------------------------------------
if [[ "$SKIP_WEIGHTS" -eq 1 ]]; then
    info "Пропускаю загрузку весов (--skip-weights)"
else
    mkdir -p "$MODELS_DIR"
    info "Загружаю веса модели из Hugging Face ($HF_MODEL_ID)..."
    info "(Это может занять 20-40 минут в зависимости от скорости интернета — модель ~15 GB)"

    # Новый CLI от Hugging Face называется `hf`. Старый `huggingface-cli` deprecated.
    if ! command -v hf >/dev/null 2>&1; then
        python -m pip install --upgrade "huggingface_hub[cli]"
    fi

    # Выбираем доступную команду (hf — новая, huggingface-cli — старая fallback)
    HF_CMD=""
    if command -v hf >/dev/null 2>&1; then
        HF_CMD="hf"
    elif command -v huggingface-cli >/dev/null 2>&1; then
        HF_CMD="huggingface-cli"
    else
        err "Не найден ни 'hf', ни 'huggingface-cli'"
        exit 1
    fi
    info "Использую CLI: $HF_CMD"

    # По одному вызову на шаблон. У части версий CLI второй и последующие --include
    # уходят в позиционные FILENAMES, и тогда часть весов молча не скачивается при
    # полностью «успешном» выводе — именно так однажды пропал DiT.
    for PATTERN in "hunyuan3d-dit-v2-1/*" "hunyuan3d-vae-v2-1/*" "README*"; do
        "$HF_CMD" download "$HF_MODEL_ID" \
            --local-dir "${MODELS_DIR}/Hunyuan3D-2.1" \
            --include "$PATTERN" \
            || { err "Не удалось скачать веса ($PATTERN). Проверь соединение и HF_TOKEN, если модель gated."; exit 1; }
    done

    ok "Shape-модель скачана в ${MODELS_DIR}/Hunyuan3D-2.1"

    # ---- Paint-модель ----------------------------------------------------------
    # Раньше это был вопрос с «не рекомендуется» и умолчанием «нет»: считалось, что
    # текстурирование на Mac идёт через медленный CPU-fallback. Это оказалось не так
    # — растеризация идёт на MPS, диффузия тоже, — поэтому paint ставится по
    # умолчанию, а отказаться можно флагом --shape-only.
    if [[ "$SHAPE_ONLY" -eq 1 ]]; then
        info "Пропускаю paint-веса (--shape-only): текстурирование будет недоступно"
    else
        info "Загружаю paint-модель для текстурирования (~6.5 GB)"
        "$HF_CMD" download "$HF_MODEL_ID" \
            --local-dir "${MODELS_DIR}/Hunyuan3D-2.1" \
            --include "hunyuan3d-paintpbr-v2-1/*" \
            || warn "Не удалось скачать paint-модель"

        # DINOv2-giant — кондиционер референсного изображения в paint-UNet.
        # Апстрим тянет его через AutoModel.from_pretrained("facebook/dinov2-giant")
        # в кеш HF прямо во время первого текстурирования. Замерено: прогон
        # молча стоит несколько минут внутри paint-стадии, уже после того как
        # оплачена генерация формы, и ни одной строчки в лог не пишет. Лучше
        # оплатить это на установке.
        info "Загружаю facebook/dinov2-giant (~4.5 GB, кондиционер paint-модели)"
        "$HF_CMD" download facebook/dinov2-giant \
            || warn "dinov2-giant не скачался — он подтянется сам при первом текстурировании"

        # RealESRGAN — апскейл ×4 многоракурсных изображений перед запеканием.
        # Живёт не на HF, а в релизе на GitHub, и апстрим его не качает вообще.
        CKPT_DIR="${REPO_DIR}/hy3dpaint/ckpt"
        mkdir -p "$CKPT_DIR"
        if [[ ! -f "${CKPT_DIR}/RealESRGAN_x4plus.pth" ]]; then
            info "Загружаю RealESRGAN_x4plus.pth (~64 MB)"
            URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            curl -fL "$URL" -o "${CKPT_DIR}/RealESRGAN_x4plus.pth" \
                || warn "Не удалось скачать RealESRGAN"
        fi
        # Не `[[ -f ... ]] && ok ...`: под `set -e` ложное условие последней команды
        # в блоке валит весь скрипт, хотя отсутствие апскейлера — это предупреждение.
        if [[ -f "${CKPT_DIR}/RealESRGAN_x4plus.pth" ]]; then
            ok "Веса RealESRGAN на месте"
        fi
    fi
fi

# ---- Финальная проверка ----------------------------------------------------
# doctor.py вместо списка импортов: он проверяет бэкенд, расширения, веса и —
# главное — отвечает на вопрос «оба ли пайплайна реально работоспособны»,
# а не «импортируется ли gradio».
info "Проверяю установку..."
python "${PROJECT_DIR}/scripts/doctor.py" || warn "doctor.py сообщил о проблемах — см. выше"
ok "Установка завершена"

echo ""
echo "============================================================"
echo "  Hunyuan3D 2.1 установлен в $PROJECT_DIR"
echo "  Запусти интерфейс: $PROJECT_DIR/launch.sh"
echo "  После запуска открой: http://localhost:7860"
echo "============================================================"
