<#
.SYNOPSIS
    Install Hunyuan3D 2.1 for AMD ROCm on Windows.

.DESCRIPTION
    Windows-native ROCm is a public preview, so this script is deliberately
    conservative: if a working ROCm PyTorch is already importable it reuses it
    and never reinstalls torch over the top. That matters because the wheel that
    works for a given GPU is architecture-specific — official pytorch.org ROCm
    wheels are built against ROCm 6.x and segfault on ROCm 7.2, and gfx1151
    (Radeon 8060S / Ryzen AI Max+) is not in AMD's support matrix at all and
    needs the TheRock nightly index.

    What this does NOT do on Windows: build the native custom_rasterizer.
    It cannot be built here (see scripts/build_extensions.py for the two
    independent reasons), so the pure-PyTorch rasterizer is used instead. That
    is a supported configuration, not a degraded one.

.PARAMETER Venv
    Create and use .\venv instead of the current interpreter. The venv is made
    with --system-site-packages so an already-working ROCm torch stays visible.

.PARAMETER InstallTorch
    Install torch from the ROCm wheel index even if one is already present.

.PARAMETER SkipWeights
    Skip the Hugging Face downloads (~30 GB with the paint model).

.PARAMETER ShapeOnly
    Download only the shape model, not the paint (texturing) weights.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_rocm_windows.ps1
#>
param(
    [switch]$Venv,
    [switch]$InstallTorch,
    [switch]$SkipWeights,
    [switch]$ShapeOnly
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RepoDir    = Join-Path $ProjectDir "Hunyuan3D-2.1"
$WeightsDir = Join-Path $ProjectDir "weights\Hunyuan3D-2.1"
$RepoUrl    = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git"
$HfModelId  = "tencent/Hunyuan3D-2.1"

function Write-Info { param($m) Write-Host "[INFO]  $m" -ForegroundColor Cyan }
function Write-Ok   { param($m) Write-Host "[ OK ]  $m" -ForegroundColor Green }
function Write-Warn2{ param($m) Write-Host "[WARN]  $m" -ForegroundColor Yellow }
function Write-Err  { param($m) Write-Host "[ERR ]  $m" -ForegroundColor Red }

Write-Host "============================================================"
Write-Host "  Hunyuan3D 2.1 - ROCm install (Windows)"
Write-Host "  Project: $ProjectDir"
Write-Host "============================================================"

# ---- Python ---------------------------------------------------------------
$Python = "python"
& $Python --version *> $null
if (-not $?) { Write-Err "python not found in PATH"; exit 1 }
Write-Ok "Python: $(& $Python --version 2>&1)"

if ($Venv) {
    $VenvDir = Join-Path $ProjectDir "venv"
    if (-not (Test-Path $VenvDir)) {
        Write-Info "Creating venv (with --system-site-packages so an existing ROCm torch stays visible)"
        & $Python -m venv --system-site-packages $VenvDir
    }
    $Python = Join-Path $VenvDir "Scripts\python.exe"
    Write-Ok "Using venv interpreter: $Python"
}

& $Python -m pip install --upgrade pip setuptools wheel

# ---- GPU / ROCm detection -------------------------------------------------
$probe = @'
import json, sys
info = {"ok": False}
try:
    import torch
    info["torch"] = torch.__version__
    info["hip"] = getattr(torch.version, "hip", None)
    info["cuda_available"] = torch.cuda.is_available()
    if info["cuda_available"]:
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["arch"] = getattr(p, "gcnArchName", None)
        info["mem_gb"] = round(p.total_memory / 1024**3, 1)
    info["ok"] = bool(info["hip"]) and info["cuda_available"]
except Exception as e:
    info["error"] = f"{type(e).__name__}: {e}"
print(json.dumps(info))
'@
$probeFile = Join-Path $env:TEMP "hy3d_probe.py"
Set-Content -Path $probeFile -Value $probe -Encoding utf8
$torchInfo = & $Python $probeFile | ConvertFrom-Json
Remove-Item $probeFile -Force -ErrorAction SilentlyContinue

if ($torchInfo.ok -and -not $InstallTorch) {
    Write-Ok "ROCm PyTorch already working: torch $($torchInfo.torch), HIP $($torchInfo.hip)"
    Write-Ok "GPU: $($torchInfo.gpu) [$($torchInfo.arch)], $($torchInfo.mem_gb) GiB visible"
    Write-Info "Leaving torch untouched. Pass -InstallTorch to force a reinstall."
} else {
    if ($torchInfo.error) { Write-Warn2 "torch probe: $($torchInfo.error)" }
    elseif (-not $torchInfo.hip) { Write-Warn2 "the installed torch is not a ROCm build" }
    elseif (-not $torchInfo.cuda_available) { Write-Warn2 "torch is a ROCm build but sees no GPU" }

    Write-Host ""
    Write-Warn2 "Windows ROCm PyTorch is a public preview and the correct wheel is"
    Write-Warn2 "architecture-specific. This script will not guess for you."
    Write-Host ""
    Write-Host "  For gfx1151 (Radeon 8060S / Ryzen AI Max+ 395), gfx1200/gfx1201:"
    Write-Host "    $Python -m pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch torchvision torchaudio"
    Write-Host ""
    Write-Host "  For Radeon 7000/9000 series, AMD's ROCm 7.x preview wheels:"
    Write-Host "    see https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/"
    Write-Host ""
    Write-Host "  Do NOT use the pytorch.org ROCm wheels on a ROCm 7.x runtime:"
    Write-Host "  they are built against ROCm 6.x and segfault on GPU memory access."
    Write-Host ""
    $answer = Read-Host "Install the gfx1151 nightly wheels now? [y/N]"
    if ($answer -match '^(y|yes)$') {
        & $Python -m pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch torchvision torchaudio
        Write-Ok "torch installed - re-run this script to continue"
    } else {
        Write-Err "Install a working ROCm torch first, then re-run."
    }
    exit 1
}

# rocm-sdk ships its devel contents packed; expanding is harmless if already done
# and is what makes `rocm-sdk path --bin` usable.
& $Python -c "import rocm_sdk" *> $null
if ($?) {
    Write-Info "Expanding rocm-sdk devel contents (rocm-sdk init)"
    & $Python -m rocm_sdk init *> $null
}

# ---- Upstream repo --------------------------------------------------------
& git --version *> $null
if (-not $?) { Write-Err "git not found in PATH"; exit 1 }

if (-not (Test-Path $RepoDir)) {
    Write-Info "Cloning $RepoUrl"
    & git clone --depth 1 $RepoUrl $RepoDir
    if (-not $?) { Write-Err "clone failed"; exit 1 }
    Write-Ok "Cloned into $RepoDir"
} else {
    Write-Info "Upstream repo already present; trying a fast-forward pull"
    Push-Location $RepoDir
    & git pull --ff-only
    if (-not $?) { Write-Warn2 "pull failed, continuing with the current checkout" }
    Pop-Location
}

# ---- Python dependencies --------------------------------------------------
Write-Info "Installing dependencies from requirements_rocm.txt"
& $Python -m pip install -r (Join-Path $ProjectDir "requirements_rocm.txt")
if (-not $?) { Write-Warn2 "some packages failed; see the output above" }
Write-Ok "Dependencies installed"

# ---- Native extensions ----------------------------------------------------
Write-Info "Building native extensions"
Write-Info "(custom_rasterizer cannot be compiled on Windows/ROCm; the pure-PyTorch"
Write-Info " rasterizer will be used. mesh_inpaint_processor does build here.)"
& $Python (Join-Path $ProjectDir "scripts\build_extensions.py")
if (-not $?) { Write-Warn2 "extension build reported problems; texturing may be unavailable" }

# ---- Weights --------------------------------------------------------------
if ($SkipWeights) {
    Write-Info "Skipping weight downloads (-SkipWeights)"
} else {
    # Resolve the CLI: `hf` is current, `huggingface-cli` is the deprecated
    # alias. Both may live in the venv's Scripts dir rather than on PATH.
    $HfCmd = $null
    foreach ($cand in @(
        (Join-Path (Split-Path -Parent $Python) "hf.exe"),
        (Join-Path (Split-Path -Parent $Python) "huggingface-cli.exe")
    )) { if (-not $HfCmd -and (Test-Path $cand)) { $HfCmd = $cand } }
    if (-not $HfCmd) { $HfCmd = (Get-Command hf -ErrorAction SilentlyContinue).Source }
    if (-not $HfCmd) { $HfCmd = (Get-Command huggingface-cli -ErrorAction SilentlyContinue).Source }
    if (-not $HfCmd) {
        & $Python -m pip install --upgrade "huggingface_hub[cli]"
        $HfCmd = (Join-Path (Split-Path -Parent $Python) "hf.exe")
    }
    Write-Info "Hugging Face CLI: $HfCmd"

    New-Item -ItemType Directory -Force $WeightsDir | Out-Null

    # `--include` takes exactly ONE pattern. Passing several in a row makes the
    # extras positional FILENAMES instead, which silently downloads the wrong
    # subset — that is how the DiT ends up missing while the VAE arrives.
    Write-Info "Downloading the shape model (DiT + VAE, ~15 GB). Resumable."
    & $HfCmd download $HfModelId --local-dir $WeightsDir `
        --include "hunyuan3d-dit-v2-1/*" --include "hunyuan3d-vae-v2-1/*" --include "README*"
    if (-not $?) { Write-Err "shape weights failed (401/403 means you need 'hf auth login')"; exit 1 }
    Write-Ok "Shape model in $WeightsDir"

    if (-not $ShapeOnly) {
        Write-Info "Downloading the paint model for texturing (~10 GB)"
        & $HfCmd download $HfModelId --local-dir $WeightsDir --include "hunyuan3d-paintpbr-v2-1/*"
        if (-not $?) { Write-Warn2 "paint weights failed; texturing will be unavailable" }

        $ckptDir = Join-Path $RepoDir "hy3dpaint\ckpt"
        New-Item -ItemType Directory -Force $ckptDir | Out-Null
        $esrgan = Join-Path $ckptDir "RealESRGAN_x4plus.pth"
        if (-not (Test-Path $esrgan)) {
            Write-Info "Downloading RealESRGAN_x4plus.pth (super-resolution, ~64 MB)"
            $url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            try {
                Invoke-WebRequest -Uri $url -OutFile $esrgan -UseBasicParsing
                Write-Ok "RealESRGAN weights in $esrgan"
            } catch {
                Write-Warn2 "RealESRGAN download failed: $_"
            }
        } else {
            Write-Ok "RealESRGAN weights already present"
        }
    }
}

# ---- Verify ---------------------------------------------------------------
Write-Info "Verifying the installation"
& $Python (Join-Path $ProjectDir "scripts\doctor.py")

Write-Host ""
Write-Host "============================================================"
Write-Host "  Done. Start the UI with:  .\launch.ps1"
Write-Host "  Then open http://localhost:7860"
Write-Host "============================================================"
