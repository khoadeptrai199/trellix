<#
.SYNOPSIS
    Launch the Hunyuan3D 2.1 web UI on Windows (ROCm or CPU).

.DESCRIPTION
    Sets the backend-specific environment before starting Python. The ROCm bits
    matter for real:

      PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
          The paint pipeline allocates and frees large, differently-sized
          buffers (2048x2048 renders, 4096x4096 textures). Without expandable
          segments the caching allocator fragments and OOMs with plenty of
          memory still free.

      MIOPEN_USER_DB_PATH / MIOPEN_CUSTOM_CACHE_DIR
          MIOpen autotunes convolution kernels on first use and caches the
          result. Pointing the cache at the project keeps that work across runs
          instead of paying the warm-up on every launch.

    HSA_OVERRIDE_GFX_VERSION is deliberately NOT set: it is only needed when the
    torch build has no kernels for your GPU, and setting it when native kernels
    do exist makes things slower or wrong. Set it yourself if you know you need it.

.PARAMETER Raster
    Override the rasterizer backend: gpu, cpu or torch.

.PARAMETER Backend
    Override device detection: rocm, cuda, mps or cpu. Mostly for debugging.

.PARAMETER Port
    Gradio port (default 7860).

.PARAMETER FastAttention
    Set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1.

    On consumer/APU AMD GPUs PyTorch disables the flash and mem-efficient SDPA
    backends and says so at runtime: "Flash Efficient attention on Current AMD
    GPU is still experimental." Everything therefore runs on the math backend.
    This switch enables the AOTriton kernels, which can speed attention up
    substantially. It is opt-in because it is exactly the kind of experimental
    path that can produce wrong numbers — if you enable it, check the output.
#>
param(
    [ValidateSet("gpu", "cpu", "torch")] [string]$Raster,
    [ValidateSet("rocm", "cuda", "mps", "cpu")] [string]$Backend,
    [int]$Port = 7860,
    [switch]$FastAttention
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$Python = "python"
$VenvPython = Join-Path $ProjectDir "venv\Scripts\python.exe"
if (Test-Path $VenvPython) { $Python = $VenvPython }

# ROCm / allocator tuning
$env:PYTORCH_HIP_ALLOC_CONF = if ($env:PYTORCH_HIP_ALLOC_CONF) { $env:PYTORCH_HIP_ALLOC_CONF } else { "expandable_segments:True" }
# MIOPEN_FIND_MODE is intentionally left unset — see the comment in backend.py.
# FAST routes MIOpen through a fallback path that passes solvers a null workspace
# and crashes the process with an access violation.
$cacheDir = Join-Path $ProjectDir ".cache\miopen"
New-Item -ItemType Directory -Force $cacheDir | Out-Null
$env:MIOPEN_USER_DB_PATH    = $cacheDir
$env:MIOPEN_CUSTOM_CACHE_DIR = $cacheDir

$env:TOKENIZERS_PARALLELISM = "false"
$env:GRADIO_PORT = "$Port"
if ($Raster)  { $env:HY3D_RASTER = $Raster }
if ($Backend) { $env:HY3D_BACKEND = $Backend }
if ($FastAttention) { $env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL = "1" }

Write-Host "=============================================="
Write-Host "  Hunyuan3D 2.1 - Web UI"
Write-Host "  Project: $ProjectDir"
Write-Host "  URL:     http://localhost:$Port"
if ($env:HY3D_RASTER)  { Write-Host "  Raster:  $env:HY3D_RASTER (forced)" }
if ($env:HY3D_BACKEND) { Write-Host "  Backend: $env:HY3D_BACKEND (forced)" }
if ($FastAttention)    { Write-Host "  Attention: AOTriton experimental kernels ENABLED" }
Write-Host "=============================================="

Push-Location $ProjectDir
try {
    & $Python gradio_app.py
} finally {
    Pop-Location
}
