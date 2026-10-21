<#
.SYNOPSIS
    Install optional SAM ViT-B into this project's Windows venv.
.DESCRIPTION
    Requires the main dependencies and a working torch/torchvision first.
    Pins SAM source, checks the checkpoint SHA256, and leaves torch untouched.
#>
$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $ProjectDir "venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing venv\Scripts\python.exe. Follow docs/HUONG_DAN_WINDOWS.md first."
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required. Install Git for Windows, then reopen PowerShell."
}

& $Python -c "import torch, torchvision, cv2, numpy; print('torch:', torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Install working torch/torchvision and project dependencies first." }

& $Python -m pip install --no-deps "git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf"
if ($LASTEXITCODE -ne 0) { throw "SAM package installation failed." }

$WeightsDir = Join-Path $ProjectDir "weights"
New-Item -ItemType Directory -Force $WeightsDir | Out-Null
$Checkpoint = Join-Path $WeightsDir "sam_vit_b_01ec64.pth"
$ExpectedSha = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
$Valid = (Test-Path -LiteralPath $Checkpoint) -and (
    (Get-FileHash -LiteralPath $Checkpoint -Algorithm SHA256).Hash -eq $ExpectedSha
)
if (-not $Valid) {
    $Download = "$Checkpoint.download"
    $PreviousProgressPreference = $ProgressPreference
    try {
        $ProgressPreference = "SilentlyContinue"
        Write-Host "Downloading SAM ViT-B (~375 MB)..."
        Invoke-WebRequest -UseBasicParsing -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -OutFile $Download
        if ((Get-FileHash -LiteralPath $Download -Algorithm SHA256).Hash -ne $ExpectedSha) {
            throw "SAM checkpoint SHA256 mismatch. Re-run this script to download again."
        }
        Move-Item -LiteralPath $Download -Destination $Checkpoint -Force
    } finally {
        $ProgressPreference = $PreviousProgressPreference
        if (Test-Path -LiteralPath $Download) { Remove-Item -LiteralPath $Download -Force }
    }
}

& $Python -c "from segment_anything import SamAutomaticMaskGenerator, sam_model_registry; print('SAM import OK')"
if ($LASTEXITCODE -ne 0) { throw "SAM import check failed." }
Write-Host "SAM ready; checkpoint SHA256 verified: $Checkpoint"
Write-Host "Start launch.ps1 and select SAM AI candidates in Split 2D image into parts."
