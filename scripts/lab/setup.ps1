# One-time setup on a Windows lab machine with an NVIDIA GPU.
#
#   powershell -ExecutionPolicy Bypass -File scripts\lab\setup.ps1 -CudaTag cu126
#
# On Windows the PyPI torch wheel is CPU-only, so a CUDA build must come from
# download.pytorch.org: read the CUDA version from `nvidia-smi` and the matching tag from
# https://pytorch.org/get-started/locally/ (e.g. cu118, cu121, cu124, cu126, cu128).
# Creates .venv\ in the repository, installs torch and the package, then runs
# scripts\lab\check_gpu.py.
param(
    [Parameter(Mandatory = $true)][string]$CudaTag,
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")

& $Python -c "import sys; assert sys.version_info >= (3, 10), 'Python >= 3.10 is required'"
if ($LASTEXITCODE -ne 0) { exit 1 }
& $Python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url "https://download.pytorch.org/whl/$CudaTag"
if ($LASTEXITCODE -ne 0) { exit 1 }
python -m pip install -e ".[rl,dev]"
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
python scripts\lab\check_gpu.py
Write-Host ""
Write-Host "Setup done. Activate the environment in new shells with: .\.venv\Scripts\Activate.ps1"
