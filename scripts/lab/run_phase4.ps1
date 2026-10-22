# Phase 4 on a Windows GPU machine, start to finish (see scripts/lab/README.md).
#
#   .\.venv\Scripts\Activate.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\lab\run_phase4.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\lab\run_phase4.ps1 -Seeds 0-2 -Steps 1000000
#
# Same five stages as run_phase4.sh. Re-running is safe: finished seeds are skipped.
param(
    [string]$Seeds = "0-9",
    [int]$Steps = 5000000,
    [string]$Device = "cuda",
    [int]$Episodes = 5000,
    [string]$Experiment = "e1_vec"
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")
New-Item -ItemType Directory -Force results\lab_logs | Out-Null
$log = "results\lab_logs\phase4_$(Get-Date -Format yyyyMMdd_HHmmss).log"
Start-Transcript -Path $log -Append | Out-Null

function Step([string]$title, [scriptblock]$block) {
    Write-Host "== $title"
    & $block
    if ($LASTEXITCODE -ne 0) { Stop-Transcript | Out-Null; throw "failed: $title" }
}

try {
    Write-Host "== Phase 4 on $env:COMPUTERNAME, $(Get-Date), commit $(git rev-parse --short HEAD)"
    Step "1/5 bring-up check" { python scripts\lab\check_gpu.py }
    Step "2/5 CPU parity tests" { python -m pytest -q -p no:cacheprovider tests\test_vec_env.py }
    Step "3/5 throughput benchmark" { python scripts\bench_throughput.py --out results\bench }
    Step "4/5 VecPPO on the E1 market ($Seeds, $Steps steps, $Device)" {
        python -m rl_execution.cli train --agent vecppo --device $Device --seeds $Seeds `
            --timesteps $Steps --regime abm_ac_optimal --experiment $Experiment --progress
    }
    Step "4/5 evaluate on the numpy env" {
        python -m rl_execution.cli evaluate --experiment $Experiment --episodes $Episodes --antithetic
    }
    Step "4/5 report" { python -m rl_execution.cli report --experiment $Experiment }
    Write-Host "== 5/5 archive"
    & (Join-Path $PSScriptRoot "pack_results.ps1") $Experiment bench
    Write-Host "== done; log: $log"
}
finally {
    Stop-Transcript | Out-Null
}
