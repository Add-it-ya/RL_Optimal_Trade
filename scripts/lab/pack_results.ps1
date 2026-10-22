# Bundle lab output into one archive to bring back (runs/ and results/ are not in git).
#
#   powershell -ExecutionPolicy Bypass -File scripts\lab\pack_results.ps1 e1_vec bench
#
# Includes runs\<exp> and results\<exp> for each name given, plus results\lab_logs.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")

$items = @()
foreach ($exp in $args) {
    foreach ($d in @("runs\$exp", "results\$exp")) {
        if (Test-Path $d) { $items += $d }
    }
}
if (Test-Path results\lab_logs) { $items += "results\lab_logs" }
if ($items.Count -eq 0) { throw "nothing to pack" }
$out = "lab_results_$($env:COMPUTERNAME)_$(Get-Date -Format yyyyMMdd_HHmm).zip"
Compress-Archive -Path $items -DestinationPath $out -Force
Write-Host "Wrote $out with: $($items -join ', ')"
