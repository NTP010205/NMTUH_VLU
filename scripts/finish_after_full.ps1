param([int]$RunnerPid)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
$logPath = Join-Path $projectRoot 'docs/review/finalization.log'
if ($RunnerPid -gt 0) {
    Wait-Process -Id $RunnerPid -ErrorAction SilentlyContinue
}
# Analysis refuses incomplete attempted budgets or invalid sources. It is the
# gate even if the original worker process terminated unexpectedly.
& $pythonPath -m src.analyze_results *>> $logPath
if ($LASTEXITCODE -ne 0) { throw 'Full export audit failed; see finalization.log. No finalization performed.' }
& $pythonPath -m src.finalize_experiments *>> $logPath
if ($LASTEXITCODE -ne 0) { throw 'Refit/sensitivity failed; see finalization.log.' }
& $pythonPath scripts/build_final_report.py *>> $logPath
if ($LASTEXITCODE -ne 0) { throw 'Report source validation failed; see finalization.log.' }
& $pythonPath scripts/check_v1_integrity.py *>> $logPath
if ($LASTEXITCODE -ne 0) { throw 'Historical evidence hash mismatch.' }
Write-Output 'Full analysis, refit, sensitivity and report content complete. DOCX render QA and clean-checkout acceptance remain.' | Add-Content -LiteralPath $logPath
