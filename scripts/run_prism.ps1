# PRISM Runner Wrapper
# ====================
# Loads .env, runs preflight env check, then starts the python runner.
#
# Deployed on the Vultr VPS as the target of the PRISM-Runner scheduled
# task. Source-controlled here so future deploys cannot drift silently.
#
# History: 2026-05-18 — first commit. Replaces the on-VPS-only version
# that caused the 4-week audit-log blackout (see
# docs/audits/2026-05-18_signal_audit_gap.md). The old version had no
# preflight check; if a critical env key was masked by a typo in .env,
# the runner started anyway and silently misbehaved. This version fails
# loud at startup if PRISM_SIGNAL_AUDIT_ENABLED or other CRITICAL_KEYS
# are missing.

$ErrorActionPreference = "Stop"
$PrismRoot = "C:\Prism"
$LogFile = Join-Path $PrismRoot "logs\runner.log"
$EnvFile = Join-Path $PrismRoot ".env"

Set-Location $PrismRoot

function Append-RunnerLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogFile -Value "[$stamp] [run_prism.ps1] $Message"
}

# --- Load .env -------------------------------------------------------
# Strip whole-line comments AND inline `  # comment` suffixes.
# Skip blank lines and lines without `=`. Trim key + value.
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_
        if ($line -match '^\s*#') { return }
        if ($line -notmatch '=') { return }
        $line = $line -replace '\s+#.*$', ''
        $kv = $line -split '=', 2
        $k = $kv[0].Trim()
        $v = if ($kv.Count -ge 2) { $kv[1].Trim() } else { '' }
        if ($k) { [Environment]::SetEnvironmentVariable($k, $v) }
    }
    Append-RunnerLog "Loaded .env from $EnvFile"
} else {
    Append-RunnerLog "WARNING: $EnvFile not found; relying on machine/user env only"
}

# --- Preflight env check --------------------------------------------
# Fails loud if a CRITICAL_KEY is missing. Without this, a typo or
# accidental comment in .env produces a silent runtime degradation.
Append-RunnerLog "Running env preflight..."
$preflight = & python "$PrismRoot\scripts\preflight_env.py" 2>&1
$preflight | ForEach-Object { Add-Content -Path $LogFile -Value $_ }
if ($LASTEXITCODE -ne 0) {
    Append-RunnerLog "PREFLIGHT FAILED (exit $LASTEXITCODE) — refusing to start runner"
    Append-RunnerLog "Fix the missing critical env keys (see [preflight] FAIL lines above) then restart PRISM-Runner."
    exit 1
}
Append-RunnerLog "Preflight OK"

# --- Start runner ---------------------------------------------------
Append-RunnerLog "PRISM runner starting"
& python -m prism.delivery.runner *>> $LogFile
$exit = $LASTEXITCODE
Append-RunnerLog "PRISM runner exited with code $exit"
exit $exit
