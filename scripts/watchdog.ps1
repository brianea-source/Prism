# PRISM-Watchdog startup wrapper. Loads .env, runs the watchdog loop.
# Re-spawned by Task Scheduler at boot; restart-on-failure is configured
# at task-creation time in install_watchdog.bat.

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

# Load .env into the current process (KEY=VALUE lines, # comments allowed)
$envFile = Join-Path $Repo ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*#') { return }
        if ($_ -match '^\s*$') { return }
        $kv = $_ -split '=', 2
        if ($kv.Length -eq 2) {
            [System.Environment]::SetEnvironmentVariable(
                $kv[0].Trim(), $kv[1].Trim(), "Process"
            )
        }
    }
}

python -m prism.watchdog.watchdog
