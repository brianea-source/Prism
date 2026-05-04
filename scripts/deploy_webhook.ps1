# PRISM-DeployWebhook startup wrapper.
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

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

python scripts\deploy_webhook.py
