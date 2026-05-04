@echo off
:: Install PRISM self-sufficiency scheduled tasks on the Windows VPS.
::
:: Requires: Administrator shell.  Creates four scheduled tasks alongside
:: the existing PRISM-Runner so the runner is supervised, auto-deployed,
:: drift-monitored, and digested without human intervention.
::
::   PRISM-Watchdog       — every 5 min, restarts runner if down
::   PRISM-DeployWebhook  — at startup, listens for GitHub push events
::   PRISM-DriftMonitor   — daily 03:00 UTC, retrains on drift
::   PRISM-DailyDigest    — daily 08:00 UTC, posts health report
::
:: Run from repo root: scripts\install_watchdog.bat

setlocal
set REPO=%~dp0..
pushd "%REPO%" || (echo Could not cd to %REPO% & exit /b 1)
set REPO_ABS=%CD%
popd

set PSCMD=powershell.exe -NoProfile -ExecutionPolicy Bypass -File

echo Installing PRISM-Watchdog from %REPO_ABS%
schtasks /create /F /TN "PRISM-Watchdog" ^
  /TR "%PSCMD% \"%REPO_ABS%\scripts\watchdog.ps1\"" ^
  /SC ONSTART /RL HIGHEST /RU SYSTEM /DELAY 0001:00
if errorlevel 1 (echo Failed to create PRISM-Watchdog & exit /b 1)

echo Installing PRISM-DeployWebhook
schtasks /create /F /TN "PRISM-DeployWebhook" ^
  /TR "%PSCMD% \"%REPO_ABS%\scripts\deploy_webhook.ps1\"" ^
  /SC ONSTART /RL HIGHEST /RU SYSTEM /DELAY 0002:00
if errorlevel 1 (echo Failed to create PRISM-DeployWebhook & exit /b 1)

echo Installing PRISM-DriftMonitor (daily 03:00 UTC)
schtasks /create /F /TN "PRISM-DriftMonitor" ^
  /TR "%PSCMD% \"%REPO_ABS%\scripts\drift_monitor.ps1\"" ^
  /SC DAILY /ST 03:00 /RL HIGHEST /RU SYSTEM
if errorlevel 1 (echo Failed to create PRISM-DriftMonitor & exit /b 1)

echo Installing PRISM-DailyDigest (daily 08:00 UTC)
schtasks /create /F /TN "PRISM-DailyDigest" ^
  /TR "%PSCMD% \"%REPO_ABS%\scripts\daily_digest.ps1\"" ^
  /SC DAILY /ST 08:00 /RL HIGHEST /RU SYSTEM
if errorlevel 1 (echo Failed to create PRISM-DailyDigest & exit /b 1)

echo.
echo ✔ All 4 scheduled tasks installed. Verify with:  schtasks /query /tn PRISM-*
exit /b 0
