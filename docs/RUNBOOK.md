# PRISM Operator Runbook

> End-to-end playbook: from bare host to live trading.
> Last updated: 2026-04-24

---

## 1. Overview

**PRISM** is a multi-layer ML signal system for FX and gold (XAUUSD, EURUSD, GBPUSD).
It generates directional signals from FVG confluence, macro overlays, and news intelligence,
then routes them to a connected MetaTrader 5 account via a Slack confirmation/execution workflow.

This runbook covers: host setup, model retraining, pre-live health check, starting the runner,
observability, failure recovery, and demo-to-live promotion criteria.

---

## 2. Host Setup

**Requirements:**
- OS: Windows 10/11 (MT5 Python API requires Windows; checks SKIP on macOS/Linux)
- Python 3.10+
- MetaTrader 5 terminal installed and logged into your Exness demo account

```bash
pip install -r requirements.txt
git clone https://github.com/brianea-source/Prism.git
cd Prism
cp .env.example .env
# Fill in .env with your credentials (see section 3)
```

---

## 3. Environment Configuration

| Variable | Required | Description |
|---|---|---|
| PRISM_SLACK_TOKEN | Yes | Bot token for PRISM Signals Slack app (xoxb-...) |
| PRISM_SLACK_CHANNEL | Yes | Channel ID where signals are posted |
| MT5_LOGIN | Yes | Exness MT5 account number |
| MT5_SERVER | Yes | MT5 server name (e.g. Exness-MT5Trial) |
| MT5_PASSWORD | Yes | MT5 account password |
| TIINGO_API_KEY | Yes | Price data and news sentiment |
| FRED_API_KEY | Yes | Macro features (CPI, yield curve, etc.) |
| PRISM_STATE_DIR | optional | State persistence directory (default: state/) |
| PRISM_RISK_PCT | optional | Fraction of balance per trade (default: 0.01) |
| PRISM_MAX_DAILY_LOSS_PCT | optional | Daily drawdown threshold (default: 0.03) |
| PRISM_SUN_OPEN_SKIP_MIN | optional | Minutes to skip after Sunday FX re-open (default: 30) |
| PRISM_INSTRUMENTS | optional | Comma-separated instruments (default: XAUUSD,EURUSD,GBPUSD) |
| PRISM_SCAN_INTERVAL | optional | Seconds between scans (default: 60) |
| PRISM_EXECUTION_MODE | optional | CONFIRM / AUTO / NOTIFY (default: CONFIRM) |
| PRISM_BAR_COUNT | optional | Bars per request (default: 500) |
| PRISM_ALLOW_APPROX_PIP_VALUE | optional | Set 1 to allow fallback pip value (not for live) |

---

## 4. Model Retraining

```bash
python -m prism.model.retrain
```

Expected output: 12 pkl files in models/ (4 layers x 3 instruments).
Sanity check: models/XAUUSD_layer0.pkl through GBPUSD_layer3.pkl must all exist.

---

## 5. Health Check

```bash
python scripts/health_check.py              # full check
python scripts/health_check.py --check environment   # single check
python scripts/health_check.py --no-slack  # skip Slack probe
python scripts/health_check.py --json      # JSON output
python scripts/health_check.py --list      # list all checks
```

Exit codes: 0=all PASS, 1=any FAIL, 2=any WARN no FAIL.

**What each check tests:**

| Check | What it verifies |
|---|---|
| environment | All required env vars set; warns on optional misses |
| state_dir | PRISM_STATE_DIR exists and can round-trip JSON |
| models | All 4 layer pkl files present per instrument |
| session_clock | is_kill_zone() and is_sunday_open_gap() return bool |
| inflight_persistence | In-flight dedup keys survive persist to load round-trip |
| mt5_connect | MT5Bridge.connect() succeeds (SKIP on macOS) |
| mt5_bars | H4/H1/M5 bars fetchable for all instruments |
| pip_value | Live pip value differs from approx fallback |
| reconnect | heartbeat and ensure_connected idempotent; no spurious events |
| drawdown_guard | Guard trips at correct threshold; state file written |
| slack | Tagged probe posted to Slack channel |

---

## 6. Starting the Runner

```bash
python -m prism.delivery.runner
```

**Execution modes:**
- CONFIRM (default): waits for Slack reaction before executing — use for first 5 days
- AUTO: executes immediately on signal
- NOTIFY: posts signal, never executes (observation mode)

**Healthy startup log lines:**
```
[INFO] prism.delivery.runner — PRISM runner starting (mode=CONFIRM, ...)
[INFO] prism.delivery.drawdown_guard — Drawdown guard: new day ..., SOD balance=...
[INFO] prism.execution.mt5_bridge — MT5 connected to Exness-MT5Trial
[INFO] prism.delivery.runner — Scan loop started (interval=60s)
```

Stop gracefully: Ctrl+C or SIGTERM.

---

## 7. Observability

Slack messages posted to PRISM_SLACK_CHANNEL:

| Message | When |
|---|---|
| Signal card | New signal with entry zone, TP, SL, lot size |
| Kill-zone skip | London (07:00-11:00 UTC) or NY (13:00-17:00 UTC) active |
| Sunday-gap skip | FX re-opened less than PRISM_SUN_OPEN_SKIP_MIN minutes ago |
| Drawdown trip | Daily loss crossed threshold; new entries halted |
| Disconnect alert | MT5 unreachable > PRISM_MT5_DISCONNECT_ALERT_SEC seconds |
| Recovery alert | MT5 reconnected after disconnect alert |

---

## 8. Recovery Procedures

### MT5 Disconnect
The bridge retries with exponential backoff (base 10s, max 300s). Slack alert after 2 minutes.
If terminal stays down: restart MetaTrader 5; bridge reconnects on next heartbeat.
If credentials changed: update .env, restart runner.

### Drawdown Trip
Do NOT auto-reset without reviewing what triggered the trip.
To reset after review:
```bash
# 1. Stop runner
# 2. Review MT5 trade history for the session
# 3. If safe to resume:
rm state/drawdown_state.json
python -m prism.delivery.runner
```
Guard resets automatically at UTC midnight even without deleting state.

### Wedged In-Flight Key
Symptom: runner not re-generating signals for an instrument despite active session.
Fix:
```bash
python3 -c "import json; print(json.load(open('state/in_flight_keys.json')))"
echo '{}' > state/in_flight_keys.json
# No restart needed; file re-read on next scan
```

### Revoked Slack Token
1. Go to api.slack.com/apps -> PRISM Signals -> OAuth & Permissions
2. Reinstall to Workspace, copy new xoxb-... token
3. Update .env PRISM_SLACK_TOKEN=xoxb-new-token
4. Restart runner

---

## 9. Demo to Live Promotion Criteria

All must be confirmed before promoting:

1. health_check.py exits 0 with live MT5 credentials
2. 72 hours continuous demo operation with zero unintended trades
3. Drawdown guard tripped and reset at least once during demo
4. MT5 disconnect/reconnect cycle tested; Slack alerts received for both events
5. PRISM_EXECUTION_MODE=CONFIRM for first 5 live trading days
6. Daily drawdown thresholds reviewed relative to live account size (default 3%)

---

## 10. File Location Reference

| File | Purpose | Gitignored |
|---|---|---|
| .env | Runtime secrets and config | Yes |
| .env.example | Config template | No |
| models/*.pkl | Trained model artefacts | Yes |
| state/last_brief_date.txt | Daily brief dedup | Yes |
| state/drawdown_state.json | Drawdown kill-switch state | Yes |
| state/in_flight_keys.json | In-flight dedup (survives restart) | Yes |
| scripts/health_check.py | Pre-live smoke test | No |
| docs/RUNBOOK.md | This document | No |
| prism/delivery/runner.py | Main runner loop | No |
| prism/delivery/drawdown_guard.py | Daily drawdown kill-switch | No |
| prism/delivery/session_filter.py | Kill-zone and Sunday-gap predicates | No |
| prism/execution/mt5_bridge.py | MT5 connection and reconnect logic | No |
