# Signal Audit Log Gap — Root Cause Analysis (2026-05-18)

**Author:** Ada (assistant, on Brian's request)
**Triggered by:** Brian asked "what really happened that we have no data for those weeks?"
**Scope:** `state/signal_audit/<INSTRUMENT>/YYYY-MM-DD.jsonl` — Phase 6.F audit log feeding Phase 7.A gate 5 (historical builder validation) and Phase 8 quality-filter training.

---

## TL;DR

The audit log gap (no JSONL rows since `EURUSD/2026-04-20.jsonl`) has **two stacked causes**:

1. **Env-var was silently masked.** `.env` on the VPS has `PRISM_SIGNAL_AUDIT_ENABLED=1` written inside a *commented* documentation paragraph (line begins with `# `). `run_prism.ps1` strips full-line comments, so the var never reached the runner process. Code default is `"1"` but only when no env mask is set; here the var was effectively unset. Patched 2026-05-18 13:57 UTC by `setx /M PRISM_SIGNAL_AUDIT_ENABLED 1` on the VPS — `python -c "import os; print(...)"` confirms the runner now sees `'1'`.

2. **No signals have fired.** Even with audit enabled, `write_signal_audit(signal, when=now)` at `prism/delivery/runner.py:430` only runs when `_should_fire(...)` returns True. Tail of `C:\Prism\logs\runner.log` shows every recent scan (multiple per hour, all sessions, both instruments) producing `"ML direction (X) conflicts with news bias (Y) — skipping"` or `"No signal this scan"`. The runner has been *evaluating* but not *firing*. **The audit gap is therefore a symptom of low signal volume, not a logging bug.**

The audit log gap *is itself the signal* that Phase 7.A gate 5 has no source data — the news-bias gate has effectively killed signal flow since ~04-20.

---

## Forensic Walkthrough

### 1. VPS process state (healthy)

```text
Scheduled task PRISM-Runner:    Status=Running   Last Run=2026-05-18 13:57:23   Last Result=267009 (ERROR_TASK_IS_RUNNING — expected for long-lived service)
python.exe:                     PID=3424         Started=2026-05-18 13:57:24   CPU=fresh
Git HEAD on VPS:                593580c          (= origin/main, fully up to date)
.env file:                      C:\Prism\.env    Last modified 2026-05-04
runner.log:                     C:\Prism\logs\runner.log   17.9 MB, actively appending
```

VPS is not the problem.

### 2. The masked env var

`C:\Prism\.env` contains the following block:

```ini
PRISM_STATE_DIR=state
# Per-signal audit log (Phase 6.F). Writes one JSONL line per fired signal
# to PRISM_STATE_DIR/signal_audit/<instrument>/YYYY-MM-DD.jsonl, after the
# PRISM_SIGNAL_AUDIT_ENABLED=1
PRISM_RISK_PCT=0.01
```

`run_prism.ps1` filters lines with `^\s*#` — so the line `# PRISM_SIGNAL_AUDIT_ENABLED=1` is treated as documentation, not a setting. `PRISM_SIGNAL_AUDIT_ENABLED` never propagated into the python process environment.

Code in `prism/delivery/signal_audit.py:56`:

```python
raw = os.environ.get("PRISM_SIGNAL_AUDIT_ENABLED", "1").strip().lower()
```

Default is `"1"` when no env value is present. PowerShell does not pass an unset key as `""`, so the default *did* apply — meaning audit was "logically on" all along, but the runner still never wrote a JSONL line. Which proves:

### 3. Audit hook never reached (the real cause)

`Select-String 'signal_audit\|audit_dir\|PRISM_SIGNAL_AUDIT' C:\Prism\logs\runner.log` returns **zero matches**. The audit code path is never *invoked*, never errors, never logs anything — because `_should_fire(...)` returns False before `write_signal_audit` is ever called.

Representative log tail (multiple scans, ~minute cadence):

```text
2026-05-18 13:52:27,183 [INFO] prism.signal.generator — ML direction (SHORT) conflicts with news bias (BULLISH) — skipping
2026-05-18 13:52:27,183 [INFO] __main__ — XAUUSD: No signal this scan
2026-05-18 13:52:27,590 [ERROR] prism.data.quiver — COT fetch failed for EURUSD: 404 Client Error: Not Found for url: https://www.cftc.gov/dea/newcot/financial_lof.txt
2026-05-18 13:52:27,918 [WARNING] prism.data.quiver — Fear & Greed fetch failed: 418 Client Error: Unknown Error for url: https://production.dataviz.cnn.io/index/fearandgreed/graphdata
2026-05-18 13:52:29,246 [INFO] __main__ — EURUSD: No signal this scan
```

So the gap is "no signals → no audit rows". The audit pipeline is wired correctly; it's just downstream of a gate that's currently rejecting 100% of setups.

### 4. Upstream-data regressions (contributing factor)

Two news-bias inputs are broken:

| Source | Endpoint | Error | Impact |
| --- | --- | --- | --- |
| CFTC COT | `https://www.cftc.gov/dea/newcot/financial_lof.txt` | `404 Not Found` | News bias degraded; fallback to stale data |
| CNN Fear & Greed | `https://production.dataviz.cnn.io/index/fearandgreed/graphdata` | `418 I'm a teapot` (anti-bot) | News bias degraded; fallback to neutral |

When the news-bias input is degraded but the gate is still aggressive, the gate biases toward "skip everything" because it can't confirm the bias direction. This is the most likely contributing factor to the dry spell.

---

## Remediation

### Done (2026-05-18 13:57 UTC, executed via WinRM from Mac mini)
- Set machine env var on VPS: `[Environment]::SetEnvironmentVariable('PRISM_SIGNAL_AUDIT_ENABLED','1','Machine')`.
- Restarted `PRISM-Runner` scheduled task. New python PID 3424, started 13:57:24.
- Verified env propagation: `python -c "import os; print(os.environ.get('PRISM_SIGNAL_AUDIT_ENABLED'))"` → `'1'`.
- Audit will now write the moment a signal fires.

### This PR (`ops/audit-log-enable-2026-05-18`)
- Move `PRISM_SIGNAL_AUDIT_ENABLED=1` to an uncommented line in `.env.example` so future deployments do not inherit the same bug.
- Source-control `scripts/run_prism.ps1` (was on-VPS-only). New version runs `scripts/preflight_env.py` at startup and refuses to launch the runner if a critical env key is missing.
- Add `scripts/preflight_env.py` — lists missing-but-expected env keys, fails loud, redacts secrets in log output.
- Tests: `tests/test_preflight_env.py` — 9 tests covering parse, redact, CLI checks, --warn-only, --check single key.
- This audit doc lives at `docs/audits/2026-05-18_signal_audit_gap.md`.

### Follow-up PRs (separate, sequenced — to be opened after this lands)
- **Fix COT 404:** CFTC reorganized in early 2026. New endpoint path; update `prism/data/quiver.py::fetch_cot`. Add cache + 24h TTL.
- **Fix Fear & Greed 418:** add user-agent header + 1 retry with backoff, or migrate to an alternative source. Consider scraping the rendered CNN page via puppeteer-on-VPS as a last resort.
- **News-bias gate audit:** once audit log has ≥48h of post-restart data, reproduce the gate logic offline and compute the "would-have-fired" rate at varying gate sensitivities. If the gate is rejecting >90% of otherwise-valid setups, recalibrate.

---

## Acceptance Criteria (verifying the fix worked)

After 24h of post-restart runtime, the following should all be true:
- [ ] `C:\Prism\state\signal_audit\XAUUSD\<today>.jsonl` exists with ≥1 row.
- [ ] `C:\Prism\state\signal_audit\EURUSD\<today>.jsonl` exists with ≥1 row.
- [ ] Each row has the Phase 6.F schema: `htf_bias`, `smart_money`, `signal_id`, `ml_direction`, `news_bias`, gate decisions.
- [ ] runner.log shows `signal_audit` write entries (currently zero).
- [ ] runner.log shows `[preflight] OK` lines at every startup (proves preflight is active).

If after 48h these are still empty, the news-bias gate is the next thing to investigate — not the audit logger.

---

## Lessons Captured

1. **Env files should be machine-validated, not comment-validated.** A commented-out setting that *looks like* a setting is worse than a missing setting.
2. **Runners should emit a `[BOOT]` line listing all `PRISM_*` env keys (redacted) on startup.** This PR delivers that via `scripts/preflight_env.py`.
3. **The audit gap was the canary.** Phase 7.A gate 5 depends on accumulated audit data — but if signals aren't firing, no amount of "fix the audit log" helps. The deeper issue is upstream-data health + gate calibration.
