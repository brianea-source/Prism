# DEC-001 — MT5 Execution Host: Vultr Windows VPS

**Status:** Decided
**Date:** 2026-05-18
**Decision-maker:** Brian (confirmed via Slack DM, 2026-05-18 07:56 CST: "Lock vps")
**Closes:** ACTION_PLAN.md Task 0.2

---

## Context

PRISM needs an execution host for MT5 trades once Phase 7.A gates pass and we move from NOTIFY-only to live trading. Three candidates were on the table per [ACTION_PLAN.md §Task 0.2](../ACTION_PLAN.md):

| Option | Pros | Cons |
| --- | --- | --- |
| (A) Vultr Windows VPS | 24/7 uptime; native MT5; no Mac dependency; already provisioned (`209.250.248.171`); WinRM management surface validated; ~$13.50/mo | Windows licensing footprint; remote-admin overhead |
| (B) Wine on Mac | No new infra cost | Ties execution to Brian's laptop being awake + connected; Wine MT5 reliability is mixed; single-host risk |
| (C) NOTIFY-only (interim) | Zero infra; manual approval gate | Not a long-term answer; signal-to-fill latency dominated by Brian's reaction time |

## Decision

**Lock in Option A — Vultr Windows VPS at `209.250.248.171`.**

## Rationale

1. **Already paid for and operational.** Vultr VPS has been live since 2026-05-04. Exness MT5 installed. `PRISM-Runner` scheduled task running. PRISM-Watchdog (PR #32) deployed. Health-check loop from Mac mini (`scripts/check_prism_runner.py`) operational. Switching away would discard ~2 weeks of integration work.
2. **Audit log RCA validated the host.** [The 2026-05-18 audit-log investigation](../audits/2026-05-18_signal_audit_gap.md) confirmed the VPS host is *not* the problem — the 4-week audit gap was an env-mask bug, not a runner-availability bug. Process uptime is ~36 hours of continuous run as of the latest check.
3. **Operating cost is rounding error.** $13.50/mo against a system targeting 1% risk per trade on a multi-thousand-dollar account is negligible. Wine-on-Mac would be cheaper, but uptime risk is the real cost.
4. **WinRM gives us a programmatic management surface.** Mac mini → Vultr VPS over NTLM-authenticated WinRM is already in production for the heartbeat health check. Diagnostics, env-var updates, scheduled-task restarts are all scriptable. Wine on Mac would be ad-hoc local commands with no remote-management story.
5. **Single-host clarity.** Co-locating runner + MT5 + the upcoming TradingView webhook ingress on one VPS keeps the network topology simple. Wine-on-Mac would still need a public-facing webhook host eventually.

## Consequences

### Unblocks
- Task 0.3 (Provision Vultr Windows VPS) — already done; mark complete in ACTION_PLAN.md.
- Task 0.4 (GitHub Issue trade journal) — `GH_TOKEN` env can be set on the VPS via the same `setx /M` path used for the audit-log fix.
- Task 2.1 (TradingView webhook intake) — hosting target is settled.
- Task 4.1 (Paper trading on Exness demo) — VPS points at demo MT5 server; later flip to live.

### Operating burden accepted
- Vultr invoice (~$13.50/mo) on Brian's account.
- Windows patch cadence — accept Microsoft's monthly Patch Tuesday cycle. Watchdog covers the reboot case.
- Single-region risk (currently Atlanta per IP geolocation). Acceptable; Exness has servers globally and reconnects on disconnect. Multi-region failover is a future concern, not a Phase 0 concern.

### Explicitly NOT decided here
- Live vs demo MT5 account — that's a Phase 4 decision tied to gate-pass completion, not infrastructure.
- Position-sizing or risk caps — already covered by `PRISM_RISK_PCT`, `PRISM_MAX_CONCURRENT`, `PRISM_MAX_DAILY_LOSS_PCT`.
- Disaster recovery — separate decision once the VPS is carrying real capital. Snapshots are cheap.

## Implementation Status

All steps already complete:
- ✅ Vultr Windows Server 2022 VPS provisioned at `209.250.248.171`
- ✅ MetaTrader 5 (Exness) installed
- ✅ Python 3.11 installed
- ✅ Git clone of `brianea-source/Prism` at `C:\Prism`
- ✅ `PRISM-Runner` scheduled task registered as the long-running runner
- ✅ `PRISM-Watchdog` scheduled task for auto-restart (PR #32)
- ✅ `PRISM-DriftMonitor` + `PRISM-DailyDigest` scheduled tasks
- ✅ Mac mini → VPS WinRM health check (every 30 min)
- ✅ Deploy webhook for git-pull-on-push (PR #28, #33)

This decision retroactively documents what Brian has confirmed.

## Review Trigger

Re-evaluate if any of the following become true:
- Vultr region outage exceeds 4 hours in a 30-day window
- VPS monthly cost increases >2× current
- Exness shuts down the Atlanta server or migrates regions in a way that creates >50ms additional latency
- A hardware co-location option appears with materially better latency to Exness servers

Otherwise: no scheduled re-review.
