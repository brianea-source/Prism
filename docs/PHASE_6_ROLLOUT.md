# Phase 6 Rollout Runbook — Smart Money Observability

**Merge:** `3f08061` | **Tests:** 468 passed | **Status:** Production-ready

## Pre-flight (5 min)

```bash
# On the prod runner
git pull origin main && git log -1 --oneline  # expect 3f08061
python3 -m pytest tests/ -q                   # expect 468 passed
```

Confirm:
- `PRISM_SLACK_TOKEN` is live (the new 🎯 Smart Money block lands in #prism-signals)
- Log destination is durable — we now `logger.error(exc_info=True)` so tracebacks must be grep-able

## Stage 1 — Observability Only

Set in `.env`:
```
PRISM_SMART_MONEY_ENABLED=1
PRISM_SWEEP_REQUIRED=0
PRISM_PO3_REQUIRED=0
```

Restart the runner. The next signal that fires should produce a Slack card with the `🎯 Smart Money` block beneath the existing HTF Bias section.

## What to Watch (Days 1–5)

### Health Signals

- **Slack:** Every signal card shows the `🎯 Smart Money` block (even when sub-dicts are `None` — you'll see `OB: none in range`, `Sweep: none recent`, etc.)
- **Logs:** Zero `OrderBlockDetector failed` / `SweepDetector failed` / `Po3Detector failed` ERROR lines
- **In-process:** `gen.detector_failure_counts` reads `{ob: 0, sweep: 0, po3: 0}` after a full session

### Empirical Data to Capture per Signal (feeds Phase 7.A)

- Did `sm.ob` populate? Direction, `distance_pips`, `state`?
- Did `sm.sweep` populate? `type`, `bars_ago`, `displacement_followed`?
- Did `sm.po3` populate? Which `phase`? Was `is_entry_phase` true?

### Sanity Targets (from research corpus)

- Po3 `DISTRIBUTION` should fire ~1–3× per London/NY kill zone, ~0 in Asian
- Sweeps roughly correlate with kill-zone openings (London 07:00, NY 13:00 UTC)
- OB `distance_pips` for in-range OBs: typically 5–30 on EURUSD; XAUUSD will be larger but uses `PIP_SIZE=0.01` so same scale roughly applies

## Stage Advancement Criteria

### Stage 1 → Stage 2 (`PRISM_SWEEP_REQUIRED=1`)

- 48h zero detector exceptions
- Manual review of 5+ Slack cards with `Sweep: ✅` confirms swept levels look like real swing highs/lows (not noise wicks)
- Sweep fire rate isn't degenerate (not 0 across the day, not on every signal)

### Stage 2 → Stage 3 (`PRISM_PO3_REQUIRED=1`)

- Stage 2 block-rate is plausible (PRD2 implies 30–50%; >70% means thresholds need tuning, not gating)
- Win-rate of surviving signals is ≥ pre-Stage-2 baseline (no degradation)
- 5+ "blocked-by-sweep" signals reviewed manually and confirmed correctly filtered

## Rollback

Single env flip, no code revert needed. Existing positions untouched (gates only affect new entries):

```
PRISM_SMART_MONEY_ENABLED=0
```

Restart the runner.

## Parallel Tracks (run alongside observability)

| Track | Deliverable | Blocks on |
|-------|-------------|-----------|
| **A** | `docs/PHASE_7A_SCOPE.md` — 5 ICT features specced (`htf_alignment`, `kill_zone_strength`, `sweep_confirmed`, `ob_distance_pips`, `po3_phase`); normalisation, walk-forward harness, deferred Phase 7.B work | Stage 1 data |
| **B** | `prism/audit/smart_money_export.py` — reads per-signal audit log, emits CSV/parquet of `{signal_time, instrument, direction, ob_*, sweep_*, po3_*}` | Track A merge |
| **C** | This doc | — |

Recommended order: C → A → B.
