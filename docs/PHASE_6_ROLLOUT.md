# Phase 6 Smart Money — Rollout Runbook

> Operator playbook for turning on the OB / Sweep / Po3 confluence layer
> shipped in PRs #13–#17. Designed for **observability-first rollout**:
> populate the data, watch it, then turn on gates one at a time.
>
> Last updated: 2026-04-27

---

## 1. Why this doc exists

Phase 6 ships the smart-money detectors (`OrderBlockDetector`, `SweepDetector`,
`Po3Detector`) and wires them through `SignalGenerator` behind a master env
switch that **defaults off**. PRD2's spec defaults the sub-flag gates to `1`,
but flipping all four at once on day one would mean we never see what the
detectors actually produce on real bars before they start blocking trades —
and worse, when Phase 7 (ML feature engineering) retrains on smart-money
features, it would be fitting on synthetic test distributions instead of
real ones.

This runbook locks in the three-stage rollout that prevents both problems:

1. **Stage 1 — observability**: detectors run, packet is enriched, Slack shows
   the `🎯 Smart Money` block, **nothing is blocked**.
2. **Stage 2 — sweep gate**: signals require a recent direction-matching
   liquidity sweep with displacement.
3. **Stage 3 — Po3 gate**: signals require Po3 entry phase (sweep +
   displacement on the entry timeframe).

Each stage is a single env-var flip with single-env-var rollback.

---

## 2. Pre-flight checklist (run on the production runner)

```bash
git pull origin main && git log -1 --oneline   # expect >= 3f08061
python3 -m pytest tests/ -q                    # expect >= 468 passed
```

Confirm:

- `PRISM_SLACK_TOKEN` is live and the runner can post — the Stage 1 indicator
  is the new `🎯 Smart Money` block landing in `#prism-signals`. If the Slack
  card looks unchanged after Stage 1, **stop** and check the deployment.
- Logs land somewhere durable (file, Datadog, CloudWatch — anywhere greppable).
  We now log detector exceptions at `ERROR` level with `exc_info=True`, so the
  traceback needs to be capturable. A console-only deployment will lose them.
- `gen.detector_failure_counts` is exposed wherever the runner dumps
  process-level state. If you don't already log per-instrument counters at
  scan-end, **wire that before flipping the switch** — silent failures are
  the only way Phase 6 can hurt you, and the counters are the canary.

---

## 3. The three stages

### Stage 1 — observability only

```env
PRISM_SMART_MONEY_ENABLED=1
PRISM_SWEEP_REQUIRED=0
PRISM_PO3_REQUIRED=0
```

Restart the runner. Every signal that fires now carries `signal.smart_money`
populated with `ob`/`sweep`/`po3` sub-dicts and the Slack card includes the
new section. **Zero signals are blocked** by Phase 6 in this stage.

#### What to watch (Days 1–5)

| Health signal | Where to check | Action threshold |
|---|---|---|
| `OrderBlockDetector failed` ERROR log | log destination | any occurrence → investigate before Stage 2 |
| `SweepDetector failed` ERROR log | log destination | any occurrence → investigate before Stage 2 |
| `Po3Detector failed` ERROR log | log destination | any occurrence → investigate before Stage 2 |
| `gen.detector_failure_counts[*]` | runner state dump | any non-zero value → investigate before Stage 2 |
| `🎯 Smart Money` block in Slack | `#prism-signals` | absent on any signal → deployment didn't pick up the env change |

#### Empirical data to capture (this is what feeds Phase 7.A)

For every signal that fires, record:

- `sm.ob`: did it populate? `state`, `direction`, `effective_direction`,
  `distance_pips`, `is_rejection_block`, `in_range`
- `sm.sweep`: `type`, `bars_ago`, `displacement_followed`, `qualifies`
- `sm.po3`: `phase`, `session`, `is_entry_phase`, `range_size_pips`

Track B (planned: `prism/audit/smart_money_export.py`) will automate this
extraction from the existing audit log. Until then, eyeballing 10–20 Slack
cards across one London + one NY session per day is enough to catch obvious
distributional weirdness.

#### Sanity targets from the research corpus

These are rough — the whole point of Stage 1 is replacing them with real
numbers — but they're the failure flags:

- Po3 `DISTRIBUTION` should fire **~1–3× per London/NY kill zone**, **~0×
  in Asian**. Zero across a full week of London sessions = detector bug.
  Every signal in `DISTRIBUTION` = thresholds are too loose.
- Liquidity sweeps roughly cluster around kill-zone openings (London 07:00,
  NY 13:00 UTC). A sweep firing 50× per hour at 03:00 UTC = noise wicks
  passing the filter; tighten `min_displacement_pips`.
- OB `distance_pips` for `in_range=true` blocks: typically 5–30 on EURUSD.
  All distances >100 = stale OBs not being consumed; check `update_states`
  cursor logic.

---

### Stage 2 — sweep gate

```env
PRISM_SWEEP_REQUIRED=1
```

Restart. Now signals without a recent qualifying sweep are blocked at the
smart-money layer (between Layer 3 FVG retest and Layer 4 SL/TP). Block
reason in logs: `Smart-money gate blocked: no recent qualifying sweep for
direction`.

#### Advancement criteria — Stage 1 → Stage 2

Do **not** advance until all of:

- 48 consecutive hours with zero entries in `gen.detector_failure_counts`
- Manual review of 5+ Slack cards with `Sweep: ✅` confirms swept levels
  look like real swing highs/lows (not single-tick noise wicks)
- Sweep fire rate isn't degenerate: not 0 across a full trading day, not
  on 100% of signals

#### What to watch (Days 6–8)

| Metric | Threshold | If exceeded |
|---|---|---|
| Stage-2 block rate (% signals filtered by sweep gate) | 30–50% (PRD2 baseline) | >70% → tighten `lookback` or relax `min_displacement_pips`; <10% → tighten `min_displacement_pips` |
| Win rate of surviving signals vs. pre-Stage-2 baseline | ≥ baseline | <baseline → roll back to Stage 1, the gate is filtering wrong |

---

### Stage 3 — Po3 gate

```env
PRISM_PO3_REQUIRED=1
```

Restart. Signals also require Po3 to be in entry phase (`sweep_detected AND
displacement_detected`). This is the most aggressive filter — Po3
DISTRIBUTION is rare by design.

#### Advancement criteria — Stage 2 → Stage 3

- Stage 2 block rate is plausible (30–50%, not >70%)
- Manual review of 5+ "blocked-by-sweep" log entries confirms they were
  correctly filtered (the missing sweep would have been a low-quality entry)
- Win rate is non-degraded

#### What to watch (Days 9+)

`is_entry_phase=True` is rare. Expect Stage 3 to filter out a meaningful
fraction even of Stage 2 survivors. If the combined block rate (Stage 2 +
Stage 3) crosses 90%, you've stacked filters too tight and need to relax
`displacement_pips` on `Po3Detector`.

---

## 4. Rollback

Single env flip:

```env
PRISM_SMART_MONEY_ENABLED=0
```

Restart. No code revert needed. Existing positions are not affected — gates
only ever apply to *new* entries. Ladder downward stage-by-stage if you only
want to back off one filter:

| Symptom | Action |
|---|---|
| Stage 3 filtering too hard | `PRISM_PO3_REQUIRED=0` (back to Stage 2) |
| Stage 2 filtering too hard | `PRISM_SWEEP_REQUIRED=0` (back to Stage 1) |
| Detector crashing in production | `PRISM_SMART_MONEY_ENABLED=0` (full rollback) — then fix the detector under master-off, ship a follow-up PR, restart at Stage 1 |

---

## 5. Where the code lives

| Surface | File | What it does |
|---|---|---|
| Master switch + sub-flag reads | `prism/signal/generator.py:_evaluate_smart_money` (L238) | Returns `None` when master is off; otherwise runs all three detectors |
| Failure counters | `prism/signal/generator.py:74` (`self.detector_failure_counts`) | Per-detector counts, ops-scrapeable |
| Gate ordering rationale | `prism/signal/generator.py:373` (`elif po3_required` + preceding comment) | Inline comment explains fail-fast: sweep before Po3 in ICT order |
| Failure contract tests | `tests/test_phase6e_resilience.py` | 7 tests covering OB-observability-only, sweep/po3 fail-closed-when-required |
| Integration tests | `tests/test_phase6d_integration.py` | 9 tests covering populated packet, gate behaviour, Slack rendering |
| Slack `🎯 Smart Money` block | `prism/delivery/slack_notifier.py:_format_signal_blocks` | Reads `signal.smart_money`; falls back to "none" lines per sub-dict |

---

## 6. Open questions / future work

- **Track A — `docs/PHASE_7A_SCOPE.md`**: scope doc for the ML feature
  engineering retrain that consumes Stage 1 data. Five features available
  pre-Phase-8: `htf_alignment`, `kill_zone_strength`, `sweep_confirmed`,
  `ob_distance_pips`, `po3_phase`. Two more (`fvg_quality_score`, `ote_zone`)
  deferred to Phase 7.B after Phase 8 lands.
- **Track B — `prism/audit/smart_money_export.py`**: extract `smart_money`
  fields from the audit log into CSV/parquet so empirical distributions can
  be plotted without hand-eyeballing Slack cards.
- **Future — production metrics surface**: `gen.detector_failure_counts` is
  a primitive. Once Stage 1 has logged a few weeks, decide whether it earns
  a proper StatsD/Prometheus metric or stays as a logged scalar.
