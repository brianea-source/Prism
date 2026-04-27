# Phase 7.A — ML Feature Enhancement Results

> Per-instrument retraining outcomes for the five ICT features defined in
> [PHASE_7A_SCOPE.md](./PHASE_7A_SCOPE.md). Each section is filled in
> after the candidate model passes (or fails) the gates and the artifact
> at `models/phase7a_walkforward_<instrument>.json` is committed.
>
> Status: shell — populated as instruments retrain after the audit log
> accumulates ≥ 2 weeks of multi-session data.
>
> Last updated: shell created 2026-04-27, no instruments retrained yet.

---

## Acceptance gates (carried from scope §6)

| # | Metric | Gate | Source of truth |
|---|---|---|---|
| 1 | Walk-forward median F1 | new ≥ baseline | `phase7a_walkforward_<instrument>.json` → `decision.gates.gate_1_f1` |
| 2 | Walk-forward median Sharpe | new ≥ baseline × 0.95 | same artifact → `gate_2_sharpe` |
| 3 | Walk-forward median Max DD | new ≤ baseline × 1.10 (in magnitude) | same artifact → `gate_3_max_drawdown` |
| 4 | Top-20 SHAP feature stability | ≥ 60% overlap with baseline top-20 | SHAP harness output |
| 5 | Stage 1 vs. historical distribution drift | ≤ 1 of 5 features rejects at α = 0.01 (Bonferroni-corrected α/5 of 0.05 family) | `python -m prism.audit.smart_money_export diff` output |
| 6 | Live A/B in NOTIFY mode | 2 weeks, ≥ 30 signals | runner Slack history |

A feature set is promoted to `models/` only when **all 6 gates pass**.
Failure at any gate keeps the candidate in `prism_v2_dev/` per PRD2.

---

## Reproducing a result

For each instrument, the pipeline that produces the rows below is:

```bash
# 1. Stage 1 audit log already accumulating in state/signal_audit/<instrument>/
#    (see docs/PHASE_6_ROLLOUT.md §2 pre-flight)

# 2a. Build the FULL historical state sidecar (all bars — for training)
python -m prism.data.historical_state \
    --instrument EURUSD \
    --h4 data/EURUSD_h4.parquet \
    --h1 data/EURUSD_h1.parquet \
    --entry data/EURUSD_h1.parquet \
    --output models/historical_state_EURUSD.parquet

# 2b. Build the SIGNAL-CONDITIONED sidecar (gate-5 comparison population)
#     Gate-5 compares live audit-log bars (signal-producing only) against
#     the historical builder. The live side is already signal-conditioned;
#     the historical side must be too, otherwise selection-bias differences
#     dominate the drift test.
python -m prism.data.historical_state \
    --instrument EURUSD \
    --h4 data/EURUSD_h4.parquet \
    --h1 data/EURUSD_h1.parquet \
    --entry data/EURUSD_h1.parquet \
    --signal-conditioned-only \
    --output models/historical_state_EURUSD_conditioned.parquet

# 3. Run gate 5 first — if it fails, fix the historical builder before
#    spending compute on retraining.
#    ⚠️  Use the CONDITIONED sidecar from step 2b, not the full sidecar.
python -m prism.audit.smart_money_export diff \
    --start 2026-04-27 --end 2026-05-25 \
    --instrument EURUSD \
    --historical models/historical_state_EURUSD_conditioned.parquet \
    --features htf_alignment:int_ordinal \
    --features kill_zone_strength:int_ordinal \
    --features sweep_confirmed:bool \
    --features ob_distance_pips:continuous \
    --features po3_phase:categorical

# 4. Retrain with walk-forward, gates 1-3 evaluation.
#    ⚠️  Confirm PRISM_OB_MAX_DISTANCE_PIPS matches the value used
#    during Stage 1 (check sidecar metadata with:
#      python -c "from prism.data.historical_state import read_sidecar_metadata; \
#                 print(read_sidecar_metadata('models/historical_state_EURUSD.parquet'))")
PRISM_OB_MAX_DISTANCE_PIPS=50 \
python prism/model/retrain.py \
    --instrument EURUSD \
    --walkforward \
    --phase7a-sidecar models/historical_state_EURUSD.parquet

# 5. Manually run SHAP harness for gate 4 (out of scope for this PR)

# 6. Flip NOTIFY mode for the candidate model and let it accumulate
#    2 weeks of A/B comparison vs. the production model
```

Once `models/phase7a_walkforward_<instrument>.json` is committed, fill
in the corresponding section below.

---

## EURUSD

> Status: pending. Populate after first retrain.

```text
TBD — copy the artifact summary block here once
models/phase7a_walkforward_EURUSD.json is committed.
```

| Gate | Baseline | Candidate | Threshold | Pass? |
|---|---|---|---|---|
| 1 — F1 | TBD | TBD | new ≥ baseline | TBD |
| 2 — Sharpe | TBD | TBD | new ≥ baseline × 0.95 | TBD |
| 3 — Max DD | TBD | TBD | new ≤ baseline × 1.10 | TBD |
| 4 — SHAP stability | TBD | TBD | ≥ 60% top-20 overlap | TBD |
| 5 — Drift | TBD | — | ≤ 1/5 reject @ α=0.01 | TBD |
| 6 — Live A/B | TBD | — | ≥ 30 signals over 2wk | TBD |

**Decision:** TBD.

---

## XAUUSD

> Status: pending.

| Gate | Baseline | Candidate | Threshold | Pass? |
|---|---|---|---|---|
| 1 — F1 | TBD | TBD | new ≥ baseline | TBD |
| 2 — Sharpe | TBD | TBD | new ≥ baseline × 0.95 | TBD |
| 3 — Max DD | TBD | TBD | new ≤ baseline × 1.10 | TBD |
| 4 — SHAP stability | TBD | TBD | ≥ 60% top-20 overlap | TBD |
| 5 — Drift | TBD | — | ≤ 1/5 reject @ α=0.01 | TBD |
| 6 — Live A/B | TBD | — | ≥ 30 signals over 2wk | TBD |

**Decision:** TBD.

---

## GBPUSD

> Status: pending.

| Gate | Baseline | Candidate | Threshold | Pass? |
|---|---|---|---|---|
| 1 — F1 | TBD | TBD | new ≥ baseline | TBD |
| 2 — Sharpe | TBD | TBD | new ≥ baseline × 0.95 | TBD |
| 3 — Max DD | TBD | TBD | new ≤ baseline × 1.10 | TBD |
| 4 — SHAP stability | TBD | TBD | ≥ 60% top-20 overlap | TBD |
| 5 — Drift | TBD | — | ≤ 1/5 reject @ α=0.01 | TBD |
| 6 — Live A/B | TBD | — | ≥ 30 signals over 2wk | TBD |

**Decision:** TBD.

---

## Cross-instrument observations

> Populate after at least 2 of 3 instruments complete the harness. Per
> scope §8.4: passing instruments may be promoted independently — the
> per-instrument model files mean the asymmetry doesn't compound. Note
> any features where lift was instrument-specific (e.g. `kill_zone_strength`
> contributing more on indices than majors).

TBD.

---

## Phase 7.B follow-up (after Phase 8)

When Phase 8's `check_ote_zone()` lands and the FVG `formed_bar` →
timestamp migration is complete, re-run this harness with the same
fold geometry to evaluate `fvg_quality_score` + `ote_zone`. The clean
A/B between Phase 7.A and Phase 7.A+B keeps the marginal contribution
of the two deferred features attributable.

Track this as a separate results doc (`PHASE_7B_RESULTS.md`) when
Phase 8 closes.
