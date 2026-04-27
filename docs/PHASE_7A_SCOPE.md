# Phase 7.A — ML Feature Enhancement (Pre-Phase-8 Scope)

> Scope and acceptance criteria for the first half of PRD2's Phase 7.
> Defines which ICT-derived features can be added to the training pipeline
> **before Phase 8 ships**, the data path for retraining, and the gates
> that promote a new feature set to `main`.
>
> Status: scoped, ready to merge the moment Stage 1 of the Phase 6 rollout
> produces enough live data to validate the historical reconstruction.
>
> Last updated: 2026-04-27

---

## 1. Why split Phase 7

PRD2 §7 specifies seven new features: `fvg_quality_score`, `sweep_confirmed`,
`ob_distance_pips`, `po3_phase`, `ote_zone`, `kill_zone_strength`,
`htf_alignment`. Two of those (`fvg_quality_score`, `ote_zone`) depend on
machinery that **does not exist on `main` yet**:

- `fvg_quality_score` needs OB-confluence-within-10-pips, ATR-relative size,
  and FVG age — and the FVG age tracker doesn't survive across runner
  restarts because `prism/signal/fvg.py` doesn't persist its history.
- `ote_zone` needs the swing-high/low + retracement logic that
  `prism/signal/quality_filter.py` (Phase 8) is meant to introduce. PRD2
  says the function signature should match Phase 8's `check_ote_zone()` —
  building it standalone now risks signature drift.

Trying to ship all seven at once means waiting for Phase 8, and the lift
from the other five is large enough on its own to justify a retrain.
**Phase 7.A** = the five buildable-now features. **Phase 7.B** = the two
deferred features, merged after Phase 8 lands.

---

## 2. The five features (Phase 7.A)

All five can be computed today from modules already on `main` at `3f08061`.

### 2.1 `htf_alignment` — int (0–3)

| Aspect | Value |
|---|---|
| Source (offline / training) | `prism/signal/htf_bias.py:get_htf_bias(df_1h, df_4h)` → `HTFBiasResult.bias_1h`, `bias_4h`. Standalone function — what `historical_state.py` calls per-bar. |
| Source (live / validation) | `SignalPacket.htf_bias` (separate field — **not** inside `smart_money`). Audit log writes both `htf_bias` and `smart_money` per scan; gate 5 reads `htf_bias.bias_1h` / `bias_4h` directly. |
| Inputs | 1H bias enum, 4H bias enum, signal direction (`LONG`/`SHORT`) |
| Function | `compute_htf_alignment(bias_1h, bias_4h, direction)` per PRD2 §7 |
| Encoding | 0 = both against, 1 = one against, 2 = one aligned + one neutral, 3 = both aligned |
| Normalization | leave as ordinal int (already bounded, monotone with quality) |
| Notes | `Bias` enum has `BULLISH`/`BEARISH`/`RANGING`. Treat `RANGING` as neutral in the alignment scoring; never as agreement. The stateful `HTFBiasEngine.refresh()` wrapper exists too but is for the live runner's per-session caching — historical builder uses the stateless `get_htf_bias()` to avoid replaying the cache state. |

### 2.2 `kill_zone_strength` — int (0–3)

| Aspect | Value |
|---|---|
| Source | pure function of bar UTC hour; no module dependency |
| Function | `compute_kill_zone_strength(hour_utc)` per PRD2 §7 |
| Encoding | 0 = off-session, 1 = Asian, 2 = London/NY edge, 3 = London/NY core |
| Normalization | leave as ordinal int |
| Notes | Existing `prism/data/pipeline.py:_session()` produces a different scheme (0–4 with overlap separately). **Don't reuse it.** Add `kill_zone_strength` as a *new* column; let the model learn from both if there's signal in both, drop one in feature selection if not. |

### 2.3 `sweep_confirmed` — bool (0/1)

| Aspect | Value |
|---|---|
| Source | `prism/signal/sweeps.py:SweepDetector.has_recent_sweep(direction)` |
| Inputs | sweep history list (built by `detect()` over the lookback window), direction |
| Encoding | 1 if a direction-matching sweep with `displacement_followed=True` exists within the last `lookback` bars; 0 otherwise |
| Normalization | already 0/1 |
| Notes | The detector returns a richer `LiquiditySweep` object — for the feature matrix, collapse to bool. Keep `bars_ago` and `displacement_followed` as side columns if the audit pipeline (Track B) emits them; can be promoted to features in 7.B if shown to be predictive. |

### 2.4 `ob_distance_pips` — float (0 to ~200)

| Aspect | Value |
|---|---|
| Source | `prism/signal/order_blocks.py:OrderBlockDetector.distance_to_ob(price, direction)` |
| Inputs | current bar close, signal direction |
| Encoding | float pips to nearest *active* direction-matching OB; `None` when no active OB |
| Normalization | encode `None` as `-1` sentinel; **also add `ob_in_range = (dist <= 30)` companion bool**. Reasoning: the model can't learn the discontinuity between "no OB" and "OB at 200 pips" from a numeric column alone. The bool gives it a clean cliff. |
| Notes | Tied to `PRISM_OB_MAX_DISTANCE_PIPS` (default 30). If the env var changes between training and inference, the distribution will shift — **lock the env value into the trained model artifact** (write to the model JSON sidecar so retrain detects mismatch). |

### 2.5 `po3_phase` — categorical (4 one-hot columns)

| Aspect | Value |
|---|---|
| Source | `prism/signal/po3.py:Po3Detector.detect_phase().phase` |
| Inputs | session window OHLC, instrument |
| Encoding | 4 binary columns: `po3_accumulation`, `po3_manipulation`, `po3_distribution`, `po3_unknown`. Exactly one is `1` per row. |
| Normalization | one-hot **not** ordinal — the phases aren't a quality ladder, they're categorical states. ACCUMULATION isn't "halfway to DISTRIBUTION"; an ordinal `0/1/2/3` would mislead the gradient boosting splits. |
| Notes | PRD2 §7 specifies `int (0-2)` because the spec didn't account for `Po3Phase.UNKNOWN` (returned when there's not enough data, which happens at session boundaries). Carrying UNKNOWN as its own column is correct; collapsing it to `-1` or merging into ACCUMULATION would force the splitter to learn the boundary indirectly. |

---

## 3. Deferred to Phase 7.B (after Phase 8)

| Feature | Blocker | Earliest possible |
|---|---|---|
| `fvg_quality_score` | Needs OB-confluence-within-10-pips check, ATR-relative size, and a stable FVG age. The first two are trivial. The age problem is subtler than "persistence is missing" — `FVGDetector.save()` exists at `prism/signal/fvg.py:204` and `load()` at line 219. The actual blocker is that `FVGZone.formed_bar` is a **df-relative integer index** (line 37), not a timestamp. Across runner restarts the input dataframe's row 0 shifts, so `age_bars = len(df) - formed_bar` (line 136) silently drifts. Fixing this means migrating `formed_bar` to a UTC timestamp and recomputing `age_bars` against the current bar's timestamp — a small surgery, but one that touches FVG semantics enough to be its own PR. Bundling with Phase 8 keeps the FVG changes in one review. | Phase 8 |
| `ote_zone` | Needs OTE retracement logic that Phase 8's `check_ote_zone()` will introduce. PRD2 §8 requires the signatures to stay aligned to avoid drift; building two implementations risks divergence. | Phase 8 |

Phase 7.B should be a single PR that adds both features once Phase 8 lands.
Re-running the same retrain harness from Phase 7.A then gives a clean A/B
between the two feature-set sizes.

---

## 4. Data paths

### 4.1 Training data path (the only one that matters for the retrain)

**Phase 7.A retrains on historical bars, not on live Stage 1 output.** A
six-week observability window cannot supply enough labelled examples for
a model retrain — we need years. The historical path is therefore the only
viable one for the actual training, and PRD2 already specifies it:

> *PRD2 §7.2:* "Cache historical detector state — Phase 6 detectors (OB,
> sweep, Po3) need bar-by-bar state to feed Phase 7's enrichment for offline
> retraining. Add a `prism/data/historical_state.py` builder that walks the
> training set once and writes detector outputs to a parquet sidecar."

This stays the canonical approach. The implementation PR for Phase 7.A
must include `prism/data/historical_state.py` (PRD2 §7) and wire its
sidecar parquet into `PRISMFeaturePipeline._engineer_features()`.

### 4.2 Validation data path (what Stage 1 actually buys us)

Stage 1 (`PRISM_SMART_MONEY_ENABLED=1`, both gates `0`) doesn't produce
training data — but it produces something equally important: a **live
distribution we can compare the historical reconstruction against**. If
the historical builder thinks `po3_phase=DISTRIBUTION` fires 1.8× per
London kill zone on average, but Stage 1 logs say it fires 0.3×, the
historical reconstruction has a bug — and we want to find that *before*
training a model on it.

**This means Stage 1 must produce structured per-signal records.** Right
now it doesn't (see §4.3). Fix it before flipping the switch.

### 4.3 Stage 1 prerequisite: structured signal audit log (new)

`prism/delivery/runner.py` currently emits signals to two sinks:

1. Slack (human-readable; the smart-money block is text, not data)
2. MT5 (only execution-relevant fields)

**There is no structured persistent log of `signal.smart_money` per scan.**
A `grep -r "audit_log\|signal_audit" prism/` returns no matches. Without
this, Track B's planned `prism/audit/smart_money_export.py` has nothing to read, and
the validation path in §4.2 has no source data.

**Required addition before Stage 1 starts** — a small new module:

```
prism/delivery/signal_audit.py    (new, ~50 lines)
    • Path: state/signal_audit/<instrument>/YYYY-MM-DD.jsonl
    • One line per scan that produced a non-None signal
    • Schema: {timestamp, instrument, direction, confidence, signal_id,
      htf_bias, smart_money}  (asdict() of the SignalPacket subset)
    • Append-only, daily-rotated, no compression at write time
    • Wire-in: runner.py:374 immediately after `gen.generate()`,
      gated on `PRISM_SIGNAL_AUDIT_ENABLED` (default 1)
```

This is **NOT** part of Phase 7.A's PR — it's a small companion PR that
ships alongside the rollout doc (or as a Phase 6.F follow-up) so Stage 1
data is usable when it arrives. Track B (`prism/audit/smart_money_export.py`, matching the path declared in PR #19's rollout doc) is then
a pure consumer of this log and can ship after Phase 7.A.

If the audit log isn't wired before Stage 1, Phase 7.A still ships — but
the validation path in §4.2 collapses to "trust the historical builder",
which is exactly the failure mode this rollout is structured to avoid.

---

## 5. Walk-forward harness

Inherits the chronological split logic already in
`PRISMFeaturePipeline.split_train_test()` (`prism/data/pipeline.py:291`).
Phase 7.A adds rolling-window walk-forward on top, because a single 80/20
chronological split conflates "did the model learn?" with "did the
features happen to help in 2025 specifically?".

**Spec:**

| Parameter | Value | Rationale |
|---|---|---|
| Window length | 18 months | Long enough to span >1 macro regime |
| Train/test ratio inside window | 80/20 chronological → 14.4-month train, 3.6-month test | Match existing split in `pipeline.py:split_train_test` |
| Step size | 1 month | Cadence the runner already retrains at (PRD2 §4) |
| Total folds | 24 monthly anchors → 24 distinct OOS test starts spanning **2 years**, **~3.5 years total data required** (last fold ends at month 41) | Matches `retrain.py`'s 3-year default window with ~6 months of slack for the trailing test segment |
| Comparison | per-fold: baseline features vs. baseline + Phase 7.A features | Want both absolute lift AND lift consistency |
| Aggregate | median lift across folds, with min/max envelope | Median resists single-fold luck; envelope catches "great on average, bad in 6 of 24" |

**Independence caveat (don't oversell the fold count).** Adjacent test
windows overlap by 3.6 − 1 = 2.6 months. The 24 fold results are NOT 24
independent samples — they're a smoothed local-stability estimator with
~5× effective independence at best. Treat the median as
"feature lift is stable across this slice of history" rather than
"24 statistically independent confirmations." If a stricter independence
property is needed later, drop to non-overlapping 4-month test windows
(step = 4) — that gives ~10 truly independent folds at the cost of less
temporal resolution.

**Output artifact:** `models/phase7a_walkforward_<instrument>.json` with per-fold rows and an aggregate summary, mirrored to `docs/PHASE_7A_RESULTS.md` once retrain runs (PRD2 §7 calls for `PHASE_7_AB_RESULTS.md`; renaming to `7A_RESULTS` keeps the A/B split clear).

---

## 6. Acceptance criteria

PRD2 §7 already specifies the gates. Phase 7.A inherits them, with one
addition (gate 5):

| # | Metric | Gate | Rationale |
|---|---|---|---|
| 1 | Walk-forward median F1 | new ≥ baseline | If F1 regresses across the median fold, the new features are net noise |
| 2 | Walk-forward Sharpe | new ≥ baseline × 0.95 | 5% slack for variance |
| 3 | Walk-forward max drawdown | new ≤ baseline × 1.10 | Don't trade Sharpe for blow-up risk |
| 4 | Top-20 SHAP feature stability | ≥ 60% overlap with baseline top-20 | Wholesale churn = data-leakage suspect |
| 5 | **Stage 1 vs. historical distribution drift** | At most **1 of 5** features rejects the null at the per-feature α (see §6.1 for tests + correction) | If 2+ features have significantly different live vs. historical distributions, the historical builder is wrong and the model is fitting on a synthetic distribution |
| 6 | Live A/B in NOTIFY mode | 2 weeks, ≥ 30 signals | Live distribution must match backtest |

If gate 5 fails, **fix the historical builder before retraining** — don't
just retrain on more data. The whole point of the validation data path
(§4.2) is to catch reconstruction bugs at this gate, not in production.

If any gate fails, the new feature set stays in `prism_v2_dev/` per PRD2.

### 6.1 Gate 5 — drift test selection and population conditioning

**Wrong test = wrong gate.** KS assumes continuous unbounded distributions
and is invalid for the four categorical features in Phase 7.A. One test
per feature, chosen by feature type:

| Feature | Type | Test | Bins / DoF |
|---|---|---|---|
| `htf_alignment` | int 0–3 | Chi-squared goodness-of-fit | 4 bins, 3 DoF |
| `kill_zone_strength` | int 0–3 | Chi-squared goodness-of-fit | 4 bins, 3 DoF |
| `sweep_confirmed` | bool | Fisher's exact (2×2 contingency) | — |
| `po3_phase` | one-hot 4-cat | Chi-squared goodness-of-fit on the underlying 4-cat (one test, not four) | 4 bins, 3 DoF |
| `ob_distance_pips` | float | KS two-sample | — |

**Multiple-comparisons correction.** Five independent tests at α = 0.05
gives a family-wise false-positive rate of `1 − 0.95⁵ ≈ 22.6%` for "at
least one rejection" under no real drift. To keep gate 5 honest:

- **Per-feature α = 0.01** (Bonferroni: 0.05 / 5). Family-wise α stays at
  0.05.
- **Pass rule:** ≤1 feature rejects at α = 0.01. The "≤1" is *also* a
  deliberate relaxation on top of Bonferroni — one feature drifting
  significantly is more likely a small-sample artefact in Stage 1 than
  a builder bug, and the SHAP stability gate (4) catches it from the
  other direction.

**Population conditioning (don't compare apples to oranges).** The live
audit log written at `runner.py:374` records features only on bars that
*produced a signal* (i.e., bars that passed every prior layer:
HTF-aligned, FVG retest hit, smart-money guard cleared if gates are on).
The historical reconstruction is computed *per-bar*. The two populations
are different:

- **Live distribution** ≈ distribution of features `| signal_fired = True`
- **Naive historical distribution** ≈ unconditional per-bar distribution

A drift on `kill_zone_strength` between those two populations would just
re-confirm "signals fire more often during kill zones" — which is
designed-in, not a builder bug. Gate 5 must therefore condition both
populations on the same selection criterion. Two ways to do this, both
acceptable:

1. **Replay-mode historical builder.** `historical_state.py` runs the
   full signal pipeline (HTF gate + FVG retest + smart-money observability
   evaluation) per-bar and emits a feature row only when the simulated
   signal would have fired. Output is comparable to the audit log row-by-row.
2. **Bar-keyed audit log.** The live audit log writes one record per *scan
   that reached the smart-money layer*, regardless of whether the signal
   ultimately fired downstream. Then both populations are conditioned on
   "smart-money was evaluated" — a higher-up selection point, but
   identical between live and historical.

Phase 7.A spec adopts **option 1** (replay-mode historical builder)
because it stays consistent with PRD2 §7's bar-level walk and doesn't
change the audit log schema after Stage 1 begins. Option 2 is the
fallback if option 1 turns out to be too slow to recompute weekly.

---

## 7. Implementation PR — files

When Phase 7.A is ready to merge (Stage 1 data validated, historical
builder converged, walk-forward green), the implementation PR includes:

| File | Status | Lines (est.) |
|---|---|---|
| `prism/data/feature_engineering.py` | new | ~150 (the 5 functions + `ICTFeatureEngineer.enrich_features` for 5 features) |
| `prism/data/historical_state.py` | new | ~200 (per-bar walk + parquet sidecar writer) |
| `prism/data/pipeline.py` | modified | +30 (`_engineer_features` reads sidecar, calls enricher) |
| `prism/model/walkforward.py` | new | ~150 (24-fold rolling harness) |
| `prism/model/retrain.py` | modified | +20 — `--walkforward` flag, plus the `PRISM_OB_MAX_DISTANCE_PIPS` lock-in to the model JSON sidecar from §2.4 (write at training time; load-time mismatch warns once, errors after one full retrain cycle) |
| `prism/model/predict.py` | modified | +10 — load-time check that the runtime `PRISM_OB_MAX_DISTANCE_PIPS` matches the value in the loaded model's sidecar |
| `tests/test_feature_engineering.py` | new | ~250 (PRD2 §7 specifies 15–18 tests; we ship the 5-feature subset, ~10 tests) |
| `tests/test_historical_state.py` | new | ~120 (parquet round-trip, per-bar correctness, regime change continuity) |
| `tests/test_walkforward.py` | new | ~80 (fold boundaries, no lookahead, JSON output) |
| `docs/PHASE_7A_RESULTS.md` | new | ~100 (filled in post-retrain with actual numbers) |
| `models/phase7a_walkforward_*.json` | new artefact | (one per instrument) |

**Estimated total:** ~1120 lines code + tests + a results doc with numbers.
Single PR, single review pass, single merge — same shape as Phase 6.D.

---

## 8. Open questions / risks

1. **`PRISM_OB_MAX_DISTANCE_PIPS` lock-in.** The OB distance feature
   depends on this env var. If it differs between training and live, the
   model's `ob_in_range` companion bool will fire on a different
   distribution. Resolution: write the env value into the model artifact
   sidecar at training time; runtime asserts equality at load (warn-only
   for the first deploy, error after one full retrain cycle).
2. **Po3 UNKNOWN frequency.** If the historical builder produces 50%
   UNKNOWN at session boundaries but Stage 1 produces 5%, gate 5 fails
   on the Po3 column alone. Mitigation: pad the historical session
   windows symmetrically — match the live runner's bar buffering exactly.
3. **`htf_alignment` definition with RANGING.** PRD2's `compute_htf_alignment`
   treats RANGING as neither aligned nor against. If the production HTF
   detector's RANGING fires more often than the historical builder's
   (e.g. due to a different `lookback`), `htf_alignment=2` rates will
   diverge. Inherit `lookback` from the same env var on both paths.
4. **Single-instrument retraining.** Phase 7.A as scoped retrains one
   instrument at a time (`--instrument` is required in `retrain.py`).
   If two of three (XAU/EUR/GBP) pass gates and one fails, do we promote
   the passing ones? Recommendation: yes, but document the asymmetry in
   `PHASE_7A_RESULTS.md`. The model files are per-instrument anyway.
5. **Track B race.** If `prism/audit/smart_money_export.py` (Track B) starts before
   the audit logger from §4.3 is wired, Track B has no input. Sequence:
   audit-logger PR → Stage 1 flip → Track A PR (this scope) → Track B PR.

---

## 9. Sequencing

```
[Phase 6 rollout]
    ↓
audit-logger PR (Phase 6.F or addendum to rollout doc PR)
    ↓
Stage 1 ENABLED — observability begins, audit log accumulates
    ↓ (≥2 weeks of multi-session data)
Track B PR — prism/audit/smart_money_export.py: parses audit log, emits parquet,
             diff vs. historical builder output
    ↓
historical_state.py builder converges with live (gate 5 passes)
    ↓
Phase 7.A implementation PR — features, walkforward, retrain, results doc
    ↓
[Stages 2 & 3 of Phase 6 rollout proceed in parallel]
    ↓
Phase 8 ships
    ↓
Phase 7.B PR — adds fvg_quality_score + ote_zone, re-runs harness
```

Phase 7.A's implementation PR is **gated on Stage 1 data + audit logger +
historical builder convergence** — not on calendar time. The scope itself
(this doc) is mergeable now and locks the surface area before
implementation begins.
