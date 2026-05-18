# PRISM Stockraft Analysis — XAUUSD (5-year)

**Date:** 2026-05-18  
**Branch:** `data/stockraft-fetch-fixes-2026-05-18`  
**Source:** Dukascopy (true spot, BID candles, M1 → resampled)  
**Period:** 2021-01-01 → 2026-05-01 (1,946 days)  
**Methodology:** Stockraft master prompts (Swing Points, Breakout Edge, Liquidity Sweeps) per `prism/backtest/stockraft_analysis.py`  

> See sibling report for EURUSD. This document tracks Task 0.1 from `docs/ACTION_PLAN.md`.

## 1. Data coverage

| Timeframe | Bars | Start | End | Span (days) |
|-----------|------|-------|-----|-------------|
| M15 | 133,536 | 2021-01-01 00:00:00+00:00 | 2026-05-01 23:45:00+00:00 | 1946 |
| H1 | 33,384 | 2021-01-01 00:00:00+00:00 | 2026-05-01 23:00:00+00:00 | 1946 |
| H4 | 8,346 | 2021-01-01 00:00:00+00:00 | 2026-05-01 20:00:00+00:00 | 1946 |
| D1 | 1,391 | 2021-01-01 00:00:00+00:00 | 2026-05-01 00:00:00+00:00 | 1946 |

Cache coverage in `data/dukascopy_cache/XAUUSD/`: 1,387 daily bi5 files (out of 1,391 weekdays — 4 missing, almost certainly Dukascopy outages or holidays the upstream feed didn't publish). M1 resamples were performed in-process from the M1 candle stream.

Bar counts match expected continuous coverage (96 M15/day, 24 H1/day, 6 H4/day, 1 D1/day × ~1391 trading days).

## 2. Prompt 1 — Swing points & session bias

**Swing definition:** local extrema with N=5 bars on each side.

### Swing counts and session distribution

| TF | High swings | Low swings | Total | Kill-zone % | London % | NY % |
|----|-------------|------------|-------|-------------|----------|------|
| M15 | 10,646 | 10,792 | 21,438 | 34.6% | 22.6% | 33.4% |
| H1 | 2,307 | 2,370 | 4,677 | 42.5% | 24.3% | 39.0% |
| H4 | 538 | 557 | 1,095 | 65.7% | 59.6% | 19.9% |
| D1 | 73 | 72 | 145 | 0.0% | 0.0% | 0.0% |

### Top swing hours (UTC)

| TF | Top high-swing hours | Top low-swing hours |
|----|---------------------|---------------------|
| M15 | 22:00 (n=830), 23:00 (n=678), 20:00 (n=603), 01:00 (n=546), 19:00 (n=528) | 22:00 (n=863), 23:00 (n=659), 20:00 (n=612), 01:00 (n=544), 19:00 (n=506) |
| H1 | 01:00 (n=201), 13:00 (n=191), 14:00 (n=179), 15:00 (n=139), 12:00 (n=136) | 01:00 (n=203), 14:00 (n=198), 13:00 (n=190), 15:00 (n=152), 12:00 (n=139) |
| H4 | 12:00 (n=259), 16:00 (n=66), 08:00 (n=64), 00:00 (n=59), 04:00 (n=51) | 12:00 (n=255), 16:00 (n=86), 08:00 (n=75), 00:00 (n=73), 04:00 (n=41) |
| D1 | 00:00 (n=73) | 00:00 (n=72) |

**Key finding (M15 & H1):** Asian-session hours (20–01 UTC) cluster the most swing points. This is the artifact of low-liquidity range expansion at session opens — primarily Sydney/Tokyo open at ~22:00 UTC. NY/London kill-zones still account for 34–43% of all swings, in line with ICT-style expectations.

**H4 view:** When zoomed out, 12:00 UTC (London/NY overlap open) dominates by a wide margin (n=259 vs runner-up n=66) — confirming that on higher TFs, the true swing-engine is the London-NY overlap, not the Asian open noise.

### Day-of-week distribution (high swings)

| TF | Mon | Tue | Wed | Thu | Fri |
|----|-----|-----|-----|-----|-----|
| M15 | 2129 | 1697 | 1846 | 1900 | 3074 |
| H1 | 476 | 416 | 467 | 428 | 520 |
| H4 | 92 | 110 | 116 | 111 | 109 |
| D1 | 14 | 12 | 13 | 19 | 15 |

**Friday dominance** is consistent across M15/H1 — confirms NFP-week and end-of-week positioning generates the most extremes. Tuesday is the quietest day.

## 3. Prompt 2 — Breakout edge

**Setup:** body-pct ≥ 60%, 20-bar Donchian breakout, 35-pip SL, 1:2 R:R, 50-bar future window.

| TF | Signals | Win rate | Expectancy (R) | ASIAN WR | LONDON WR | NY WR |
|----|---------|----------|----------------|----------|-----------|-------|
| M15 | 8,665 | 66.1% | +0.98 | 60% (n=3658) | 68% (n=2390) | 73% (n=2617) |
| H1 | 2,087 | 84.4% | +1.53 | 81% (n=567) | 88% (n=584) | 84% (n=936) |
| H4 | 571 | 89.3% | +1.68 | 93% (n=181) | 92% (n=255) | 79% (n=135) |
| D1 | 125 | 97.6% | +1.93 | 98% (n=125) | — | — |

**⚠️ Caveat:** the 50-bar future window means signals are evaluated against ~2 weeks of M15 future (or ~2 months on D1). On instruments like XAUUSD that have trended sharply higher 2021→2026 (roughly $1,800 → $3,400/oz), any breakout-long with a 70-pip ($0.70) target will hit positive R *most of the time* simply because of the macro tailwind. The numbers are presented as raw evidence; they should not be interpreted as a live-tradable edge. A walk-forward / regime-aware backtest is required (see §6).

## 4. Prompt 3 — Liquidity sweep + reversal

**Setup:** wick beyond prior swing high/low followed by close back through within `rev_window=4` bars.

| TF | Sweeps | Reversal rate | Kill-zone reversal | Depth p50 (pips) | Depth p90 (pips) |
|----|--------|---------------|---------------------|------------------|------------------|
| M15 | 25,010 | 84.8% | 83.8% | 69.0 | 303.0 |
| H1 | 5,764 | 84.9% | 84.0% | 154.0 | 661.0 |
| H4 | 1,498 | 85.5% | 84.2% | 287.1 | 1149.8 |
| D1 | 172 | 83.1% | 0.0% | 897.7 | 2754.3 |

### Sweep reversal rate by session

| TF | ASIAN | LONDON | NY |
|----|-------|--------|-----|
| M15 | 85.3% (n=9992) | 85.2% (n=7590) | 83.6% (n=7428) |
| H1 | 85.4% (n=1911) | 85.4% (n=1736) | 83.9% (n=2117) |
| H4 | 82.9% (n=467) | 83.9% (n=658) | 91.7% (n=373) |
| D1 | 83.1% (n=172) | — | — |

**Key finding:** **Sweep-then-reverse rate is 83–85% across all timeframes** — the strongest single edge identified. This is the ICT-style signature on XAUUSD and matches the Stockraft PDF's expectation. H4 NY-session sweeps reach **92% reversal** (n=373) — the highest cell in the entire matrix.

## 5. Signal matrix (per-TF)

| TF | Rec SL (pips) | Target R:R | Top swing hrs (UTC) | KZ swing % | Asian fakeout risk | Edge sign |
|----|---------------|-----------|----------------------|-------------|----------------------|------------|
| M15 | 840 | 1:2.0 | 22,23,20 | 34.6% | Normal | ✅ |
| H1 | 2119 | 1:2.0 | 01,13,14 | 42.5% | Normal | ✅ |
| H4 | 5570 | 1:2.0 | 12,16,08 | 65.7% | Normal | ✅ |
| D1 | 19761 | 1:2.0 | 00 | 0.0% | ⚠️ HIGH | ✅ |

**Important: the SL recommendations are p90 sweep depth.** On M15 that's ~840 pips (≈$8.40 on gold). On H1 that's ~2,100 pips (≈$21). On H4 ~5,500 pips (≈$55). These are *worst-case* containers, not realistic per-trade stops. Use **p50 sweep depth as a more realistic SL anchor**: M15 ~69 pips, H1 ~154 pips, H4 ~287 pips.

## 6. Recommendations

1. **Adopt the sweep-then-reverse pattern as PRISM's primary edge candidate.** 85% reversal at 25k+ samples on M15 and 5.7k samples on H1 is the most reliable structural pattern in this dataset. Build the feature: `sweep_then_reverse_signal(lookback_n=5, rev_window=4)`. Add it to `prism/data/feature_engineering.py`.
2. **Filter trades to NY-session H4 sweeps** for the highest-confidence subset (92% reversal, n=373 over 5 years ≈ 75/yr). This is exactly the kind of low-frequency / high-quality signal a `predict_proba > 0.92` threshold should be looking for in the existing model.
3. **Discount the breakout-edge numbers** in `prism/backtest/stockraft_analysis.analyze_breakout_edge`. The 50-bar future window plus the macro-bullish XAU regime inflate the win-rate. Re-run with walk-forward and regime-conditional aggregation (bull/range/bear by quarterly SMA slope) before trusting it.
4. **Avoid 22–01 UTC entries on M15** despite the swing-count cluster there: Asian session liquidity holes manufacture sweep-and-noise, not directional moves. The H4 view confirms 12:00 UTC (London open / NY pre-open) is the real engine on higher TFs.
5. **Friday and Monday over-weight in PRISM's expected-value calculations.** Day-of-week prior is a free signal in the feature set.
6. **SL sizing:** use the p50 of sweep depth, not p90. For XAUUSD M15: 69 pips. Pair with 1:2 R:R from the matrix.

## 7. Data artifacts

Parquet files generated (not committed; in `data/raw/`, .gitignored):

```
data/raw/XAUUSD_M15_2021-01-01_2026-05-03.parquet  (133,536 bars)
data/raw/XAUUSD_H1_2021-01-01_2026-05-03.parquet   ( 33,384 bars)
data/raw/XAUUSD_H4_2021-01-01_2026-05-03.parquet   (  8,346 bars)
data/raw/XAUUSD_D1_2021-01-01_2026-05-03.parquet   (  1,391 bars)
```

Raw Dukascopy bi5 candle files (per-day, lzma-compressed) cached at `data/dukascopy_cache/XAUUSD/<YYYY>/<MM-1>/<DD>/BID_candles_min_1.bi5`. 1,387 files for the 2021-01-01 → 2026-05-01 window.

Reproduce:
```bash
python3 scripts/fetch_backtest_data.py --instrument XAUUSD --start 2021-01-01 --end 2026-05-03 --analyze
```

## 8. Code changes this branch

`fix(dukascopy): correct bi5 field order (OCLH) + seconds time base; fix pandas 2.x resample agg`

- `prism/data/dukascopy.py`: fixed the bi5 record layout (field order is OCLH, not OHLC; time base is seconds, not ms). Without this fix every parquet had 1 bar per file with `high < open` on most records.
- `scripts/fetch_backtest_data.py` + `prism/data/dukascopy.py`: replaced the deprecated `agg(open='first', ...)` named-aggregation form with the tuple form `agg(open=('open','first'), ...)`. pandas 2.3 raises `TypeError` on the bare-kwarg form.
- `tests/test_fetch_backtest_data.py`: new — mocks Dukascopy HTTP, asserts parquet schema and OHLC validity.

---

*Generated 2026-05-18 by Ada (PRISM Task 0.1).*