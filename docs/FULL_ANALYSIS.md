# PRISM Full Analysis — Smart Money Trading System

**Generated:** 2026-05-03
**Analyst:** Ada Sandpaw
**Branch:** `data/dukascopy-stockraft-2026-05-03`
**Scope:** State of PRISM, alignment with the "Good Trading 2026" stack, MT5-on-Mac, deeper backtesting, GitHub-as-journal, TradingView intake, AI analysis, phase gaps, automation roadmap

> Replaces the 2026-04-26 analysis (ICT/SMC source-by-source teaching synthesis). That document is preserved in the prior commit history. This version is operational — what to build, in what order, with what tools.

---

## 1. Current State Assessment

### 1.1 What PRISM has shipped

Code is on `main` and on the working branch `data/dukascopy-stockraft-2026-05-03`. Test baseline: **>468 tests green**.

| Layer | Module | Status |
|---|---|---|
| Live data | `prism/data/tiingo.py` (legacy), `prism/execution/mt5_bridge.py:get_bars()` | ✅ Production |
| **True spot historical data** | `prism/data/dukascopy.py` (XAU/EUR/GBP/JPY/BTC, 2003→today, free, no key) | ✅ New on this branch |
| **Stockraft analysis layer** | `prism/backtest/stockraft_analysis.py` (3 master prompts as code) | ✅ New on this branch |
| ML direction (H4) | `prism/model/predict.py`, `prism/model/retrain.py` | ✅ Production |
| FVG detection + retest | `prism/signal/fvg.py` (228 LOC) | ✅ Production |
| ICC structure (H1) | `prism/signal/icc.py` (285 LOC) | ✅ Production |
| **HTF Bias engine (Phase 5)** | `prism/signal/htf_bias.py` (276 LOC) | ✅ Merged |
| **Order Block detector (Phase 6)** | `prism/signal/order_blocks.py` (486 LOC, full lifecycle state machine) | ✅ Merged |
| **Liquidity Sweep detector (Phase 6)** | `prism/signal/sweeps.py` (240 LOC) | ✅ Merged |
| **Po3 phase detector (Phase 6)** | `prism/signal/po3.py` (180 LOC) | ✅ Merged |
| **Smart-money wiring (Phase 6.D/E)** | `prism/signal/generator.py` — sub-flag gates, observability | ✅ Merged |
| **Phase 7.A ML features** | 5 ICT features (`htf_alignment`, `kill_zone_strength`, `sweep_confirmed`, `ob_distance_pips`, `po3_phase`) | ✅ Code merged, **gates pending live data** |
| Audit log | `prism/delivery/signal_audit.py` (125 LOC, JSONL per-signal) | ✅ Merged |
| Smart-money export / drift diff | `prism/audit/smart_money_export.py` | ✅ Merged |
| Session filter | `prism/delivery/session_filter.py` (London 07-11 UTC, NY 13-17 UTC) | ✅ Production |
| Slack delivery + confirm flow | `prism/delivery/slack_notifier.py`, `confirm_handler.py` | ✅ Production |
| Drawdown guard | `prism/delivery/drawdown_guard.py` | ✅ Production |
| MT5 execution bridge | `prism/execution/mt5_bridge.py` (1082 LOC) | ✅ Production (Windows only) |
| News intelligence | `prism/news/intelligence.py` | ✅ Production |
| News + macro feeds | `prism/data/fred.py`, `prism/data/quiver.py` | ✅ Production |

### 1.2 What's working

- **Signal pipeline is complete end-to-end** from H4 regime → ICC → FVG retest → SL/TP → SignalPacket → Slack → MT5.
- **Smart-money observability is live** (Stage 1): every signal carries `signal.smart_money` and `signal.htf_bias`, written to JSONL audit log under `state/signal_audit/<instrument>/YYYY-MM-DD.jsonl`.
- **Dukascopy fetcher is production-quality**: LZMA bi5 binary parsing, weekend skip, instrument config for XAU/EUR/GBP/JPY/BTC, local cache. No API key, no rate limits.
- **Stockraft 3-prompt methodology is code, not a PDF**: swing detection, breakout edge, sweep depth statistics — all callable on Dukascopy DataFrames.
- **Phase 7.A ML harness is complete**: walk-forward (24 monthly folds × 18-month windows), sidecar parquet for historical detector state, env-var lock-in to model artifacts.

### 1.3 What's missing or fragile

| Gap | Severity | Notes |
|---|---|---|
| **Phase 7.A gates not yet evaluated** | High | Gate 6 needs ≥30 signals over 2 weeks NOTIFY mode. Gate 5 needs ≥2 weeks of multi-session audit log. Both are calendar-bound. |
| **No Mac-native execution path** | High (Brian-blocking) | MT5 is Windows-only. Bridge code runs but the terminal doesn't on a Mac. Signals can fire — execution can't. |
| **Backtesting is ad-hoc** | Medium | `stockraft_analysis.py` reports edge stats, but there's no portfolio-level simulator with slippage, commissions, position sizing, drawdown curves. Lean/QuantConnect or `vectorbt` would close this. |
| **No structured trade journal** | Medium | Slack history is ephemeral. Audit log is signal-level, not trade-level (no entry → exit → P&L → notes loop). |
| **No TradingView ingestion** | Medium | Brian uses TradingView for chart context. PRISM has no webhook intake to bring TV alerts (Pine Script outputs) into the signal stream. |
| **AI analysis is shallow** | Medium | Claude is used implicitly via OpenClaw for research/dev. No Claude calls inside the live signal loop (e.g., "explain why this signal fired in plain English" or "score the chart context against this exemplar"). |
| **Phase 7.B (`fvg_quality_score`, `ote_zone`) deferred** | Low | Blocked on Phase 8 (`check_ote_zone()` in `quality_filter.py`) and FVG `formed_bar` → timestamp migration. |
| **Phase 8 (Trade Quality Filter) not started** | Medium | Per PRD2: FVG quality scoring, OTE zone check, multi-confirmation stacking, dynamic R:R targeting. ~2-4 weeks of work. |

---

## 2. Stack Alignment — "Good Trading 2026"

The TikTok image Brian shared is the community-consensus tool stack. Here is exactly how each maps onto PRISM, and what to adopt vs. skip.

| Category | Tool shown | PRISM today | Recommendation |
|---|---|---|---|
| **Ideas** | Claude (Anthropic) | ✅ Used via OpenClaw for research/dev | **Keep.** Add Opus for deep research tasks (per Brian: "use Opus for all analysis"). |
| **Market Analysis** | TradingView | ❌ No integration | **Adopt as alert source.** Pine Script alerts → TradingView webhook → PRISM `/tv-webhook` endpoint. Use TV for human chart context; PRISM still owns signal logic. |
| **AI Analysis** | Anthropic API | ✅ Available via OpenClaw + raw `ANTHROPIC_API_KEY` in env | **Deepen.** Add Claude calls inside the signal loop (Section 7). |
| **Backtesting** | Lean / QuantConnect | ⚠️ Ad-hoc Stockraft analysis only | **Augment, don't replace.** Keep Stockraft for edge statistics. Add `vectorbt` or `backtesting.py` for portfolio simulation. Skip QuantConnect for now (cloud-tied, learning curve, paid tier for live). Re-evaluate at Phase 8. |
| **Journal / Tracking** | TradersSync / TradeZella | ❌ None | **Skip the SaaS, use GitHub Issues.** Per Brian's stated preference. One Issue per trade, labels for outcome/instrument/phase, milestones for trading weeks, Project board for pipeline view. |
| **Database / Organization** | Notion | ❌ None | **GitHub-only.** Issues + Projects + Wiki + Discussions. No Notion. Per Brian. |
| **Automation** | Python | ✅ Core PRISM language | **Keep.** No change. |
| **Execution** | MetaTrader 5 | ✅ Bridge code (Windows-only) | **Keep + cloud VPS.** See Section 3. |

### 2.1 What to adopt now
1. **TradingView webhook intake** — Pine Script → `/tv-webhook` HTTP endpoint → SignalPacket-compatible record → Slack notification. ~1 day's work.
2. **GitHub Issues as trade journal** — issue template, label set, Project board, automation script that opens an issue per `SignalPacket` fire and closes it on MT5 exit. ~1-2 days' work.
3. **Cloud VPS for MT5** — see Section 3.

### 2.2 What to skip
- **Notion** — Brian explicitly does not want it. GitHub covers all collaboration needs.
- **TradeZella / TradersSync** — paid SaaS that GitHub Issues replicates for free, with full git history and Project board automation.
- **QuantConnect cloud platform** — vendor lock-in, learning curve, paid live trading. Lean (the open-source engine underneath) could be adopted in Phase 9+ if portfolio backtesting becomes the bottleneck. Not now.

---

## 3. MT5 on Mac — Recommended Solution

**Brian's question (paraphrased):** "Is there a way to do this without me running the Windows MT5 thing on my Mac manually?"

**Answer:** Yes. Three viable paths, in order of recommended:

### 3.1 Option A — Cloud Windows VPS (RECOMMENDED)

A small Windows Server VPS runs MT5 24/7. PRISM connects via the existing `mt5_bridge.py` over SSH tunnel + Python API. Brian's Mac never touches MT5.

- **Provider:** Vultr or Contabo (Windows Server 2022, 2 vCPU / 4 GB / 80 GB SSD)
- **Cost:** $12-20/month (Vultr ~$13.50, Contabo ~$8 for 4-vCPU Windows VPS)
- **Latency:** Pick a region close to the broker's MT5 server (Exness uses London/Singapore). ~5-30 ms typical.
- **Setup:** 1 hour. Install MT5, log in to Exness, install Python 3.11, pull PRISM repo, run `prism/delivery/runner.py` as a Windows Service.
- **Reliability:** 99.9%+ uptime SLA, automatic reboot on failure, snapshots for rollback.
- **Brian's involvement:** zero after initial setup. Watches Slack signals on his Mac as before.

**Why this wins:**
- Native MT5 (no Wine quirks).
- 24/7 (Brian's Mac doesn't have to be on).
- Cheap enough to be a rounding error.
- Snapshot-based rollback if the runner crashes catastrophically.

### 3.2 Option B — Docker + Wine on Mac (NOT recommended)

Run MT5 inside a Wine container on Mac. Works but:
- Wine + MT5 has known stability issues (random crashes, slow startup).
- Brian's Mac has to be awake when markets are live.
- Wine can break on every macOS update.
- Apple Silicon needs Rosetta + x86_64 Wine — adds another fragility layer.

Use only if Brian explicitly refuses a cloud VPS.

### 3.3 Option C — Signal-only mode (interim)

Skip MT5 entirely. PRISM stops at the SignalPacket → Slack step. Brian executes trades manually. Useful for:
- The 2-week Phase 7.A NOTIFY mode A/B test (no execution needed for gate 6).
- Learning the system before committing real capital.
- Avoiding the cloud VPS cost during the unproven phase.

**Recommendation for Brian:** Start in **signal-only mode this week** (no infrastructure change needed — already supported by `PRISM_MODE=NOTIFY`). Once Phase 7.A gates pass and Brian wants execution, **provision the cloud VPS in one afternoon** (Section 3.1). Skip Option B entirely.

### 3.4 Why NOT Interactive Brokers / Alpaca

Brian asked about broker alternatives that work natively on Mac. Short answer: not worth switching for this use case.
- **Interactive Brokers** has solid Python API (`ib_insync`) but XAUUSD on IB is via futures (GC=F), not true spot. PRISM's whole edge is true spot Dukascopy data → spot execution mismatch hurts the model.
- **Alpaca** is solid for US equities/crypto, weak for forex/gold. Doesn't offer XAUUSD spot at all.
- **OANDA** has decent forex/gold + Python API. Could be a real alternative if MT5/Exness becomes a problem. Park it as a Plan B; don't switch now.

The MT5/Exness pipeline is already built and working. Cloud VPS solves the "Brian's Mac" problem for $13.50/month. Don't fix what isn't broken.

---

## 4. Backtesting Strategy

PRISM has two backtesting concepts colliding: **edge analysis** (Stockraft 3-prompt) and **portfolio simulation** (executes the actual signal generator over historical bars and tracks P&L). They serve different purposes.

### 4.1 Layer A — Stockraft Edge Analysis (already built)

`prism/backtest/stockraft_analysis.py` runs three master prompts on Dukascopy DataFrames:

1. **Swing Points & Pattern Detector** — finds swings, classifies session bias, time clusters, day-of-week edge, sweep depth distribution, kill-zone activity %.
2. **Candle Structure & Breakout Edge** — detects strong-body breakouts, win rate, expectancy, by-session breakdown.
3. **Liquidity Sweep Analysis** — sweep count, reversal rate, by-session/day-of-week, depth percentiles (p50, p90 → calibrate SL).

**Output:** a `StockraftReport` with edge statistics. This is the **input to model training** (what features to engineer, what SL distance to gate at, when to trade).

**Status:** Code exists. Not yet run end-to-end on 5 years of Dukascopy data on this branch.

### 4.2 Layer B — Portfolio Simulation (TO BUILD)

The signal generator + execution + slippage + commission + position sizing simulated over historical bars, producing:
- Equity curve
- Sharpe / Sortino / Calmar
- Max drawdown (absolute + duration)
- Per-trade P&L distribution
- Win rate by session / day / instrument / Po3 phase
- R:R achieved vs. R:R targeted
- Realistic execution: spread + slippage + commission + funding cost

**Recommended tool:** `vectorbt` (Python, fast, vectorized, no cloud dependency). Alternatively `backtesting.py` (simpler, slower). Both work natively on Mac.

**NOT recommended:** Lean/QuantConnect — the cloud platform is overkill. The open-source Lean engine itself is fine but C# + Python interop is a tax. Revisit only if PRISM grows to multi-strategy / multi-asset portfolio scale.

### 4.3 Layer C — Walk-Forward (already built — Phase 7.A)

`prism/model/walkforward.py` does rolling-window walk-forward on the **ML model**, not on the full strategy. 24 folds × 18-month windows × 1-month step. Output: per-fold and aggregate F1/Sharpe/MaxDD.

This validates the **model**, not the **strategy**. Layer B above validates the strategy.

### 4.4 When to use which

| Question | Use |
|---|---|
| "Does this market have edge under the Stockraft framework?" | Layer A (Stockraft Edge) |
| "What features should the ML model train on?" | Layer A (informs Phase 7) |
| "Does the full PRISM strategy make money over 5 years?" | Layer B (Portfolio Simulation — to build) |
| "Is the ML model stable across regime changes?" | Layer C (Walk-Forward — already built) |

### 4.5 Backtesting roadmap
1. Run Layer A on 5 years of XAUUSD Dukascopy data (1 hour, fully automated). Commit `data/raw/XAUUSD_*_dukascopy_*.parquet` and `data/raw/stockraft_report_XAUUSD.json`. **This week.**
2. Build Layer B on top of `vectorbt`: `prism/backtest/portfolio_sim.py` — replays SignalGenerator over historical bars with realistic execution. **2 weeks.**
3. Phase 7.A walk-forward (Layer C) waits on live audit data — calendar-bound to mid-May.

---

## 5. GitHub as Trade Journal

Per Brian's stated preference: GitHub Issues + Projects, not Notion, not TradeZella.

### 5.1 Issue template

`.github/ISSUE_TEMPLATE/trade.yaml`:

```yaml
name: Trade
description: A live or paper trade fired by PRISM
title: "[<INSTRUMENT>] <DIRECTION> @ <ENTRY> — <SESSION> <DATE>"
labels: ["trade"]
body:
  - type: input
    id: signal_id
    attributes:
      label: Signal ID
      description: PRISM signal_id from SignalPacket
  - type: dropdown
    id: instrument
    attributes:
      label: Instrument
      options: [XAUUSD, EURUSD, GBPUSD, USDJPY]
  - type: dropdown
    id: direction
    attributes:
      label: Direction
      options: [LONG, SHORT]
  - type: input
    id: entry
    attributes:
      label: Entry
  - type: input
    id: sl
    attributes:
      label: Stop Loss
  - type: input
    id: tp1
    attributes:
      label: TP1 (1:1 partial)
  - type: input
    id: tp2
    attributes:
      label: TP2 (next liquidity)
  - type: textarea
    id: reasoning
    attributes:
      label: Why this fired
      description: HTF bias, smart-money block, FVG details, news context
  - type: textarea
    id: outcome
    attributes:
      label: Outcome
      description: Filled in at exit. Win/Loss/BE, R achieved, notes
```

### 5.2 Label taxonomy

- **Instrument:** `xauusd`, `eurusd`, `gbpusd`, `usdjpy`
- **Direction:** `long`, `short`
- **Session:** `london`, `ny`, `asian`, `overlap`
- **Phase:** `phase-6` (smart-money observability), `phase-7a` (ML retrain), `phase-8`
- **Outcome:** `win`, `loss`, `breakeven`, `pending`
- **Mode:** `paper`, `live`, `notify`
- **Quality:** `a-grade`, `b-grade`, `c-grade` (manually reviewed post-trade)

### 5.3 Project board layout

`Projects → New project → Board view`:
- **Columns:** Pending → Triggered → Open → TP1 hit → Closed → Reviewed
- **Automation:** GitHub Actions workflow on `signal_audit` JSONL writes opens Issues automatically; on MT5 close events, moves them to `Closed` and fills outcome.
- **Insights:** auto-generated R-by-session, win-rate-by-Po3-phase charts via the GitHub Projects insight panel.

### 5.4 Implementation

```python
# prism/journal/github_issues.py (NEW — ~150 LOC)
"""
Bidirectional sync between PRISM signal/trade events and GitHub Issues.

  on_signal_fired(signal_packet):  open issue, label by instrument/session/phase
  on_trade_filled(ticket):         comment with fill details
  on_tp1_hit(ticket):              comment, move project card to "TP1 hit"
  on_trade_closed(ticket, pnl):    close issue, fill outcome label, move to Reviewed
"""
```

Driven by `gh` CLI under the hood (already in OpenClaw skills). Auth via `GH_TOKEN` env var.

### 5.5 Why GitHub beats TradeZella

- **Free.** TradeZella is $30/month; over 12 months that's the cost of two cloud VPSes.
- **Structured + queryable.** `gh issue list --label win,xauusd,london --json` returns JSON. SaaS journals don't.
- **Versioned.** Git history of every label change, every comment.
- **Already in Brian's workflow.** No new tool to learn.
- **AI-ready.** Claude can read the entire trade journal via the `gh` CLI; a SaaS journal sits behind a vendor API.

---

## 6. TradingView Integration

Brian uses TradingView for chart context. PRISM should ingest TV alerts as a **complementary signal channel**, not replace its own logic.

### 6.1 The pattern

```
Pine Script alert  →  TradingView webhook  →  PRISM HTTP endpoint  →  Slack signal block
                                                      ↓
                                          (optional) cross-check vs PRISM signal:
                                          "TV says LONG at 2318, PRISM also has LONG with
                                          confidence 0.72 — confluence boost."
```

### 6.2 Implementation

```python
# prism/delivery/tv_webhook.py  (NEW — ~120 LOC)
"""
TradingView webhook intake. Stand up a small HTTPS endpoint
(FastAPI on the same host as the runner) that:
  1. Authenticates by shared secret in URL path or HMAC header.
  2. Parses Pine Script alert JSON {ticker, direction, entry, sl, tp, note}.
  3. Records to state/tv_alerts/<instrument>/YYYY-MM-DD.jsonl.
  4. Posts to Slack with #prism-tv-alerts channel formatting.
  5. (Phase 9+) Cross-check against active PRISM signal cache; if confluence,
     flag the next PRISM signal as 'TV-confluence boosted'.
"""
```

### 6.3 Pine Script alert template

```pinescript
//@version=5
indicator("PRISM-TV bridge", overlay=true)
// ... your favorite ICT/SMC indicator logic here ...

if (long_setup)
    alert('{"ticker":"' + syminfo.ticker + '","direction":"LONG","entry":' + str.tostring(close) + ',"note":"FVG retest + sweep"}', alert.freq_once_per_bar_close)
```

### 6.4 Hosting

The webhook endpoint runs on the same Windows VPS as MT5 (FastAPI + Uvicorn + Cloudflare Tunnel for HTTPS without a fixed IP). Cost: free (Cloudflare Tunnel is free tier).

### 6.5 What this is and isn't

- **It IS:** a way for Brian to feed his own Pine Script ideas into PRISM's notification stream and (later) confluence model.
- **It IS NOT:** a replacement for PRISM's signal generator. PRISM owns the model. TradingView is augmentation, not source of truth.

---

## 7. AI Analysis Enhancement

Currently Claude is used implicitly via OpenClaw for development tasks. The signal loop itself doesn't call any LLM. There are three high-leverage places to wire it in.

### 7.1 Per-signal narrative ("explain this trade in plain English")

When a SignalPacket fires, send the structured signal data (HTF bias, smart-money block, FVG details, news context, ML confidence) to Claude Opus and get back a 3-4 sentence narrative for Slack:

> "📈 **LONG XAUUSD @ 2318.50** — H4 and H1 are both bullish (HH/HL stacked), and the London kill zone produced a clean liquidity sweep below 2316.20 with 18-pip displacement back up. Price is now retesting the FVG at 2318.20-2318.80 with an active bullish OB 12 pips below at 2317.10. ML confidence 0.71. Risk 22 pips, target 2321 (1:1) → 2325.40 (next liquidity)."

**Implementation:** ~50 LOC, ~$0.005 per signal at Opus pricing, async (doesn't block MT5).

```python
# prism/ai/narrate.py
async def narrate_signal(packet: SignalPacket) -> str:
    prompt = build_narrate_prompt(packet)  # template with structured fields
    return await anthropic.messages.create(model="claude-opus-4-5", ...)
```

### 7.2 Post-trade review ("why did this lose?")

When a trade closes for a loss, Claude reviews the audit log + bar context + smart-money block and writes a comment on the GitHub Issue:

> "Loss likely caused by news event 14 minutes after entry (FOMC member speech, sentiment flipped). PRISM news intelligence has the event but it was scored MEDIUM impact and not blocked. Recommend tightening news blackout to MEDIUM+ during NY kill zone."

**Implementation:** ~80 LOC, runs as part of `on_trade_closed` GitHub Issue automation.

### 7.3 Weekly meta-analysis

Once a week, Claude reads the past 7 days of audit log + trade journal and produces a `docs/weekly/YYYY-WW.md` summary:
- Best-performing signal type
- Worst-performing setup
- Drift in any feature (cross-check against gate 5)
- Suggested parameter tweaks

**Implementation:** GitHub Actions cron, calls Claude via `ANTHROPIC_API_KEY`, opens a PR with the weekly file.

### 7.4 What NOT to do with AI

- **Do NOT have Claude generate signals.** That's what the ML model is for. Claude has no edge over a trained gradient-boosted model on candlestick prediction; using it that way is hallucination-by-design.
- **Do NOT have Claude approve trades in real time.** The CONFIRM mode goes to the human, not to an LLM. LLMs are for narrative, review, and meta-analysis — not execution authority.

---

## 8. Phase Gaps Analysis

### 8.1 Phase 7.A — Live data wait

The 5 ICT features and walk-forward harness are merged. Gates 5 and 6 require:
- **Gate 5** (drift): ≥2 weeks of multi-session audit log + a passing chi-squared/KS comparison against the historical builder. Calendar: ~2026-05-11 if Stage 1 turned on 2026-04-27. Chi-squared sensitivity: depends on signal volume per kill zone — likely needs 30+ signals per feature bin.
- **Gate 6** (live A/B): ≥30 signals over 2 weeks NOTIFY mode. Calendar-bound. With current signal frequency (~2-3 per kill zone per instrument), 30 signals = ~5-7 trading days at 2 instruments. **Likely passes by 2026-05-13.**

**Action:** Stage 1 is already running (assumed — verify before next steps). Wait. This is the only thing in PRISM that can't be parallelized.

### 8.2 Phase 7.B — `fvg_quality_score` + `ote_zone`

Blocked on:
1. **Phase 8** — need `check_ote_zone()` in `prism/signal/quality_filter.py` to align signatures.
2. **FVG `formed_bar` migration** — `FVGZone.formed_bar` is currently a df-relative integer index (`prism/signal/fvg.py:37`). Across runner restarts the index shifts, so `age_bars = len(df) - formed_bar` silently drifts. Must migrate to UTC timestamp.

Bundle as one PR after Phase 8. Re-run the same walk-forward harness for clean A/B.

### 8.3 Phase 8 — Trade Quality Filter

Per PRD2 §8. Not yet started. Scope:
- `prism/signal/quality_filter.py` (~250 LOC)
- `check_ote_zone(price, swing_high, swing_low, direction)` → bool
- FVG quality scoring (multi-factor: confluence with OB, ATR-relative size, kill-zone formation, age)
- Multi-confirmation stacking (reversal candle + EMA cross, finastictrading-style)
- Dynamic R:R targeting (next liquidity pool, not fixed multiplier)
- Tests: ~30
- Estimated effort: 8-12 hours implementation + review

**Recommended sequencing:** Phase 8 is unblocked NOW. Don't wait on Phase 7.A gates. Run in parallel.

### 8.4 Phases 9+ (post-PRD2)

Out of scope for PRD2. Forward-look only:
- **Phase 9:** TradingView webhook intake (Section 6).
- **Phase 10:** Multi-instrument concurrent execution + portfolio-level drawdown guard.
- **Phase 11:** AI narrative + post-trade review (Section 7).
- **Phase 12:** Portfolio simulation harness (Section 4.2).

---

## 9. Automation Roadmap

What runs without Brian touching anything, vs. what needs his hands.

### 9.1 Fully automated (today)

- Live MT5 bar ingest + feature engineering + ML inference + signal generation
- Slack signal posting
- Audit log writes
- Drawdown guard
- Sunday gap guard
- Signal dedup
- News blackout
- (When Phase 6 gates flip) sweep + Po3 entry filtering

### 9.2 Fully automated (after this branch's action plan)

- Dukascopy 5-year XAU/EUR backfill (`scripts/fetch_backtest_data.py --analyze`)
- Stockraft edge report regeneration (cron weekly)
- GitHub Issue auto-creation per signal (`prism/journal/github_issues.py`)
- Per-signal Claude narrative (Section 7.1, async)
- TradingView webhook ingest (Section 6)

### 9.3 Needs Brian's hands

- **Initial cloud VPS provisioning** (1 hour, one time)
- **Slack `/approve` button click** in CONFIRM mode (intentional — never automate trade approval)
- **Weekly review** of GitHub Project board (15 min/week, eyes-on-glass)
- **Phase 7.A gate decision** (Brian makes the call to flip Stage 2 / Stage 3 sweep + Po3 gates after seeing audit data)
- **Capital management** (account funding, withdrawal — outside PRISM's authority)

### 9.4 The principle

PRISM is a **signal generator + execution gateway**, not an autonomous trader. CONFIRM mode is the default. AUTO mode exists but is a deliberate, opt-in choice per session, not the default. The human stays in the loop on the trade decision; everything else runs without intervention.

---

## 10. Risks & Open Questions

| # | Risk | Mitigation |
|---|---|---|
| 1 | Phase 7.A gate 5 fails — historical builder diverges from live audit | Audit log accumulating since 2026-04-27. Replay-mode historical builder (Option 1 in scope §6.1) prevents apples-to-oranges comparison. If gate 5 fails, FIX the builder before retraining. |
| 2 | Cloud VPS gets compromised (broker creds leak) | Restrict VPS to inbound SSH from Brian's IP only. Rotate Exness password monthly. MT5 magic number filters limit blast radius. |
| 3 | TradingView webhook becomes a vector for spoofed alerts | HMAC-signed payloads or shared-secret URL path. Rate-limit at Cloudflare. |
| 4 | Claude API outage breaks live narrative | Narrative is async + best-effort. SignalPacket fires regardless. Slack posts the structured block; the LLM narrative is a Slack thread reply that may or may not arrive. |
| 5 | GitHub Actions outage breaks trade journal | Issue creation is async. Local fallback writes to `state/trade_journal/<instrument>/YYYY-MM-DD.jsonl`. Reconcile on Actions recovery. |
| 6 | Brian gets FOMO, manually intervenes mid-trade | Documented: PRISM is the system, Brian is the safety. Manual closes are logged in the journal and feed back into post-trade review. Not a bug; a feature. |
| 7 | Dukascopy data has gaps or format changes | Cache is local, immutable once written. Format-change detection in `_decode_bi5()` raises early. Fallback to yfinance is wired but lower quality. |

---

## Appendix A — File map (current branch)

```
prism/
  data/
    dukascopy.py            ← NEW: true spot historical fetcher
    feature_engineering.py
    historical_state.py     ← Phase 7.A sidecar builder
    pipeline.py
    tiingo.py               ← legacy live/historical
    fred.py, quiver.py      ← macro / news
  signal/
    htf_bias.py             ← Phase 5
    order_blocks.py         ← Phase 6 (with full lifecycle state machine)
    sweeps.py               ← Phase 6
    po3.py                  ← Phase 6
    fvg.py                  ← Phase 2 (formed_bar migration pending)
    icc.py                  ← Phase 2
    generator.py            ← orchestrator (smart-money sub-flag gates)
  backtest/
    stockraft_analysis.py   ← NEW: 3-prompt edge analysis
  delivery/
    runner.py               ← signal loop
    signal_audit.py         ← per-signal JSONL audit
    slack_notifier.py
    drawdown_guard.py
    confirm_handler.py
    session_filter.py
  audit/
    smart_money_export.py   ← Phase 7.A drift diff
  execution/
    mt5_bridge.py           ← Windows-only execution
  model/
    predict.py, retrain.py, walkforward.py
  news/
    intelligence.py
docs/
  PRD.md, PRD2.md, PRISM_PRD.md
  RESEARCH_BRIEF.md
  PHASE_6_ROLLOUT.md
  PHASE_7A_SCOPE.md
  PHASE_7A_RESULTS.md
  RUNBOOK.md
  SUMMARY_FOR_BRIAN.md
  FULL_ANALYSIS.md          ← THIS DOCUMENT
  ACTION_PLAN.md            ← companion sequenced plan
scripts/
  fetch_backtest_data.py    ← --analyze runs Stockraft end-to-end
  download_historical_data.py
  health_check.py
config/
  instruments.yaml
data/
  raw/                      ← parquet OHLCV
  dukascopy_cache/          ← bi5 binary cache
```

## Appendix B — Key targets

From PRD2 §1 (carried into this analysis):

- Win rate: ≥62% (current model baseline ~55%)
- Average R:R: ≥1:2
- Sharpe (live): ≥1.5
- Max drawdown: ≤15%
- 24-fold walk-forward median F1: new ≥ baseline
- 24-fold walk-forward median Sharpe: new ≥ baseline × 0.95
- 24-fold walk-forward median MaxDD: new ≤ baseline × 1.10
- Phase 7.A gate 5 drift: ≤1 of 5 features rejects at α=0.01

---

*End of FULL_ANALYSIS.md (2026-05-03 revision).*
