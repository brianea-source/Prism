# PRISM Action Plan — 2026-05-03

> Companion to [FULL_ANALYSIS.md](./FULL_ANALYSIS.md). Tasks are sequenced into 4 phases. Effort estimates are wall-clock for the assigned owner. "Ada" = subagent / sub-PR work; "Brian" = decisions, account creds, capital. Each task has explicit blockers. Roll hard; ship in PR-sized chunks.

**Branch:** `data/dukascopy-stockraft-2026-05-03`
**Labels for issues:** `phase-0`, `phase-1`, `phase-2`, `phase-3`

---

## Phase 0 — Immediate (this week, 2026-05-03 → 2026-05-10)

Goals: Get true-spot 5-year backtest data on disk + Stockraft baseline numbers in a doc, decide MT5 path, set up the journal.

### Task 0.1: Run Dukascopy fetch + Stockraft analysis on 5 years of XAUUSD
**What:** Pull XAUUSD M1 → M5 / H1 / H4 / D1 from Dukascopy from 2021-01-01 → 2026-05-03, then run the 3-prompt Stockraft analysis end-to-end.
**Why:** This is the data foundation for everything downstream — Phase 7.A retrain, Phase 8 quality filter, the portfolio simulator. Without it we're modeling on Tiingo proxies + GLD futures, which is what the PRD2 retrain is meant to fix.
**How:**
```bash
cd ~/work/Prism
python3 scripts/fetch_backtest_data.py --instrument XAUUSD --start 2021-01-01 --analyze
# Expect: data/raw/XAUUSD_{M5,H1,H4,D1}_2021-01-01_2026-05-03.parquet
# Plus: docs/STOCKRAFT_REPORT_XAUUSD.md (or similar) with edge stats
```
Repeat for EURUSD. Commit parquet outputs to git LFS or a separate data branch (don't bloat main with 5-year tick data).
**Owner:** Ada
**Effort:** 1 hour run-time + 1 hour to write the markdown report. Mostly waiting on Dukascopy downloads (~50 MB / instrument / year compressed).
**Blocker:** None. Code is on this branch; just execute.

### Task 0.2: Decide MT5-on-Mac path with Brian
**What:** Brian picks one: (A) Cloud Windows VPS, (B) Wine on Mac, (C) Signal-only NOTIFY mode (interim).
**Why:** Phase 7.A gate 6 doesn't need execution (NOTIFY-only A/B). But once gates pass and Brian wants live trades, the answer has to be ready. Decision now → no scramble in 2 weeks.
**How:** One-line Slack reply to Ada: "VPS", "Wine", or "Notify-only for now". Recommended: VPS.
**Owner:** Brian
**Effort:** 5 minutes (decision). Implementation effort follows.
**Blocker:** None. Information needed is in FULL_ANALYSIS.md §3.

### Task 0.3: Provision Vultr Windows VPS (if 0.2 = VPS)
**What:** Spin up a Vultr Windows Server 2022 VPS (Frankfurt or London region, ~$13.50/mo), install MT5, log in to Exness, install Python 3.11, deploy PRISM as a Windows Service.
**Why:** Eliminates Brian's Mac as the execution host. 24/7 uptime. Native MT5.
**How:**
1. Brian creates Vultr account, provides Ada with API token (or shares screen for setup).
2. Ada provisions: 2 vCPU / 4 GB / 80 GB SSD Windows Server 2022, region nearest Exness MT5 server.
3. RDP in, install: MetaTrader 5 (Exness installer), Python 3.11, Git, NSSM (service wrapper).
4. `git clone` PRISM repo, `pip install -r requirements.txt`, copy `.env.production` from secure store.
5. Register `prism-runner` as a Windows Service via NSSM. Auto-start, auto-restart.
6. Smoke test: `python -m prism.delivery.runner --once --instrument XAUUSD --mode NOTIFY`.
**Owner:** Ada (with Brian's broker creds)
**Effort:** 1.5 hours
**Blocker:** Task 0.2 decision + Brian's Exness MT5 creds + Vultr account.

### Task 0.4: Stand up GitHub Issue trade journal
**What:** Create issue template, label set, Project board. Write `prism/journal/github_issues.py` for auto-issue-on-signal-fired.
**Why:** Brian's stated preference. Free. AI-readable. Already in the workflow.
**How:**
1. Add `.github/ISSUE_TEMPLATE/trade.yaml` (template in FULL_ANALYSIS.md §5.1).
2. Create labels via `gh label create` (instrument, direction, session, phase, outcome, mode, quality categories).
3. Create Project board "PRISM Trades" with columns: Pending → Triggered → Open → TP1 hit → Closed → Reviewed.
4. Build `prism/journal/github_issues.py`:
   - `on_signal_fired(packet)` → `gh issue create` with structured body
   - `on_trade_filled(ticket)` → comment on issue
   - `on_tp1_hit(ticket)` → label move + Project card move
   - `on_trade_closed(ticket, pnl)` → close issue + outcome label
5. Wire into `prism/delivery/runner.py` next to `signal_audit.py` (same hook point, line ~374).
6. Tests: `tests/test_github_issues.py` — mock `gh` CLI, assert issue body schema.
**Owner:** Ada
**Effort:** 6-8 hours (1 PR)
**Blocker:** `GH_TOKEN` env var on the runner host (Vultr VPS once 0.3 lands; localhost in interim).

### Task 0.5: Verify Stage 1 audit log is healthy
**What:** Confirm `state/signal_audit/<instrument>/YYYY-MM-DD.jsonl` is being written on the live runner with multi-session data accumulating since ~2026-04-27.
**Why:** Phase 7.A gate 5 depends on this audit log. If it's not writing, gate 5 has no source data → can't validate the historical builder → can't retrain safely.
**How:**
```bash
# On whichever host runs the live runner today (Brian's Mac in NOTIFY mode? or already on a VPS?)
ls -lh state/signal_audit/XAUUSD/
tail -3 state/signal_audit/XAUUSD/2026-05-03.jsonl | jq .
# Confirm: file exists, recent timestamp, schema has htf_bias + smart_money + signal_id
```
If empty: check `PRISM_SIGNAL_AUDIT_ENABLED=1` is set. Check disk write perms. Check `prism/delivery/signal_audit.py` is on the runner's PYTHONPATH.
**Owner:** Ada
**Effort:** 30 min
**Blocker:** Need to know which host is running the live runner.

---

## Phase 1 — Short Term (2 weeks, 2026-05-10 → 2026-05-24)

Goals: Phase 8 unblocked + Phase 7.A gates ready to evaluate + portfolio simulator on disk.

### Task 1.1: Build Phase 8 — Trade Quality Filter
**What:** New module `prism/signal/quality_filter.py`. Implements OTE zone check, FVG quality scoring (multi-factor), multi-confirmation stacking (reversal candle + EMA cross), dynamic R:R targeting (next liquidity pool).
**Why:** PRD2 §8. Unblocks Phase 7.B (`fvg_quality_score`, `ote_zone`). Lifts win rate per FULL_ANALYSIS.md §1.3.
**How:**
1. Create `prism/signal/quality_filter.py` (~250 LOC).
   - `check_ote_zone(price, swing_high, swing_low, direction) -> tuple[bool, float]` → returns (in_zone, fib_pct).
   - `score_fvg_quality(fvg, ob_list, atr, kill_zone, age_bars) -> float` (0.0-1.0).
   - `check_candle_confirmation(df, direction, ema_period=8) -> bool`.
   - `next_liquidity_pool(swing_points, direction, current_price) -> float | None`.
2. Wire into `generator.py` after Layer 3 (FVG) and before final SignalPacket assembly. Behind `PRISM_QUALITY_FILTER_ENABLED` env (default 0 for Stage 1 of Phase 8 rollout).
3. Tests: `tests/test_quality_filter.py` (~30 tests).
**Owner:** Ada
**Effort:** 8-12 hours (1 PR)
**Blocker:** None — fully unblocked. Don't wait on Phase 7.A gates.

### Task 1.2: Migrate `FVGZone.formed_bar` to UTC timestamp
**What:** `FVGZone.formed_bar` is currently a df-relative integer index. Across runner restarts the index shifts. Migrate to `formed_at: datetime` (UTC) and recompute `age_bars` against the current bar's timestamp.
**Why:** Blocker for Phase 7.B (`fvg_quality_score` needs stable age). Also a latent bug today — long-running runners have silent age drift.
**How:**
1. Edit `prism/signal/fvg.py`:
   - Add `formed_at: datetime` field to `FVGZone` (alongside the existing `formed_bar`).
   - Compute `age_bars` as `(current_time - formed_at) // bar_period`.
   - Keep `formed_bar` for backward compat for one release; deprecation log if used.
2. Update `FVGDetector.save()` / `load()` JSON schema with version bump.
3. Migration script: `scripts/migrate_fvg_state_v2.py` — reads old FVG state, computes `formed_at` from session start + `formed_bar × bar_period`, writes new format.
4. Tests: round-trip, version detection, age computation across restart simulation.
**Owner:** Ada
**Effort:** 4-6 hours (1 PR)
**Blocker:** None.

### Task 1.3: Phase 7.B — `fvg_quality_score` + `ote_zone` features
**What:** Once Task 1.1 lands, add the two deferred Phase 7 features to the ML feature pipeline. Re-run the same walk-forward harness for clean A/B (Phase 7.A vs Phase 7.A+B).
**Why:** Closes PRD2 Phase 7. Two more features that the model can learn from.
**How:**
1. Add `compute_fvg_quality_score(fvg, ob_list, atr, kill_zone, age_bars)` and `compute_ote_zone(price, swing_high, swing_low, direction)` to `prism/data/feature_engineering.py`.
2. Update `prism/data/historical_state.py` to emit both columns.
3. Re-run `prism/model/walkforward.py` per instrument.
4. Document in `docs/PHASE_7B_RESULTS.md`.
**Owner:** Ada
**Effort:** 4-6 hours implementation + 1-2 hours retrain wall-clock (per instrument, ×3 instruments)
**Blocker:** Tasks 1.1 + 1.2 merged.

### Task 1.4: Build portfolio simulator (Layer B backtesting)
**What:** New module `prism/backtest/portfolio_sim.py`. Replays the live `SignalGenerator` over historical Dukascopy bars with realistic slippage, commission, position sizing. Outputs equity curve + Sharpe / Sortino / MaxDD.
**Why:** FULL_ANALYSIS.md §4.2. Stockraft layer gives edge stats; this gives strategy-level "does it actually make money" answers.
**How:**
1. Pick `vectorbt` over `backtesting.py` (faster, vectorized, Mac-native).
2. New module wraps `vectorbt.Portfolio.from_signals()` with PRISM-specific config:
   - Slippage: 0.5 pip XAU, 0.3 pip majors
   - Commission: $7 per round-turn lot (Exness Standard)
   - Position size from `RISK_PCT × balance / SL_pips`
   - Concurrent trades: respect `MAX_CONCURRENT`
3. CLI: `python -m prism.backtest.portfolio_sim --instrument XAUUSD --start 2021-01-01 --end 2026-05-03 --output models/portfolio_xauusd_2026-05-03.json`
4. Output: equity curve PNG, JSON metrics, per-trade CSV.
5. Tests: `tests/test_portfolio_sim.py` (~15 tests, with synthetic OHLC fixtures).
**Owner:** Ada
**Effort:** 12-16 hours (1 PR)
**Blocker:** Task 0.1 (Dukascopy data on disk).

### Task 1.5: Phase 7.A gate evaluation
**What:** Run the full Phase 7.A gate harness per instrument once audit log has ≥2 weeks of data and ≥30 NOTIFY-mode signals.
**Why:** This is the gate-keeper for promoting the new feature set to `models/`. Without passing all 6 gates, the new model stays in `prism_v2_dev/`.
**How:** Follow the runbook in `docs/PHASE_7A_RESULTS.md` lines 38-92. Specifically:
1. Build full historical sidecar + signal-conditioned sidecar.
2. Run `python -m prism.audit.smart_money_export diff` for gate 5.
3. If gate 5 fails: fix the historical builder, do NOT retrain.
4. Run walk-forward retrain with `PRISM_OB_MAX_DISTANCE_PIPS=50`.
5. Run SHAP harness for gate 4.
6. Promote per-instrument if all 6 gates pass.
**Owner:** Ada
**Effort:** 2-3 hours of compute + 1-2 hours analysis
**Blocker:** Calendar — needs ≥2 weeks of Stage 1 data + ≥30 NOTIFY signals. Earliest: 2026-05-13 (assuming Stage 1 turned on 2026-04-27).

### Task 1.6: Stockraft edge report — auto-regenerate weekly
**What:** GitHub Actions cron that re-runs Stockraft analysis on the latest Dukascopy data weekly and commits an updated `docs/STOCKRAFT_REPORT_<INSTRUMENT>_<YYYY-MM-DD>.md`.
**Why:** Edge statistics drift with regime. We want to know when sweep depth p90 moves from 32 pips → 45 pips before the model does.
**How:**
1. `.github/workflows/stockraft_weekly.yml`: cron `0 6 * * MON` (Monday 06:00 UTC).
2. Workflow steps: checkout, install deps, run `scripts/fetch_backtest_data.py --analyze --instrument XAUUSD`, commit + open PR.
3. PR auto-assigned to Ada for review; merge if no regressions.
**Owner:** Ada
**Effort:** 2 hours
**Blocker:** Task 0.1 baseline must exist first.

---

## Phase 2 — Medium Term (1 month, 2026-05-24 → 2026-06-21)

Goals: Brian's preferred tools wired in, AI in the loop, paper trading milestone.

### Task 2.1: TradingView webhook intake
**What:** Stand up `prism/delivery/tv_webhook.py` — FastAPI endpoint on the Vultr VPS that ingests Pine Script alerts, writes them to `state/tv_alerts/`, and posts to a `#prism-tv-alerts` Slack channel. Cloudflare Tunnel for HTTPS without public IP.
**Why:** FULL_ANALYSIS.md §6. Brings Brian's TradingView ideas into the PRISM observability stream.
**How:**
1. New module ~120 LOC. FastAPI + Uvicorn + HMAC auth on `X-PRISM-Signature` header.
2. Pine Script alert template in `docs/PINE_SCRIPT_TEMPLATE.md`.
3. Cloudflare Tunnel config: `cloudflared tunnel create prism-tv` + `cloudflared service install`.
4. Tests: `tests/test_tv_webhook.py` — HMAC validation, JSON parsing, schema mismatch handling.
5. Slack channel: create `#prism-tv-alerts`, configure Slack notifier to route TV-source signals there (separate from `#prism-signals`).
**Owner:** Ada
**Effort:** 8-10 hours (1 PR)
**Blocker:** Task 0.3 (Vultr VPS) for hosting.

### Task 2.2: Per-signal Claude narrative
**What:** Async Claude Opus call after every SignalPacket fires, returning a 3-4 sentence plain-English summary. Posted as a Slack thread reply to the structured signal block.
**Why:** FULL_ANALYSIS.md §7.1. Faster cognition for Brian during the kill-zone window.
**How:**
1. New module `prism/ai/narrate.py` (~100 LOC).
2. Prompt template: structured JSON in, narrative out. Hardcoded ICT/SMC vocabulary.
3. Async — fires-and-forgets (`asyncio.create_task`); SignalPacket → MT5 path is unchanged.
4. Falls back to no-op on Anthropic API timeout (>5s).
5. Cost cap: rate-limit to ~$5/day via daily counter in state file.
6. Tests: mocked Anthropic client, prompt structure assertions.
**Owner:** Ada
**Effort:** 4-6 hours (1 PR)
**Blocker:** `ANTHROPIC_API_KEY` env var on runner host.

### Task 2.3: Post-trade Claude review
**What:** When a trade closes for a loss, Claude reads the audit log + bar context + smart-money block and writes a comment on the GitHub Issue. Hooks off `on_trade_closed`.
**Why:** FULL_ANALYSIS.md §7.2. Compounds learning from every loss.
**How:**
1. Extend `prism/journal/github_issues.py:on_trade_closed()` to call `prism/ai/post_trade_review.py`.
2. Review prompt includes: full audit row + 50 bars before + 20 bars after + news events in window.
3. Output is appended as Issue comment with `claude-review` label.
4. Cost: ~$0.02 per loss review at Opus pricing. Negligible.
**Owner:** Ada
**Effort:** 3-4 hours (1 PR)
**Blocker:** Task 0.4 (GitHub Issue automation merged).

### Task 2.4: Weekly meta-analysis
**What:** GitHub Actions cron — every Sunday 22:00 UTC — Claude reads past 7 days of audit + journal and produces `docs/weekly/<YYYY-WW>.md`. Opens PR.
**Why:** FULL_ANALYSIS.md §7.3. Turns ad-hoc review into a habit.
**How:**
1. `.github/workflows/weekly_meta.yml`.
2. Workflow calls `python -m prism.ai.weekly_meta --week-ending YYYY-MM-DD`.
3. Claude prompt template in `prism/ai/prompts/weekly_meta.md`.
4. PR auto-labeled `weekly-review`, auto-assigned to Brian.
**Owner:** Ada
**Effort:** 4-5 hours
**Blocker:** Task 0.4 (Issue journal needs to have history to summarize).

### Task 2.5: Paper trading milestone
**What:** 4 weeks of paper trading on Exness demo via the Vultr VPS, all instruments enabled, Phase 6 sweep + Po3 gates ON, Phase 8 quality filter ON, full automation pipeline live (TradingView intake, GitHub journal, Claude narrative). NO real capital.
**Why:** Validates end-to-end before risking money. Tests every integration: Dukascopy → ML → smart money → quality filter → MT5 → journal → AI review.
**How:**
1. Provision Exness Demo account (Brian, ~5 min).
2. Configure Vultr VPS to point at demo MT5 server.
3. Set `PRISM_MODE=AUTO`, all gates ON.
4. Run for 4 weeks. Daily check-in on Slack `#prism-signals`. Weekly meta from Task 2.4.
5. Acceptance: real signal cadence (≥3/day across instruments), no runner crashes, journal complete, no NaN P&L numbers, paper P&L is positive expectancy.
**Owner:** Brian (capital decision) + Ada (execution)
**Effort:** 4 weeks calendar; ~1 hour Brian/week, automated otherwise.
**Blocker:** Tasks 0.3, 0.4, 1.1, 1.4, 2.1, 2.2, 2.3 all merged.

---

## Phase 3 — Live Trading Milestone (after Phase 2 success, ~2026-06-21+)

Goals: Real capital, contained, with full kill-switch.

### Task 3.1: Phase 6 gates flip — Stage 2 → Stage 3
**What:** Following `docs/PHASE_6_ROLLOUT.md`. Flip `PRISM_SWEEP_REQUIRED=1` (Stage 2). One week of observation. If stable, flip `PRISM_PO3_REQUIRED=1` (Stage 3).
**Why:** Smart-money gates ship dark by default. We earn the right to gate by observing first. This is the deliberate path PRD2 specifies.
**How:** Single env var change on Vultr VPS. Runner restart. Watch Slack for signal volume drop. Compare audit log distributions before/after.
**Owner:** Brian (decision) + Ada (execution)
**Effort:** 5 minutes per stage flip; 1 week observation between.
**Blocker:** Phase 7.A gates passed (Task 1.5).

### Task 3.2: Live trading — small capital
**What:** Fund Exness live account with $5K. Run PRISM AUTO mode with `PRISM_RISK_PCT=0.005` (0.5% per trade, half the default). All instruments. All gates on.
**Why:** Reality check. Demo doesn't have real spread + slippage volatility. $5K capped at 0.5% = $25 max risk per trade.
**How:**
1. Brian funds Exness live ($5K).
2. Switch Vultr VPS env from demo creds to live creds.
3. Restart runner.
4. Daily P&L review. Weekly meta from Task 2.4.
5. Capital ramp gates: 4 weeks profitable + ≥40 trades + Sharpe ≥1.2 → step up to $20K and 1% risk. Repeat at $50K.
**Owner:** Brian (capital) + Ada (execution + monitoring)
**Effort:** 4 weeks observation, mostly automated.
**Blocker:** Task 2.5 (paper trading milestone passed).

### Task 3.3: Add 2nd + 3rd instrument to live
**What:** Once XAUUSD live is stable, expand to EURUSD + GBPUSD live. Each new instrument enters as a 4-week paper-then-live cycle of its own.
**Why:** Diversification. Reduces single-instrument concentration risk. Tests the model's portability.
**How:** Repeat Tasks 2.5 + 3.2 per instrument. Use the per-instrument walk-forward results from Task 1.5 to gate which to bring live first.
**Owner:** Brian + Ada
**Effort:** 4-8 weeks per instrument
**Blocker:** Task 3.2 stable for ≥4 weeks.

### Task 3.4: Multi-instrument portfolio drawdown guard
**What:** Extend `prism/delivery/drawdown_guard.py` from per-instrument to portfolio-level. If aggregate daily drawdown exceeds the kill threshold, halt ALL instruments, not just the offender.
**Why:** Three concurrent instruments = three independent risk pools. Aggregate exposure could exceed comfort even if no single instrument hits its kill.
**How:**
1. New `PortfolioDrawdownGuard` class wrapping the per-instrument guards.
2. Daily P&L aggregation across all open + closed trades.
3. Kill threshold: `PRISM_PORTFOLIO_DD_KILL_PCT` (default 5%).
4. Tests: synthetic multi-instrument loss scenarios.
**Owner:** Ada
**Effort:** 4-6 hours (1 PR)
**Blocker:** Task 3.3 (multi-instrument live).

---

## Cross-cutting / continuous

### Task X.1: Commit discipline
**What:** Per `~/.openclaw/workspace/AGENTS.md`: every substantive session, commit + push to `ada-memory`, open PRs on every code repo with changes. No silent accumulation.
**Owner:** Ada
**Effort:** 5 min/session
**Blocker:** None.

### Task X.2: Heartbeat / status surface
**What:** Daily Slack digest at 09:00 UTC: open trades, pending signals, journal stats, model drift indicators, gate 5 distribution health, runner uptime, news intelligence latency.
**Why:** Gregory's principle — "supreme excellence is invisible". Brian sees status without asking.
**How:** Extend `prism/delivery/slack_notifier.py` with a `daily_digest()` function. Cron via OpenClaw heartbeat.
**Owner:** Ada
**Effort:** 4 hours
**Blocker:** Task 0.4 (journal exists to summarize).

### Task X.3: Documentation discipline
**What:** Every PR updates the relevant doc (PRD2, PHASE_*_RESULTS.md, RUNBOOK.md). No code lands without docs. Already a project norm; keep it.
**Owner:** Ada
**Effort:** Ongoing
**Blocker:** None.

### Task X.4: PRISM Self-Sufficiency
**What:** Four-component infrastructure that runs PRISM without human supervision on the Windows VPS:

- **VPS Watchdog** (`prism/watchdog/watchdog.py`) — polls `python.exe` every 5 min, restarts `PRISM-Runner` via `schtasks` with up to 3 attempts (5 min apart), Slack-alerts on success and on terminal failure. Logs to `logs/watchdog.log`. Scheduled task: `PRISM-Watchdog`.
- **GitHub Webhook Auto-Deploy** (`scripts/deploy_webhook.py`) — Flask listener on `PRISM_DEPLOY_PORT` (default 9000). HMAC-validates `X-Hub-Signature-256` against `PRISM_DEPLOY_SECRET`, on push-to-main runs `git pull && pip install -r requirements.txt -q`, then `schtasks /end && /run` to bounce the runner. Scheduled task: `PRISM-DeployWebhook`.
- **Drift Monitor + Auto-Retrain** (`prism/watchdog/drift_monitor.py`) — daily 03:00 UTC. Reads 7d of `state/signal_audit/<INST>/*.jsonl`. Trips if signals/day < `PRISM_DRIFT_MIN_SIGNALS` (3), %NEUTRAL > `PRISM_DRIFT_NEUTRAL_PCT` (0.60), or mean confidence < `PRISM_DRIFT_MIN_CONFIDENCE` (0.45). On trip → `python -m prism.model.retrain --instrument <INST>`, validates via `predict.missing_model_files`, restarts runner, Slack summary. Scheduled task: `PRISM-DriftMonitor`.
- **Daily Digest** (`prism/watchdog/daily_digest.py`) — daily 08:00 UTC. Posts uptime %, signal counts, confidence distribution, last retrain per instrument, current execution mode. Scheduled task: `PRISM-DailyDigest`.

**Why:** Independence goal — PRISM survives Brian being offline for a week. Watchdog catches OS-level crashes; auto-deploy means PRs land in production within a minute of merge; drift monitor self-heals model decay; daily digest makes status invisible (Slack tells you, you don't ask).
**How:** Run `scripts/install_watchdog.bat` from an Administrator shell on the VPS after this PR merges. Configure GitHub webhook → `https://<vps>:9000/webhook` with the same `PRISM_DEPLOY_SECRET`.
**Owner:** Ada (built)
**Effort:** ~1 day total (built in this PR)
**Blocker:** None.

---

## Sequencing diagram

```
Week 1 (Phase 0):
  └─ 0.1 Dukascopy/Stockraft baseline ──┬─ 1.4 Portfolio sim
  └─ 0.2 MT5 decision ─────── 0.3 VPS ──┴─ 2.1 TV webhook ── 2.5 Paper milestone
  └─ 0.4 GitHub journal ─────────────── 2.3 Post-trade review
  └─ 0.5 Audit log health ── 1.5 Phase 7.A gates ── 3.1 Stage 2/3 flip

Week 2-3 (Phase 1):
  └─ 1.1 Phase 8 quality filter ────┐
  └─ 1.2 FVG timestamp migration ───┴── 1.3 Phase 7.B retrain
  └─ 1.4 Portfolio sim
  └─ 1.5 Phase 7.A gates (calendar-bound)
  └─ 1.6 Stockraft weekly cron

Week 4-7 (Phase 2):
  └─ 2.1 TV webhook
  └─ 2.2 Claude narrative ─── 2.3 Post-trade review ─── 2.4 Weekly meta
  └─ 2.5 Paper trading milestone (4 weeks calendar)

Week 8+ (Phase 3):
  └─ 3.1 Phase 6 gate flips
  └─ 3.2 Live $5K (4 weeks)
  └─ 3.3 Multi-instrument live (rolling)
  └─ 3.4 Portfolio drawdown guard
```

---

## Owner / effort summary

| Owner | Total ~hours | Notes |
|---|---|---|
| Ada | ~75 hours implementation | Sub-PR sized chunks; parallelizable across agents |
| Brian | ~3 hours decisions + 4 weeks calendar observation | Approval gates, capital, MT5/Exness creds |
| Compute | ~30 hours backtest + retrain wall-clock | Fully automated; runs on Vultr VPS or runner host |

Total wall-clock from 2026-05-03 to live trading milestone: ~8-10 weeks. Most of that is the calendar-bound paper trading observation (Task 2.5).

---

*End of ACTION_PLAN.md (2026-05-03).*
