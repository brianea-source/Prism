# PRISM Research Brief — April 26 2026
## Sources Analyzed by Ada (Opus 4.6 synthesis task)

---

## 1. SCREENSHOT CORPUS (18 images)

### Source A: @finastictrading (Mannan) — XAUUSD 5M Scalping System (Slides 1–8)
Full 4-part system:

**Part 1 — The Bias Check**
- Problem: scalpers trade against the bigger trend and get chopped
- Action: Lock direction on 1H BEFORE touching 5M
- Rules:
  1. Open 1H chart on XAUUSD
  2. Mark last 3 swing highs and lows
  3. Higher highs/lows = BUYS ONLY | Lower highs/lows = SELLS ONLY
  4. Bias locked for entire session — no flipping
- Quote: "The 1H is your compass. Without it you are guessing."

**Part 2 — The Trendline Setup**
- Problem: entering on impulse moves is how scalpers blow accounts
- Action: Wait for price to retrace to a diagonal trendline on 5M
- Rules:
  1. Draw diagonal line connecting recent higher lows (uptrend) or lower highs (downtrend)
  2. Wait for price to pull back and TOUCH this line
  3. If price not retracing — wait. That is the job.
  4. The trendline is your filter — no touch, no setup
- Quote: "The retracement is the trade. Everything else is noise."

**Part 3 — Double Confirmation**
- Problem: price touching the trendline is a warning, NOT an entry
- Action: Stack TWO confirmations before clicking
- Rules:
  1. Confirmation 1: Reversal candle at the trendline (engulfing, hammer, doji, morning/evening star)
  2. Confirmation 2: Candle closes ABOVE 8 EMA for buys (BELOW for sells)
  3. Both must align on same or consecutive candles
  4. One without the other = SKIP the trade
- Quote: "Two signals. One entry. No exceptions."

**Part 4 — Tight Risk, Quick Exit**
- Rules:
  1. SL below trendline with 5–10 pip buffer — NEVER wider
  2. Partial close at 10 pips — locks in profit early
  3. Full exit at 20 pips — never chase further
  4. Move SL to breakeven the moment 1:1 hits — protect every win
- Quote: "Small. Fast. Repeated. That is the entire game."

---

### Source B: @samsidney_ — 8-Step Trading Education Playlist
1. Bitcoin Playboy: Full trading course — $139K+ profit at 99.7% return *(source claim, unverified)*
2. TTrades: Top-down HTF→LTF analysis framework
3. The Traveling Trader: Trading psychology — mental game, 5 years failure before breakthrough
4. JeaFx (861K views): Indecision → Reversal → Strength candle → BUY entry
5. Bitcoin Playboy: Sweeps vs Displacements — ICT/SMC core distinction
6. Justin Werlein (127K views): FVG quality scoring — WHICH FVGs to trade vs skip
7. Umar Ashraf (1.1M views): Asymmetric R:R — risk small for outsized reward (~70:1)
8. Ewen Collier: 9:30 AM NYSE open — two repeatable daily setups

Additional: Jason Graystone (5.1M views) — 3-step chart pattern system; Wysetrade — OHLC anatomy

---

## 2. GATIETRADES (Ameer Gatie) — GEMS System

**Identity:** "GEMS Po3 Trader" — full transparency journey, daily livestreams 9:15am EST
**System:** GEMS — built from ICT @TheInnerCircleTrader + Po3 Mastery
**Instruments:** Futures on funded accounts

**GEMS = ICT-derived smart money framework:**

**Po3 (Power of Three) — The Core:**
1. Accumulation — Consolidation/ranging (smart money building positions quietly)
2. Manipulation — False move sweeping retail stops (liquidity hunt)
3. Distribution — True directional move (the real trade)

**ICT Concepts in GEMS:**
- Order Blocks (OBs): Last opposing candle before strong institutional move
- Fair Value Gaps (FVGs): 3-candle imbalances — price returns to fill
- Liquidity Sweeps: Price takes out obvious highs/lows → reverses (retail stop hunt)
- Kill Zones: London (2-5am EST), NY Open (7-10am EST), especially 9:15-9:30am
- Displacement: Strong impulsive candle confirming smart money direction
- Break of Structure (BOS): Swing point broken confirming trend continuation
- Change of Character (CHOCH): First opposing BOS — potential reversal signal
- Premium/Discount zones: Fib 50% = equilibrium; above = premium (sell), below = discount (buy)
- Optimal Trade Entry (OTE): 61.8%-78.6% fib retracement — ideal entry zone

---

## 3. CONSOLIDATED STRATEGY DNA (universal consensus across all sources)

### The Full Entry Checklist:
1. **HTF Bias** (Daily→4H→1H): swing structure → lock direction before session
2. **Kill Zone**: London (2-5am EST) or NY (7-10am EST) — do NOT trade outside
3. **Po3 Phase**: Wait for Manipulation phase (sweep) to complete
4. **Displacement**: Strong candle post-sweep confirming direction
5. **OB/FVG Zone**: Price retraces into OB or FVG created by displacement
6. **Candle Confirmation**: Reversal candle (engulfing/hammer/doji) + 8 EMA close
7. **Entry**: At OB/FVG boundary, SL below OB low
8. **Target**: Next liquidity pool, partial at 1:1, full at 2:1+

### Risk DNA:
- SL: tight, below OB/structure, never wide
- TP1 (partial 50%): 1:1 R or 10 pips
- TP2: Next liquidity / 2:1+ R
- Breakeven: at 1:1 hit
- Daily kill: -3R (already built in Phase 4)
- No direction flip mid-session once bias is locked

---

## 4. PRISM PHASE 4 (current) vs PRD2 TARGETS

### What PRISM Has:
- Live MT5 bars (Tiingo Phase 4)
- Model-trained signal generation (Phase 1-2)
- FVG detection with retest persistence (Phase 2)
- Session filter London/NY (Phase 3)
- Slack confirmation + poll flow (Phase 3)
- Daily drawdown kill-switch (Phase 4)
- In-flight signal dedup (Phase 4)
- MT5 reconnect loop (Phase 4)
- Sunday gap guard (Phase 4)

### PRD2 — New Capability Targets:
- **Phase 5**: HTF Bias Engine — 1H/4H/Daily swing structure → trade direction gate
- **Phase 6**: Smart Money Layer — OB detection, Sweep detection, CHOCH/BOS, Po3 phase tagging, Displacement confirmation
- **Phase 7**: ML Enhancement — retrain model features with ICT-derived inputs (FVG quality score, sweep-to-OB proximity, kill zone context, Po3 phase label, OTE distance)
- **Phase 8**: Trade Quality Filter — FVG quality scoring (Justin Werlein), OTE zone check, multi-confirmation stacking, asymmetric R:R targeting based on next liquidity pool

### PRD2 vs Updated PRD — Recommendation:
**PRD2.** Phase 1-4 are complete, tested, and merged to main. These new features represent an entirely new intelligence layer — transforming PRISM from a signal generator into a true smart money model. PRD2 starts clean, references Phase 1-4 as foundation, and owns the SMC/ICT upgrade arc.
