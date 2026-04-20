# PRISM — Trading Model
**Predictive Regime Intelligence & Signal Model**

> *"Trade the signal. Not the noise."*

AI/ML-powered trading intelligence for Forex pairs and Gold (XAU/USD). Generates directional bias, entry signals, and optimized Stop Loss / Take Profit levels. Integrates with MetaTrader 5 (MT5) and delivers signals via OpenClaw → Slack.

---

## Quick Reference

- [PRD](docs/PRD.md) — Full Product Requirements Document (v1.1)

## Research Sources

| Source | Contribution |
|--------|-------------|
| @mar_antaya | ML methodology, data sources (FRED, Tiingo, Quiver) |
| @tradesbysci | ICC entry framework (Indication → Correction → Continuation) |
| @richkuo7 | TBD — pending analysis |

## Architecture

```
Data Layer          →  Model Layer         →  Signal Layer      →  Execution
Tiingo (price)         XGBoost classifier     ICC Phase detect     MT5 Python bridge
FRED (macro)           LightGBM ensemble      SL/TP calculator     OpenClaw → Slack
Quiver (sentiment)     Random Forest risk     Risk validator        Auto/Confirm mode
CFTC COT (flow)        SHAP audit loop        AOI confluence        Position sizing
```

## Instruments

Phase 1: XAU/USD, EUR/USD, GBP/USD
Phase 2: USD/JPY, NAS100, SPX500, BTC/USD

## Repository Structure

```
Trading-model/
├── docs/
│   └── PRD.md              # Product Requirements Document
├── notebooks/              # Jupyter research notebooks
├── prism/                  # Core Python package
│   ├── data/               # Data fetchers (Tiingo, FRED, Quiver)
│   ├── model/              # ML training, inference, evaluation
│   ├── signal/             # Entry/SL/TP generation, ICC detection
│   ├── mt5/                # MetaTrader5 bridge + order execution
│   └── openclaw/           # Slack notifications, commands
├── data/
│   ├── raw/                # Downloaded market data (gitignored)
│   └── processed/          # Feature-engineered datasets (gitignored)
├── config/                 # Instrument configs, model params
├── scripts/                # Backtest, retrain, live trading scripts
└── tests/                  # Unit tests
```

## Build Status

| Phase | Status |
|-------|--------|
| Phase 0: Research + Infrastructure | In Progress |
| Phase 1: Core ML Model | Pending |
| Phase 2: MT5 Integration | Pending |
| Phase 3: OpenClaw Integration | Pending |
| Phase 4: Gold + Multi-pair | Pending |

---

Owner: Brian Stiehl | AI: Ada Sandpaw | Started: 2026-04-20
