# PRISM — Documentation Index

> One stop for every PRISM design and operations doc. If you're new, read top-to-bottom.

---

## Product Requirements

| Doc | Scope | Status |
|-----|-------|--------|
| [PRD.md](./PRD.md) | PRISM v1.0 — original PRD (Phases 0-4: data pipeline, ML model, MT5 integration, OpenClaw delivery) | ✅ Frozen baseline |
| [PRISM_PRD.md](./PRISM_PRD.md) | PRD v1.2 amendment — multi-timeframe top-down + FVG confluence + Exness broker recommendation | ✅ Frozen baseline |
| [PRD2.md](./PRD2.md) | PRISM v2.0 — Phases 5-8: HTF Bias Engine, Smart Money (OB/RB lifecycle, sweeps, Po3), ML feature enhancement, trade quality filter | 🟡 DRAFT (under review) |
| [PRD2_APPENDIX_RESEARCH.md](./PRD2_APPENDIX_RESEARCH.md) | PRD2 supplementary — 5 trader profiles (powell.trades, fearingtrades, nasdaqab, jeron, barontrades) + Phase 9 TradingView MCP vision | 🟡 RESEARCH |

If you're implementing Phase 5+, **read `PRD2.md` first** and treat the appendix as supporting context — Phase 6's RB lifecycle state machine lives in `PRD2.md`, not the appendix.

---

## Research Corpus (inputs to PRD2)

These are the synthesis artifacts that fed PRD2. Read in this order if you want the full picture:

1. [RESEARCH_BRIEF.md](./RESEARCH_BRIEF.md) — Source corpus index: 18 screenshots (finastictrading, GatieTrades GEMS, samsidney_ playlist) and the gap map against current PRISM code.
2. [FULL_ANALYSIS.md](./FULL_ANALYSIS.md) — Per-source teaching breakdown, cross-source consensus, 10 PRISM gaps with ranked impact estimates.
3. [SUMMARY_FOR_BRIAN.md](./SUMMARY_FOR_BRIAN.md) — Executive synthesis: top 10 insights, PRD2 vs. updated-PRD recommendation, expected win-rate delta, baseline test result.

> Note on overlap: the "10 insights" list appears in slight variations across all three docs. `FULL_ANALYSIS.md` is the most exhaustive; the other two are condensations for different audiences.

---

## Operations

| Doc | Scope |
|-----|-------|
| [RUNBOOK.md](./RUNBOOK.md) | Operator playbook — host setup, retraining, pre-live health check, runner start, observability, recovery, demo→live promotion |

---

## Phase Planning

Forward-looking scope docs that lock the surface area of an upcoming phase before the implementation PR opens. Each one specifies inputs, outputs, acceptance gates, and deferred-work splits so the implementation review is a confirmation rather than a redesign.

| Doc | Scope |
|-----|-------|
| [PHASE_7A_SCOPE.md](./PHASE_7A_SCOPE.md) | Pre-Phase-8 scope for the ML feature-engineering retrain. The 5 buildable-now ICT features, training-data path (historical reconstruction + Stage 1 validation), walk-forward harness spec, acceptance gates. |

---

## Versioning convention

- **PRD = v1.x** (Phases 0-4) — signal foundation, shipped, 360 tests green on `main`
- **PRD2 = v2.x** (Phases 5-8) — intelligence layer, in spec
- **Phase 9+** — TradingView MCP visual confirmation, deferred until PRD2 ships

When a phase merges to main, mark its PRD section ✅ Done in this index.
