"""
prism/backtest/stockraft_analysis.py
=====================================
Stockraft master prompt methodology implemented as a PRISM analysis layer.

Implements the three-prompt workflow from the Stockraft PDF:
  Prompt 1: Swing Points & Pattern Detector
  Prompt 2: Candle Structure & Breakout Edge
  Prompt 3: Liquidity Sweep Analysis

These run on historical OHLCV data to extract edge statistics that feed into
PRISM's backtest validation and ML feature engineering.

Reference: Claude AI for Trading Strategy — Master Prompts (Stockraft, 2026)
Data source: Dukascopy (as recommended in the PDF — free, true spot, no rate limits)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("prism.backtest.stockraft")

# ── Session definitions (UTC) ──────────────────────────────────────────────────
SESSION_WINDOWS = {
    "ASIAN":  (0,  7),
    "LONDON": (7, 13),
    "NY":     (12, 21),
}

KILL_ZONES = {
    "london_open":  (7,  9),
    "ny_open":      (12, 14),
    "london_close": (14, 16),
    "ny_close":     (19, 21),
}

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


@dataclass
class StockraftReport:
    instrument:   str
    timeframe:    str
    start:        str
    end:          str
    total_bars:   int

    # Prompt 1
    swing_count:        int   = 0
    session_bias:       dict  = field(default_factory=dict)
    time_clusters:      dict  = field(default_factory=dict)
    day_of_week:        dict  = field(default_factory=dict)
    sweep_depth_stats:  dict  = field(default_factory=dict)
    kill_zone_pct:      float = 0.0

    # Prompt 2
    breakout_count:      int   = 0
    breakout_win_rate:   float = 0.0
    breakout_expectancy: float = 0.0
    breakout_by_session: dict  = field(default_factory=dict)

    # Prompt 3
    sweep_count:         int   = 0
    sweep_reversal_rate: float = 0.0
    sweep_by_session:    dict  = field(default_factory=dict)
    sweep_depth_p50:     float = 0.0
    sweep_depth_p90:     float = 0.0

    # Unified signal matrix
    signal_matrix: dict = field(default_factory=dict)


def _pip(instrument: str) -> float:
    return 0.01 if instrument in ("XAUUSD", "BTCUSD") else 0.0001


def _session(hour: int) -> str:
    if SESSION_WINDOWS["LONDON"][0] <= hour < SESSION_WINDOWS["LONDON"][1]:
        return "LONDON"
    if SESSION_WINDOWS["NY"][0] <= hour < SESSION_WINDOWS["NY"][1]:
        return "NY"
    return "ASIAN"


def _in_kill_zone(hour: int) -> bool:
    return any(s <= hour < e for s, e in KILL_ZONES.values())


def _detect_swings(df: pd.DataFrame, n: int = 5):
    highs, lows = [], []
    for i in range(n, len(df) - n):
        h = df["high"].iloc
        l = df["low"].iloc
        if all(h[i] >= h[i-k] and h[i] >= h[i+k] for k in range(1, n+1)):
            highs.append(i)
        if all(l[i] <= l[i-k] and l[i] <= l[i+k] for k in range(1, n+1)):
            lows.append(i)
    return highs, lows


def _body_pct(row: pd.Series) -> float:
    rng = row["high"] - row["low"]
    return abs(row["close"] - row["open"]) / rng if rng > 0 else 0.0


def _candle_type(row: pd.Series) -> str:
    rng  = row["high"] - row["low"]
    if rng == 0:
        return "none"
    body = abs(row["close"] - row["open"])
    bpct = body / rng
    uw   = row["high"] - max(row["open"], row["close"])
    lw   = min(row["open"], row["close"]) - row["low"]
    if bpct < 0.10:
        return "doji"
    if bpct >= 0.60 and uw < body * 0.3 and lw < body * 0.3:
        return "engulfing_candidate"
    if bpct < 0.40 and (uw > body * 2 or lw > body * 2):
        return "pin_bar"
    return "other"


# ── Prompt 1 ──────────────────────────────────────────────────────────────────

def analyze_swing_points(df: pd.DataFrame, instrument: str, n: int = 5) -> dict:
    pip_sz  = _pip(instrument)
    h_idxs, l_idxs = _detect_swings(df, n)
    h_set, l_set   = set(h_idxs), set(l_idxs)

    res = {
        "n_highs": len(h_idxs), "n_lows": len(l_idxs),
        "session_high": {s: 0 for s in ("ASIAN","LONDON","NY")},
        "session_low":  {s: 0 for s in ("ASIAN","LONDON","NY")},
        "hour_high": {}, "hour_low": {},
        "dow_high":  {d: 0 for d in DAYS},
        "dow_low":   {d: 0 for d in DAYS},
        "sweep_h": [], "sweep_l": [],
        "candle_h": {}, "candle_l": {},
        "kill_zone_n": 0,
    }

    def sweep_depth(idx: int, kind: str) -> float:
        if kind == "high":
            prev = [i for i in h_idxs if i < idx]
            return max(0.0, (df["high"].iloc[idx] - df["high"].iloc[prev[-1]]) / pip_sz) if prev else 0.0
        else:
            prev = [i for i in l_idxs if i < idx]
            return max(0.0, (df["low"].iloc[prev[-1]] - df["low"].iloc[idx]) / pip_sz) if prev else 0.0

    total = len(h_idxs) + len(l_idxs)
    kz_n  = 0

    for idx in h_idxs:
        ts  = pd.Timestamp(df["datetime"].iloc[idx])
        hr  = ts.hour; dow = ts.day_of_week; sess = _session(hr)
        ct  = _candle_type(df.iloc[idx])
        res["session_high"][sess] += 1
        res["hour_high"][hr] = res["hour_high"].get(hr, 0) + 1
        res["sweep_h"].append(sweep_depth(idx, "high"))
        res["candle_h"][ct] = res["candle_h"].get(ct, 0) + 1
        if dow < 5: res["dow_high"][DAYS[dow]] += 1
        if _in_kill_zone(hr): kz_n += 1

    for idx in l_idxs:
        ts  = pd.Timestamp(df["datetime"].iloc[idx])
        hr  = ts.hour; dow = ts.day_of_week; sess = _session(hr)
        ct  = _candle_type(df.iloc[idx])
        res["session_low"][sess] += 1
        res["hour_low"][hr] = res["hour_low"].get(hr, 0) + 1
        res["sweep_l"].append(sweep_depth(idx, "low"))
        res["candle_l"][ct] = res["candle_l"].get(ct, 0) + 1
        if dow < 5: res["dow_low"][DAYS[dow]] += 1
        if _in_kill_zone(hr): kz_n += 1

    res["kill_zone_pct"] = round(100 * kz_n / max(total, 1), 1)

    for key, arr_key in [("stats_h","sweep_h"), ("stats_l","sweep_l")]:
        arr = np.array(res[arr_key])
        if len(arr) > 0:
            p90 = float(np.percentile(arr, 90))
            res[key] = {
                "median": float(np.median(arr)),
                "mean":   float(arr.mean()),
                "p90":    p90,
                "recommended_sl": round(p90 + 3, 1),
            }
    return res


# ── Prompt 2 ──────────────────────────────────────────────────────────────────

def analyze_breakout_edge(
    df: pd.DataFrame, instrument: str,
    lookback: int = 20, body_min: float = 0.60,
    sl_pips: float = 35.0, target_rr: float = 2.0,
) -> dict:
    pip_sz   = _pip(instrument)
    roll_h   = df["high"].rolling(lookback).max().shift(1)
    roll_l   = df["low"].rolling(lookback).min().shift(1)
    wins     = []; losses = []
    by_sess  = {s: {"n":0,"wins":0} for s in ("ASIAN","LONDON","NY")}

    for i in range(lookback + 3, len(df) - 25):
        row  = df.iloc[i]
        bpct = _body_pct(row)
        if bpct < body_min:
            continue
        hr   = pd.Timestamp(row["datetime"]).hour
        sess = _session(hr)
        sl   = sl_pips * pip_sz
        bullish = row["close"] > roll_h.iloc[i]
        bearish = row["close"] < roll_l.iloc[i]
        if not (bullish or bearish):
            continue

        entry = row["close"]
        if bullish:
            tp_p, sl_p = entry + sl * target_rr, entry - sl
            def hit(fut): return fut["high"] >= tp_p, fut["low"] <= sl_p
        else:
            tp_p, sl_p = entry - sl * target_rr, entry + sl
            def hit(fut): return fut["low"] <= tp_p, fut["high"] >= sl_p

        won = False; lost = False
        for j in range(i+1, min(i+50, len(df))):
            w, l = hit(df.iloc[j])
            if w: won = True; break
            if l: lost = True; break

        by_sess[sess]["n"] += 1
        if won:
            wins.append(1); by_sess[sess]["wins"] += 1
        else:
            losses.append(1)

    n = len(wins) + len(losses)
    wr = len(wins) / n if n else 0.0
    exp = (wr * target_rr) + ((1 - wr) * -1.0) if n else 0.0
    return {
        "total": n, "win_rate": round(wr, 4),
        "expectancy_r": round(exp, 4),
        "is_positive_edge": exp > 0,
        "by_session": by_sess,
    }


# ── Prompt 3 ──────────────────────────────────────────────────────────────────

def analyze_liquidity_sweeps(
    df: pd.DataFrame, instrument: str,
    n: int = 5, rev_window: int = 4,
) -> dict:
    pip_sz = _pip(instrument)
    h_idxs, l_idxs = _detect_swings(df, n)
    h_set, l_set   = set(h_idxs), set(l_idxs)
    sweeps = []

    for i in range(n+5, len(df) - rev_window):
        row = df.iloc[i]
        hr  = pd.Timestamp(row["datetime"]).hour
        sess = _session(hr)
        kz   = _in_kill_zone(hr)

        for j in range(max(0,i-40), i):
            if j in h_set:
                sh = df["high"].iloc[j]
                if row["high"] > sh and row["close"] < sh:
                    depth = (row["high"] - sh) / pip_sz
                    rev = any(df["close"].iloc[k] < sh for k in range(i+1, min(i+rev_window+1,len(df))))
                    sweeps.append({"depth":depth,"rev":rev,"sess":sess,"kz":kz}); break
            if j in l_set:
                sl = df["low"].iloc[j]
                if row["low"] < sl and row["close"] > sl:
                    depth = (sl - row["low"]) / pip_sz
                    rev = any(df["close"].iloc[k] > sl for k in range(i+1, min(i+rev_window+1,len(df))))
                    sweeps.append({"depth":depth,"rev":rev,"sess":sess,"kz":kz}); break

    if not sweeps:
        return {"total": 0}

    dfs = pd.DataFrame(sweeps)
    depths = dfs["depth"].values
    res = {
        "total": len(sweeps),
        "reversal_rate": round(float(dfs["rev"].mean()), 4),
        "kz_reversal_rate": round(float(dfs[dfs["kz"]]["rev"].mean()), 4) if dfs["kz"].any() else 0.0,
        "depth_p50": round(float(np.median(depths)), 2),
        "depth_p90": round(float(np.percentile(depths, 90)), 2),
        "recommended_sl": round(float(np.percentile(depths, 90)) + 3, 1),
        "by_session": {},
    }
    for s in ("ASIAN","LONDON","NY"):
        sub = dfs[dfs["sess"] == s]
        if len(sub) > 0:
            res["by_session"][s] = {"n": len(sub), "reversal_rate": round(float(sub["rev"].mean()), 4)}
    return res


# ── Master runner ─────────────────────────────────────────────────────────────

def run_stockraft_analysis(
    df: pd.DataFrame, instrument: str, timeframe: str,
    sl_pips: float = 35.0, target_rr: float = 2.0,
) -> StockraftReport:
    """Run all three Stockraft master prompts and produce a unified report."""
    report = StockraftReport(
        instrument=instrument, timeframe=timeframe,
        start=str(df["datetime"].min())[:10],
        end=str(df["datetime"].max())[:10],
        total_bars=len(df),
    )
    log.info(f"Stockraft analysis: {instrument} {timeframe} ({len(df):,} bars)")

    sp = analyze_swing_points(df, instrument)
    report.swing_count   = sp["n_highs"] + sp["n_lows"]
    report.session_bias  = {"HIGH": sp["session_high"], "LOW": sp["session_low"]}
    report.time_clusters = {"HIGH": sp["hour_high"],    "LOW": sp["hour_low"]}
    report.day_of_week   = {"HIGH": sp["dow_high"],     "LOW": sp["dow_low"]}
    report.sweep_depth_stats = sp.get("stats_h", {})
    report.kill_zone_pct = sp.get("kill_zone_pct", 0.0)

    bo = analyze_breakout_edge(df, instrument, sl_pips=sl_pips, target_rr=target_rr)
    report.breakout_count      = bo["total"]
    report.breakout_win_rate   = bo["win_rate"]
    report.breakout_expectancy = bo["expectancy_r"]
    report.breakout_by_session = bo["by_session"]

    sw = analyze_liquidity_sweeps(df, instrument)
    report.sweep_count        = sw.get("total", 0)
    report.sweep_reversal_rate = sw.get("reversal_rate", 0.0)
    report.sweep_by_session   = sw.get("by_session", {})
    report.sweep_depth_p50    = sw.get("depth_p50", 0.0)
    report.sweep_depth_p90    = sw.get("depth_p90", 0.0)

    rec_sl = max(
        sw.get("recommended_sl", sl_pips),
        sp.get("stats_h", {}).get("recommended_sl", sl_pips),
    )
    top_h = sorted(sp.get("hour_high", {}).items(), key=lambda x: x[1], reverse=True)[:3]

    report.signal_matrix = {
        "recommended_sl_pips":      round(rec_sl, 1),
        "target_rr":                target_rr,
        "top_swing_hours_utc":      [h for h, _ in top_h],
        "kill_zone_swing_pct":      report.kill_zone_pct,
        "london_pct":               _dom(sp, "LONDON"),
        "ny_pct":                   _dom(sp, "NY"),
        "asian_fakeout_risk":       _asian_risk(bo),
        "sweep_kz_reversal_rate":   sw.get("kz_reversal_rate", sw.get("reversal_rate", 0.0)),
        "breakout_positive_edge":   bo.get("is_positive_edge", False),
    }
    return report


def _dom(sp, sess) -> float:
    total = sum(sp["session_high"].values()) + sum(sp["session_low"].values())
    return round((sp["session_high"].get(sess,0) + sp["session_low"].get(sess,0)) / max(total,1), 4)


def _asian_risk(bo) -> bool:
    a = bo["by_session"].get("ASIAN", {}); l = bo["by_session"].get("LONDON", {})
    if a.get("n",0) < 10 or l.get("n",0) < 10: return True
    return (a["wins"]/a["n"]) < (l["wins"]/l["n"]) * 0.8


def print_report(r: StockraftReport) -> None:
    sm = r.signal_matrix
    print(f"\n{'='*60}")
    print(f"STOCKRAFT ANALYSIS — {r.instrument} {r.timeframe}")
    print(f"Period: {r.start} → {r.end}  |  {r.total_bars:,} bars")
    print("="*60)
    print(f"\n[PROMPT 1] SWING POINTS  (n={r.swing_count})")
    print(f"  Kill zone swings:   {r.kill_zone_pct:.1f}%")
    print(f"  London dominance:   {100*sm.get('london_pct',0):.1f}%")
    print(f"  NY dominance:       {100*sm.get('ny_pct',0):.1f}%")
    if r.sweep_depth_stats:
        print(f"  Sweep depth p90:    {r.sweep_depth_stats.get('p90',0):.1f} pips  "
              f"→ SL = {r.sweep_depth_stats.get('recommended_sl',35):.0f}p")
    top = sorted(r.time_clusters.get("HIGH",{}).items(), key=lambda x:x[1], reverse=True)[:3]
    print(f"  Top swing hours:    {[f'{h:02d}:00 UTC' for h,_ in top]}")

    print(f"\n[PROMPT 2] BREAKOUT EDGE (n={r.breakout_count})")
    print(f"  Win rate:           {100*r.breakout_win_rate:.1f}%")
    print(f"  Expectancy:         {r.breakout_expectancy:+.2f}R")
    print(f"  Edge:               {'✅ positive' if r.breakout_expectancy > 0 else '❌ negative'}")
    for sess, d in r.breakout_by_session.items():
        if d["n"] > 0:
            print(f"    {sess:8s}  n={d['n']:3d}  WR={100*d['wins']/d['n']:.0f}%")

    print(f"\n[PROMPT 3] LIQUIDITY SWEEPS (n={r.sweep_count})")
    print(f"  Reversal rate:      {100*r.sweep_reversal_rate:.1f}%")
    print(f"  Kill zone rev rate: {100*sm.get('sweep_kz_reversal_rate',0):.1f}%")
    print(f"  Sweep depth:        p50={r.sweep_depth_p50:.1f}p  p90={r.sweep_depth_p90:.1f}p")

    print(f"\n[SIGNAL MATRIX]")
    print(f"  Recommended SL:     {sm.get('recommended_sl_pips',35):.0f} pips")
    print(f"  Target R:R:         1:{sm.get('target_rr',2)}")
    print(f"  Asian fakeout risk: {'⚠️  HIGH — skip Asian breakouts' if sm.get('asian_fakeout_risk') else '✅ Normal'}")
    print("="*60)
