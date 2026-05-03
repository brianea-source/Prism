#!/usr/bin/env python3
"""
PRISM Backtesting Data Fetcher
==============================
Fetches historical OHLCV data for PRISM backtesting.

Data source priority (per Stockraft PDF recommendation):
  1. Dukascopy  — TRUE spot XAUUSD/EURUSD, free, no API key, no rate limits
                  Best granularity available. Recommended in Stockraft PDF.
  2. yfinance   — Fallback only (GC=F futures for gold, less accurate)

Instruments: XAUUSD, EURUSD
Timeframes:  M15, H1, H4, D1
Period:      2021-01-01 → today (5 years)

Usage:
    python3 scripts/fetch_backtest_data.py
    python3 scripts/fetch_backtest_data.py --start 2020-01-01
    python3 scripts/fetch_backtest_data.py --instrument XAUUSD --timeframe H1
    python3 scripts/fetch_backtest_data.py --export-csv   # export Stockraft-format CSVs
    python3 scripts/fetch_backtest_data.py --analyze      # run Stockraft analysis after fetch
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prism.data.dukascopy import fetch_dukascopy, export_stockraft_csv

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

YF_TICKERS   = {"XAUUSD": "GC=F", "EURUSD": "EURUSD=X"}
YF_INTERVALS = {"M15": "15m", "H1": "1h", "H4": "1h", "D1": "1d"}
YF_MAX_DAYS  = {"M15": 60,    "H1": 730,  "H4": 730,  "D1": 9999}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fetch_backtest")


# ── yfinance fallback ─────────────────────────────────────────────────────────

def fetch_yfinance(instrument, timeframe, start, end):
    ticker = YF_TICKERS.get(instrument)
    if not ticker:
        return None
    interval = YF_INTERVALS.get(timeframe, "1h")
    max_days = YF_MAX_DAYS.get(timeframe, 9999)
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")
    from datetime import timedelta
    chunks, cursor = [], start_dt
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=max_days), end_dt)
        try:
            df = yf.Ticker(ticker).history(
                start=cursor.strftime("%Y-%m-%d"),
                end=chunk_end.strftime("%Y-%m-%d"),
                interval=interval, auto_adjust=True,
            )
            if not df.empty:
                chunks.append(df)
            time.sleep(0.5)
        except Exception as e:
            log.warning(f"  yfinance chunk {cursor.date()} failed: {e}")
        cursor = chunk_end
    if not chunks:
        return None
    df = pd.concat(chunks)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")].reset_index().rename(columns={
        "Datetime":"datetime","Date":"datetime",
        "Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume",
    })
    df = df[[c for c in ["datetime","open","high","low","close","volume"] if c in df.columns]]
    if timeframe == "H4":
        df = df.set_index("datetime").resample("4h").agg(
            open="first", high="max", low="min", close="last", volume="sum"
        ).dropna().reset_index()
    df["source"] = "yfinance_fallback"
    log.warning(f"  ⚠️  Using yfinance fallback ({instrument} {timeframe}) — "
                "this is GC=F futures, NOT true spot. Prefer Dukascopy.")
    return df


# ── Main fetch logic ──────────────────────────────────────────────────────────

def fetch_instrument(instrument, timeframe, start, end, force=False):
    out_file = DATA_DIR / f"{instrument}_{timeframe}_{start}_{end}.parquet"
    if out_file.exists() and not force:
        df = pd.read_parquet(out_file)
        log.info(f"  Cache hit: {out_file.name} ({len(df):,} bars)")
        return out_file

    log.info(f"\nFetching {instrument} {timeframe} ({start} → {end})...")

    # Primary: Dukascopy (true spot, recommended by Stockraft PDF)
    df = fetch_dukascopy(instrument, timeframe, start, end)

    if df is None or df.empty:
        log.warning(f"  Dukascopy returned nothing — falling back to yfinance")
        df = fetch_yfinance(instrument, timeframe, start, end)

    if df is None or df.empty:
        log.error(f"  ❌  No data for {instrument} {timeframe}")
        return out_file

    df.to_parquet(out_file, index=False)
    log.info(f"  ✅  Saved {len(df):,} bars → {out_file.name}")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="PRISM backtest data fetcher")
    parser.add_argument("--start",       default="2021-01-01")
    parser.add_argument("--end",         default=datetime.utcnow().strftime("%Y-%m-%d"))
    parser.add_argument("--instrument",  default=None)
    parser.add_argument("--timeframe",   default=None)
    parser.add_argument("--force",       action="store_true", help="Re-fetch even if cached")
    parser.add_argument("--export-csv",  action="store_true", help="Also export Stockraft-format CSVs")
    parser.add_argument("--analyze",     action="store_true", help="Run Stockraft analysis after fetch")
    args = parser.parse_args()

    instruments = [args.instrument] if args.instrument else ["XAUUSD", "EURUSD"]
    timeframes  = [args.timeframe]  if args.timeframe  else ["M15", "H1", "H4", "D1"]

    log.info("PRISM Backtest Data Fetcher")
    log.info(f"Primary source: Dukascopy (true spot — as specified in Stockraft PDF)")
    log.info(f"Fallback:       yfinance GC=F (futures proxy — less accurate)")
    log.info("")

    results = []
    for inst in instruments:
        for tf in timeframes:
            path = fetch_instrument(inst, tf, args.start, args.end, force=args.force)
            if path.exists():
                df  = pd.read_parquet(path)
                src = df["source"].iloc[0] if "source" in df.columns and len(df) > 0 else "unknown"
                s   = str(df["datetime"].min())[:10] if len(df) > 0 else "N/A"
                e   = str(df["datetime"].max())[:10] if len(df) > 0 else "N/A"
                results.append({"inst":inst,"tf":tf,"bars":len(df),"source":src,"start":s,"end":e})
            else:
                results.append({"inst":inst,"tf":tf,"bars":0,"source":"none","start":"N/A","end":"N/A"})

    # Optional: export Stockraft CSVs
    if args.export_csv:
        log.info("\nExporting Stockraft-format CSVs...")
        for r in results:
            if r["bars"] > 0:
                try:
                    csv_path = export_stockraft_csv(r["inst"], r["tf"], args.start, args.end)
                    log.info(f"  CSV: {csv_path.name}")
                except Exception as e:
                    log.warning(f"  CSV export failed ({r['inst']} {r['tf']}): {e}")

    # Optional: run Stockraft analysis
    if args.analyze:
        log.info("\nRunning Stockraft master prompt analysis...")
        try:
            from prism.backtest.stockraft_analysis import run_stockraft_analysis, print_report
            for r in results:
                if r["bars"] > 200 and r["tf"] in ("H1", "M15"):
                    path = DATA_DIR / f"{r['inst']}_{r['tf']}_{args.start}_{args.end}.parquet"
                    df   = pd.read_parquet(path)
                    report = run_stockraft_analysis(df, r["inst"], r["tf"])
                    print_report(report)
        except Exception as e:
            log.warning(f"Analysis failed: {e}")

    # Summary
    print(f"\n{'='*68}")
    print("PRISM Backtest Data — Fetch Summary")
    print("="*68)
    for r in results:
        flag = "✅" if r["bars"] > 0 else "❌"
        src_note = " ⚠️ PROXY" if "yfinance" in r["source"] else ""
        print(f"  {flag}  {r['inst']:8s}  {r['tf']:4s}  {r['bars']:>8,} bars  "
              f"{r['start']} → {r['end']}  [{r['source']}{src_note}]")
    print(f"  Data directory: {DATA_DIR}")
    print("="*68)

    failed = [r for r in results if r["bars"] == 0]
    if failed:
        print(f"\n  ⚠️  {len(failed)} dataset(s) failed to fetch.")
        sys.exit(1)


if __name__ == "__main__":
    main()
