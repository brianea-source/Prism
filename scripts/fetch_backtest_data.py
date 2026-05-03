#!/usr/bin/env python3
"""
PRISM Backtesting Data Fetcher
==============================
Fetches historical OHLCV data for PRISM backtesting.

Instruments: XAUUSD, EURUSD
Timeframes: M15, H1, H4, D1
Period: 2021-01-01 → today (3+ years)
Sources: Tiingo (primary) → yfinance (fallback)

Usage:
    python3 scripts/fetch_backtest_data.py
    python3 scripts/fetch_backtest_data.py --start 2019-01-01
    python3 scripts/fetch_backtest_data.py --instrument XAUUSD --timeframe H1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

# ── Config ────────────────────────────────────────────────────────────────────
TIINGO_API_KEY = os.environ.get("TIINGO_API_KEY", "")
FRED_API_KEY   = os.environ.get("FRED_API_KEY", "")

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

YF_TICKERS = {
    "XAUUSD": "GC=F",
    "EURUSD": "EURUSD=X",
}
YF_INTERVALS = {"M15": "15m", "H1": "1h", "H4": "1h", "D1": "1d"}
YF_MAX_DAYS  = {"M15": 60, "H1": 730, "H4": 730, "D1": 9999}

TIINGO_TICKERS = {"XAUUSD": "GLD", "EURUSD": "eurusd"}
TIINGO_FREQ    = {"M15": "15Min", "H1": "1Hour", "H4": "4Hour", "D1": None}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("fetch_backtest")


# ── Tiingo Fetch ──────────────────────────────────────────────────────────────
def fetch_tiingo(instrument, timeframe, start, end):
    if not TIINGO_API_KEY:
        log.warning("TIINGO_API_KEY not set — skipping Tiingo")
        return None
    try:
        import requests
        ticker = TIINGO_TICKERS.get(instrument, instrument.lower())
        headers = {"Authorization": f"Token {TIINGO_API_KEY}", "Content-Type": "application/json"}
        s = requests.Session(); s.headers.update(headers)

        if timeframe == "D1":
            url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
            params = {"startDate": start, "endDate": end, "format": "json"}
        else:
            freq = TIINGO_FREQ.get(timeframe)
            url = f"https://api.tiingo.com/iex/{ticker}/prices"
            params = {"startDate": start, "endDate": end, "resampleFreq": freq, "columns": "open,high,low,close,volume"}

        resp = s.get(url, params=params, timeout=30); resp.raise_for_status()
        data = resp.json()
        if not data:
            return None

        df = pd.DataFrame(data)
        rename = {"date": "datetime", "adjOpen": "open", "adjHigh": "high",
                  "adjLow": "low", "adjClose": "close", "adjVolume": "volume",
                  "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
        col_map = {k: v for k, v in rename.items() if k in df.columns}
        df = df[list(col_map.keys())].rename(columns=col_map)
        # De-dup columns after rename
        df = df.loc[:, ~df.columns.duplicated()]
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.sort_values("datetime").reset_index(drop=True)
        df["source"] = "tiingo"
        log.info(f"  Tiingo -> {len(df):,} bars for {instrument} {timeframe}")
        return df
    except Exception as e:
        log.warning(f"  Tiingo fetch failed ({instrument} {timeframe}): {e}")
        return None


# ── yfinance Fetch ────────────────────────────────────────────────────────────
def fetch_yfinance(instrument, timeframe, start, end):
    ticker = YF_TICKERS.get(instrument)
    if not ticker:
        log.warning(f"No yfinance ticker for {instrument}")
        return None

    interval = YF_INTERVALS.get(timeframe, "1h")
    max_days = YF_MAX_DAYS.get(timeframe, 9999)
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")

    chunks = []
    cursor = start_dt
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=max_days), end_dt)
        try:
            t = yf.Ticker(ticker)
            df = t.history(start=cursor.strftime("%Y-%m-%d"), end=chunk_end.strftime("%Y-%m-%d"),
                           interval=interval, auto_adjust=True)
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
    df = df[~df.index.duplicated(keep="last")]
    df = df.reset_index().rename(columns={
        "Datetime": "datetime", "Date": "datetime",
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
    })
    df = df[[c for c in ["datetime","open","high","low","close","volume"] if c in df.columns]].copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    if timeframe == "H4":
        df = df.set_index("datetime").resample("4h").agg(
            {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
        ).dropna().reset_index()

    df["source"] = "yfinance"
    log.info(f"  yfinance -> {len(df):,} bars for {instrument} {timeframe}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def fetch_instrument(instrument, timeframe, start, end, force=False):
    out_file = DATA_DIR / f"{instrument}_{timeframe}_{start}_{end}.parquet"
    if out_file.exists() and not force:
        df = pd.read_parquet(out_file)
        log.info(f"  Cache hit: {out_file.name} ({len(df):,} bars)")
        return out_file

    log.info(f"Fetching {instrument} {timeframe} ({start} -> {end})...")
    df = fetch_tiingo(instrument, timeframe, start, end)
    if df is None or df.empty:
        log.info("  Tiingo returned nothing -- falling back to yfinance")
        df = fetch_yfinance(instrument, timeframe, start, end)

    if df is None or df.empty:
        log.error(f"  No data for {instrument} {timeframe}")
        return out_file

    df.to_parquet(out_file, index=False)
    log.info(f"  Saved {len(df):,} bars -> {out_file.name}")
    return out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",      default="2021-01-01")
    parser.add_argument("--end",        default=datetime.utcnow().strftime("%Y-%m-%d"))
    parser.add_argument("--instrument", default=None)
    parser.add_argument("--timeframe",  default=None)
    parser.add_argument("--force",      action="store_true")
    args = parser.parse_args()

    instruments = [args.instrument] if args.instrument else ["XAUUSD", "EURUSD"]
    timeframes  = [args.timeframe]  if args.timeframe  else ["M15", "H1", "H4", "D1"]

    results = []
    for inst in instruments:
        for tf in timeframes:
            path = fetch_instrument(inst, tf, args.start, args.end, force=args.force)
            if path.exists():
                df = pd.read_parquet(path)
                bars = len(df)
                source = df["source"].iloc[0] if "source" in df.columns and bars > 0 else "unknown"
                actual_start = str(df["datetime"].min())[:10] if bars > 0 else "N/A"
                actual_end   = str(df["datetime"].max())[:10] if bars > 0 else "N/A"
            else:
                bars, source, actual_start, actual_end = 0, "none", "N/A", "N/A"
            results.append({"instrument": inst, "timeframe": tf, "bars": bars,
                             "source": source, "start": actual_start, "end": actual_end})

    print("\n" + "="*62)
    print("PRISM Backtest Data — Fetch Summary")
    print("="*62)
    for r in results:
        s = "OK" if r["bars"] > 0 else "FAIL"
        print(f"  [{s:4s}]  {r['instrument']:8s}  {r['timeframe']:4s}  {r['bars']:>8,} bars"
              f"  {r['start']} -> {r['end']}  [{r['source']}]")
    print("="*62)
    print(f"  Data stored in: {DATA_DIR}\n")

    failed = [r for r in results if r["bars"] == 0]
    if failed:
        print(f"  WARNING: {len(failed)} dataset(s) failed to fetch.")
        sys.exit(1)


if __name__ == "__main__":
    main()
