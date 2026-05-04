"""
prism/data/dukascopy.py
=======================
Dukascopy historical OHLCV data fetcher.

Dukascopy provides FREE true spot data (not ETF proxies, not futures):
  - XAUUSD: true spot gold, 2003 → today
  - EURUSD: true spot FX, 2003 → today
  - M1, M5, M15, H1, H4, D1 candles
  - No API key, no rate limits, no registration required

Data format: LZMA-compressed bi5 binary files.
URL: https://datafeed.dukascopy.com/datafeed/{INSTRUMENT}/{year}/{month:02d}/{day:02d}/BID_candles_min_1.bi5
     (month is 0-indexed on Dukascopy!)

bi5 record struct (big-endian, 24 bytes each):
  uint32  ms_from_midnight  - milliseconds from midnight UTC
  uint32  open              - price * POINT_FACTOR
  uint32  high              - price * POINT_FACTOR
  uint32  low               - price * POINT_FACTOR
  uint32  close             - price * POINT_FACTOR
  float32 volume            - tick volume

Data source spec from Stockraft PDF:
  "Dukascopy — Free historical tick data. Requires basic registration.
   Best granularity available without paying — useful for very precise
   sweep analysis."
"""
from __future__ import annotations

import lzma
import logging
import struct
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

log = logging.getLogger("prism.data.dukascopy")

# ── Instrument config ─────────────────────────────────────────────────────────
INSTRUMENT_CONFIG: dict[str, dict] = {
    "XAUUSD": {"symbol": "XAUUSD", "point_factor": 1000.0},
    "EURUSD": {"symbol": "EURUSD", "point_factor": 100000.0},
    "GBPUSD": {"symbol": "GBPUSD", "point_factor": 100000.0},
    "USDJPY": {"symbol": "USDJPY", "point_factor": 1000.0},
    "BTCUSD": {"symbol": "BTCUSD", "point_factor": 1000.0},
}

RESAMPLE_RULES: dict[str, str] = {
    "M1":  "1min",
    "M5":  "5min",
    "M15": "15min",
    "H1":  "1h",
    "H4":  "4h",
    "D1":  "1D",
}

DUKA_BASE   = "https://datafeed.dukascopy.com/datafeed"
RECORD_SIZE = 24        # bytes per M1 bar
RECORD_FMT  = ">IIIIIf"  # big-endian: 5×uint32 + float32
WEEKEND_DAYS = {5, 6}   # Saturday=5, Sunday=6

CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "dukascopy_cache"

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
})


# ── Internal helpers ──────────────────────────────────────────────────────────

def _cache_path(symbol: str, year: int, month_0idx: int, day: int) -> Path:
    return CACHE_DIR / symbol / str(year) / f"{month_0idx:02d}" / f"{day:02d}" / "BID_candles_min_1.bi5"


def _fetch_day_raw(instrument: str, dt: datetime) -> Optional[bytes]:
    """Download one day's M1 BID candles bi5 file. Returns bytes or None."""
    cfg    = INSTRUMENT_CONFIG[instrument]
    symbol = cfg["symbol"]
    year   = dt.year
    month  = dt.month - 1   # Dukascopy months are 0-indexed!
    day    = dt.day

    cache_file = _cache_path(symbol, year, month, day)
    if cache_file.exists() and cache_file.stat().st_size > 0:
        return cache_file.read_bytes()

    url = f"{DUKA_BASE}/{symbol}/{year}/{month:02d}/{day:02d}/BID_candles_min_1.bi5"
    try:
        resp = _SESSION.get(url, timeout=20)
        if resp.status_code == 200 and len(resp.content) > 0:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(resp.content)
            return resp.content
        return None
    except Exception as e:
        log.debug(f"_fetch_day_raw error ({symbol} {dt.date()}): {e}")
        return None


def _parse_bi5(raw: bytes, date: datetime, point_factor: float) -> pd.DataFrame:
    """Decompress and parse one day's bi5 file into a DataFrame of M1 bars."""
    try:
        data = lzma.decompress(raw)
    except Exception as e:
        log.debug(f"lzma decompress failed: {e}")
        return pd.DataFrame()

    n = len(data) // RECORD_SIZE
    if n == 0:
        return pd.DataFrame()

    midnight_ms = int(date.replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
    ).timestamp() * 1000)

    rows = []
    for i in range(n):
        chunk = data[i * RECORD_SIZE: (i + 1) * RECORD_SIZE]
        if len(chunk) < RECORD_SIZE:
            break
        ms, o, h, l, c, vol = struct.unpack(RECORD_FMT, chunk)
        ts = pd.Timestamp(midnight_ms + ms, unit="ms", tz="UTC")
        rows.append({
            "datetime": ts,
            "open":     o / point_factor,
            "high":     h / point_factor,
            "low":      l / point_factor,
            "close":    c / point_factor,
            "volume":   float(vol),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)


def _resample_df(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample M1 data to the requested timeframe."""
    if timeframe == "M1":
        return df
    rule = RESAMPLE_RULES[timeframe]
    df2 = df.set_index("datetime")
    resampled = df2.resample(rule, label="left", closed="left").agg(
        open="first", high="max", low="min", close="last", volume="sum"
    ).dropna().reset_index()
    return resampled


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_dukascopy(
    instrument: str,
    timeframe:  str,
    start:      str,
    end:        str,
    delay_s:    float = 0.1,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data from Dukascopy.

    Args:
        instrument: "XAUUSD" | "EURUSD" | "GBPUSD" | "USDJPY" | "BTCUSD"
        timeframe:  "M1" | "M5" | "M15" | "H1" | "H4" | "D1"
        start:      "YYYY-MM-DD"
        end:        "YYYY-MM-DD"
        delay_s:    Polite delay between requests (default 0.1s)

    Returns:
        DataFrame[datetime, open, high, low, close, volume, source] or None.
    """
    cfg = INSTRUMENT_CONFIG.get(instrument)
    if cfg is None:
        log.warning(f"Dukascopy: unsupported instrument '{instrument}'")
        return None
    if timeframe not in RESAMPLE_RULES:
        log.warning(f"Dukascopy: unsupported timeframe '{timeframe}'")
        return None

    start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt   = datetime.strptime(end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)
    point_f  = cfg["point_factor"]

    total_days  = max((end_dt - start_dt).days, 1)
    all_frames  = []
    fetched     = 0
    current     = start_dt

    log.info(f"Dukascopy: fetching {instrument} {timeframe} ({start} → {end})...")

    while current < end_dt:
        if current.weekday() in WEEKEND_DAYS:
            current += timedelta(days=1)
            continue

        raw = _fetch_day_raw(instrument, current)
        if raw:
            df_day = _parse_bi5(raw, current, point_f)
            if not df_day.empty:
                all_frames.append(df_day)
        else:
            log.debug(f"  No data for {instrument} {current.date()}")
            time.sleep(delay_s)

        fetched += 1
        if fetched % 50 == 0:
            pct = int(100 * fetched / total_days)
            log.info(f"  {pct}% ({fetched}/{total_days} trading days)")

        current += timedelta(days=1)

    if not all_frames:
        log.warning(f"Dukascopy: zero M1 bars returned for {instrument}")
        return None

    df = pd.concat(all_frames, ignore_index=True)
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    df = _resample_df(df, timeframe)
    df["source"] = "dukascopy"

    log.info(f"Dukascopy: {len(df):,} {timeframe} bars — {instrument} ({start} → {end})")
    return df


def export_stockraft_csv(
    instrument: str,
    timeframe:  str,
    start:      str,
    end:        str,
    out_dir:    Optional[Path] = None,
) -> Path:
    """
    Fetch data and export as a clean CSV in the format expected by
    the Stockraft master prompts (for uploading to Claude).

    CSV format:
        Date, Time, Open, High, Low, Close, Volume
        2025-10-15, 09:30, 4485.20, 4488.90, 4483.10, 4487.50, 1842
        ...

    Returns the path to the CSV file.
    """
    df = fetch_dukascopy(instrument, timeframe, start, end)
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {instrument} {timeframe}")

    df_out = pd.DataFrame({
        "Date":   df["datetime"].dt.strftime("%Y-%m-%d"),
        "Time":   df["datetime"].dt.strftime("%H:%M"),
        "Open":   df["open"].round(5),
        "High":   df["high"].round(5),
        "Low":    df["low"].round(5),
        "Close":  df["close"].round(5),
        "Volume": df["volume"].astype(int),
    })

    if out_dir is None:
        out_dir = Path(__file__).parent.parent.parent / "data" / "csv_exports"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{instrument}_{timeframe}_{start}_{end}_stockraft.csv"
    df_out.to_csv(out_path, index=False)
    log.info(f"Stockraft CSV: {out_path.name} ({len(df_out):,} rows)")
    return out_path
