"""
prism/data/quiver.py
--------------------
Quiver Quantitative alternative data fetcher for PRISM.

Provides COT (Commitment of Traders) reports, Fear & Greed Index,
and Reddit sentiment data.  Falls back to public CFTC data when the
Quiver API key is not available.
"""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/raw")

QUIVER_BASE_URL = "https://api.quiverquant.com/beta"

# PRISM instrument → CFTC/Quiver contract name mapping
COT_SYMBOL_MAP: dict[str, str] = {
    "XAUUSD": "Gold",
    "EURUSD": "Euro FX",
    "GBPUSD": "British Pound",
    "USDJPY": "Japanese Yen",
    "NAS100": "Nasdaq Mini",
    "SPX500": "S&P 500 Mini",
}

# Public CFTC Financial Traders report (no API key required)
CFTC_COT_URL = "https://www.cftc.gov/dea/options/financial_lof.htm"

# CNN Fear & Greed proxy via alternative.me (public)
FEAR_GREED_URL = "https://api.alternative.me/fng/"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> Optional[str]:
    """Load Quiver API key; return None if not set (triggers fallback paths)."""
    return os.environ.get("QUIVER_API_KEY") or None


def _cache_path(name: str, symbol: str = "", extra: str = "") -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parts = ["quiver", name]
    if symbol:
        parts.append(symbol.replace(" ", "_"))
    if extra:
        parts.append(extra)
    return CACHE_DIR / ("_".join(parts) + ".parquet")


def _quiver_get(endpoint: str, api_key: str, params: Optional[dict] = None) -> list | dict:
    """GET from Quiver API with basic error handling."""
    headers = {
        "Accept": "application/json",
        "X-CSRFToken": api_key,
        "Cookie": f"csrftoken={api_key}",
    }
    url = f"{QUIVER_BASE_URL}/{endpoint}"
    resp = requests.get(url, headers=headers, params=params or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# COT Report
# ---------------------------------------------------------------------------

def get_cot_report(symbol: str) -> pd.DataFrame:
    """
    Fetch Commitment of Traders (COT) report for the given instrument.

    Tries Quiver Quantitative first; falls back to CFTC public CSV.

    Parameters
    ----------
    symbol : str
        PRISM instrument code, e.g. 'EURUSD', 'XAUUSD'.

    Returns
    -------
    pd.DataFrame
        Columns: date, net_long_speculative, net_short_commercial.
        Weekly frequency (CFTC publishes every Friday).
    """
    cot_name = COT_SYMBOL_MAP.get(symbol.upper(), symbol)
    cache = _cache_path("cot", cot_name)

    if cache.exists():
        logger.info("Cache hit: %s", cache)
        return pd.read_parquet(cache)

    api_key = _get_api_key()
    df: Optional[pd.DataFrame] = None

    if api_key:
        df = _fetch_cot_quiver(cot_name, api_key)

    if df is None or df.empty:
        logger.info("Falling back to CFTC public COT data for %s", cot_name)
        df = _fetch_cot_cftc(cot_name)

    if df is None or df.empty:
        logger.warning("No COT data available for %s", symbol)
        return pd.DataFrame(columns=["date", "net_long_speculative", "net_short_commercial"])

    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_parquet(cache, index=False)
    logger.info("Cached COT → %s (%d rows)", cache, len(df))
    return df


def _fetch_cot_quiver(cot_name: str, api_key: str) -> Optional[pd.DataFrame]:
    """Fetch COT data from Quiver Quantitative API."""
    try:
        data = _quiver_get("live/cot", api_key, params={"ticker": cot_name})
        if not data:
            return None
        df = pd.DataFrame(data)
        # Quiver field mapping (adjust if schema changes)
        col_map = {
            "Date": "date",
            "NoncommercialLong": "net_long_speculative",
            "CommercialShort": "net_short_commercial",
        }
        df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "net_long_speculative", "net_short_commercial"]]
    except Exception as exc:
        logger.warning("Quiver COT fetch failed: %s", exc)
        return None


def _fetch_cot_cftc(cot_name: str) -> Optional[pd.DataFrame]:
    """
    Fetch COT data from CFTC public HTML table.

    The CFTC Disaggregated Futures-and-Options page provides weekly reports.
    We parse the table and filter by market name.
    """
    try:
        resp = requests.get(CFTC_COT_URL, timeout=60)
        resp.raise_for_status()

        tables = pd.read_html(io.StringIO(resp.text))
        if not tables:
            logger.warning("No tables found on CFTC COT page")
            return None

        df = tables[0]

        # Typical CFTC columns vary; attempt common names
        name_col = next((c for c in df.columns if "market" in str(c).lower() or "name" in str(c).lower()), None)
        date_col = next((c for c in df.columns if "date" in str(c).lower()), None)
        long_col = next((c for c in df.columns if "noncomm" in str(c).lower() and "long" in str(c).lower()), None)
        short_col = next((c for c in df.columns if "comm" in str(c).lower() and "short" in str(c).lower()), None)

        if not all([name_col, date_col, long_col, short_col]):
            logger.warning("Could not identify expected CFTC COT columns; skipping fallback.")
            return None

        mask = df[name_col].astype(str).str.lower().str.contains(cot_name.lower(), na=False)
        filtered = df[mask].copy()
        if filtered.empty:
            logger.warning("No CFTC rows matching '%s'", cot_name)
            return None

        filtered = filtered[[date_col, long_col, short_col]].copy()
        filtered.columns = ["date", "net_long_speculative", "net_short_commercial"]
        filtered["date"] = pd.to_datetime(filtered["date"])
        filtered["net_long_speculative"] = pd.to_numeric(filtered["net_long_speculative"], errors="coerce")
        filtered["net_short_commercial"] = pd.to_numeric(filtered["net_short_commercial"], errors="coerce")
        return filtered

    except Exception as exc:
        logger.warning("CFTC COT fetch/parse failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Fear & Greed Index
# ---------------------------------------------------------------------------

def get_fear_greed(limit: int = 365) -> pd.DataFrame:
    """
    Fetch the Fear & Greed Index.

    Uses alternative.me public API (CNN F&G proxy) regardless of Quiver key.
    Falls back to Quiver if alternative.me is unreachable.

    Parameters
    ----------
    limit : int
        Number of historical data points to fetch (max ~900 from alternative.me).

    Returns
    -------
    pd.DataFrame
        Columns: date, fear_greed_index (int, 0–100).
    """
    cache = _cache_path("fear_greed", extra=str(limit))
    if cache.exists():
        logger.info("Cache hit: %s", cache)
        return pd.read_parquet(cache)

    df: Optional[pd.DataFrame] = None

    # Try alternative.me first (public, reliable)
    try:
        resp = requests.get(FEAR_GREED_URL, params={"limit": limit, "format": "json"}, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", [])
        if data:
            rows = [
                {
                    "date": pd.to_datetime(int(d["timestamp"]), unit="s").date(),
                    "fear_greed_index": int(d["value"]),
                }
                for d in data
            ]
            df = pd.DataFrame(rows)
            logger.info("Fetched %d Fear & Greed rows from alternative.me", len(df))
    except Exception as exc:
        logger.warning("alternative.me Fear & Greed fetch failed: %s", exc)

    if df is None or df.empty:
        api_key = _get_api_key()
        if api_key:
            df = _fetch_fear_greed_quiver(api_key)

    if df is None or df.empty:
        logger.warning("No Fear & Greed data available")
        return pd.DataFrame(columns=["date", "fear_greed_index"])

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_parquet(cache, index=False)
    logger.info("Cached Fear & Greed → %s (%d rows)", cache, len(df))
    return df


def _fetch_fear_greed_quiver(api_key: str) -> Optional[pd.DataFrame]:
    """Fetch Fear & Greed from Quiver API as fallback."""
    try:
        data = _quiver_get("live/feargreed", api_key)
        if not data:
            return None
        df = pd.DataFrame(data)
        df.rename(columns={"Date": "date", "FearGreed": "fear_greed_index"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "fear_greed_index"]]
    except Exception as exc:
        logger.warning("Quiver Fear & Greed fetch failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Reddit Sentiment
# ---------------------------------------------------------------------------

def get_reddit_sentiment(symbol: str) -> pd.DataFrame:
    """
    Fetch daily Reddit mention/sentiment scores from Quiver Quantitative.

    Requires QUIVER_API_KEY.  Returns an empty DataFrame if key is absent.

    Parameters
    ----------
    symbol : str
        PRISM instrument code or ticker (e.g. 'EURUSD', 'GLD').

    Returns
    -------
    pd.DataFrame
        Columns: date, reddit_score (float, higher = more bullish mentions).
    """
    cache = _cache_path("reddit", symbol)
    if cache.exists():
        logger.info("Cache hit: %s", cache)
        return pd.read_parquet(cache)

    api_key = _get_api_key()
    if not api_key:
        logger.warning("QUIVER_API_KEY not set; skipping Reddit sentiment for %s", symbol)
        return pd.DataFrame(columns=["date", "reddit_score"])

    ticker = COT_SYMBOL_MAP.get(symbol.upper(), symbol.upper())

    try:
        data = _quiver_get(f"historical/reddit/{ticker}", api_key)
        if not data:
            return pd.DataFrame(columns=["date", "reddit_score"])

        df = pd.DataFrame(data)
        df.rename(columns={"Date": "date", "Score": "reddit_score"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df["reddit_score"] = pd.to_numeric(df.get("reddit_score", pd.Series(dtype=float)), errors="coerce")
        df = df[["date", "reddit_score"]].copy()
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        df.to_parquet(cache, index=False)
        logger.info("Cached Reddit sentiment → %s (%d rows)", cache, len(df))
        return df

    except Exception as exc:
        logger.warning("Quiver Reddit sentiment fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame(columns=["date", "reddit_score"])
