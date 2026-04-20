"""
prism/data/tiingo.py
--------------------
Tiingo API price data fetcher for PRISM trading model.

Supports daily OHLCV, intraday OHLCV, and news sentiment data.
Applies instrument mapping for FX/metals (e.g. XAUUSD → GLD).
Caches results as Parquet files to avoid redundant API calls.

Rate limit: 50 requests/min (enforced via time.sleep).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIINGO_BASE_URL = "https://api.tiingo.com"
TIINGO_NEWS_URL = "https://api.tiingo.com/tiingo/news"

# Forex / metal → Tiingo ticker mapping
INSTRUMENT_MAP: dict[str, str] = {
    "XAUUSD": "GLD",   # Gold ETF proxy (Tiingo metals not yet available)
    "EURUSD": "FXE",   # Euro FX ETF
    "GBPUSD": "FXB",   # British Pound ETF
    "USDJPY": "FXY",   # Japanese Yen ETF
}

# Rate limit: 50 req/min → 1.2 s/req minimum
_MIN_INTERVAL_SECS: float = 60.0 / 50

CACHE_DIR = Path("data/raw")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Load Tiingo API key from environment variable TIINGO_API_KEY."""
    key = os.environ.get("TIINGO_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "TIINGO_API_KEY is not set. "
            "Export it before running: export TIINGO_API_KEY=your_key_here"
        )
    return key


def _default_headers(api_key: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Token {api_key}",
    }


def _map_instrument(symbol: str) -> str:
    """Convert a PRISM instrument code to the Tiingo ticker symbol."""
    return INSTRUMENT_MAP.get(symbol.upper(), symbol.upper())


def _cache_path(prefix: str, symbol: str, timeframe: str, start: str, end: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"tiingo_{prefix}_{symbol}_{timeframe}_{start}_{end}.parquet"
    return CACHE_DIR / fname


def _rate_sleep(last_call_time: list[float]) -> None:
    """Enforce minimum interval between API calls (rate limiting)."""
    if last_call_time:
        elapsed = time.monotonic() - last_call_time[0]
        if elapsed < _MIN_INTERVAL_SECS:
            time.sleep(_MIN_INTERVAL_SECS - elapsed)
    last_call_time.clear()
    last_call_time.append(time.monotonic())


def _get(
    url: str,
    params: dict,
    api_key: str,
    last_call: list[float],
) -> dict | list:
    """Perform a rate-limited GET request and return parsed JSON."""
    _rate_sleep(last_call)
    resp = requests.get(
        url,
        params=params,
        headers=_default_headers(api_key),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_ohlcv(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = "daily",
) -> pd.DataFrame:
    """
    Fetch daily (or other non-intraday) OHLCV data from Tiingo.

    Parameters
    ----------
    symbol : str
        PRISM instrument code (e.g. 'EURUSD', 'XAUUSD') or raw Tiingo ticker.
    start_date : str
        Start date in ISO format 'YYYY-MM-DD'.
    end_date : str
        End date in ISO format 'YYYY-MM-DD'.
    timeframe : str
        Tiingo resampleFreq value: 'daily', 'weekly', 'monthly', 'annually'.

    Returns
    -------
    pd.DataFrame
        Columns: datetime (UTC, tz-aware), open, high, low, close, volume.
        Sorted ascending by datetime.
    """
    ticker = _map_instrument(symbol)
    cache = _cache_path("ohlcv", ticker, timeframe, start_date, end_date)

    if cache.exists():
        logger.info("Cache hit: %s", cache)
        return pd.read_parquet(cache)

    api_key = _get_api_key()
    last_call: list[float] = []

    url = f"{TIINGO_BASE_URL}/tiingo/daily/{ticker}/prices"
    params = {
        "startDate": start_date,
        "endDate": end_date,
        "resampleFreq": timeframe,
        "token": api_key,
    }

    logger.info(
        "Fetching Tiingo OHLCV: ticker=%s timeframe=%s %s→%s",
        ticker, timeframe, start_date, end_date,
    )

    try:
        data = _get(url, params, api_key, last_call)
    except requests.HTTPError as exc:
        logger.error("Tiingo OHLCV request failed: %s", exc)
        raise

    if not data:
        logger.warning("Tiingo returned empty response for %s", ticker)
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(data)

    # Prefer adjusted columns; fall back to unadjusted
    col_map: dict[str, str] = {}
    for adj, raw, target in [
        ("adjOpen",   "open",   "open"),
        ("adjHigh",   "high",   "high"),
        ("adjLow",    "low",    "low"),
        ("adjClose",  "close",  "close"),
        ("adjVolume", "volume", "volume"),
    ]:
        if adj in df.columns:
            col_map[adj] = target
        elif raw in df.columns:
            col_map[raw] = target

    if "date" in df.columns:
        col_map["date"] = "datetime"

    df.rename(columns=col_map, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    final_cols = ["datetime", "open", "high", "low", "close", "volume"]
    df = df[[c for c in final_cols if c in df.columns]].copy()
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_parquet(cache, index=False)
    logger.info("Cached OHLCV → %s (%d rows)", cache, len(df))
    return df


def get_intraday(
    symbol: str,
    start_date: str,
    end_date: str,
    freq: str = "1hour",
) -> pd.DataFrame:
    """
    Fetch intraday OHLCV data from Tiingo IEX endpoint.

    Parameters
    ----------
    symbol : str
        PRISM instrument code or Tiingo ticker.
    start_date : str
        Start date in ISO format 'YYYY-MM-DD'.
    end_date : str
        End date in ISO format 'YYYY-MM-DD'.
    freq : str
        Resampling frequency, e.g. '1min', '5min', '1hour'.

    Returns
    -------
    pd.DataFrame
        Columns: datetime (UTC, tz-aware), open, high, low, close, volume.
    """
    ticker = _map_instrument(symbol)
    cache = _cache_path("intraday", ticker, freq, start_date, end_date)

    if cache.exists():
        logger.info("Cache hit: %s", cache)
        return pd.read_parquet(cache)

    api_key = _get_api_key()
    last_call: list[float] = []

    url = f"{TIINGO_BASE_URL}/iex/{ticker}/prices"
    params = {
        "startDate": start_date,
        "endDate": end_date,
        "resampleFreq": freq,
        "columns": "open,high,low,close,volume",
        "token": api_key,
    }

    logger.info(
        "Fetching Tiingo intraday: ticker=%s freq=%s %s→%s",
        ticker, freq, start_date, end_date,
    )

    try:
        data = _get(url, params, api_key, last_call)
    except requests.HTTPError as exc:
        logger.error("Tiingo intraday request failed: %s", exc)
        raise

    if not data:
        logger.warning("Tiingo returned empty intraday response for %s", ticker)
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(data)
    if "date" in df.columns:
        df.rename(columns={"date": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    cols = [c for c in ["datetime", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[cols].copy()
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_parquet(cache, index=False)
    logger.info("Cached intraday → %s (%d rows)", cache, len(df))
    return df


def get_news_sentiment(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch news articles from Tiingo and compute a daily sentiment score.

    Sentiment is a heuristic average over per-article tag-based polarity.
    Replace ``_score_article`` with a transformer model for production.

    Parameters
    ----------
    symbol : str
        PRISM instrument code or Tiingo ticker.
    start_date : str
        Start date 'YYYY-MM-DD'.
    end_date : str
        End date 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        Columns: date (date), sentiment_score (float, range −1 to +1).
    """
    ticker = _map_instrument(symbol)
    cache = _cache_path("news", ticker, "daily", start_date, end_date)

    if cache.exists():
        logger.info("Cache hit: %s", cache)
        return pd.read_parquet(cache)

    api_key = _get_api_key()
    last_call: list[float] = []

    params = {
        "tickers": ticker,
        "startDate": start_date,
        "endDate": end_date,
        "limit": 1000,
        "token": api_key,
    }

    logger.info("Fetching Tiingo news: ticker=%s %s→%s", ticker, start_date, end_date)

    try:
        articles = _get(TIINGO_NEWS_URL, params, api_key, last_call)
    except requests.HTTPError as exc:
        logger.error("Tiingo news request failed: %s", exc)
        raise

    if not articles:
        logger.warning("No news articles returned for %s", ticker)
        return pd.DataFrame(columns=["date", "sentiment_score"])

    rows = []
    for article in articles:
        pub_raw = article.get("publishedDate")
        if not pub_raw:
            continue
        pub = pd.to_datetime(pub_raw, utc=True).date()
        score = _score_article(article)
        rows.append({"date": pub, "sentiment_score": score})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    daily = df.groupby("date")["sentiment_score"].mean().reset_index()
    daily.sort_values("date", inplace=True)
    daily.reset_index(drop=True, inplace=True)

    daily.to_parquet(cache, index=False)
    logger.info("Cached news sentiment → %s (%d days)", cache, len(daily))
    return daily


def _score_article(article: dict) -> float:
    """
    Derive a sentiment score in [−1, +1] from a Tiingo news article dict.

    Uses tag-based heuristic.  Replace with transformer model as needed.
    """
    tags: list[str] = [t.lower() for t in article.get("tags", [])]
    positive_tags = {"bullish", "positive", "upgrade", "beat", "rally", "surge", "gain"}
    negative_tags = {"bearish", "negative", "downgrade", "miss", "crash", "plunge", "selloff", "decline"}

    pos = sum(1 for t in tags if t in positive_tags)
    neg = sum(1 for t in tags if t in negative_tags)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total
