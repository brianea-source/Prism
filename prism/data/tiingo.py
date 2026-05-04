"""
prism/data/tiingo.py
Tiingo API price data fetcher.
API key: env var TIINGO_API_KEY
"""
from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from datetime import datetime
import requests
import pandas as pd

logger = logging.getLogger(__name__)

# Instrument mapping: MT5 symbol -> Tiingo ticker
INSTRUMENT_MAP = {
    # NOTE: XAUUSD mapped to GLD (SPDR Gold ETF) as Tiingo proxy for spot gold.
    # GLD tracks spot XAU/USD closely but has minor ETF-specific dynamics.
    # Switch to "XAUUSD=X" via yfinance for spot-accurate data (see get_ohlcv fallback).
    "XAUUSD": "GLD",
    "EURUSD": "EURUSD", # Tiingo forex
    "GBPUSD": "GBPUSD",
    "USDJPY": "USDJPY",
}


# yfinance ticker overrides — used when Tiingo is unavailable.
# These differ from INSTRUMENT_MAP because yfinance uses its own symbol conventions.
YF_MAP = {
    "XAUUSD": "GC=F",    # Gold futures — spot-accurate proxy on yfinance (GLD=X does not exist)
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
}
CACHE_DIR = Path("data/raw")

class TiingoClient:
    BASE_URL = "https://api.tiingo.com"
    RATE_LIMIT_DELAY = 1.2  # seconds between requests (50/min max)

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("TIINGO_API_KEY", "")
        if not self.api_key:
            logger.warning("TIINGO_API_KEY not set. Data fetches will fail.")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        })
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _get(self, endpoint: str, params: dict) -> list | dict:
        url = f"{self.BASE_URL}/{endpoint}"
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        time.sleep(self.RATE_LIMIT_DELAY)
        return resp.json()

    def get_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "daily",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars from Tiingo.
        symbol: MT5 symbol (e.g. 'XAUUSD') or Tiingo ticker
        timeframe: 'daily', '1hour', '15min', '5min', '1min'
        Returns DataFrame with columns: datetime, open, high, low, close, volume
        """
        ticker = INSTRUMENT_MAP.get(symbol, symbol.lower())
        cache_file = CACHE_DIR / f"tiingo_{ticker}_{timeframe}_{start_date}_{end_date}.parquet"

        if cache_file.exists():
            logger.info(f"Loading cached data: {cache_file}")
            cached = pd.read_parquet(cache_file)
            # Guard against legacy/poisoned caches that were written before
            # the column-projection fix landed (datetime-only frames).
            required = {"datetime", "open", "high", "low", "close"}
            if required.issubset(cached.columns):
                return cached
            logger.warning(
                "Cached parquet %s is missing %s — re-fetching from API",
                cache_file, sorted(required - set(cached.columns)),
            )
            try:
                cache_file.unlink()
            except OSError:
                pass

        try:
            if timeframe == "daily":
                data = self._get(
                    f"tiingo/daily/{ticker}/prices",
                    {"startDate": start_date, "endDate": end_date, "format": "json"},
                )
            else:
                # NOTE: do NOT pass a ``columns`` query param to the IEX
                # endpoint. When that param is set, Tiingo returns only the
                # raw (non-``adj*``) column variants and the rename below
                # silently drops every OHLCV field, leaving a ``datetime``-
                # only DataFrame that gets cached and then blows up in
                # _engineer_features with KeyError: \'close\'. Let Tiingo
                # return the full row; we project to OHLCV ourselves.
                freq_map = {"1hour": "1Hour", "4hour": "4Hour", "15min": "15Min", "5min": "5Min", "1min": "1Min"}
                data = self._get(
                    f"iex/{ticker}/prices",
                    {
                        "startDate": start_date,
                        "endDate": end_date,
                        "resampleFreq": freq_map.get(timeframe, "1Hour"),
                    },
                )

            df = pd.DataFrame(data)
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return df

            # Tiingo daily/prices returns ``adj*`` variants; iex/prices
            # returns plain ``open``/``high``/``low``/``close``/``volume``.
            # Accept either flavour, preferring ``adj*`` when both are
            # present so split/dividend adjustments win on equity tickers.
            rename: dict[str, str] = {"date": "datetime"}
            for adj_src, plain_src, dst in [
                ("adjOpen",   "open",   "open"),
                ("adjHigh",   "high",   "high"),
                ("adjLow",    "low",    "low"),
                ("adjClose",  "close",  "close"),
                ("adjVolume", "volume", "volume"),
            ]:
                if adj_src in df.columns:
                    rename[adj_src] = dst
                elif plain_src in df.columns:
                    rename[plain_src] = dst
            keep = [c for c in rename if c in df.columns]
            df = df[keep].rename(columns=rename)
            # Drop any duplicate-target columns the rename collapsed.
            df = df.loc[:, ~df.columns.duplicated()]

            required = {"datetime", "open", "high", "low", "close"}
            missing = required - set(df.columns)
            if missing:
                # Don\'t cache a half-baked frame — that\'s exactly how the
                # KeyError: \'close\' regression survived for months.
                raise RuntimeError(
                    f"Tiingo response for {ticker} missing required columns "
                    f"{sorted(missing)}; got {list(df.columns)}"
                )

            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df = df.sort_values("datetime").reset_index(drop=True)
            df.to_parquet(cache_file, index=False)
            logger.info(f"Fetched {len(df)} bars for {ticker} ({timeframe})")
            return df

        except Exception as e:
            logger.error(f"Tiingo fetch failed for {ticker}: {e}")
            raise

    def get_news_sentiment(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch news sentiment scores from Tiingo.
        Returns DataFrame: date, sentiment_score (rolling 24h average)
        """
        ticker = INSTRUMENT_MAP.get(symbol, symbol.lower())
        cache_file = CACHE_DIR / f"tiingo_sentiment_{ticker}_{start_date}_{end_date}.parquet"

        if cache_file.exists():
            return pd.read_parquet(cache_file)

        try:
            data = self._get(
                "tiingo/news",
                {"tickers": ticker, "startDate": start_date, "endDate": end_date, "limit": 1000},
            )
            if not data:
                return pd.DataFrame(columns=["date", "sentiment_score"])

            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["publishedDate"]).dt.date
            # Average sentiment per day (Tiingo returns positive/negative scores)
            if "sentiment" in df.columns:
                df["sentiment_score"] = df["sentiment"].apply(
                    lambda x: x.get("compound", 0) if isinstance(x, dict) else 0
                )
            else:
                df["sentiment_score"] = 0.0

            daily = df.groupby("date")["sentiment_score"].mean().reset_index()
            daily.to_parquet(cache_file, index=False)
            return daily

        except Exception as e:
            logger.warning(f"Sentiment fetch failed for {ticker}: {e}. Returning empty.")
            return pd.DataFrame(columns=["date", "sentiment_score"])


def get_ohlcv(symbol: str, start_date: str, end_date: str, timeframe: str = "daily") -> pd.DataFrame:
    """Module-level convenience function."""
    return TiingoClient().get_ohlcv(symbol, start_date, end_date, timeframe)


def get_news_sentiment(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Module-level convenience function."""
    return TiingoClient().get_news_sentiment(symbol, start_date, end_date)
