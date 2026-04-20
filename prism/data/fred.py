"""
prism/data/fred.py
------------------
FRED (Federal Reserve Economic Data) macro data fetcher for PRISM.

Fetches key macroeconomic series via the fredapi library and assembles
a daily feature DataFrame suitable for ML model training and inference.

All monthly/quarterly series are forward-filled to daily frequency.
Results are cached as Parquet to minimise API round-trips.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/raw")

# Series definitions: (series_id, column_name, transform)
# transform: None | 'yoy_pct' | 'qoq_pct' | 'diff'
MACRO_SERIES: list[tuple[str, str, Optional[str]]] = [
    ("FEDFUNDS",  "fed_funds_rate",    None),
    ("SOFR",      "sofr",              None),
    ("CPIAUCSL",  "cpi_yoy",           "yoy_pct"),
    ("GDP",       "gdp_growth",        "qoq_pct"),
    ("UNRATE",    "unemployment_rate", None),
    ("DGS10",     "yield_10y",         None),
    ("DGS2",      "yield_2y",          None),
    ("DTWEXBGS",  "dxy",               None),
    ("VIXCLS",    "vix",               None),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Load FRED API key from environment variable FRED_API_KEY."""
    key = os.environ.get("FRED_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "FRED_API_KEY is not set. "
            "Export it before running: export FRED_API_KEY=your_key_here"
        )
    return key


def _cache_path(name: str, start: str, end: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"fred_{name}_{start}_{end}.parquet"


def _apply_transform(series: pd.Series, transform: Optional[str]) -> pd.Series:
    """Apply the requested statistical transform to a raw FRED series."""
    if transform is None:
        return series
    if transform == "yoy_pct":
        # Year-over-year percent change (monthly data → 12-period lag)
        return series.pct_change(12) * 100
    if transform == "qoq_pct":
        # Quarter-over-quarter percent change (quarterly data → 1-period lag)
        return series.pct_change(1) * 100
    if transform == "diff":
        return series.diff()
    raise ValueError(f"Unknown transform: {transform}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_series(
    series_id: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch a single FRED series.

    Parameters
    ----------
    series_id : str
        FRED series identifier, e.g. 'FEDFUNDS', 'DGS10'.
    start_date : str
        Start date 'YYYY-MM-DD'.
    end_date : str
        End date 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        Columns: date (DatetimeTZNaive date), value (float).
        Sorted ascending by date.
    """
    try:
        from fredapi import Fred  # type: ignore
    except ImportError as exc:
        raise ImportError("fredapi not installed. Run: pip install fredapi") from exc

    cache = _cache_path(series_id.lower(), start_date, end_date)
    if cache.exists():
        logger.info("Cache hit: %s", cache)
        return pd.read_parquet(cache)

    api_key = _get_api_key()
    fred = Fred(api_key=api_key)

    logger.info("Fetching FRED series %s from %s to %s", series_id, start_date, end_date)
    try:
        raw = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    except Exception as exc:
        logger.error("FRED request failed for %s: %s", series_id, exc)
        raise

    df = raw.rename("value").reset_index()
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_parquet(cache, index=False)
    logger.info("Cached FRED %s → %s (%d rows)", series_id, cache, len(df))
    return df


def get_macro_features(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Download all PRISM macro series and assemble a daily feature DataFrame.

    Monthly/quarterly series are resampled to daily frequency using forward-fill.
    A ``yield_spread`` column (10Y − 2Y) and ``fed_funds_delta_wk`` column
    (weekly change in fed funds rate) are computed and appended.

    Parameters
    ----------
    start_date : str
        Start date 'YYYY-MM-DD'.
    end_date : str
        End date 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        Index: date (daily). Columns as listed in MACRO_SERIES plus derived
        columns ``yield_spread`` and ``fed_funds_delta_wk``.
        Values are forward-filled; no back-fill is applied to prevent leakage.
    """
    cache = _cache_path("macro", start_date, end_date)
    if cache.exists():
        logger.info("Cache hit: %s", cache)
        return pd.read_parquet(cache)

    # Build a daily date index spanning the full requested range
    date_index = pd.date_range(start=start_date, end=end_date, freq="D")
    result = pd.DataFrame(index=date_index)
    result.index.name = "date"

    for series_id, col_name, transform in MACRO_SERIES:
        logger.info("Loading %s → %s", series_id, col_name)
        try:
            df = get_series(series_id, start_date, end_date)
        except Exception as exc:
            logger.warning("Skipping %s due to error: %s", series_id, exc)
            result[col_name] = float("nan")
            continue

        s = df.set_index("date")["value"]
        s.index = pd.to_datetime(s.index)
        s = _apply_transform(s, transform)

        # Reindex to daily and forward-fill (no back-fill — prevents look-ahead)
        s = s.reindex(date_index).ffill()
        result[col_name] = s

    # Derived columns
    if "yield_10y" in result.columns and "yield_2y" in result.columns:
        result["yield_spread"] = result["yield_10y"] - result["yield_2y"]

    if "fed_funds_rate" in result.columns:
        result["fed_funds_delta_wk"] = result["fed_funds_rate"].diff(7)

    if "dxy" in result.columns:
        result["dxy_return_5d"] = result["dxy"].pct_change(5) * 100

    result.reset_index(inplace=True)
    result.to_parquet(cache, index=False)
    logger.info("Cached macro features → %s (%d rows, %d cols)", cache, len(result), result.shape[1])
    return result
