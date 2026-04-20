"""
prism/data/fred.py
FRED (Federal Reserve Economic Data) macro fetcher.
API key: env var FRED_API_KEY (free at fred.stlouisfed.org)
"""
import os
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)
CACHE_DIR = Path("data/raw")

SERIES = {
    "fed_funds_rate": "FEDFUNDS",
    "sofr": "SOFR",
    "cpi": "CPIAUCSL",
    "gdp": "GDP",
    "unemployment_rate": "UNRATE",
    "yield_10y": "DGS10",
    "yield_2y": "DGS2",
    "dxy": "DTWEXBGS",
    "vix": "VIXCLS",
}


class FREDClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            from fredapi import Fred
            self.fred = Fred(api_key=self.api_key) if self.api_key else None
        except ImportError:
            logger.warning("fredapi not installed. Run: pip install fredapi")
            self.fred = None

    def get_series(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch a single FRED series. Returns DataFrame: date, value."""
        cache_file = CACHE_DIR / f"fred_{series_id}_{start_date}_{end_date}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)

        if self.fred is None:
            logger.error("FRED client not initialized (missing key or fredapi package)")
            return pd.DataFrame(columns=["date", "value"])

        try:
            s = self.fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            df = s.reset_index()
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"])
            df.to_parquet(cache_file, index=False)
            return df
        except Exception as e:
            logger.error(f"FRED fetch failed for {series_id}: {e}")
            return pd.DataFrame(columns=["date", "value"])

    def get_macro_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Build combined macro feature DataFrame, daily frequency.
        Columns: fed_funds_rate, fed_funds_delta_wk, sofr, cpi_yoy,
                 gdp_growth, unemployment_rate, yield_10y, yield_2y,
                 yield_spread, dxy, dxy_return_5d, vix
        """
        cache_file = CACHE_DIR / f"fred_macro_{start_date}_{end_date}.parquet"
        if cache_file.exists():
            return pd.read_parquet(cache_file)

        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        df = pd.DataFrame(index=date_range)
        df.index.name = "date"

        # Fetch each series and reindex to daily (forward-fill)
        for col, series_id in SERIES.items():
            s = self.get_series(series_id, start_date, end_date)
            if s.empty:
                df[col] = float("nan")
                continue
            s = s.set_index("date")["value"]
            s = s.reindex(date_range).ffill()
            df[col] = s.values

        # Derived features
        df["fed_funds_delta_wk"] = df["fed_funds_rate"].diff(5)  # 5-day delta
        df["cpi_yoy"] = df["cpi"].pct_change(252) * 100           # ~1 year
        df["gdp_growth"] = df["gdp"].pct_change(63) * 100         # ~1 quarter
        df["yield_spread"] = df["yield_10y"] - df["yield_2y"]
        df["dxy_return_5d"] = df["dxy"].pct_change(5) * 100
        df = df.drop(columns=["cpi", "gdp"], errors="ignore")

        df = df.reset_index()
        df.to_parquet(cache_file, index=False)
        logger.info(f"FRED macro features built: {len(df)} rows, {len(df.columns)} columns")
        return df


def get_macro_features(start_date: str, end_date: str) -> pd.DataFrame:
    """Module-level convenience function."""
    return FREDClient().get_macro_features(start_date, end_date)
