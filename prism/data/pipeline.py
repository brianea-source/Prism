"""
prism/data/pipeline.py
----------------------
PRISM Feature Engineering Pipeline.

Assembles a complete ML feature matrix from price data (Tiingo),
macro data (FRED), and alternative/sentiment data (Quiver/Tiingo news).

Key design principles:
- No future leakage: all transforms are causal (past-only window)
- Chronological splits only — never shuffle time-series data
- StandardScaler fitted on train set only, applied to test set
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session classification helpers (UTC hour → market session)
# ---------------------------------------------------------------------------
#  0 = Tokyo   (00:00–06:00 UTC)
#  1 = London  (07:00–12:00 UTC)
#  2 = New York (13:00–17:00 UTC)
#  3 = Overlap (any other or multi-session)

def _classify_session(hour_utc: int) -> int:
    if 0 <= hour_utc < 7:
        return 0  # Tokyo
    elif 7 <= hour_utc < 12:
        return 1  # London
    elif 12 <= hour_utc < 18:
        return 2  # New York
    else:
        return 3  # Off-hours / overlap


# ---------------------------------------------------------------------------
# FOMC & NFP calendar helpers
# These return hard-coded approximate dates; replace with live FRED calendar
# for production.
# ---------------------------------------------------------------------------

_FOMC_2026 = [
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 5, 6),
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 10, 28), date(2026, 12, 16),
]

_NFP_2026 = [
    date(2026, 1, 9), date(2026, 2, 6), date(2026, 3, 6),
    date(2026, 4, 3), date(2026, 5, 8), date(2026, 6, 5),
    date(2026, 7, 10), date(2026, 8, 7), date(2026, 9, 4),
    date(2026, 10, 2), date(2026, 11, 6), date(2026, 12, 4),
]


def _days_to_next(ref_date: date, event_dates: list[date]) -> int:
    """Return number of calendar days until the next event on or after ref_date."""
    future = [d for d in event_dates if d >= ref_date]
    if not future:
        return 999  # sentinel when no upcoming dates available
    return (min(future) - ref_date).days


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-12)
    d = k.rolling(d_period).mean()
    return k, d


def _macd(close: pd.Series,
          fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    fast_ema = _ema(close, fast)
    slow_ema = _ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bollinger_pct(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.Series:
    """Return %B: position of close within Bollinger Bands (0=lower, 1=upper)."""
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + std_dev * std
    lower = ma - std_dev * std
    return (close - lower) / (upper - lower + 1e-12)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class PRISMFeaturePipeline:
    """
    Feature engineering pipeline for PRISM ML models.

    Parameters
    ----------
    instrument : str
        PRISM instrument code, e.g. 'EURUSD', 'XAUUSD'.
    timeframe : str
        Primary chart timeframe.  Currently 'H1' (hourly) is the default.
        Valid values: 'M15', 'H1', 'H4', 'D1'.
    """

    def __init__(self, instrument: str, timeframe: str = "H1") -> None:
        self.instrument = instrument.upper()
        self.timeframe = timeframe
        self._scaler: Optional[object] = None   # fitted sklearn StandardScaler
        self._feature_cols: list[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Build the full PRISM feature matrix.

        Fetches and merges: price (Tiingo), macro (FRED), sentiment
        (Tiingo news + Quiver COT + Fear & Greed).  Adds calendar features
        and the training target variables.

        Parameters
        ----------
        start_date : str
            'YYYY-MM-DD' — data fetch window start.
        end_date : str
            'YYYY-MM-DD' — data fetch window end.

        Returns
        -------
        pd.DataFrame
            Feature matrix indexed by datetime (hourly, UTC).
            See module docstring for column descriptions.
        """
        from prism.data.tiingo import get_intraday, get_news_sentiment
        from prism.data.fred import get_macro_features
        from prism.data.quiver import get_cot_report, get_fear_greed

        freq_map = {"M15": "15min", "H1": "1hour", "H4": "4hour", "D1": "daily"}
        tiingo_freq = freq_map.get(self.timeframe, "1hour")

        # ---- Price data ------------------------------------------------
        logger.info("[pipeline] Fetching price data: %s %s", self.instrument, tiingo_freq)
        price_df = get_intraday(self.instrument, start_date, end_date, freq=tiingo_freq)
        if price_df.empty:
            raise ValueError(f"No price data returned for {self.instrument}")

        price_df = price_df.set_index("datetime").sort_index()
        df = self._build_price_features(price_df)

        # ---- Macro features (FRED) -------------------------------------
        logger.info("[pipeline] Fetching macro features from FRED")
        try:
            macro_df = get_macro_features(start_date, end_date)
            macro_df["date"] = pd.to_datetime(macro_df["date"])
            macro_df = macro_df.set_index("date")

            macro_cols = [
                "fed_funds_rate", "fed_funds_delta_wk", "yield_spread",
                "cpi_yoy", "dxy_return_5d", "vix",
            ]
            for col in macro_cols:
                if col in macro_df.columns:
                    # Align on date (forward-fill intraday from daily macro)
                    df[col] = df.index.normalize().map(
                        macro_df[col].to_dict()
                    )
                    df[col] = df[col].ffill()
        except Exception as exc:
            logger.warning("[pipeline] FRED macro fetch failed, skipping: %s", exc)
            for col in ["fed_funds_rate", "fed_funds_delta_wk", "yield_spread",
                        "cpi_yoy", "dxy_return_5d", "vix_level"]:
                df[col] = np.nan

        if "vix" in df.columns:
            df.rename(columns={"vix": "vix_level"}, inplace=True)

        # ---- News sentiment (Tiingo) -----------------------------------
        logger.info("[pipeline] Fetching news sentiment")
        try:
            news_df = get_news_sentiment(self.instrument, start_date, end_date)
            news_df["date"] = pd.to_datetime(news_df["date"])
            news_map = news_df.set_index("date")["sentiment_score"].to_dict()
            df["news_sentiment_24h"] = df.index.normalize().map(news_map)
            df["news_sentiment_24h"] = df["news_sentiment_24h"].ffill().fillna(0.0)
        except Exception as exc:
            logger.warning("[pipeline] News sentiment fetch failed: %s", exc)
            df["news_sentiment_24h"] = np.nan

        # ---- COT data (Quiver / CFTC) ---------------------------------
        logger.info("[pipeline] Fetching COT report")
        try:
            cot_df = get_cot_report(self.instrument)
            if not cot_df.empty:
                cot_df["date"] = pd.to_datetime(cot_df["date"])
                cot_map = cot_df.set_index("date")["net_long_speculative"].to_dict()
                df["cot_net_speculative"] = df.index.normalize().map(cot_map)
                df["cot_net_speculative"] = df["cot_net_speculative"].ffill()
            else:
                df["cot_net_speculative"] = np.nan
        except Exception as exc:
            logger.warning("[pipeline] COT fetch failed: %s", exc)
            df["cot_net_speculative"] = np.nan

        # ---- Fear & Greed ---------------------------------------------
        logger.info("[pipeline] Fetching Fear & Greed index")
        try:
            fg_df = get_fear_greed()
            if not fg_df.empty:
                fg_df["date"] = pd.to_datetime(fg_df["date"])
                fg_map = fg_df.set_index("date")["fear_greed_index"].to_dict()
                df["fear_greed"] = df.index.normalize().map(fg_map)
                df["fear_greed"] = df["fear_greed"].ffill()
            else:
                df["fear_greed"] = np.nan
        except Exception as exc:
            logger.warning("[pipeline] Fear & Greed fetch failed: %s", exc)
            df["fear_greed"] = np.nan

        # ---- Calendar features ----------------------------------------
        df["day_of_week"] = df.index.dayofweek
        df["days_to_fomc"] = [
            _days_to_next(ts.date(), _FOMC_2026) for ts in df.index
        ]
        df["days_to_nfp"] = [
            _days_to_next(ts.date(), _NFP_2026) for ts in df.index
        ]
        df["is_high_impact_week"] = (
            (df["days_to_fomc"] <= 5) | (df["days_to_nfp"] <= 5)
        ).astype(int)

        # ---- Target variables -----------------------------------------
        df = self._build_targets(df, price_df)

        # Drop warmup rows (rolling indicators need warmup)
        df = df.iloc[200:].copy()
        df.reset_index(inplace=True)

        self._feature_cols = [
            c for c in df.columns
            if c not in ["datetime", "direction_4h", "magnitude_pips"]
        ]

        return df

    def normalize(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Apply StandardScaler to all numeric feature columns.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix as returned by ``build_features``.
        fit : bool
            If True, fit the scaler on this data (use for training set).
            If False, apply a previously fitted scaler (use for test/live).

        Returns
        -------
        pd.DataFrame
            Copy of df with numeric feature columns standardised.
        """
        from sklearn.preprocessing import StandardScaler  # type: ignore

        non_feature = {"datetime", "direction_4h", "magnitude_pips"}
        feat_cols = [c for c in self._feature_cols if c in df.columns and c not in non_feature]

        result = df.copy()
        if fit or self._scaler is None:
            self._scaler = StandardScaler()
            result[feat_cols] = self._scaler.fit_transform(result[feat_cols].astype(float))
        else:
            result[feat_cols] = self._scaler.transform(result[feat_cols].astype(float))

        return result

    def split_train_test(
        self,
        df: pd.DataFrame,
        test_ratio: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chronological train / test split.

        NEVER shuffles — maintains temporal order to prevent leakage.

        Parameters
        ----------
        df : pd.DataFrame
            Full feature matrix.
        test_ratio : float
            Fraction of rows reserved for the test set (tail of the series).

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (train_df, test_df) — both retain original column structure.
        """
        n = len(df)
        split_idx = int(n * (1 - test_ratio))
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        logger.info(
            "Train/test split: %d / %d rows (%.0f%% / %.0f%%)",
            len(train), len(test), 100 * (1 - test_ratio), 100 * test_ratio,
        )
        return train, test

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_price_features(self, price: pd.DataFrame) -> pd.DataFrame:
        """Compute all price-derived technical features."""
        df = pd.DataFrame(index=price.index)
        c = price["close"]
        h = price["high"]
        l = price["low"]
        v = price.get("volume", pd.Series(0, index=price.index))

        # Log returns
        log_c = np.log(c)
        df["return_1"]  = log_c.diff(1)
        df["return_5"]  = log_c.diff(5)
        df["return_20"] = log_c.diff(20)
        df["return_50"] = log_c.diff(50)

        # Volatility
        df["atr_14"] = _atr(h, l, c, 14)
        df["atr_50"] = _atr(h, l, c, 50)
        df["hv_20"]  = log_c.diff().rolling(20).std() * np.sqrt(252 * 24)  # annualised HV

        # Trend — EMAs
        df["ema_9"]   = _ema(c, 9)
        df["ema_21"]  = _ema(c, 21)
        df["ema_50"]  = _ema(c, 50)
        df["ema_200"] = _ema(c, 200)

        # Momentum — EMA slopes (1-period change)
        df["ema_9_slope"]  = df["ema_9"].diff()
        df["ema_21_slope"] = df["ema_21"].diff()

        # RSI
        df["rsi_14"] = _rsi(c, 14)

        # MACD
        df["macd"], df["macd_signal"], df["macd_hist"] = _macd(c)

        # Stochastic
        df["stoch_k"], df["stoch_d"] = _stochastic(h, l, c)

        # OBV change (1-period diff to avoid runaway scale)
        df["obv_change"] = _obv(c, v).diff()

        # Bollinger Band %B
        df["bb_pct"] = _bollinger_pct(c)

        # Daily range position: where does current close sit in today's range?
        daily_high = h.resample("D").transform("max")
        daily_low  = l.resample("D").transform("min")
        df["daily_range_position"] = (c - daily_low) / (daily_high - daily_low + 1e-12)

        # Session
        df["session"] = price.index.hour.map(_classify_session)

        return df

    def _build_targets(
        self,
        df: pd.DataFrame,
        price: pd.DataFrame,
        flat_threshold_pct: float = 0.1,
        horizon_bars: int = 4,
        mfe_bars: int = 20,
    ) -> pd.DataFrame:
        """
        Compute training target variables.

        direction_4h : int
            1 = price rose > flat_threshold_pct% over next horizon_bars bars.
           -1 = price fell > flat_threshold_pct% over next horizon_bars bars.
            0 = flat (within ±threshold).

        magnitude_pips : float
            Max Favorable Excursion (MFE) over the next mfe_bars bars,
            expressed in pips (×10000 for FX pairs, ×100 for JPY, ×10 for Gold).
        """
        pip_multiplier = self._pip_multiplier()
        close = price["close"]
        high  = price["high"]

        future_close = close.shift(-horizon_bars)
        pct_change = (future_close - close) / close * 100

        direction = np.where(
            pct_change > flat_threshold_pct, 1,
            np.where(pct_change < -flat_threshold_pct, -1, 0),
        )
        df["direction_4h"] = direction

        # MFE: max high over next mfe_bars bars minus entry close (for longs)
        mfe = pd.Series(index=close.index, dtype=float)
        for i in range(len(close)):
            window = high.iloc[i + 1: i + 1 + mfe_bars]
            if len(window) == 0:
                mfe.iloc[i] = np.nan
            else:
                mfe.iloc[i] = (window.max() - close.iloc[i]) * pip_multiplier

        df["magnitude_pips"] = mfe.values
        return df

    def _pip_multiplier(self) -> float:
        """Return pip multiplier for the current instrument."""
        if self.instrument in ("USDJPY", "EURJPY", "GBPJPY"):
            return 100.0
        if self.instrument in ("XAUUSD",):
            return 10.0
        return 10_000.0  # standard FX pair
