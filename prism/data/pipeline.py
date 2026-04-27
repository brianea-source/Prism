"""
prism/data/pipeline.py
PRISM Feature Engineering Pipeline.
Builds the full feature matrix for ML training and inference.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast = _ema(series, 12)
    slow = _ema(series, 26)
    macd = fast - slow
    signal = _ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist

def _bollinger_pct(series: pd.Series, period: int = 20) -> pd.Series:
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (series - lower) / (upper - lower + 1e-10)

def _session(dt_index: pd.DatetimeIndex) -> pd.Series:
    """Encode trading session: 0=off, 1=tokyo, 2=london, 3=ny, 4=overlap"""
    hour = dt_index.hour
    session = pd.Series(0, index=dt_index)
    session[(hour >= 0) & (hour < 8)] = 1    # Tokyo
    session[(hour >= 8) & (hour < 12)] = 4   # London/Tokyo overlap
    session[(hour >= 12) & (hour < 16)] = 3  # NY
    session[(hour >= 16) & (hour < 20)] = 2  # London
    return session


class PRISMFeaturePipeline:
    """
    Builds the complete feature matrix for a given instrument.

    Usage:
        pipeline = PRISMFeaturePipeline("EURUSD", "H1")
        df = pipeline.build_features("2022-01-01", "2025-12-31")
        X_train, X_test, y_train, y_test = pipeline.train_test_split(df)
    """

    def __init__(
        self,
        instrument: str,
        timeframe: str = "H1",
        *,
        phase7a_sidecar_path: "Path | str | None" = None,
        phase7a_ob_max_distance_pips: float | None = None,
    ):
        self.instrument = instrument
        self.timeframe = timeframe
        self._scaler = None
        self._feature_cols: list[str] = []
        # Pip size for SL/TP calculations
        self.pip_size = 0.01 if any(x in instrument for x in ["XAU", "JPY"]) else 0.0001

        # Phase 7.A: optional historical state sidecar (parquet) wired in via
        # ``prism.data.historical_state.build_replay_sidecar``. When set, the
        # training feature matrix gains the five Phase 7.A ICT columns merged
        # by ``datetime``. Left as None on the live path — predict.py runs
        # the engineer directly off ``signal.smart_money`` instead.
        self.phase7a_sidecar_path: "Path | None" = (
            Path(phase7a_sidecar_path) if phase7a_sidecar_path is not None else None
        )
        # ``ob_max_distance_pips`` is the lock-in surface — see
        # docs/PHASE_7A_SCOPE.md §2.4 / §8.1. ``None`` means "skip the
        # Phase 7.A enrichment" (legacy callers that don't know about
        # the sidecar). Explicit value gets passed through to
        # ``ICTFeatureEngineer`` and written into the model artifact
        # sidecar by retrain.py.
        self.phase7a_ob_max_distance_pips: float | None = (
            float(phase7a_ob_max_distance_pips)
            if phase7a_ob_max_distance_pips is not None
            else None
        )

    def build_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Build the full feature matrix for training/backtest.
        Returns DataFrame with all features + target columns.
        """
        df = self._load_price_data(start_date, end_date)
        if df.empty:
            raise ValueError(f"No price data for {self.instrument} ({start_date} to {end_date})")
        return self._engineer_features(
            df,
            macro_start=start_date,
            macro_end=end_date,
            include_targets=True,
        )

    def build_features_from_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the technical + macro + alt feature pipeline on an already-loaded
        OHLCV DataFrame. This is the live path: the runner pulls real bars
        from MT5 via ``MT5Bridge.get_bars`` and hands them straight in.

        Targets (direction_fwd_4, magnitude_pips) are intentionally omitted —
        they are training-only labels derived by looking forward, which has
        no meaning for a live latest-bar scan.

        Macro/COT/fear-greed features are refreshed inline using a sensible
        [first-bar, last-bar + 1 day] window so trained models still see a
        populated macro column.
        """
        if df.empty:
            raise ValueError(
                f"build_features_from_bars({self.instrument}): received empty DataFrame"
            )
        if "datetime" not in df.columns:
            raise ValueError(
                "build_features_from_bars requires a 'datetime' column; "
                f"got {list(df.columns)}"
            )
        # Derive a matching macro window from the bars we were handed.
        dt_series = pd.to_datetime(df["datetime"])
        macro_start = dt_series.min().strftime("%Y-%m-%d")
        macro_end = (dt_series.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return self._engineer_features(
            df.copy(),
            macro_start=macro_start,
            macro_end=macro_end,
            include_targets=False,
        )

    # ---------------------------------------------------------------
    # Shared feature-engineering core — used by BOTH the training
    # path (build_features) and the live path (build_features_from_bars).
    # The ONLY differences are (1) data loading happens upstream and
    # (2) targets are skipped for live scans since they require
    # forward-looking data.
    # ---------------------------------------------------------------
    def _engineer_features(
        self,
        df: pd.DataFrame,
        macro_start: str,
        macro_end: str,
        include_targets: bool,
    ) -> pd.DataFrame:
        df = df.sort_values("datetime").reset_index(drop=True)
        close = df["close"]

        # --- Price features ---
        for n in [1, 5, 20, 50]:
            df[f"return_{n}"] = np.log(close / close.shift(n))

        df["atr_14"] = _atr(df, 14)
        df["atr_50"] = _atr(df, 50)
        df["hv_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)

        for p in [9, 21, 50, 200]:
            df[f"ema_{p}"] = _ema(close, p)
        df["ema_9_slope"] = df["ema_9"].diff(3)
        df["ema_21_slope"] = df["ema_21"].diff(3)
        df["ema_trend"] = np.sign(df["ema_9"] - df["ema_21"])

        df["rsi_14"] = _rsi(close)
        df["macd"], df["macd_signal"], df["macd_hist"] = _macd(close)

        low_14 = df["low"].rolling(14).min()
        high_14 = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        df["bb_pct"] = _bollinger_pct(close)
        df["daily_range_position"] = (close - df["low"]) / (df["high"] - df["low"] + 1e-10)

        if "volume" in df.columns:
            df["obv"] = (np.sign(close.diff()) * df["volume"]).cumsum()
            df["obv_change"] = df["obv"].pct_change(5)
        else:
            df["obv_change"] = 0.0

        if pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["session"] = _session(pd.DatetimeIndex(df["datetime"])).values
        df["day_of_week"] = pd.to_datetime(df["datetime"]).dt.dayofweek

        # --- Macro features (FRED) ---
        try:
            from prism.data.fred import get_macro_features
            macro = get_macro_features(macro_start, macro_end)
            macro["date"] = pd.to_datetime(macro["date"])
            df["date"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None).dt.normalize()
            df = df.merge(macro, on="date", how="left")
            df = df.drop(columns=["date"], errors="ignore")
            for col in macro.columns:
                if col != "date" and col in df.columns:
                    df[col] = df[col].ffill()
        except Exception as e:
            logger.warning(f"FRED macro features unavailable: {e}")
            for col in ["fed_funds_rate", "fed_funds_delta_wk", "yield_spread",
                        "cpi_yoy", "dxy_return_5d", "vix"]:
                df[col] = np.nan

        # --- COT / sentiment features ---
        try:
            from prism.data.quiver import get_cot_report, get_fear_greed
            cot = get_cot_report(self.instrument)
            if not cot.empty:
                cot["date"] = pd.to_datetime(cot["date"])
                df["date"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None).dt.normalize()
                df = df.merge(cot[["date", "net_speculative"]], on="date", how="left")
                df["cot_net_speculative"] = df["net_speculative"].ffill()
                df = df.drop(columns=["date", "net_speculative"], errors="ignore")
            fg = get_fear_greed()
            if not fg.empty:
                fg["date"] = pd.to_datetime(fg["date"])
                df["date"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None).dt.normalize()
                df = df.merge(fg, on="date", how="left")
                df["fear_greed"] = df["fear_greed"].ffill()
                df = df.drop(columns=["date"], errors="ignore")
        except Exception as e:
            logger.warning(f"Alt data unavailable: {e}")
            df["cot_net_speculative"] = np.nan
            df["fear_greed"] = np.nan

        # --- Target variables (training path only) ---
        if include_targets:
            df["direction_fwd_4"] = np.sign(close.shift(-4) - close).astype(int)
            df["magnitude_pips"] = 0.0
            for i in range(len(df) - 20):
                future = df.iloc[i + 1:i + 21]
                if df["direction_fwd_4"].iloc[i] >= 0:
                    df.at[df.index[i], "magnitude_pips"] = (
                        future["high"].max() - close.iloc[i]
                    ) / self.pip_size
                else:
                    df.at[df.index[i], "magnitude_pips"] = (
                        close.iloc[i] - future["low"].min()
                    ) / self.pip_size
            df = df.dropna(subset=["direction_fwd_4"]).reset_index(drop=True)

        # Phase 7.A: optional ICT feature enrichment via historical state
        # sidecar. Joined by ``datetime`` to the technical feature matrix —
        # see docs/PHASE_7A_SCOPE.md §4.1. Sidecar must be produced with the
        # same timeframe as ``self.timeframe`` (typically H1) so the join
        # keys line up; the helper warns and skips on count mismatch
        # rather than silently merging on a partial overlap.
        df = self._merge_phase7a_sidecar(df)

        exclude = {"datetime", "open", "high", "low", "close", "volume",
                   "direction_fwd_4", "magnitude_pips"}
        # Phase 7.A's string label column is for gate-5 only, not for the
        # ML feature matrix — splitters can't consume strings without an
        # encoder, and the four one-hot booleans cover the same signal.
        exclude.add("po3_phase")
        self._feature_cols = [c for c in df.columns if c not in exclude]

        logger.info(
            "Feature matrix built: %d rows × %d features%s",
            len(df), len(self._feature_cols),
            " (no targets)" if not include_targets else "",
        )
        return df

    def _merge_phase7a_sidecar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merge the Phase 7.A ICT feature columns into ``df`` if a
        sidecar parquet path was supplied to ``__init__``. Idempotent —
        no-op when the path is unset.

        The merge is a left join on ``datetime`` against the sidecar's
        ``audit_ts`` field, with each Phase 7.A column reduced to its
        per-bar derivation via ``ICTFeatureEngineer.enrich``. Rows in
        the technical frame that have no matching sidecar timestamp
        retain the "no signal" sentinels documented in
        :data:`prism.data.feature_engineering.PHASE_7A_FEATURE_COLUMNS`.
        """
        if self.phase7a_sidecar_path is None:
            return df

        try:
            from prism.data.feature_engineering import (
                ICTFeatureEngineer, PHASE_7A_FEATURE_COLUMNS,
            )
            from prism.data.historical_state import read_replay_sidecar
        except Exception as exc:  # pragma: no cover — import-time failure path
            logger.error(
                "Phase 7.A sidecar requested but module imports failed: %s",
                exc,
            )
            return df

        sidecar_path = self.phase7a_sidecar_path
        if not sidecar_path.exists():
            logger.warning(
                "Phase 7.A sidecar %s not found — skipping ICT enrichment",
                sidecar_path,
            )
            return df

        sidecar = read_replay_sidecar(sidecar_path)
        threshold = self.phase7a_ob_max_distance_pips
        if threshold is None:
            engineer = ICTFeatureEngineer.from_env()
        else:
            engineer = ICTFeatureEngineer(ob_max_distance_pips=threshold)
        enriched_sidecar = engineer.enrich(sidecar)

        # Normalise both join keys to UTC-naive timestamps to avoid the
        # tz-aware vs tz-naive comparison failure that would otherwise
        # produce a fully empty merge silently.
        merge_cols = ["datetime"] + list(PHASE_7A_FEATURE_COLUMNS)
        sidecar_view = enriched_sidecar.copy()
        sidecar_view["datetime"] = pd.to_datetime(
            sidecar_view["audit_ts"], utc=True,
        ).dt.tz_convert(None)
        sidecar_view = sidecar_view[merge_cols]

        df_join = df.copy()
        df_join["datetime"] = pd.to_datetime(df_join["datetime"], utc=True).dt.tz_convert(None)
        merged = df_join.merge(sidecar_view, on="datetime", how="left")

        # Fill the "no sidecar match" sentinels per scope §2 so the model
        # never sees raw NaNs in the ICT columns.
        defaults = {
            "htf_alignment": 1,
            "kill_zone_strength": 0,
            "sweep_confirmed": False,
            "ob_distance_pips": -1.0,
            "ob_in_range": False,
            "po3_phase": "unknown",
            "po3_accumulation": False,
            "po3_manipulation": False,
            "po3_distribution": False,
            "po3_unknown": True,
        }
        for col, default in defaults.items():
            if col in merged.columns:
                merged[col] = merged[col].fillna(default)

        join_count = merged[PHASE_7A_FEATURE_COLUMNS[0]].notna().sum()
        logger.info(
            "Phase 7.A sidecar merged: %d / %d rows matched a sidecar timestamp",
            join_count, len(merged),
        )

        # Restore the original datetime dtype so downstream code that
        # expects tz-aware timestamps (or the caller's original dtype)
        # is not surprised by the tz-strip we did for the join.
        merged["datetime"] = df["datetime"].values
        return merged

    def _load_price_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Try Tiingo first, fall back to yfinance."""
        tf_map = {"H4": "4hour", "H1": "1hour", "M15": "15min", "D1": "daily"}
        tiingo_tf = tf_map.get(self.timeframe, "1hour")

        try:
            from prism.data.tiingo import get_ohlcv
            df = get_ohlcv(self.instrument, start_date, end_date, tiingo_tf)
            if not df.empty:
                return df
        except Exception as e:
            logger.warning(f"Tiingo unavailable: {e}. Trying yfinance...")

        try:
            import yfinance as yf
            from prism.data.tiingo import YF_MAP
            ticker = YF_MAP.get(self.instrument, self.instrument + "=X")
            yf_tf = {"H4": "4h", "H1": "1h", "M15": "15m", "D1": "1d"}.get(self.timeframe, "1h")
            raw = yf.download(ticker, start=start_date, end=end_date, interval=yf_tf, progress=False)
            if raw.empty:
                return pd.DataFrame()
            raw = raw.reset_index()
            raw.columns = [c.lower() for c in raw.columns]
            raw = raw.rename(columns={"index": "datetime", "date": "datetime",
                                       "datetime": "datetime"})
            raw["datetime"] = pd.to_datetime(raw["datetime"])
            return raw[["datetime", "open", "high", "low", "close"]].copy()
        except Exception as e:
            logger.error(f"yfinance also failed: {e}")
            return pd.DataFrame()

    def normalize(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """StandardScaler normalization per feature. Chronological fit only."""
        from sklearn.preprocessing import StandardScaler
        df = df.copy()
        if fit:
            self._scaler = StandardScaler()
            df[self._feature_cols] = self._scaler.fit_transform(
                df[self._feature_cols].fillna(0)
            )
        else:
            if self._scaler is None:
                raise RuntimeError("Call normalize(fit=True) first")
            df[self._feature_cols] = self._scaler.transform(
                df[self._feature_cols].fillna(0)
            )
        return df

    def split_train_test(
        self, df: pd.DataFrame, test_ratio: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Chronological train/test split. NEVER shuffles.
        Returns: X_train, X_test, y_train, y_test
        """
        split_idx = int(len(df) * (1 - test_ratio))
        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]
        X_train = train[self._feature_cols].fillna(0)
        X_test = test[self._feature_cols].fillna(0)
        y_train = train["direction_fwd_4"]
        y_test = test["direction_fwd_4"]
        logger.info(f"Train: {len(train)} rows | Test: {len(test)} rows | "
                    f"Features: {len(self._feature_cols)}")
        return X_train, X_test, y_train, y_test
