"""
tests/test_pr1_pr2_fixes.py
Regression tests for bugs discovered during post-merge review of PR#1 and PR#2.

Covers:
  - train.py referenced a non-existent column 'direction_4h' (should be 'direction_fwd_4')
  - train.py saved model filenames that PRISMPredictor could not load
    (layer1_lgbm vs layer1_lgb, layer2_reg vs layer2_magnitude,
     layer3_rf vs layer3_confidence)
  - tiingo.py routed FX intraday through the IEX endpoint (equities-only)
  - tiingo.py column mapping only handled adjusted equity-daily columns,
    so intraday and FX responses collapsed to empty DataFrames
  - fred.py computed cpi_yoy / gdp_growth on the forward-filled daily panel
    instead of the raw monthly/quarterly series
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# 1. train → predict end-to-end (catches both the column-name and filename bugs)
# ---------------------------------------------------------------------------

def _fake_features(n: int = 80, n_features: int = 12, seed: int = 7) -> pd.DataFrame:
    """Build a DataFrame that mirrors the shape PRISMTrainer expects."""
    rng = np.random.default_rng(seed)
    feats = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    feats["direction_fwd_4"] = rng.choice([-1, 0, 1], size=n)
    feats["magnitude_pips"] = rng.uniform(5, 40, size=n)
    return feats


class _FakePipeline:
    """Minimal stand-in for PRISMFeaturePipeline — avoids network I/O."""

    def __init__(self, instrument: str, timeframe: str = "H1"):
        self.instrument = instrument
        self.timeframe = timeframe
        self._feature_cols: list[str] = []
        self._df: pd.DataFrame | None = None

    def build_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        df = _fake_features()
        self._feature_cols = [c for c in df.columns if c.startswith("feat_")]
        self._df = df
        return df

    def split_train_test(self, df, test_ratio: float = 0.2):
        split = int(len(df) * (1 - test_ratio))
        train = df.iloc[:split]
        test = df.iloc[split:]
        return (
            train[self._feature_cols],
            test[self._feature_cols],
            train["direction_fwd_4"],
            test["direction_fwd_4"],
        )


def test_train_all_layers_then_predict_roundtrip(monkeypatch):
    """End-to-end contract: PRISMTrainer.train_all_layers() must save models
    under filenames that PRISMPredictor._load_models() can find.

    This single test catches both the PR#2 bugs:
      1. train.py dropna(subset=["direction_4h"]) → KeyError
      2. train saves layer1_lgbm_* / layer2_reg_* / layer3_rf_* but predict
         loads layer1_lgb_* / layer2_magnitude_* / layer3_confidence_*.
    """
    from prism.model import train as train_mod
    from prism.model import predict as predict_mod

    with tempfile.TemporaryDirectory() as tmpdir:
        mdir = Path(tmpdir)
        # Redirect both trainer and predictor to the temp models directory.
        monkeypatch.setattr(train_mod, "MODELS_DIR", mdir)
        monkeypatch.setattr(predict_mod, "MODEL_DIR", mdir)
        # Avoid importing the real feature pipeline (requires network/FRED/etc).
        monkeypatch.setattr(
            "prism.data.pipeline.PRISMFeaturePipeline", _FakePipeline
        )

        trainer = train_mod.PRISMTrainer("EURUSD", timeframe="H1")
        results = trainer.train_all_layers("2024-01-01", "2024-06-01")

        # All four layers trained.
        assert len(results) == 4
        layer_names = {r.layer for r in results}
        assert layer_names == {
            "layer1_xgb", "layer1_lgb", "layer2_magnitude", "layer3_confidence"
        }

        # Each saved file exists under the name PRISMPredictor will look for.
        for name in ("layer1_xgb", "layer1_lgb",
                     "layer2_magnitude", "layer3_confidence"):
            assert (mdir / f"{name}_EURUSD.joblib").exists(), (
                f"Trainer did not save {name}_EURUSD.joblib — "
                "predictor will FileNotFoundError."
            )

        # Predictor loads cleanly and produces per-row output.
        predictor = predict_mod.PRISMPredictor("EURUSD")
        X = _fake_features(n=10)[[f"feat_{i}" for i in range(12)]]
        out = predictor.predict(X)
        assert len(out["direction"]) == 10
        assert set(out["direction_str"]).issubset({"LONG", "SHORT", "NEUTRAL"})


# ---------------------------------------------------------------------------
# 2. Tiingo FX endpoint routing
# ---------------------------------------------------------------------------

def test_tiingo_fx_instrument_hits_fx_endpoint(tmp_path, monkeypatch):
    """EURUSD intraday must hit /tiingo/fx/eurusd/prices, not /iex/."""
    from prism.data import tiingo as tiingo_mod

    monkeypatch.setattr(tiingo_mod, "CACHE_DIR", tmp_path)

    client = tiingo_mod.TiingoClient(api_key="test-key")
    captured: dict = {}

    def fake_get(endpoint: str, params: dict):
        captured["endpoint"] = endpoint
        captured["params"] = params
        return [
            {"date": "2024-01-01T00:00:00Z",
             "open": 1.10, "high": 1.11, "low": 1.09, "close": 1.105,
             "volume": 0},
            {"date": "2024-01-01T01:00:00Z",
             "open": 1.105, "high": 1.12, "low": 1.10, "close": 1.115,
             "volume": 0},
        ]

    monkeypatch.setattr(client, "_get", fake_get)
    df = client.get_ohlcv("EURUSD", "2024-01-01", "2024-01-02", timeframe="1hour")

    assert captured["endpoint"] == "tiingo/fx/eurusd/prices"
    assert captured["params"]["resampleFreq"] == "1hour"
    assert not df.empty
    assert list(df.columns) == ["datetime", "open", "high", "low", "close", "volume"]
    assert len(df) == 2


def test_tiingo_fx_daily_uses_1day_resample(tmp_path, monkeypatch):
    """Daily FX requests must go to the FX endpoint with resampleFreq=1day."""
    from prism.data import tiingo as tiingo_mod

    monkeypatch.setattr(tiingo_mod, "CACHE_DIR", tmp_path)
    client = tiingo_mod.TiingoClient(api_key="test-key")
    captured: dict = {}

    def fake_get(endpoint, params):
        captured["endpoint"] = endpoint
        captured["params"] = params
        return [{"date": "2024-01-01T00:00:00Z",
                 "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15,
                 "volume": 0}]

    monkeypatch.setattr(client, "_get", fake_get)
    df = client.get_ohlcv("GBPUSD", "2024-01-01", "2024-01-31", timeframe="daily")

    assert captured["endpoint"] == "tiingo/fx/gbpusd/prices"
    assert captured["params"]["resampleFreq"] == "1day"
    assert not df.empty


def test_tiingo_intraday_without_adj_columns_returns_ohlcv(tmp_path, monkeypatch):
    """Intraday responses have no adj* fields; the renamer must fall back to
    raw open/high/low/close/volume rather than silently returning empty."""
    from prism.data import tiingo as tiingo_mod

    monkeypatch.setattr(tiingo_mod, "CACHE_DIR", tmp_path)
    client = tiingo_mod.TiingoClient(api_key="test-key")

    def fake_get(endpoint, params):
        # Mimic Tiingo IEX / FX intraday payload (no adj* fields)
        return [
            {"date": "2024-01-01T12:00:00Z", "open": 100.0, "high": 101.0,
             "low": 99.5, "close": 100.5, "volume": 1000},
            {"date": "2024-01-01T13:00:00Z", "open": 100.5, "high": 101.5,
             "low": 100.0, "close": 101.0, "volume": 1500},
        ]

    monkeypatch.setattr(client, "_get", fake_get)
    df = client.get_ohlcv("XAUUSD", "2024-01-01", "2024-01-02", timeframe="1hour")

    assert not df.empty, "Expected raw OHLCV fallback, got empty DataFrame"
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
    assert df["close"].iloc[-1] == pytest.approx(101.0)


def test_tiingo_daily_prefers_adjusted_columns(tmp_path, monkeypatch):
    """Equity daily responses expose adj* fields — those should be preferred."""
    from prism.data import tiingo as tiingo_mod

    monkeypatch.setattr(tiingo_mod, "CACHE_DIR", tmp_path)
    client = tiingo_mod.TiingoClient(api_key="test-key")

    def fake_get(endpoint, params):
        return [
            {"date": "2024-01-01T00:00:00Z",
             "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1_000,
             "adjOpen": 95.0, "adjHigh": 96.0, "adjLow": 94.0, "adjClose": 95.5,
             "adjVolume": 1_100},
        ]

    monkeypatch.setattr(client, "_get", fake_get)
    df = client.get_ohlcv("XAUUSD", "2024-01-01", "2024-01-31", timeframe="daily")

    assert df["close"].iloc[0] == pytest.approx(95.5), (
        "Daily XAUUSD (GLD) should use adjClose, not raw close"
    )
    assert df["volume"].iloc[0] == 1_100  # adjVolume


# ---------------------------------------------------------------------------
# 3. FRED cpi_yoy computed from raw monthly series
# ---------------------------------------------------------------------------

def test_fred_cpi_yoy_computed_on_raw_monthly_series(tmp_path, monkeypatch):
    """cpi_yoy must equal (CPI_this_month / CPI_12_months_ago - 1) * 100,
    independent of whether the daily panel has been forward-filled.

    Previously pct_change(252) on the daily ffilled series gave a DIFFERENT
    answer whenever start_date landed mid-month or the monthly release
    cadence shifted by a business day or two.
    """
    from prism.data import fred as fred_mod

    monkeypatch.setattr(fred_mod, "CACHE_DIR", tmp_path)

    # 24 monthly CPI observations: flat at 100 for year 1, then exactly +5%
    # for every month of year 2 (YoY for any row in year 2 = 5%).
    months = pd.date_range("2022-01-01", periods=24, freq="MS")
    cpi_monthly = pd.DataFrame({
        "date": months,
        "value": [100.0] * 12 + [105.0] * 12,
    })
    # Quarterly GDP: flat then +2% growth quarter-on-quarter.
    quarters = pd.date_range("2022-01-01", periods=8, freq="QS")
    gdp_q = pd.DataFrame({
        "date": quarters,
        "value": [100.0, 100.0, 100.0, 100.0, 102.0, 104.04, 106.12, 108.24],
    })

    def fake_get_series(self, series_id, start_date, end_date):
        if series_id == "CPIAUCSL":
            return cpi_monthly.copy()
        if series_id == "GDP":
            return gdp_q.copy()
        # Everything else: empty → NaN column, fine for this test
        return pd.DataFrame(columns=["date", "value"])

    # Skip fredapi entirely — we stub get_series.
    monkeypatch.setattr(fred_mod.FREDClient, "get_series", fake_get_series)

    client = fred_mod.FREDClient(api_key="")
    # Force fred to be non-None so the ImportError guard doesn't short-circuit
    client.fred = object()

    macro = client.get_macro_features("2022-01-01", "2023-12-31")

    # Pick a row from the second year — YoY should be 5% (per our synthetic series)
    macro["date"] = pd.to_datetime(macro["date"])
    late = macro[macro["date"] >= "2023-06-01"].iloc[0]
    assert late["cpi_yoy"] == pytest.approx(5.0, rel=0.02), (
        f"Expected cpi_yoy ~5.0, got {late['cpi_yoy']}"
    )

    # GDP QoQ on the raw series is 2% in the growth phase
    late_gdp = macro[macro["date"] >= "2023-03-01"].iloc[0]
    assert late_gdp["gdp_growth"] == pytest.approx(2.0, rel=0.05), (
        f"Expected gdp_growth ~2.0, got {late_gdp['gdp_growth']}"
    )


# ---------------------------------------------------------------------------
# 4. Sanity: INSTRUMENT_MAP is consistent with FX routing
# ---------------------------------------------------------------------------

def test_instrument_map_fx_tickers_lowercase():
    """Tiingo FX tickers are lowercase. Keep INSTRUMENT_MAP aligned so we
    don't accidentally request `/tiingo/fx/EURUSD/prices` (which may 404)."""
    from prism.data.tiingo import INSTRUMENT_MAP, _FX_INSTRUMENTS

    for sym in _FX_INSTRUMENTS:
        assert INSTRUMENT_MAP[sym] == INSTRUMENT_MAP[sym].lower()
