"""Tests for ICC pattern detection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd
import pytest
from prism.signal.icc import detect_icc_phase, get_icc_entry, AOIDetector

def make_ohlcv(closes):
    n = len(closes)
    closes = np.array(closes)
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": closes * 0.999, "high": closes * 1.002,
        "low": closes * 0.998, "close": closes, "volume": [1000.0] * n,
    })

def test_detect_none_flat_market():
    df = make_ohlcv([1.1000] * 40)
    assert detect_icc_phase(df) == "NONE"

def test_get_icc_entry_returns_valid_structure():
    closes = list(np.linspace(1.1000, 1.1200, 30)) + list(np.linspace(1.1200, 1.1100, 10))
    result = get_icc_entry(make_ohlcv(closes), instrument="EURUSD")
    assert result is None or isinstance(result, dict)
    if result:
        for key in ("direction", "entry", "sl", "correction_pct"):
            assert key in result
        assert result["direction"] in ("LONG", "SHORT")

def test_long_sl_below_entry():
    closes = list(np.linspace(1.1000, 1.1200, 25)) + list(np.linspace(1.1200, 1.1120, 8)) + [1.1145, 1.1160]
    result = get_icc_entry(make_ohlcv(closes), instrument="EURUSD")
    if result and result["direction"] == "LONG":
        assert result["sl"] < result["entry"]

def test_aoi_detector():
    df = make_ohlcv([1.1000 + i * 0.001 for i in range(30)])
    aoi = AOIDetector(df)
    assert isinstance(aoi.get_nearby_aoi(1.1290), list)
    assert isinstance(aoi.is_at_aoi(1.1290), bool)
