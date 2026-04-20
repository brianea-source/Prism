"""
tests/test_signal_generator.py
-------------------------------
Unit tests for SignalGenerator (prism/signal/generator.py).

These tests are designed to run without live API keys or trained models.
They exercise the calculate_sl_tp method and signal validation logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prism.signal.generator import SignalGenerator, MIN_RR_RATIO, PIP_VALUES, MIN_SL_PIPS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eurusd_bars(n: int = 100) -> pd.DataFrame:
    """Generate synthetic EURUSD H1 bars."""
    rng = np.random.default_rng(42)
    closes = 1.08 + np.cumsum(rng.normal(0, 0.0002, n))
    highs  = closes + rng.uniform(0.0001, 0.0005, n)
    lows   = closes - rng.uniform(0.0001, 0.0005, n)
    opens  = np.roll(closes, 1)
    opens[0] = closes[0]
    index = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": np.ones(n) * 1000,
    }, index=index)


def _make_generator(instrument: str = "EURUSD") -> SignalGenerator:
    return SignalGenerator(instrument=instrument, model_dir="/tmp/nonexistent_models/")


# ---------------------------------------------------------------------------
# calculate_sl_tp
# ---------------------------------------------------------------------------

class TestCalculateSLTP:

    @pytest.mark.parametrize("instrument,entry,direction,correction_low,expected_dir", [
        ("EURUSD", 1.0800, "LONG",  1.0750, "sl_below_entry"),
        ("EURUSD", 1.0800, "SHORT", 1.0850, "sl_above_entry"),
        ("XAUUSD", 2000.0, "LONG",  1990.0, "sl_below_entry"),
        ("XAUUSD", 2000.0, "SHORT", 2010.0, "sl_above_entry"),
        ("USDJPY", 150.00, "LONG",  149.50, "sl_below_entry"),
    ])
    def test_sl_direction(self, instrument, entry, direction, correction_low, expected_dir):
        gen = _make_generator(instrument)
        atr = entry * 0.003  # 0.3% ATR
        sl, tp1, tp2 = gen.calculate_sl_tp(direction, entry, correction_low, atr, instrument)

        if expected_dir == "sl_below_entry":
            assert sl < entry, f"LONG SL {sl:.5f} should be below entry {entry:.5f}"
        else:
            assert sl > entry, f"SHORT SL {sl:.5f} should be above entry {entry:.5f}"

    def test_tp1_tp2_ordering_long(self):
        gen = _make_generator("EURUSD")
        entry = 1.0800
        sl, tp1, tp2 = gen.calculate_sl_tp("LONG", entry, 1.0750, 0.002, "EURUSD")
        assert tp1 > entry,  f"LONG TP1 {tp1} should be above entry {entry}"
        assert tp2 > tp1,   f"LONG TP2 {tp2} should be above TP1 {tp1}"

    def test_tp1_tp2_ordering_short(self):
        gen = _make_generator("EURUSD")
        entry = 1.0800
        sl, tp1, tp2 = gen.calculate_sl_tp("SHORT", entry, 1.0850, 0.002, "EURUSD")
        assert tp1 < entry, f"SHORT TP1 {tp1} should be below entry {entry}"
        assert tp2 < tp1,   f"SHORT TP2 {tp2} should be below TP1 {tp1}"

    def test_tp2_is_2x_tp1_distance(self):
        gen = _make_generator("EURUSD")
        entry = 1.0800
        sl, tp1, tp2 = gen.calculate_sl_tp("LONG", entry, 1.0750, 0.002, "EURUSD")
        dist_tp1 = abs(tp1 - entry)
        dist_tp2 = abs(tp2 - entry)
        assert abs(dist_tp2 / dist_tp1 - 2.0) < 0.01, \
            f"TP2 distance ({dist_tp2:.5f}) should be 2× TP1 distance ({dist_tp1:.5f})"

    def test_minimum_sl_distance_enforced(self):
        """SL must never be closer than MIN_SL_PIPS to entry."""
        gen = _make_generator("EURUSD")
        entry = 1.0800
        pip = PIP_VALUES["EURUSD"]
        min_pips = MIN_SL_PIPS["EURUSD"]
        # Pass a correction_low that is only 1 pip away (too tight)
        tiny_correction = entry - 1 * pip
        sl, _, _ = gen.calculate_sl_tp("LONG", entry, tiny_correction, 1e-6, "EURUSD")
        sl_pips = (entry - sl) / pip
        assert sl_pips >= min_pips - 0.01, \
            f"SL distance {sl_pips:.1f} pips should be >= {min_pips} pips minimum"

    def test_rr_ratio_at_least_min(self):
        """TP1 should give at least MIN_RR_RATIO when using a normal ATR."""
        gen = _make_generator("EURUSD")
        entry = 1.0800
        sl, tp1, _ = gen.calculate_sl_tp("LONG", entry, 1.0750, 0.002, "EURUSD")
        sl_dist  = abs(entry - sl)
        tp1_dist = abs(tp1 - entry)
        rr = tp1_dist / sl_dist
        assert rr >= 1.4, f"R:R {rr:.2f} unexpectedly low"


# ---------------------------------------------------------------------------
# SignalGenerator.generate — without models / API
# ---------------------------------------------------------------------------

class TestSignalGeneratorGenerate:

    def test_returns_none_for_short_df(self):
        gen = _make_generator("EURUSD")
        df = _make_eurusd_bars(n=10)
        assert gen.generate(df) is None

    def test_returns_none_or_dict_for_valid_df(self):
        """With no models loaded, result is either None or a valid dict."""
        gen = _make_generator("EURUSD")
        df = _make_eurusd_bars(n=200)
        result = gen.generate(df)
        assert result is None or isinstance(result, dict)

    def test_signal_dict_has_required_keys(self):
        """If a signal is returned it must have all required PRD fields."""
        required = {
            "instrument", "signal_time", "direction", "confidence",
            "entry", "sl", "tp1", "tp2", "rr_ratio", "risk_level",
            "regime", "icc_phase", "aoi_confluence", "model_version",
        }
        gen = _make_generator("EURUSD")
        df = _make_eurusd_bars(n=200)
        result = gen.generate(df)
        if result is not None:
            assert required.issubset(set(result.keys())), \
                f"Missing keys: {required - set(result.keys())}"

    def test_signal_direction_valid(self):
        gen = _make_generator("EURUSD")
        df = _make_eurusd_bars(n=200)
        result = gen.generate(df)
        if result is not None:
            assert result["direction"] in ("LONG", "SHORT")

    def test_signal_rr_above_minimum(self):
        gen = _make_generator("EURUSD")
        df = _make_eurusd_bars(n=200)
        result = gen.generate(df)
        if result is not None:
            assert result["rr_ratio"] >= MIN_RR_RATIO, \
                f"Signal R:R {result['rr_ratio']} below minimum {MIN_RR_RATIO}"

    def test_signal_confidence_above_minimum(self):
        from prism.signal.generator import MIN_CONFIDENCE
        gen = _make_generator("EURUSD")
        df = _make_eurusd_bars(n=200)
        result = gen.generate(df)
        if result is not None:
            assert result["confidence"] >= MIN_CONFIDENCE

    def test_xauusd_instrument(self):
        gen = _make_generator("XAUUSD")
        n = 200
        rng = np.random.default_rng(7)
        closes = 2000 + np.cumsum(rng.normal(0, 0.5, n))
        highs  = closes + rng.uniform(0.1, 0.5, n)
        lows   = closes - rng.uniform(0.1, 0.5, n)
        opens  = np.roll(closes, 1); opens[0] = closes[0]
        index = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": np.ones(n) * 1000,
        }, index=index)
        result = gen.generate(df)
        assert result is None or result["instrument"] == "XAUUSD"
