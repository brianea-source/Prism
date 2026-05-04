"""
PRISM Phase 4 — live bars + model presence + in-flight guard.

Covers:
* MT5Bridge.get_bars schema + stale/offline behaviour
* MT5Bridge.bars_are_fresh boundary conditions
* MockMT5Bridge.get_bars cache fallback + supports_live_bars flag
* PRISMFeaturePipeline.build_features_from_bars (in-memory feature path)
* missing_model_files helper used by runner startup
* runner._should_fire in-flight guard
* runner._scan_instrument live-vs-demo branching
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fake MT5 module — a minimal stand-in that exposes the few attributes
# MT5Bridge.get_bars needs (TIMEFRAME_* constants + copy_rates_from_pos).
# Installing MetaTrader5 in the test environment isn't feasible (Windows-only
# wheels), so we fake it for the one method under test.
# ---------------------------------------------------------------------------

class _FakeMt5:
    def __init__(self, rates=None, raise_on_rates: bool = False):
        self.TIMEFRAME_M5 = "tf_m5"
        self.TIMEFRAME_M15 = "tf_m15"
        self.TIMEFRAME_H1 = "tf_h1"
        self.TIMEFRAME_H4 = "tf_h4"
        self.TIMEFRAME_D1 = "tf_d1"
        self._rates = rates
        self._raise = raise_on_rates
        self.calls: list = []

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        self.calls.append((symbol, timeframe, start, count))
        if self._raise:
            raise RuntimeError("simulated MT5 error")
        return self._rates

    def symbol_info(self, _symbol):
        return None


def _synthetic_rates(count: int, start: datetime, period_min: int) -> np.ndarray:
    """Build a MT5-compatible structured array of ``count`` bars."""
    dtype = np.dtype([
        ("time", "i8"),
        ("open", "f8"),
        ("high", "f8"),
        ("low", "f8"),
        ("close", "f8"),
        ("tick_volume", "i8"),
        ("spread", "i4"),
        ("real_volume", "i8"),
    ])
    arr = np.zeros(count, dtype=dtype)
    for i in range(count):
        ts = start + timedelta(minutes=period_min * i)
        arr["time"][i] = int(ts.timestamp())
        arr["open"][i] = 1.1000 + 0.0001 * i
        arr["high"][i] = 1.1010 + 0.0001 * i
        arr["low"][i] = 1.0990 + 0.0001 * i
        arr["close"][i] = 1.1005 + 0.0001 * i
        arr["tick_volume"][i] = 100 + i
    return arr


# ===========================================================================
# MT5Bridge.get_bars
# ===========================================================================

class TestGetBarsSchema:
    def _bridge_with_fake_mt5(self, fake):
        from prism.execution.mt5_bridge import MT5Bridge
        b = MT5Bridge(mode="CONFIRM")
        b._mt5 = fake
        b._connected = True
        return b

    def test_returns_expected_columns(self):
        rates = _synthetic_rates(
            50, datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc), 240,
        )
        bridge = self._bridge_with_fake_mt5(_FakeMt5(rates))
        df = bridge.get_bars("EURUSD", "H4", count=50)
        assert list(df.columns) == ["datetime", "open", "high", "low", "close", "volume"]
        assert len(df) == 50

    def test_datetime_is_utc_tz_aware(self):
        rates = _synthetic_rates(
            10, datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc), 240,
        )
        bridge = self._bridge_with_fake_mt5(_FakeMt5(rates))
        df = bridge.get_bars("EURUSD", "H4")
        assert df["datetime"].iloc[0].tzinfo is not None
        assert str(df["datetime"].iloc[0].tzinfo) == "UTC"

    def test_tick_volume_mapped_to_volume(self):
        rates = _synthetic_rates(
            5, datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc), 240,
        )
        bridge = self._bridge_with_fake_mt5(_FakeMt5(rates))
        df = bridge.get_bars("EURUSD", "H4")
        assert df["volume"].iloc[0] == 100
        assert df["volume"].iloc[-1] == 104

    def test_returns_empty_when_disconnected(self):
        from prism.execution.mt5_bridge import MT5Bridge
        b = MT5Bridge(mode="CONFIRM")
        # Not connected
        df = b.get_bars("EURUSD", "H4")
        assert df.empty

    def test_returns_empty_on_mt5_error(self):
        bridge = self._bridge_with_fake_mt5(_FakeMt5(rates=None, raise_on_rates=True))
        df = bridge.get_bars("EURUSD", "H4")
        assert df.empty

    def test_returns_empty_on_none_rates(self):
        bridge = self._bridge_with_fake_mt5(_FakeMt5(rates=None))
        df = bridge.get_bars("EURUSD", "H4")
        assert df.empty

    def test_unknown_timeframe_raises_before_mt5_call(self):
        bridge = self._bridge_with_fake_mt5(_FakeMt5(rates=[]))
        # Empty rates would normally return empty-df; but invalid TF should
        # never reach that path. We still catch the exception and return
        # empty — the key property is "no signal" rather than "crash".
        df = bridge.get_bars("EURUSD", "BADTF")
        assert df.empty


# ===========================================================================
# Bar freshness guard
# ===========================================================================

class TestBarsAreFresh:
    def _df(self, last_dt):
        return pd.DataFrame({
            "datetime": [last_dt],
            "open": [1.0], "high": [1.0], "low": [1.0],
            "close": [1.0], "volume": [0],
        })

    def _bridge(self):
        from prism.execution.mt5_bridge import MT5Bridge
        return MT5Bridge(mode="CONFIRM")

    def test_fresh_h4_bar_accepted(self):
        now = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)
        # Last bar 2 hours ago — well inside a single H4 period.
        df = self._df(now - timedelta(hours=2))
        assert self._bridge().bars_are_fresh(df, "H4", now=now) is True

    def test_h4_bar_just_inside_tolerance(self):
        now = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)
        # 1.5x the H4 period = 6h tolerance. 5h old should still pass.
        df = self._df(now - timedelta(hours=5))
        assert self._bridge().bars_are_fresh(df, "H4", now=now) is True

    def test_h4_bar_past_tolerance_rejected(self):
        now = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)
        # 7h old > 1.5 * 4h = 6h tolerance → stale.
        df = self._df(now - timedelta(hours=7))
        assert self._bridge().bars_are_fresh(df, "H4", now=now) is False

    def test_m5_bar_past_tolerance_rejected(self):
        now = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)
        # 1.5 * 5min = 7.5min. 15min old → stale.
        df = self._df(now - timedelta(minutes=15))
        assert self._bridge().bars_are_fresh(df, "M5", now=now) is False

    def test_empty_df_is_never_fresh(self):
        now = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)
        assert self._bridge().bars_are_fresh(pd.DataFrame(), "H4", now=now) is False

    def test_unknown_timeframe_is_never_fresh(self):
        now = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)
        df = self._df(now - timedelta(minutes=1))
        assert self._bridge().bars_are_fresh(df, "BADTF", now=now) is False

    def test_tz_naive_datetime_treated_as_utc(self):
        """
        Defensive: a tz-naive 'datetime' column (common when reading parquet
        written without tz info) must still produce a sane freshness answer
        rather than raising TypeError on subtract.
        """
        now = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)
        df = pd.DataFrame({
            "datetime": [pd.Timestamp("2026-04-20 11:00:00")],
            "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [0],
        })
        assert self._bridge().bars_are_fresh(df, "H4", now=now) is True


# ===========================================================================
# MockMT5Bridge live-bars flag + cache fallback
# ===========================================================================

class TestMockBridgeLiveBars:
    def test_supports_live_bars_is_false(self):
        from prism.execution.mt5_bridge import MockMT5Bridge
        b = MockMT5Bridge()
        b.connect()
        assert b.supports_live_bars() is False

    def test_get_bars_falls_back_to_cache(self, tmp_path, monkeypatch):
        from prism.execution.mt5_bridge import MockMT5Bridge

        cache_dir = tmp_path / "data" / "raw"
        cache_dir.mkdir(parents=True)
        df = pd.DataFrame({
            "datetime": pd.date_range("2026-04-01", periods=10, freq="4h", tz="UTC"),
            "open": np.arange(10, dtype=float),
            "high": np.arange(10, dtype=float) + 0.5,
            "low": np.arange(10, dtype=float) - 0.5,
            "close": np.arange(10, dtype=float) + 0.1,
            "volume": np.arange(10, dtype=int),
        })
        df.to_parquet(cache_dir / "tiingo_eurusd_4hour_2026-01-01_2026-04-20.parquet")

        monkeypatch.chdir(tmp_path)
        b = MockMT5Bridge()
        b.connect()
        out = b.get_bars("EURUSD", "H4", count=50)
        assert not out.empty
        assert "datetime" in out.columns

    def test_get_bars_returns_empty_when_no_cache(self, tmp_path, monkeypatch):
        from prism.execution.mt5_bridge import MockMT5Bridge
        monkeypatch.chdir(tmp_path)  # empty cache
        b = MockMT5Bridge()
        b.connect()
        out = b.get_bars("EURUSD", "H4")
        assert out.empty


# ===========================================================================
# In-memory feature pipeline (live path)
# ===========================================================================

class TestBuildFeaturesFromBars:
    def _bars(self, n=300):
        dates = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
        rng = np.random.default_rng(42)
        close = 1.1 + rng.normal(0, 0.001, n).cumsum()
        return pd.DataFrame({
            "datetime": dates,
            "open": close - 0.0005,
            "high": close + 0.0010,
            "low": close - 0.0010,
            "close": close,
            "volume": rng.integers(100, 500, n),
        })

    def test_engineers_expected_columns(self):
        from prism.data.pipeline import PRISMFeaturePipeline
        pipe = PRISMFeaturePipeline("EURUSD", "H4")
        df = pipe.build_features_from_bars(self._bars())
        # Key technical features the models expect:
        for col in ["atr_14", "atr_50", "rsi_14", "macd", "ema_9", "ema_200",
                    "stoch_k", "bb_pct", "session", "day_of_week"]:
            assert col in df.columns, f"{col} missing from live-path features"

    def test_omits_training_targets(self):
        from prism.data.pipeline import PRISMFeaturePipeline
        pipe = PRISMFeaturePipeline("EURUSD", "H4")
        df = pipe.build_features_from_bars(self._bars())
        # Targets are forward-looking — must NOT be computed on live bars.
        assert "direction_fwd_4" not in df.columns
        assert "magnitude_pips" not in df.columns

    def test_empty_bars_raises(self):
        from prism.data.pipeline import PRISMFeaturePipeline
        pipe = PRISMFeaturePipeline("EURUSD", "H4")
        with pytest.raises(ValueError, match="empty"):
            pipe.build_features_from_bars(pd.DataFrame())

    def test_missing_datetime_column_raises(self):
        from prism.data.pipeline import PRISMFeaturePipeline
        pipe = PRISMFeaturePipeline("EURUSD", "H4")
        df = pd.DataFrame({"close": [1.0, 1.1, 1.2]})
        with pytest.raises(ValueError, match="datetime"):
            pipe.build_features_from_bars(df)

    def test_preserves_ohlcv_input(self):
        """Input bars must not be mutated — pipeline operates on a copy."""
        from prism.data.pipeline import PRISMFeaturePipeline
        bars = self._bars(100)
        before_cols = set(bars.columns)
        pipe = PRISMFeaturePipeline("EURUSD", "H4")
        pipe.build_features_from_bars(bars)
        assert set(bars.columns) == before_cols, \
            "build_features_from_bars must not mutate the input DataFrame"


# ===========================================================================
# Model-presence check
# ===========================================================================

class TestModelPresenceCheck:
    def test_missing_returns_paths_for_untrained_instrument(self, tmp_path):
        from prism.model.predict import missing_model_files, MODEL_LAYER_NAMES
        missing = missing_model_files(["EURUSD", "XAUUSD"], model_dir=tmp_path)
        # 4 joblibs + 1 feature_cols sidecar per instrument × 2 instruments = 10
        assert len(missing) == 10
        for path in missing:
            assert path.suffix == ".joblib" or path.name.startswith("feature_cols_")

    def test_present_returns_empty(self, tmp_path):
        from prism.model.predict import missing_model_files, MODEL_LAYER_NAMES
        for name in MODEL_LAYER_NAMES:
            (tmp_path / f"{name}_EURUSD.joblib").write_bytes(b"x")
        # feature_cols sidecar is also required after fix/predict-feature-alignment
        (tmp_path / "feature_cols_EURUSD.json").write_text("{}")
        missing = missing_model_files(["EURUSD"], model_dir=tmp_path)
        assert missing == []

    def test_partial_coverage(self, tmp_path):
        """Only report the actually-missing files, not all artefacts per instrument."""
        from prism.model.predict import missing_model_files
        (tmp_path / "layer1_xgb_EURUSD.joblib").write_bytes(b"x")
        (tmp_path / "layer1_lgbm_EURUSD.joblib").write_bytes(b"x")
        missing = missing_model_files(["EURUSD"], model_dir=tmp_path)
        # layer2_reg + layer3_rf joblibs + feature_cols sidecar = 3 missing
        assert len(missing) == 3
        names = {p.stem if p.suffix == ".joblib" else p.name for p in missing}
        assert names == {
            "layer2_reg_EURUSD",
            "layer3_rf_EURUSD",
            "feature_cols_EURUSD.json",
        }

    def test_layer_names_constant(self):
        """Guard against silent renames that would bypass the check."""
        from prism.model.predict import MODEL_LAYER_NAMES
        assert set(MODEL_LAYER_NAMES) == {
            "layer1_xgb", "layer1_lgbm", "layer2_reg", "layer3_rf",
        }


# ===========================================================================
# In-flight signal guard
# ===========================================================================

class _StubSignal:
    def __init__(self, direction="LONG", signal_time="2026-04-20T08:00:00"):
        self.direction = direction
        self.signal_time = signal_time
        self.signal_id = "abc123"


class TestInFlightGuard:
    def setup_method(self):
        import prism.delivery.runner as runner_module
        runner_module._last_signal_key.clear()

    def _h4_df(self, last_ts="2026-04-20 08:00:00"):
        return pd.DataFrame({
            "datetime": [pd.Timestamp(last_ts, tz="UTC")],
        })

    def test_first_signal_on_bar_fires(self):
        from prism.delivery.runner import _should_fire
        assert _should_fire("EURUSD", _StubSignal("LONG"), self._h4_df()) is True

    def test_duplicate_signal_same_bar_suppressed(self):
        from prism.delivery.runner import _should_fire
        df = self._h4_df()
        assert _should_fire("EURUSD", _StubSignal("LONG"), df) is True
        assert _should_fire("EURUSD", _StubSignal("LONG"), df) is False

    def test_opposite_direction_same_bar_fires(self):
        """LONG → SHORT on the same bar is a meaningful flip; not suppressed."""
        from prism.delivery.runner import _should_fire
        df = self._h4_df()
        assert _should_fire("EURUSD", _StubSignal("LONG"), df) is True
        assert _should_fire("EURUSD", _StubSignal("SHORT"), df) is True

    def test_new_bar_reopens_gate(self):
        from prism.delivery.runner import _should_fire
        assert _should_fire("EURUSD", _StubSignal("LONG"), self._h4_df("2026-04-20 08:00:00")) is True
        assert _should_fire("EURUSD", _StubSignal("LONG"), self._h4_df("2026-04-20 12:00:00")) is True

    def test_per_instrument_isolation(self):
        """Firing EURUSD doesn't block XAUUSD on the same bar timestamp."""
        from prism.delivery.runner import _should_fire
        df = self._h4_df()
        assert _should_fire("EURUSD", _StubSignal("LONG"), df) is True
        assert _should_fire("XAUUSD", _StubSignal("LONG"), df) is True

    def test_missing_datetime_falls_back_to_signal_time(self):
        """If H4 df has no datetime column we still need a stable key."""
        from prism.delivery.runner import _should_fire
        df = pd.DataFrame({"close": [1.0]})
        sig = _StubSignal("LONG", signal_time="2026-04-20T08:00:00")
        assert _should_fire("EURUSD", sig, df) is True
        assert _should_fire("EURUSD", sig, df) is False


# ===========================================================================
# Runner startup gate — missing-model refusal
# ===========================================================================

class TestRunnerStartupGate:
    def test_missing_models_raises_systemexit_in_confirm_mode(self, tmp_path, monkeypatch):
        import prism.model.predict as predict_module

        # Empty model dir → every instrument is missing every layer.
        monkeypatch.setattr(predict_module, "MODEL_DIR", tmp_path)
        monkeypatch.setenv("PRISM_INSTRUMENTS", "EURUSD")
        monkeypatch.setenv("PRISM_EXECUTION_MODE", "CONFIRM")
        # Prevent the rest of run() from executing (SystemExit is enough)
        import prism.delivery.runner as runner_module
        monkeypatch.setattr(runner_module, "_build_bridge",
                            lambda *a, **kw: pytest.fail("_build_bridge must not be called when models are missing"))

        with pytest.raises(SystemExit):
            runner_module.run()

    def test_notify_mode_allowed_without_models(self, tmp_path, monkeypatch):
        """
        NOTIFY mode never executes — it's the canonical dry-run. Missing
        models shouldn't block a smoke test.
        """
        import prism.model.predict as predict_module
        import prism.delivery.runner as runner_module

        monkeypatch.setattr(predict_module, "MODEL_DIR", tmp_path)
        monkeypatch.setenv("PRISM_INSTRUMENTS", "EURUSD")
        monkeypatch.setenv("PRISM_EXECUTION_MODE", "NOTIFY")

        # Stop run() immediately after the startup gate so we don't actually
        # enter the scan loop — raising inside _build_bridge is enough.
        sentinel = RuntimeError("reached _build_bridge past the gate — good")
        monkeypatch.setattr(runner_module, "_build_bridge",
                            lambda *a, **kw: (_ for _ in ()).throw(sentinel))

        with pytest.raises(RuntimeError, match="reached _build_bridge"):
            runner_module.run()


# ===========================================================================
# Runner scanner wiring — live vs demo path
# ===========================================================================

class _FakeBridgeLive:
    mode = "NOTIFY"

    def __init__(self, bars_by_tf):
        self._bars = bars_by_tf
        self.get_bars_calls: list = []

    def supports_live_bars(self):
        return True

    def get_bars(self, instrument, tf, count=500):
        self.get_bars_calls.append((instrument, tf, count))
        return self._bars.get(tf, pd.DataFrame())

    def bars_are_fresh(self, df, tf, now=None):
        return True  # always fresh in tests


class _FakeBridgeDemo:
    mode = "NOTIFY"

    def __init__(self, cache_df):
        self._df = cache_df

    def supports_live_bars(self):
        return False

    def get_bars(self, instrument, tf, count=500):
        return self._df


class _StubNotifier:
    def __init__(self):
        self.sent: list = []
        self.channel = "#t"
        self.client = object()
        self.confirm_timeout_sec = 300

    def send_signal(self, signal, mode="CONFIRM", use_buttons=False, demo_warning=None):
        self.sent.append({"mode": mode, "demo_warning": demo_warning})
        return "ts-1"

    def update_signal_status(self, *a, **kw):
        pass


class TestRunnerBranching:
    def setup_method(self):
        import prism.delivery.runner as runner_module
        runner_module._last_signal_key.clear()

    def _make_h4_bars(self, n=60):
        dates = pd.date_range("2026-04-18", periods=n, freq="4h", tz="UTC")
        return pd.DataFrame({
            "datetime": dates,
            "open": np.linspace(1.1, 1.2, n),
            "high": np.linspace(1.105, 1.205, n),
            "low": np.linspace(1.095, 1.195, n),
            "close": np.linspace(1.1, 1.2, n),
            "volume": np.full(n, 200),
        })

    def test_live_path_calls_get_bars_for_all_three_timeframes(self, monkeypatch):
        import prism.delivery.runner as runner_module

        bars = {"H4": self._make_h4_bars(), "H1": self._make_h4_bars(), "M5": self._make_h4_bars()}
        bridge = _FakeBridgeLive(bars)
        notifier = _StubNotifier()

        # Stub the generator so we don't need trained models. The scanner
        # only cares that generate() is invoked AFTER live bars are pulled
        # AND that the call chain routes through build_features_from_bars.
        from prism.signal import generator as gen_module

        class _StubGen:
            def __init__(self, *a, **kw): pass
            def generate(self, *a, **kw): return None

        monkeypatch.setattr(gen_module, "SignalGenerator", _StubGen)
        # Same pipeline stub so we don't hit FRED / COT during test runs.
        from prism.data import pipeline as pipe_module

        class _StubPipe:
            def __init__(self, *a, **kw): pass
            def build_features_from_bars(self, df): return df

        monkeypatch.setattr(pipe_module, "PRISMFeaturePipeline", _StubPipe)

        runner_module._scan_instrument(
            "EURUSD", notifier, bridge,
            datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc),
        )

        tfs = [c[1] for c in bridge.get_bars_calls]
        assert "H4" in tfs and "H1" in tfs and "M5" in tfs

    def test_demo_path_passes_demo_warning(self, monkeypatch):
        import prism.delivery.runner as runner_module

        bars = self._make_h4_bars()
        bridge = _FakeBridgeDemo(bars)
        notifier = _StubNotifier()

        from prism.signal import generator as gen_module
        from prism.execution.mt5_bridge import SignalPacket

        def _fake_generate(*a, **kw):
            # Just enough to route through _should_fire + send_signal.
            return SignalPacket(
                instrument="EURUSD", direction="LONG", entry=1.1, sl=1.09,
                tp1=1.11, tp2=1.12, rr_ratio=2.0, confidence=0.7,
                confidence_level="MEDIUM", magnitude_pips=50.0,
                regime="RISK_ON", news_bias="NEUTRAL",
                fvg_zone=None, signal_time="2026-04-20T08:00:00",
            )

        class _StubGen:
            def __init__(self, instrument, persist_fvg=True):
                assert persist_fvg is False, \
                    "Demo path must disable FVG retest (no real M5 bars)"
            def generate(self, *a, **kw): return _fake_generate()

        monkeypatch.setattr(gen_module, "SignalGenerator", _StubGen)

        runner_module._scan_instrument(
            "EURUSD", notifier, bridge,
            datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc),
        )

        assert len(notifier.sent) == 1
        assert notifier.sent[0]["demo_warning"] is not None
        assert "aliased" in notifier.sent[0]["demo_warning"].lower()

    def test_live_path_no_demo_warning(self, monkeypatch):
        import prism.delivery.runner as runner_module

        bars = {"H4": self._make_h4_bars(), "H1": self._make_h4_bars(), "M5": self._make_h4_bars()}
        bridge = _FakeBridgeLive(bars)
        notifier = _StubNotifier()

        from prism.signal import generator as gen_module
        from prism.execution.mt5_bridge import SignalPacket

        class _StubGen:
            def __init__(self, instrument, persist_fvg=True):
                assert persist_fvg is True, \
                    "Live path must keep FVG retest enabled"
            def generate(self, *a, **kw):
                return SignalPacket(
                    instrument="EURUSD", direction="LONG", entry=1.1, sl=1.09,
                    tp1=1.11, tp2=1.12, rr_ratio=2.0, confidence=0.7,
                    confidence_level="MEDIUM", magnitude_pips=50.0,
                    regime="RISK_ON", news_bias="NEUTRAL",
                    fvg_zone=None, signal_time="2026-04-20T08:00:00",
                )

        from prism.data import pipeline as pipe_module

        class _StubPipe:
            def __init__(self, *a, **kw): pass
            def build_features_from_bars(self, df): return df

        monkeypatch.setattr(gen_module, "SignalGenerator", _StubGen)
        monkeypatch.setattr(pipe_module, "PRISMFeaturePipeline", _StubPipe)

        runner_module._scan_instrument(
            "EURUSD", notifier, bridge,
            datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc),
        )

        assert len(notifier.sent) == 1
        assert notifier.sent[0]["demo_warning"] is None

    def test_stale_bars_short_circuit_scan(self, monkeypatch):
        import prism.delivery.runner as runner_module

        bars = {"H4": self._make_h4_bars(), "H1": self._make_h4_bars(), "M5": self._make_h4_bars()}

        class _StaleBridge(_FakeBridgeLive):
            def bars_are_fresh(self, df, tf, now=None):
                return False

        bridge = _StaleBridge(bars)
        notifier = _StubNotifier()

        # If the scanner didn't short-circuit, it would need the generator.
        # Fail loud if generator is ever instantiated.
        from prism.signal import generator as gen_module

        def _never(*a, **kw):
            raise AssertionError("Scanner must not invoke generator when bars are stale")

        monkeypatch.setattr(gen_module, "SignalGenerator", _never)

        runner_module._scan_instrument(
            "EURUSD", notifier, bridge,
            datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc),
        )

        assert notifier.sent == []
