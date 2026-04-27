"""Phase 6.F — per-signal structured audit log.

Covers:
  * env-gate semantics (default-on, "0" / "false" disable)
  * daily rotation by UTC date
  * per-instrument file isolation
  * PRISM_STATE_DIR override is respected
  * disk-failure / serialization-failure does NOT propagate
  * round-trip: written records can be re-parsed and contain the
    whitelisted SignalPacket subset
  * runner wire-in: _scan_instrument produces an audit row only after
    the dedup guard passes (signal-producing bars only)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np
import pytest

from prism.delivery.signal_audit import (
    AUDIT_FIELDS,
    audit_path,
    write_signal_audit,
)
from prism.execution.mt5_bridge import SignalPacket


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_signal(
    instrument: str = "EURUSD",
    direction: str = "LONG",
    *,
    smart_money: dict | None = None,
    htf_bias: dict | None = None,
) -> SignalPacket:
    return SignalPacket(
        instrument=instrument,
        direction=direction,
        entry=1.1000,
        sl=1.0950,
        tp1=1.1050,
        tp2=1.1100,
        rr_ratio=2.0,
        confidence=0.72,
        confidence_level="MEDIUM",
        magnitude_pips=50.0,
        regime="RISK_ON",
        news_bias="NEUTRAL",
        fvg_zone=None,
        signal_time="2026-04-20T08:00:00+00:00",
        htf_bias=htf_bias,
        smart_money=smart_money,
    )


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PRISM_SIGNAL_AUDIT_ENABLED", "1")
    return tmp_path


# ---------------------------------------------------------------------------
# Module-level behaviour
# ---------------------------------------------------------------------------

class TestAuditWriter:

    def test_enabled_writes_jsonl_record(self, state_dir):
        sig = _make_signal()
        when = datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc)

        path = write_signal_audit(sig, when=when)

        assert path is not None
        assert path.exists()
        assert path == state_dir / "signal_audit" / "EURUSD" / "2026-04-20.jsonl"

        record = json.loads(path.read_text().strip())
        assert record["instrument"] == "EURUSD"
        assert record["direction"] == "LONG"
        assert record["confidence"] == pytest.approx(0.72)
        assert record["audit_ts"] == "2026-04-20T08:00:00+00:00"
        # Whitelisted fields all present (htf_bias / smart_money may be None)
        for field in AUDIT_FIELDS:
            assert field in record

    def test_disabled_skips_write(self, state_dir, monkeypatch):
        monkeypatch.setenv("PRISM_SIGNAL_AUDIT_ENABLED", "0")
        sig = _make_signal()

        result = write_signal_audit(sig)

        assert result is None
        assert not (state_dir / "signal_audit").exists()

    @pytest.mark.parametrize("falsy", ["0", "false", "FALSE", "no", "off", ""])
    def test_disabled_recognizes_common_falsy_values(
        self, state_dir, monkeypatch, falsy
    ):
        monkeypatch.setenv("PRISM_SIGNAL_AUDIT_ENABLED", falsy)
        assert write_signal_audit(_make_signal()) is None

    def test_daily_rotation_uses_utc_date(self, state_dir):
        sig = _make_signal()
        d1 = datetime(2026, 4, 20, 23, 59, tzinfo=timezone.utc)
        d2 = datetime(2026, 4, 21, 0, 1, tzinfo=timezone.utc)

        p1 = write_signal_audit(sig, when=d1)
        p2 = write_signal_audit(sig, when=d2)

        assert p1 != p2
        assert p1.name == "2026-04-20.jsonl"
        assert p2.name == "2026-04-21.jsonl"

    def test_per_instrument_isolation(self, state_dir):
        when = datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc)
        p_xau = write_signal_audit(_make_signal("XAUUSD"), when=when)
        p_eur = write_signal_audit(_make_signal("EURUSD"), when=when)

        assert p_xau.parent.name == "XAUUSD"
        assert p_eur.parent.name == "EURUSD"
        assert p_xau != p_eur
        # Each file holds exactly one record
        assert len(p_xau.read_text().strip().splitlines()) == 1
        assert len(p_eur.read_text().strip().splitlines()) == 1

    def test_appends_multiple_records_to_same_day(self, state_dir):
        when = datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc)
        write_signal_audit(_make_signal(), when=when)
        write_signal_audit(_make_signal(direction="SHORT"), when=when)

        path = audit_path("EURUSD", when=when)
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["direction"] == "LONG"
        assert json.loads(lines[1])["direction"] == "SHORT"

    def test_respects_state_dir_env(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom_state"
        monkeypatch.setenv("PRISM_STATE_DIR", str(custom))
        monkeypatch.setenv("PRISM_SIGNAL_AUDIT_ENABLED", "1")

        path = write_signal_audit(
            _make_signal(),
            when=datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc),
        )

        assert path is not None
        assert custom in path.parents

    def test_disk_failure_does_not_raise(self, state_dir, caplog):
        sig = _make_signal()
        with patch(
            "prism.delivery.signal_audit.open",
            side_effect=OSError("disk full"),
        ):
            with caplog.at_level("ERROR", logger="prism.delivery.signal_audit"):
                result = write_signal_audit(sig)

        assert result is None
        assert any(
            "failed to persist" in rec.message
            for rec in caplog.records
        )

    def test_handles_non_serializable_values(self, state_dir):
        # Smart-money dicts can carry numpy scalars / pandas Timestamps
        # straight from the detectors before any explicit coercion.
        smart_money = {
            "ob": {
                "high": np.float64(1.1234),
                "low": np.float64(1.1200),
                "midpoint": np.float64(1.1217),
                "distance_pips": np.float32(12.5),
                "formed_at": pd.Timestamp("2026-04-20T04:00:00", tz="UTC"),
            },
            "sweep": None,
            "po3": None,
        }
        sig = _make_signal(smart_money=smart_money)
        when = datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc)

        path = write_signal_audit(sig, when=when)

        assert path is not None and path.exists()
        record = json.loads(path.read_text().strip())
        # default=str coerces non-serializable values without losing them
        assert record["smart_money"]["ob"]["high"] in ("1.1234", 1.1234)
        assert "2026-04-20" in str(record["smart_money"]["ob"]["formed_at"])

    def test_round_trip_preserves_whitelisted_subset(self, state_dir):
        smart_money = {
            "ob": None,
            "sweep": {"type": "BUY_SIDE", "bars_ago": 3},
            "po3": {"phase": "MANIPULATION", "is_entry_phase": True},
        }
        htf_bias = {
            "bias_1h": "BULLISH",
            "bias_4h": "BULLISH",
            "aligned": True,
            "allowed_direction": "LONG",
        }
        sig = _make_signal(smart_money=smart_money, htf_bias=htf_bias)
        when = datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc)

        path = write_signal_audit(sig, when=when)
        record = json.loads(path.read_text().strip())

        assert record["htf_bias"] == htf_bias
        assert record["smart_money"] == smart_money
        assert record["signal_id"] == sig.signal_id
        assert record["signal_time"] == sig.signal_time
        # Fields NOT in the whitelist must not leak into the audit
        assert "entry" not in record
        assert "sl" not in record
        assert "tp1" not in record
        assert "rr_ratio" not in record


# ---------------------------------------------------------------------------
# Runner wire-in
# ---------------------------------------------------------------------------

class _StubBridgeDemo:
    """Minimal demo-mode bridge — mirrors the pattern in
    test_phase4_live_bars.py::_FakeBridgeDemo. Intentionally omits
    ``ensure_connected`` so the runner's ``getattr(..., None)`` guard
    treats the bridge as always-connected (legacy-stub semantics)."""

    mode = "NOTIFY"

    def __init__(self, df):
        self._df = df

    def supports_live_bars(self):
        return False

    def get_bars(self, instrument, timeframe, count=500):
        return self._df


class _StubNotifier:
    def __init__(self):
        self.sent = []
        self.channel = "#t"
        self.client = object()
        self.confirm_timeout_sec = 300

    def send_signal(self, signal, mode="CONFIRM", use_buttons=False, demo_warning=None):
        self.sent.append({"signal": signal, "demo_warning": demo_warning})
        return "1234567890.000001"

    def update_signal_status(self, *a, **kw):
        pass

    def send_alert(self, *a, **kw):
        pass


def _make_h4_bars(n=300):
    dates = pd.date_range("2026-04-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({
        "datetime": dates,
        "open": np.linspace(1.10, 1.20, n),
        "high": np.linspace(1.105, 1.205, n),
        "low": np.linspace(1.095, 1.195, n),
        "close": np.linspace(1.10, 1.20, n),
        "volume": np.full(n, 200),
    })


class TestRunnerWireIn:

    def test_audit_written_after_dedup(self, tmp_path, monkeypatch):
        """A signal that passes _should_fire produces exactly one audit row."""
        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path))
        monkeypatch.setenv("PRISM_SIGNAL_AUDIT_ENABLED", "1")

        import prism.delivery.runner as runner_module
        from prism.signal import generator as gen_module

        sig = _make_signal("EURUSD", "LONG")

        class _StubGen:
            def __init__(self, instrument, persist_fvg=True):
                pass

            def generate(self, *a, **kw):
                return sig

        monkeypatch.setattr(gen_module, "SignalGenerator", _StubGen)
        # Reset in-flight dedup state so this test isn't flaky against
        # other tests in the same process.
        runner_module._last_signal_key.clear()

        bridge = _StubBridgeDemo(_make_h4_bars())
        notifier = _StubNotifier()

        runner_module._scan_instrument(
            "EURUSD", notifier, bridge,
            datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc),
        )

        # Exactly one signal sent + one audit row written
        assert len(notifier.sent) == 1
        audit_files = list((tmp_path / "signal_audit" / "EURUSD").glob("*.jsonl"))
        assert len(audit_files) == 1
        records = [
            json.loads(line)
            for line in audit_files[0].read_text().splitlines()
            if line.strip()
        ]
        assert len(records) == 1
        assert records[0]["signal_id"] == sig.signal_id

    def test_audit_skipped_when_dedup_suppresses(self, tmp_path, monkeypatch):
        """Re-scanning the same H4 bar produces no additional audit row."""
        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path))
        monkeypatch.setenv("PRISM_SIGNAL_AUDIT_ENABLED", "1")

        import prism.delivery.runner as runner_module
        from prism.signal import generator as gen_module

        # Each generate() call returns a *fresh* packet (new signal_id) —
        # _should_fire must dedup based on (instrument, direction, h4_bar).
        class _StubGen:
            def __init__(self, instrument, persist_fvg=True):
                pass

            def generate(self, *a, **kw):
                return _make_signal("EURUSD", "LONG")

        monkeypatch.setattr(gen_module, "SignalGenerator", _StubGen)
        runner_module._last_signal_key.clear()

        bridge = _StubBridgeDemo(_make_h4_bars())
        notifier = _StubNotifier()
        scan_at = datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc)

        runner_module._scan_instrument("EURUSD", notifier, bridge, scan_at)
        runner_module._scan_instrument("EURUSD", notifier, bridge, scan_at)

        assert len(notifier.sent) == 1  # second call dedup'd
        records = [
            json.loads(line)
            for line in (
                tmp_path / "signal_audit" / "EURUSD" / "2026-04-20.jsonl"
            ).read_text().splitlines()
            if line.strip()
        ]
        assert len(records) == 1, "dedup-suppressed scan must not write audit"

    def test_audit_skipped_when_no_signal(self, tmp_path, monkeypatch):
        """generate() returning None produces no audit row."""
        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path))
        monkeypatch.setenv("PRISM_SIGNAL_AUDIT_ENABLED", "1")

        import prism.delivery.runner as runner_module
        from prism.signal import generator as gen_module

        class _StubGen:
            def __init__(self, instrument, persist_fvg=True):
                pass

            def generate(self, *a, **kw):
                return None

        monkeypatch.setattr(gen_module, "SignalGenerator", _StubGen)
        runner_module._last_signal_key.clear()

        bridge = _StubBridgeDemo(_make_h4_bars())
        notifier = _StubNotifier()
        runner_module._scan_instrument(
            "EURUSD", notifier, bridge,
            datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc),
        )

        assert len(notifier.sent) == 0
        assert not (tmp_path / "signal_audit").exists()
