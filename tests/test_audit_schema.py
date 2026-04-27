"""Schema-parity invariant tests for the audit log.

The audit record schema is a contract between three writers/readers:

  1. :mod:`prism.delivery.signal_audit` (writer — Phase 6.F)
  2. :mod:`prism.audit.smart_money_export` (reader — Phase 7.A Track B)
  3. ``prism/data/historical_state.py`` (writer — Phase 7.A impl, future)

A schema drift between any two of these silently breaks gate 5 (live vs.
historical drift comparison). PR #21 review item #1: explicitly verify
the writer + canonical schema agree, so a future SignalPacket addition
either updates both or fails this test.
"""
from __future__ import annotations

import pytest

from prism.audit.schema import (
    ALL_FIELDS,
    AUDIT_FIELDS,
    AuditSchemaError,
    HTF_BIAS_SUBKEYS,
    SMART_MONEY_OB_SUBKEYS,
    SMART_MONEY_PO3_SUBKEYS,
    SMART_MONEY_SUBKEYS,
    SMART_MONEY_SWEEP_SUBKEYS,
    TIMESTAMP_FIELD,
    validate_record,
)


class TestSchemaConstants:

    def test_audit_fields_pinned(self):
        """The 11-field whitelist is a contract — explicit pin."""
        assert AUDIT_FIELDS == (
            "instrument",
            "direction",
            "confidence",
            "confidence_level",
            "signal_id",
            "signal_time",
            "model_version",
            "regime",
            "news_bias",
            "htf_bias",
            "smart_money",
        )
        assert len(AUDIT_FIELDS) == 11

    def test_timestamp_field(self):
        assert TIMESTAMP_FIELD == "audit_ts"

    def test_all_fields_is_timestamp_plus_audit_fields(self):
        assert ALL_FIELDS == (TIMESTAMP_FIELD, *AUDIT_FIELDS)
        assert len(ALL_FIELDS) == 12

    def test_smart_money_subkeys_pinned(self):
        assert SMART_MONEY_SUBKEYS == ("ob", "sweep", "po3")

    def test_htf_bias_subkeys_pinned(self):
        assert HTF_BIAS_SUBKEYS == (
            "bias_1h", "bias_4h", "aligned", "allowed_direction",
        )

    def test_subkey_tuples_are_tuples_not_lists(self):
        """Tuples are immutable — prevents accidental mutation of the
        canonical schema by a downstream caller."""
        for name in (
            AUDIT_FIELDS, ALL_FIELDS, SMART_MONEY_SUBKEYS, HTF_BIAS_SUBKEYS,
            SMART_MONEY_OB_SUBKEYS, SMART_MONEY_SWEEP_SUBKEYS,
            SMART_MONEY_PO3_SUBKEYS,
        ):
            assert isinstance(name, tuple)


class TestSchemaParity:
    """Writer + canonical schema must reference the same constant."""

    def test_signal_audit_reexports_canonical_audit_fields(self):
        from prism.delivery import signal_audit
        from prism.audit import schema

        # Re-exported from prism.audit.schema, not redefined
        assert signal_audit.AUDIT_FIELDS is schema.AUDIT_FIELDS

    def test_signal_audit_reexports_canonical_timestamp_field(self):
        from prism.delivery import signal_audit
        from prism.audit import schema

        assert signal_audit.TIMESTAMP_FIELD is schema.TIMESTAMP_FIELD

    def test_writer_record_keys_match_all_fields(self, tmp_path, monkeypatch):
        """End-to-end: a freshly-written audit record has exactly the
        canonical top-level keys, in the canonical order."""
        import json
        from datetime import datetime, timezone

        from prism.delivery.signal_audit import write_signal_audit
        from prism.execution.mt5_bridge import SignalPacket

        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path))
        monkeypatch.setenv("PRISM_SIGNAL_AUDIT_ENABLED", "1")

        sig = SignalPacket(
            instrument="EURUSD", direction="LONG", entry=1.1, sl=1.09,
            tp1=1.11, tp2=1.12, rr_ratio=2.0, confidence=0.7,
            confidence_level="MEDIUM", magnitude_pips=50.0,
            regime="RISK_ON", news_bias="NEUTRAL", fvg_zone=None,
            signal_time="2026-04-20T08:00:00+00:00",
        )
        path = write_signal_audit(
            sig, when=datetime(2026, 4, 20, 8, 0, tzinfo=timezone.utc),
        )
        record = json.loads(path.read_text().strip())

        # Order matters too — gate-5 parquet schemas need consistent
        # column ordering across live + historical.
        assert tuple(record.keys()) == ALL_FIELDS


class TestValidateRecord:

    def _good_record(self) -> dict:
        return {
            "audit_ts": "2026-04-20T08:00:00+00:00",
            "instrument": "EURUSD",
            "direction": "LONG",
            "confidence": 0.7,
            "confidence_level": "MEDIUM",
            "signal_id": "abc",
            "signal_time": "2026-04-20T08:00:00+00:00",
            "model_version": "prism_v2.0",
            "regime": "RISK_ON",
            "news_bias": "NEUTRAL",
            "htf_bias": None,
            "smart_money": None,
        }

    def test_valid_record_passes(self):
        validate_record(self._good_record())

    def test_missing_field_raises(self):
        rec = self._good_record()
        del rec["smart_money"]
        with pytest.raises(AuditSchemaError, match="missing required fields"):
            validate_record(rec)

    def test_non_dict_raises(self):
        with pytest.raises(AuditSchemaError, match="must be a dict"):
            validate_record("not a dict")

    def test_extra_field_lenient_by_default(self):
        rec = self._good_record()
        rec["future_field_v2"] = "ignored in lenient mode"
        validate_record(rec)

    def test_extra_field_strict_raises(self):
        rec = self._good_record()
        rec["unexpected"] = "boom"
        with pytest.raises(AuditSchemaError, match="unexpected fields"):
            validate_record(rec, strict=True)
