"""Phase 7.A feature engineering tests.

Covers the five buildable-now ICT features and the DataFrame-level
``enrich_features`` integration. Targets the same audit-log shape that
``read_audit_window`` / ``signal_audit.write_signal_audit`` produce, so
the enricher can be exercised against real Stage 1 data without
reshaping.

Each individual feature derivation is unit tested at boundaries where
the spec is explicit:

  * ``compute_htf_alignment`` — every (bias_1h, bias_4h, direction)
    combination that maps to a distinct score, plus the both-neutral
    edge case the spec table doesn't enumerate.
  * ``compute_kill_zone_strength`` — every hour partition (Asian,
    London core/edge, NY core/edge, off-session).
  * ``compute_sweep_confirmed`` — the ``qualifies`` flag, missing
    sweep, malformed dict.
  * ``compute_ob_distance_features`` — sentinel encoding, threshold
    boundary (= and < and > the threshold), missing OB, malformed.
  * ``compute_po3_phase_features`` — exactly-one-True invariant on the
    one-hot, unknown-fallback for missing/malformed input.
"""
from __future__ import annotations

import pandas as pd
import pytest

from prism.data.feature_engineering import (
    ICTFeatureEngineer,
    PHASE_7A_FEATURE_COLUMNS,
    PO3_PHASES,
    compute_htf_alignment,
    compute_kill_zone_strength,
    compute_ob_distance_features,
    compute_po3_phase_features,
    compute_sweep_confirmed,
    enrich_features,
)
from prism.signal.htf_bias import Bias


# ---------------------------------------------------------------------------
# compute_htf_alignment
# ---------------------------------------------------------------------------


class TestHtfAlignment:

    def test_both_aligned_long(self):
        assert compute_htf_alignment(Bias.BULLISH, Bias.BULLISH, "LONG") == 3

    def test_both_aligned_short(self):
        assert compute_htf_alignment(Bias.BEARISH, Bias.BEARISH, "SHORT") == 3

    def test_one_aligned_one_neutral(self):
        assert compute_htf_alignment(Bias.BULLISH, Bias.RANGING, "LONG") == 2
        assert compute_htf_alignment(Bias.RANGING, Bias.BEARISH, "SHORT") == 2

    def test_one_against_one_aligned(self):
        # 1 against, 1 aligned → 1 (per the "one against" rule)
        assert compute_htf_alignment(Bias.BULLISH, Bias.BEARISH, "LONG") == 1
        assert compute_htf_alignment(Bias.BEARISH, Bias.BULLISH, "SHORT") == 1

    def test_one_against_one_neutral(self):
        assert compute_htf_alignment(Bias.BULLISH, Bias.RANGING, "SHORT") == 1
        assert compute_htf_alignment(Bias.RANGING, Bias.BEARISH, "LONG") == 1

    def test_both_against(self):
        assert compute_htf_alignment(Bias.BEARISH, Bias.BEARISH, "LONG") == 0
        assert compute_htf_alignment(Bias.BULLISH, Bias.BULLISH, "SHORT") == 0

    def test_both_neutral_returns_one(self):
        # Spec table doesn't enumerate this — documented as 1 in the
        # docstring (same tier as one against, since neither timeframe
        # supports the trade).
        assert compute_htf_alignment(Bias.RANGING, Bias.RANGING, "LONG") == 1

    def test_string_inputs_accepted(self):
        # Audit log dicts persist Bias as a string ("BULLISH"), not the
        # enum — the function must accept both.
        assert compute_htf_alignment("BULLISH", "BULLISH", "LONG") == 3
        assert compute_htf_alignment("bearish", "ranging", "short") == 2

    def test_buy_sell_normalised_to_long_short(self):
        assert compute_htf_alignment("BULLISH", "BULLISH", "BUY") == 3
        assert compute_htf_alignment("BEARISH", "BEARISH", "SELL") == 3

    def test_unknown_direction_raises(self):
        with pytest.raises(ValueError, match="LONG/SHORT"):
            compute_htf_alignment("BULLISH", "BULLISH", "FLAT")

    def test_none_bias_treated_as_neutral(self):
        # Audit log can have htf_bias=None when HTF gate is off
        assert compute_htf_alignment(None, None, "LONG") == 1
        assert compute_htf_alignment(Bias.BULLISH, None, "LONG") == 2

    def test_unknown_bias_string_treated_as_neutral(self):
        # Forward-compat: a future Bias value lands without breaking enrich
        assert compute_htf_alignment("FUTURE_BIAS", Bias.BULLISH, "LONG") == 2


# ---------------------------------------------------------------------------
# compute_kill_zone_strength
# ---------------------------------------------------------------------------


class TestKillZoneStrength:

    @pytest.mark.parametrize("hour", [8, 9, 13, 14])
    def test_core_zones(self, hour):
        assert compute_kill_zone_strength(hour) == 3

    @pytest.mark.parametrize("hour", [7, 10, 12, 15])
    def test_edge_zones(self, hour):
        assert compute_kill_zone_strength(hour) == 2

    @pytest.mark.parametrize("hour", [0, 1, 2, 3])
    def test_asian_zone(self, hour):
        assert compute_kill_zone_strength(hour) == 1

    @pytest.mark.parametrize("hour", [4, 5, 6, 11, 16, 17, 18, 19, 20, 21, 22, 23])
    def test_off_session(self, hour):
        assert compute_kill_zone_strength(hour) == 0

    def test_hour_normalised_modulo_24(self):
        # Defensive — caller may pass a 25 or -1 from a buggy timestamp parse
        assert compute_kill_zone_strength(32) == compute_kill_zone_strength(8)
        assert compute_kill_zone_strength(-16) == compute_kill_zone_strength(8)


# ---------------------------------------------------------------------------
# compute_sweep_confirmed
# ---------------------------------------------------------------------------


class TestSweepConfirmed:

    def test_qualifies_true(self):
        sm = {"sweep": {"qualifies": True, "displacement_followed": True}}
        assert compute_sweep_confirmed(sm) is True

    def test_qualifies_false(self):
        sm = {"sweep": {"qualifies": False, "displacement_followed": True}}
        assert compute_sweep_confirmed(sm) is False

    def test_missing_qualifies_treated_as_false(self):
        sm = {"sweep": {"displacement_followed": True}}
        assert compute_sweep_confirmed(sm) is False

    def test_no_sweep_subkey(self):
        assert compute_sweep_confirmed({"ob": {}, "po3": {}}) is False

    def test_sweep_is_none(self):
        assert compute_sweep_confirmed({"sweep": None}) is False

    def test_smart_money_none(self):
        assert compute_sweep_confirmed(None) is False

    def test_smart_money_not_a_dict(self):
        # JSON garbage that escaped the validator
        assert compute_sweep_confirmed("garbage") is False
        assert compute_sweep_confirmed(42) is False


# ---------------------------------------------------------------------------
# compute_ob_distance_features
# ---------------------------------------------------------------------------


class TestObDistanceFeatures:

    def test_in_range(self):
        sm = {"ob": {"distance_pips": 12.5}}
        d, in_r = compute_ob_distance_features(sm, ob_max_distance_pips=30.0)
        assert d == 12.5
        assert in_r is True

    def test_at_threshold_boundary_inclusive(self):
        sm = {"ob": {"distance_pips": 30.0}}
        d, in_r = compute_ob_distance_features(sm, ob_max_distance_pips=30.0)
        assert d == 30.0
        assert in_r is True  # `<=` not `<`

    def test_just_over_threshold(self):
        sm = {"ob": {"distance_pips": 30.0001}}
        d, in_r = compute_ob_distance_features(sm, ob_max_distance_pips=30.0)
        assert d == pytest.approx(30.0001)
        assert in_r is False

    def test_far_out_of_range(self):
        sm = {"ob": {"distance_pips": 200.0}}
        d, in_r = compute_ob_distance_features(sm, ob_max_distance_pips=30.0)
        assert d == 200.0
        assert in_r is False

    def test_no_ob_returns_sentinel(self):
        d, in_r = compute_ob_distance_features({"ob": None}, ob_max_distance_pips=30.0)
        assert d == -1.0
        assert in_r is False

    def test_distance_pips_none_returns_sentinel(self):
        # OB present but detector returned no distance (e.g. invalid price)
        sm = {"ob": {"distance_pips": None, "midpoint": 1.05}}
        d, in_r = compute_ob_distance_features(sm, ob_max_distance_pips=30.0)
        assert d == -1.0
        assert in_r is False

    def test_negative_distance_treated_as_invalid(self):
        # Detector contract: distance_pips is non-negative; defend anyway
        sm = {"ob": {"distance_pips": -5.0}}
        d, in_r = compute_ob_distance_features(sm, ob_max_distance_pips=30.0)
        assert d == -1.0
        assert in_r is False

    def test_smart_money_none(self):
        d, in_r = compute_ob_distance_features(None, ob_max_distance_pips=30.0)
        assert d == -1.0
        assert in_r is False

    def test_threshold_lock_in_changes_in_range_only(self):
        """The threshold is the lock-in surface — the raw distance must
        not change with the threshold, only the bool."""
        sm = {"ob": {"distance_pips": 25.0}}
        d_30, in_30 = compute_ob_distance_features(sm, ob_max_distance_pips=30.0)
        d_20, in_20 = compute_ob_distance_features(sm, ob_max_distance_pips=20.0)
        assert d_30 == d_20 == 25.0
        assert in_30 is True
        assert in_20 is False


# ---------------------------------------------------------------------------
# compute_po3_phase_features
# ---------------------------------------------------------------------------


class TestPo3PhaseFeatures:

    @pytest.mark.parametrize("phase", ["ACCUMULATION", "MANIPULATION", "DISTRIBUTION", "UNKNOWN"])
    def test_each_phase_one_hot(self, phase):
        sm = {"po3": {"phase": phase}}
        label, one_hot = compute_po3_phase_features(sm)
        assert label == phase.lower()
        # Exactly one True
        assert sum(one_hot.values()) == 1
        # The True one matches the label
        assert one_hot[f"po3_{phase.lower()}"] is True

    def test_lowercase_input_accepted(self):
        sm = {"po3": {"phase": "manipulation"}}
        label, _ = compute_po3_phase_features(sm)
        assert label == "manipulation"

    def test_one_hot_keys_stable_order(self):
        sm = {"po3": {"phase": "ACCUMULATION"}}
        _, one_hot = compute_po3_phase_features(sm)
        assert tuple(one_hot.keys()) == tuple(f"po3_{p}" for p in PO3_PHASES)

    def test_no_po3_subkey_falls_back_to_unknown(self):
        label, one_hot = compute_po3_phase_features({"ob": {}, "sweep": {}})
        assert label == "unknown"
        assert one_hot["po3_unknown"] is True
        assert sum(one_hot.values()) == 1

    def test_unknown_phase_string_falls_back_to_unknown(self):
        # Forward-compat — a future Po3Phase value
        sm = {"po3": {"phase": "FUTURE_PHASE"}}
        label, one_hot = compute_po3_phase_features(sm)
        assert label == "unknown"
        assert one_hot["po3_unknown"] is True

    def test_po3_none(self):
        label, _ = compute_po3_phase_features({"po3": None})
        assert label == "unknown"

    def test_smart_money_none(self):
        label, _ = compute_po3_phase_features(None)
        assert label == "unknown"


# ---------------------------------------------------------------------------
# enrich_features (DataFrame-level integration)
# ---------------------------------------------------------------------------


def _audit_row(
    *,
    audit_ts: str = "2026-04-20T08:30:00+00:00",
    direction: str = "LONG",
    htf_bias: dict | None = None,
    smart_money: dict | None = None,
) -> dict:
    """A canonical audit-log-shaped row for enrich_features tests."""
    return {
        "audit_ts": audit_ts,
        "instrument": "EURUSD",
        "direction": direction,
        "confidence": 0.7,
        "confidence_level": "MEDIUM",
        "signal_id": "test-id",
        "signal_time": audit_ts,
        "model_version": "prism_v2.0",
        "regime": "RISK_ON",
        "news_bias": "NEUTRAL",
        "htf_bias": htf_bias or {"bias_1h": "BULLISH", "bias_4h": "BULLISH"},
        "smart_money": smart_money,
    }


class TestEnrichFeatures:

    def test_all_columns_added_in_stable_order(self):
        df = pd.DataFrame([_audit_row()])
        out = enrich_features(df, ob_max_distance_pips=30.0)
        for col in PHASE_7A_FEATURE_COLUMNS:
            assert col in out.columns, f"missing feature column: {col}"

    def test_aligned_long_in_london_core_with_smart_money(self):
        sm = {
            "ob": {"distance_pips": 8.5},
            "sweep": {"qualifies": True, "displacement_followed": True},
            "po3": {"phase": "MANIPULATION"},
        }
        df = pd.DataFrame([
            _audit_row(
                audit_ts="2026-04-20T08:30:00+00:00",
                htf_bias={"bias_1h": "BULLISH", "bias_4h": "BULLISH"},
                smart_money=sm,
            ),
        ])
        out = enrich_features(df, ob_max_distance_pips=30.0)

        assert out["htf_alignment"].iloc[0] == 3
        assert out["kill_zone_strength"].iloc[0] == 3
        assert out["sweep_confirmed"].iloc[0] is True or out["sweep_confirmed"].iloc[0] == True  # noqa: E712
        assert out["ob_distance_pips"].iloc[0] == 8.5
        assert out["ob_in_range"].iloc[0]
        assert out["po3_phase"].iloc[0] == "manipulation"
        assert out["po3_manipulation"].iloc[0]
        assert not out["po3_unknown"].iloc[0]

    def test_off_session_no_smart_money(self):
        df = pd.DataFrame([
            _audit_row(
                audit_ts="2026-04-20T20:00:00+00:00",  # off-session
                htf_bias={"bias_1h": "RANGING", "bias_4h": "RANGING"},
                smart_money=None,
            ),
        ])
        out = enrich_features(df, ob_max_distance_pips=30.0)

        assert out["htf_alignment"].iloc[0] == 1  # both neutral
        assert out["kill_zone_strength"].iloc[0] == 0
        assert not out["sweep_confirmed"].iloc[0]
        assert out["ob_distance_pips"].iloc[0] == -1.0
        assert not out["ob_in_range"].iloc[0]
        assert out["po3_phase"].iloc[0] == "unknown"
        assert out["po3_unknown"].iloc[0]

    def test_empty_dataframe_returns_empty_with_columns(self):
        df = pd.DataFrame(columns=[
            "audit_ts", "direction", "htf_bias", "smart_money",
        ])
        out = enrich_features(df, ob_max_distance_pips=30.0)
        assert len(out) == 0
        for col in PHASE_7A_FEATURE_COLUMNS:
            assert col in out.columns

    def test_missing_input_columns_use_defaults(self):
        # An audit row with no htf_bias / smart_money fields at all
        df = pd.DataFrame([
            {
                "audit_ts": "2026-04-20T13:30:00+00:00",  # NY core
                "direction": "LONG",
            },
        ])
        out = enrich_features(df, ob_max_distance_pips=30.0)
        assert out["htf_alignment"].iloc[0] == 1  # both None → neutral
        assert out["kill_zone_strength"].iloc[0] == 3
        assert not out["sweep_confirmed"].iloc[0]
        assert out["ob_distance_pips"].iloc[0] == -1.0

    def test_unknown_direction_does_not_crash(self):
        df = pd.DataFrame([_audit_row(direction="FLAT")])
        out = enrich_features(df, ob_max_distance_pips=30.0)
        # Stamps the neutral-tier score (1)
        assert out["htf_alignment"].iloc[0] == 1

    def test_threshold_lock_in_overrides_audit_in_range(self):
        """If the audit log was written with PRISM_OB_MAX_DISTANCE_PIPS=50
        but training locked 30, the feature must reflect 30 (not the
        stale persisted ``in_range`` flag). Lock-in correctness check."""
        sm_audit = {
            "ob": {"distance_pips": 40.0, "in_range": True},  # written under env=50
        }
        df = pd.DataFrame([_audit_row(smart_money=sm_audit)])
        out = enrich_features(df, ob_max_distance_pips=30.0)
        assert out["ob_distance_pips"].iloc[0] == 40.0
        assert not out["ob_in_range"].iloc[0]  # NOT True from the audit dict

    def test_does_not_mutate_input_frame(self):
        df = pd.DataFrame([_audit_row()])
        before_cols = list(df.columns)
        _ = enrich_features(df, ob_max_distance_pips=30.0)
        assert list(df.columns) == before_cols


# ---------------------------------------------------------------------------
# ICTFeatureEngineer
# ---------------------------------------------------------------------------


class TestICTFeatureEngineer:

    def test_default_constructor(self):
        eng = ICTFeatureEngineer()
        assert eng.ob_max_distance_pips == 30.0

    def test_custom_threshold(self):
        eng = ICTFeatureEngineer(ob_max_distance_pips=20.0)
        assert eng.ob_max_distance_pips == 20.0

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            ICTFeatureEngineer(ob_max_distance_pips=0)
        with pytest.raises(ValueError):
            ICTFeatureEngineer(ob_max_distance_pips=-5)

    def test_from_env_default(self, monkeypatch):
        monkeypatch.delenv("PRISM_OB_MAX_DISTANCE_PIPS", raising=False)
        eng = ICTFeatureEngineer.from_env()
        assert eng.ob_max_distance_pips == 30.0

    def test_from_env_uses_env_value(self, monkeypatch):
        monkeypatch.setenv("PRISM_OB_MAX_DISTANCE_PIPS", "42.5")
        eng = ICTFeatureEngineer.from_env()
        assert eng.ob_max_distance_pips == 42.5

    def test_from_env_with_unparseable_value_falls_back(self, monkeypatch, caplog):
        monkeypatch.setenv("PRISM_OB_MAX_DISTANCE_PIPS", "garbage")
        with caplog.at_level("WARNING", logger="prism.data.feature_engineering"):
            eng = ICTFeatureEngineer.from_env()
        assert eng.ob_max_distance_pips == 30.0
        assert any(
            "could not be parsed" in r.message for r in caplog.records
        )

    def test_enrich_uses_locked_threshold(self):
        """Engineer instance keeps the threshold stable across calls —
        a critical property for the sidecar lock-in to mean anything."""
        eng = ICTFeatureEngineer(ob_max_distance_pips=20.0)
        sm = {"ob": {"distance_pips": 25.0}}
        df = pd.DataFrame([_audit_row(smart_money=sm)])
        out = eng.enrich(df)
        assert not out["ob_in_range"].iloc[0]  # 25 > 20

        eng2 = ICTFeatureEngineer(ob_max_distance_pips=30.0)
        out2 = eng2.enrich(df)
        assert out2["ob_in_range"].iloc[0]  # 25 <= 30

    def test_feature_columns_excludes_label_string(self):
        """The string ``po3_phase`` label is for gate-5, not for the
        ML feature matrix. ``feature_columns`` must drop it."""
        eng = ICTFeatureEngineer()
        cols = eng.feature_columns()
        assert "po3_phase" not in cols
        # All four one-hots ARE there
        for p in PO3_PHASES:
            assert f"po3_{p}" in cols
        # And the four other features
        for col in (
            "htf_alignment", "kill_zone_strength", "sweep_confirmed",
            "ob_distance_pips", "ob_in_range",
        ):
            assert col in cols
