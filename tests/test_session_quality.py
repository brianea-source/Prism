"""Tests for the Session Quality Scorer (LRF equivalent)."""

import pytest
from unittest.mock import MagicMock
from datetime import date

from prism.signal.session_quality import (
    score_session,
    SessionGrade,
    SessionQuality,
)
from prism.signal.session_bias import AsianRange


def _news_signal(event_flag=False, event_name="", news_bias="NEUTRAL",
                 geopolitical_active=False, risk_regime="NEUTRAL"):
    ns = MagicMock()
    ns.event_flag = event_flag
    ns.event_name = event_name
    ns.news_bias = news_bias
    ns.geopolitical_active = geopolitical_active
    ns.risk_regime = risk_regime
    return ns


def _asian_range(range_pips=40.0):
    return AsianRange(
        high=3220.0, low=3200.0,
        midpoint=3210.0, range_pips=range_pips,
        bar_count=72, date=date(2026, 5, 19),
    )


# ---------------------------------------------------------------------------
# Grade classification
# ---------------------------------------------------------------------------

class TestGradeClassification:
    def test_no_events_is_favorable(self):
        sq = score_session(_news_signal())
        assert sq.grade == SessionGrade.FAVORABLE
        assert sq.score < 5

    def test_fomc_day_is_skip(self):
        sq = score_session(_news_signal(
            event_flag=True, event_name="FOMC Federal Funds Rate",
        ))
        assert sq.grade == SessionGrade.SKIP
        assert sq.score >= 10

    def test_nfp_day_is_skip(self):
        sq = score_session(_news_signal(
            event_flag=True, event_name="Nonfarm Payrolls",
        ))
        assert sq.grade == SessionGrade.SKIP

    def test_cpi_day_is_skip(self):
        sq = score_session(_news_signal(
            event_flag=True, event_name="CPI Consumer Price Index m/m",
        ))
        assert sq.grade == SessionGrade.SKIP

    def test_powell_speaking_is_skip(self):
        sq = score_session(_news_signal(
            event_flag=True, event_name="Fed Chair Powell Speaks",
        ))
        assert sq.grade == SessionGrade.SKIP

    def test_generic_high_impact_is_cautious(self):
        sq = score_session(_news_signal(
            event_flag=True, event_name="Trade Balance",
        ))
        assert sq.grade == SessionGrade.CAUTIOUS
        assert sq.score >= 5

    def test_geopolitical_adds_points(self):
        sq = score_session(_news_signal(geopolitical_active=True))
        assert sq.score == 3
        assert sq.grade == SessionGrade.FAVORABLE

    def test_geopolitical_plus_event_is_cautious(self):
        sq = score_session(_news_signal(
            event_flag=True, event_name="Trade Balance",
            geopolitical_active=True,
        ))
        assert sq.grade == SessionGrade.CAUTIOUS
        assert sq.score >= 8


# ---------------------------------------------------------------------------
# Asian range influence
# ---------------------------------------------------------------------------

class TestAsianRangeInfluence:
    def test_wide_asian_range_adds_points(self, monkeypatch):
        monkeypatch.setenv("PRISM_ASIAN_WIDE_PIPS", "60")
        sq = score_session(
            _news_signal(),
            asian_range=_asian_range(range_pips=100.0),
        )
        assert sq.score >= 3
        assert any("wide Asian" in r for r in sq.reasons)

    def test_normal_asian_range_no_penalty(self):
        sq = score_session(
            _news_signal(),
            asian_range=_asian_range(range_pips=40.0),
        )
        assert sq.score == 0

    def test_tight_asian_range_noted_not_penalized(self, monkeypatch):
        monkeypatch.setenv("PRISM_ASIAN_TIGHT_PIPS", "20")
        sq = score_session(
            _news_signal(),
            asian_range=_asian_range(range_pips=10.0),
        )
        assert sq.score == 0  # noted but no points added
        assert any("tight Asian" in r for r in sq.reasons)


# ---------------------------------------------------------------------------
# Prior session expansion
# ---------------------------------------------------------------------------

class TestPriorSessionExpansion:
    def test_large_prior_expansion_adds_points(self, monkeypatch):
        monkeypatch.setenv("PRISM_PRIOR_EXPANSION_PIPS", "100")
        sq = score_session(
            _news_signal(),
            prior_session_range_pips=200.0,
        )
        assert sq.score >= 2
        assert any("prior session" in r for r in sq.reasons)

    def test_normal_prior_session_no_penalty(self):
        sq = score_session(
            _news_signal(),
            prior_session_range_pips=50.0,
        )
        assert sq.score == 0


# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------

class TestConfidenceThresholds:
    def test_favorable_uses_base_confidence(self, monkeypatch):
        monkeypatch.setenv("PRISM_MIN_CONFIDENCE", "0.55")
        sq = score_session(_news_signal())
        assert sq.min_confidence == pytest.approx(0.55)

    def test_cautious_raises_confidence(self, monkeypatch):
        monkeypatch.setenv("PRISM_CAUTIOUS_MIN_CONFIDENCE", "0.72")
        sq = score_session(_news_signal(
            event_flag=True, event_name="ISM Manufacturing",
        ))
        assert sq.grade == SessionGrade.CAUTIOUS
        assert sq.min_confidence == pytest.approx(0.72)

    def test_skip_sets_unreachable_confidence(self):
        sq = score_session(_news_signal(
            event_flag=True, event_name="FOMC Federal Funds Rate",
        ))
        assert sq.grade == SessionGrade.SKIP
        assert sq.min_confidence == 1.0


# ---------------------------------------------------------------------------
# Env overrides
# ---------------------------------------------------------------------------

class TestEnvOverrides:
    def test_custom_skip_threshold(self, monkeypatch):
        monkeypatch.setenv("PRISM_LRF_SKIP_THRESHOLD", "15")
        sq = score_session(_news_signal(
            event_flag=True, event_name="FOMC Federal Funds Rate",
        ))
        # FOMC = 10 points, but skip threshold raised to 15
        assert sq.grade == SessionGrade.CAUTIOUS

    def test_custom_cautious_threshold(self, monkeypatch):
        monkeypatch.setenv("PRISM_LRF_CAUTIOUS_THRESHOLD", "2")
        sq = score_session(
            _news_signal(),
            prior_session_range_pips=200.0,
        )
        assert sq.grade == SessionGrade.CAUTIOUS
