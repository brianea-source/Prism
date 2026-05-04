"""Tests for prism.journal.github_issues — GitHub Issue trade journal.

The GitHub side is mocked at the subprocess.run boundary so the suite is
hermetic: no network, no token required. We pin gh binary detection via
``_set_gh_bin_for_tests`` so the same tests run on machines with and
without ``gh`` on ``PATH``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from prism.execution.mt5_bridge import SignalPacket
from prism.journal import github_issues


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_packet(
    instrument: str = "XAUUSD",
    direction: str = "LONG",
    *,
    signal_id: str = "test-signal-001",
    confidence_level: str = "HIGH",
    confidence: float = 0.82,
    smart_money: dict | None = None,
    htf_bias: dict | None = None,
) -> SignalPacket:
    return SignalPacket(
        instrument=instrument,
        direction=direction,
        entry=2345.20,
        sl=2342.00,
        tp1=2348.40,
        tp2=2351.60,
        rr_ratio=2.0,
        confidence=confidence,
        confidence_level=confidence_level,
        magnitude_pips=320.0,
        regime="RISK_ON",
        news_bias="NEUTRAL",
        fvg_zone={"high": 2346.0, "low": 2344.0},
        signal_time="2026-04-20T08:30:00+00:00",  # London kill zone
        signal_id=signal_id,
        htf_bias=htf_bias,
        smart_money=smart_money,
    )


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Isolate PRISM_STATE_DIR per test so journal_map.json stays scoped."""
    monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("PRISM_EXECUTION_MODE", raising=False)
    return tmp_path


@pytest.fixture
def gh_mock(monkeypatch):
    """Force the gh-CLI code path and capture every subprocess.run call.

    Returns the MagicMock so individual tests can configure return values
    and assert on call arguments.
    """
    github_issues._set_gh_bin_for_tests("/usr/local/bin/gh")  # any non-None
    mock = MagicMock()
    # Default: succeed, with an issue-create-style URL on stdout. Tests
    # that exercise other commands override side_effect.
    mock.return_value = SimpleNamespace(
        returncode=0,
        stdout="https://github.com/brianea-source/Prism/issues/42\n",
        stderr="",
    )
    monkeypatch.setattr(github_issues.subprocess, "run", mock)
    yield mock
    github_issues._set_gh_bin_for_tests(None)


# ---------------------------------------------------------------------------
# Body / labels rendering — pure functions, no GitHub calls
# ---------------------------------------------------------------------------

class TestRendering:
    def test_title_format(self):
        p = _make_packet(instrument="XAUUSD", direction="LONG")
        assert github_issues._build_title(p) == "[trade] XAUUSD LONG @ 2,345.20"

    def test_title_format_fx(self):
        p = _make_packet(instrument="EURUSD", direction="SHORT")
        p.entry = 1.08234
        assert github_issues._build_title(p) == "[trade] EURUSD SHORT @ 1.08234"

    def test_body_contains_signal_id(self):
        p = _make_packet(signal_id="abc-123")
        body = github_issues._render_body(p)
        assert "signal_id:** `abc-123`" in body

    def test_body_includes_levels(self):
        p = _make_packet()
        body = github_issues._render_body(p)
        assert "**entry:** 2,345.20" in body
        assert "**stop loss:** 2,342.00" in body
        assert "**TP1:** 2,348.40" in body
        assert "**TP2:** 2,351.60" in body
        assert "**R:R:** 2.00" in body

    def test_body_includes_smart_money_block(self):
        sm = {
            "ob": {
                "direction": "bullish",
                "midpoint": 2340.50,
                "distance_pips": 12,
                "timeframe": "H1",
            },
            "sweep": {
                "type": "sell-side",
                "swept_level": 2338.10,
                "bars_ago": 3,
                "displacement_followed": True,
            },
            "po3": {
                "phase": "manipulation",
                "session": "NY",
                "range_size_pips": 45,
                "is_entry_phase": True,
            },
        }
        p = _make_packet(smart_money=sm)
        body = github_issues._render_body(p)
        assert "Smart Money Confluence" in body
        assert "bullish" in body
        assert "sell-side" in body
        assert "manipulation" in body

    def test_body_includes_htf_bias_block(self):
        htf = {"bias_4h": "bullish", "bias_1h": "bullish", "aligned": True}
        p = _make_packet(htf_bias=htf)
        body = github_issues._render_body(p)
        assert "HTF Bias" in body
        assert "bias_4h" in body

    def test_labels_pending_phase(self):
        p = _make_packet()
        labels = github_issues._labels_for_signal(p)
        assert "phase:pending" in labels
        assert "inst:XAUUSD" in labels
        assert "dir:LONG" in labels

    def test_labels_session_inferred(self):
        # 08:30 UTC = London kill zone
        p = _make_packet()
        labels = github_issues._labels_for_signal(p)
        assert "session:london" in labels

    def test_labels_quality_from_confidence_level(self):
        a = _make_packet(confidence_level="HIGH")
        b = _make_packet(confidence_level="MEDIUM")
        c = _make_packet(confidence_level="LOW")
        assert "quality:A" in github_issues._labels_for_signal(a)
        assert "quality:B" in github_issues._labels_for_signal(b)
        assert "quality:C" in github_issues._labels_for_signal(c)

    def test_labels_quality_falls_back_to_confidence(self):
        # No confidence_level → use numeric.
        p = _make_packet(confidence_level="")
        p.confidence = 0.40
        assert "quality:C" in github_issues._labels_for_signal(p)
        p.confidence = 0.60
        assert "quality:B" in github_issues._labels_for_signal(p)
        p.confidence = 0.80
        assert "quality:A" in github_issues._labels_for_signal(p)

    def test_labels_mode_from_env(self, monkeypatch):
        monkeypatch.setenv("PRISM_EXECUTION_MODE", "AUTO")
        p = _make_packet()
        assert "mode:auto" in github_issues._labels_for_signal(p)

    def test_classify_outcome_breakeven_within_band(self):
        # |pnl| < 10% of risk → breakeven
        assert github_issues._classify_outcome(2.0, risk_usd=100.0) == "breakeven"
        assert github_issues._classify_outcome(20.0, risk_usd=100.0) == "win"
        assert github_issues._classify_outcome(-50.0, risk_usd=100.0) == "loss"

    def test_classify_outcome_default_dollar_band(self):
        # Without risk_usd, |pnl| < $1 → breakeven
        assert github_issues._classify_outcome(0.5, risk_usd=None) == "breakeven"
        assert github_issues._classify_outcome(1.5, risk_usd=None) == "win"


# ---------------------------------------------------------------------------
# State file round-trip
# ---------------------------------------------------------------------------

class TestStateRoundTrip:
    def test_store_and_load(self, state_dir):
        github_issues._store_mapping("sig-1", 101)
        github_issues._store_mapping("sig-2", 202)
        loaded = github_issues._load_map()
        assert loaded == {"sig-1": 101, "sig-2": 202}

    def test_load_missing_returns_empty(self, state_dir):
        assert github_issues._load_map() == {}

    def test_corrupt_file_returns_empty(self, state_dir):
        path = state_dir / "journal_map.json"
        path.write_text("{not json")
        assert github_issues._load_map() == {}

    def test_path_respects_state_dir(self, state_dir, monkeypatch):
        custom = state_dir / "custom"
        monkeypatch.setenv("PRISM_STATE_DIR", str(custom))
        github_issues._store_mapping("x", 1)
        assert (custom / "journal_map.json").exists()


# ---------------------------------------------------------------------------
# on_signal_fired
# ---------------------------------------------------------------------------

class TestOnSignalFired:
    def test_creates_issue_and_persists_mapping(self, state_dir, gh_mock):
        p = _make_packet(signal_id="sig-create-1")
        number = github_issues.on_signal_fired(p)
        assert number == 42

        # Persisted
        assert github_issues._load_map() == {"sig-create-1": 42}

        # gh issue create was called with the right shape
        call = gh_mock.call_args
        cmd = call.args[0]
        assert "issue" in cmd and "create" in cmd
        # Body went in via stdin
        assert call.kwargs.get("input") is not None
        assert "sig-create-1" in call.kwargs["input"]

        # Labels were attached
        assert "phase:pending" in cmd
        assert "inst:XAUUSD" in cmd
        assert "dir:LONG" in cmd

    def test_dedup_skips_when_mapping_exists(self, state_dir, gh_mock):
        github_issues._store_mapping("dup-1", 99)
        p = _make_packet(signal_id="dup-1")
        number = github_issues.on_signal_fired(p)
        assert number == 99
        # gh was never invoked — mapping short-circuited
        gh_mock.assert_not_called()

    def test_returns_none_on_gh_failure(self, state_dir, gh_mock):
        gh_mock.return_value = SimpleNamespace(
            returncode=1, stdout="", stderr="boom",
        )
        p = _make_packet(signal_id="sig-fail")
        assert github_issues.on_signal_fired(p) is None
        # Mapping NOT written for failures
        assert "sig-fail" not in github_issues._load_map()

    def test_swallows_exceptions(self, state_dir, gh_mock):
        gh_mock.side_effect = RuntimeError("network exploded")
        p = _make_packet(signal_id="sig-exc")
        # Must not raise — journal failure cannot break delivery
        assert github_issues.on_signal_fired(p) is None

    def test_no_signal_id_returns_none(self, state_dir, gh_mock):
        p = _make_packet()
        p.signal_id = ""
        assert github_issues.on_signal_fired(p) is None
        gh_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------

class TestOnTradeFilled:
    def test_comments_and_swaps_label(self, state_dir, gh_mock):
        github_issues._store_mapping("sig-fill", 50)
        # comment + edit both succeed (default mock)
        result = github_issues.on_trade_filled("sig-fill", ticket=12345)
        assert result == 50

        # 2 calls: one comment, one edit (swap label)
        assert gh_mock.call_count == 2
        cmds = [c.args[0] for c in gh_mock.call_args_list]
        assert any("comment" in cmd for cmd in cmds)
        assert any("edit" in cmd and "--add-label" in cmd for cmd in cmds)
        # Comment body mentions ticket
        comment_call = next(c for c in gh_mock.call_args_list if "comment" in c.args[0])
        assert "12345" in comment_call.kwargs["input"]

    def test_no_mapping_returns_none(self, state_dir, gh_mock):
        # No mapping → find_open_issue falls back to search. Mock returns
        # the default issue-URL stdout (not valid search JSON), so the
        # parse fails gracefully and we return None.
        from types import SimpleNamespace
        gh_mock.return_value = SimpleNamespace(returncode=0, stdout="[]", stderr="")
        result = github_issues.on_trade_filled("nope", ticket=1)
        assert result is None
        # Exactly one call: the search; no comment, no edit.
        assert gh_mock.call_count == 1
        cmd = gh_mock.call_args.args[0]
        assert "list" in cmd and "--search" in cmd


class TestOnTp1Hit:
    def test_comments_and_swaps_label(self, state_dir, gh_mock):
        github_issues._store_mapping("sig-tp1", 75)
        result = github_issues.on_tp1_hit("sig-tp1", ticket=98765)
        assert result == 75
        cmds = [c.args[0] for c in gh_mock.call_args_list]
        # An edit call swaps phase:open → phase:tp1
        edit_call = next(c for c in gh_mock.call_args_list if "edit" in c.args[0])
        cmd = edit_call.args[0]
        assert "--remove-label" in cmd
        assert "phase:open" in cmd
        assert "phase:tp1" in cmd


class TestOnTradeClosed:
    def test_win_outcome_closes_issue(self, state_dir, gh_mock):
        github_issues._store_mapping("sig-close-w", 60)
        result = github_issues.on_trade_closed("sig-close-w", ticket=111, pnl=250.0)
        assert result == 60

        cmds = [c.args[0] for c in gh_mock.call_args_list]
        # Comment, multiple phase swaps, outcome label add, and issue close.
        assert any("comment" in c for c in cmds)
        assert any("close" in c for c in cmds)
        # outcome:win label added
        edit_calls = [c for c in gh_mock.call_args_list if "edit" in c.args[0]]
        outcome_added = any("outcome:win" in c.args[0] for c in edit_calls)
        assert outcome_added

    def test_loss_outcome(self, state_dir, gh_mock):
        github_issues._store_mapping("sig-close-l", 61)
        github_issues.on_trade_closed("sig-close-l", ticket=222, pnl=-150.0)
        edit_calls = [c for c in gh_mock.call_args_list if "edit" in c.args[0]]
        assert any("outcome:loss" in c.args[0] for c in edit_calls)

    def test_breakeven_with_risk(self, state_dir, gh_mock):
        github_issues._store_mapping("sig-close-be", 62)
        # |2| < 10% × 100 = 10 → breakeven
        github_issues.on_trade_closed(
            "sig-close-be", ticket=333, pnl=2.0, risk_usd=100.0,
        )
        edit_calls = [c for c in gh_mock.call_args_list if "edit" in c.args[0]]
        assert any("outcome:breakeven" in c.args[0] for c in edit_calls)


# ---------------------------------------------------------------------------
# find_open_issue
# ---------------------------------------------------------------------------

class TestFindOpenIssue:
    def test_local_map_hit_skips_search(self, state_dir, gh_mock):
        github_issues._store_mapping("sig-cached", 7)
        assert github_issues.find_open_issue("sig-cached") == 7
        gh_mock.assert_not_called()

    def test_search_fallback_caches_result(self, state_dir, gh_mock):
        gh_mock.return_value = SimpleNamespace(
            returncode=0,
            stdout=json.dumps([{"number": 88}]),
            stderr="",
        )
        assert github_issues.find_open_issue("sig-remote") == 88
        # Cached so second call doesn't re-search
        gh_mock.reset_mock()
        assert github_issues.find_open_issue("sig-remote") == 88
        gh_mock.assert_not_called()

    def test_search_miss_returns_none(self, state_dir, gh_mock):
        gh_mock.return_value = SimpleNamespace(
            returncode=0, stdout="[]", stderr="",
        )
        assert github_issues.find_open_issue("nope") is None


# ---------------------------------------------------------------------------
# Body schema sanity — single test to lock the issue body shape
# ---------------------------------------------------------------------------

class TestBodySchema:
    def test_required_sections_present(self):
        p = _make_packet(
            smart_money={"ob": {"direction": "bullish", "midpoint": 2340.0,
                                "distance_pips": 12, "timeframe": "H1"}},
            htf_bias={"bias_4h": "bullish"},
        )
        body = github_issues._render_body(p)
        for marker in (
            "## Signal", "signal_id:",
            "## Levels", "**entry:**", "**TP1:**", "**TP2:**",
            "## Context", "**regime:**", "**confidence:**",
            "## HTF Bias",
            "## Smart Money Confluence",
            "Auto-created by PRISM",
        ):
            assert marker in body, f"body missing required marker: {marker!r}"
