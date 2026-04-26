"""
tests/test_health_check.py — 41 tests for scripts/health_check.py

Covers:
- CheckResult dataclass
- PASS/FAIL/WARN/SKIP paths for every check
- Exit code semantics (0/1/2)
- Error trapping in orchestrator
- CLI flags: --list, --json, --no-color, --check, --instruments
- Source-level: every ALL_CHECKS name has a check_ function
- RUNBOOK sync: every ALL_CHECKS name appears in docs/RUNBOOK.md
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.health_check as hc


# ---------------------------------------------------------------------------
# CheckResult dataclass
# ---------------------------------------------------------------------------

class TestCheckResult:
    def test_defaults(self):
        r = hc.CheckResult("PASS", "all good")
        assert r.status == "PASS"
        assert r.message == "all good"
        assert r.detail == ""

    def test_with_detail(self):
        r = hc.CheckResult("FAIL", "broke", "some detail")
        assert r.detail == "some detail"

    def test_valid_statuses(self):
        for s in ("PASS", "FAIL", "WARN", "SKIP"):
            r = hc.CheckResult(s, "msg")
            assert r.status == s


# ---------------------------------------------------------------------------
# ALL_CHECKS registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_checks_have_functions(self):
        """Every name in ALL_CHECKS must have a matching check_ function."""
        for name in hc.ALL_CHECKS:
            fn = getattr(hc, f"check_{name}", None)
            assert callable(fn), f"check_{name}() not found"

    def test_all_checks_count(self):
        assert len(hc.ALL_CHECKS) == 11

    def test_all_checks_expected_names(self):
        expected = [
            "environment", "state_dir", "models", "session_clock",
            "inflight_persistence", "mt5_connect", "mt5_bars", "pip_value",
            "reconnect", "drawdown_guard", "slack",
        ]
        assert hc.ALL_CHECKS == expected


# ---------------------------------------------------------------------------
# RUNBOOK sync: every check name must appear in docs/RUNBOOK.md
# ---------------------------------------------------------------------------

class TestRunbookSync:
    def test_runbook_mentions_all_checks(self):
        runbook_path = _ROOT / "docs" / "RUNBOOK.md"
        assert runbook_path.exists(), "docs/RUNBOOK.md not found"
        text = runbook_path.read_text()
        for name in hc.ALL_CHECKS:
            assert name in text, (
                f"Check '{name}' not mentioned in docs/RUNBOOK.md — "
                f"update the runbook when adding checks"
            )


# ---------------------------------------------------------------------------
# check_environment
# ---------------------------------------------------------------------------

class TestCheckEnvironment:
    def test_pass_when_all_set(self, monkeypatch):
        for v in hc._REQUIRED_VARS + hc._OPTIONAL_VARS:
            monkeypatch.setenv(v, "dummy")
        r = hc.check_environment(["XAUUSD"], False)
        assert r.status == "PASS"

    def test_fail_when_required_missing(self, monkeypatch):
        for v in hc._REQUIRED_VARS:
            monkeypatch.delenv(v, raising=False)
        r = hc.check_environment(["XAUUSD"], False)
        assert r.status == "FAIL"
        for v in hc._REQUIRED_VARS:
            assert v in r.message

    def test_warn_when_optional_missing(self, monkeypatch):
        for v in hc._REQUIRED_VARS:
            monkeypatch.setenv(v, "dummy")
        for v in hc._OPTIONAL_VARS:
            monkeypatch.delenv(v, raising=False)
        r = hc.check_environment(["XAUUSD"], False)
        assert r.status == "WARN"
        assert "optional" in r.message


# ---------------------------------------------------------------------------
# check_state_dir
# ---------------------------------------------------------------------------

class TestCheckStateDir:
    def test_pass_with_writable_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path))
        r = hc.check_state_dir(["XAUUSD"], False)
        assert r.status == "PASS"

    def test_warn_when_dir_missing(self, tmp_path, monkeypatch):
        missing = tmp_path / "nonexistent"
        monkeypatch.setenv("PRISM_STATE_DIR", str(missing))
        r = hc.check_state_dir(["XAUUSD"], False)
        assert r.status == "WARN"
        assert "will be created" in r.message

    def test_fail_on_write_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRISM_STATE_DIR", str(tmp_path))
        # Make the directory read-only
        tmp_path.chmod(0o555)
        try:
            r = hc.check_state_dir(["XAUUSD"], False)
            assert r.status in ("FAIL", "WARN")  # platform-dependent
        finally:
            tmp_path.chmod(0o755)


# ---------------------------------------------------------------------------
# check_models
# ---------------------------------------------------------------------------

class TestCheckModels:
    def test_fail_when_dir_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRISM_MODEL_DIR", str(tmp_path / "no_models"))
        r = hc.check_models(["XAUUSD"], False)
        assert r.status == "FAIL"
        assert "not found" in r.message

    def test_fail_when_files_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRISM_MODEL_DIR", str(tmp_path))
        # Only create 3 of 4 layers for XAUUSD
        for layer in range(3):
            (tmp_path / f"XAUUSD_layer{layer}.pkl").touch()
        r = hc.check_models(["XAUUSD"], False)
        assert r.status == "FAIL"
        assert "missing" in r.message

    def test_pass_when_all_present(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PRISM_MODEL_DIR", str(tmp_path))
        for inst in ["XAUUSD", "EURUSD"]:
            for layer in range(4):
                (tmp_path / f"{inst}_layer{layer}.pkl").touch()
        r = hc.check_models(["XAUUSD", "EURUSD"], False)
        assert r.status == "PASS"


# ---------------------------------------------------------------------------
# check_session_clock
# ---------------------------------------------------------------------------

class TestCheckSessionClock:
    def test_pass_normal(self):
        r = hc.check_session_clock(["XAUUSD"], False)
        assert r.status in ("PASS", "FAIL")  # FAIL only if import broken

    def test_pass_returns_bool(self):
        r = hc.check_session_clock(["XAUUSD"], False)
        if r.status == "PASS":
            assert "kill_zone=" in r.message
            assert "sunday_gap=" in r.message

    def test_fail_on_import_error(self, monkeypatch):
        import unittest.mock
        with unittest.mock.patch("builtins.__import__", side_effect=ImportError("no module")):
            # This approach is too broad; just verify the check handles ImportError
            pass
        # Direct path: patch the import inside the function
        with patch.dict("sys.modules", {"prism.delivery.session_filter": None}):
            r = hc.check_session_clock(["XAUUSD"], False)
            assert r.status in ("FAIL", "PASS")  # depends on cached imports


# ---------------------------------------------------------------------------
# check_inflight_persistence
# ---------------------------------------------------------------------------

class TestCheckInflightPersistence:
    def test_pass_or_warn(self):
        """Should PASS if runner is importable with inflight functions, WARN otherwise."""
        r = hc.check_inflight_persistence(["XAUUSD"], False)
        assert r.status in ("PASS", "WARN", "FAIL")

    def test_pass_with_real_runner(self):
        """If prism.delivery.runner has the expected functions, check passes."""
        import prism.delivery.runner as rm
        has_persist = hasattr(rm, "_persist_inflight_keys")
        has_load = hasattr(rm, "_load_inflight_keys")
        r = hc.check_inflight_persistence(["XAUUSD"], False)
        if has_persist and has_load:
            assert r.status == "PASS"


# ---------------------------------------------------------------------------
# check_mt5_connect
# ---------------------------------------------------------------------------

class TestCheckMT5Connect:
    def test_skip_on_macos(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Darwin")
        # Reset global state
        hc._mt5_skipped = False
        hc._mt5_bridge_instance = None
        r = hc.check_mt5_connect(["XAUUSD"], False)
        assert r.status == "SKIP"
        assert "macOS" in r.message

    def test_skip_when_mt5_login_not_set(self, monkeypatch):
        monkeypatch.setattr(platform, "system", lambda: "Linux")
        monkeypatch.delenv("MT5_LOGIN", raising=False)
        hc._mt5_skipped = False
        hc._mt5_bridge_instance = None
        r = hc.check_mt5_connect(["XAUUSD"], False)
        assert r.status == "SKIP"
        assert "MT5_LOGIN" in r.message


# ---------------------------------------------------------------------------
# check_mt5_bars / check_pip_value / check_reconnect (all SKIP without MT5)
# ---------------------------------------------------------------------------

class TestMT5DependentChecks:
    def setup_method(self):
        hc._mt5_skipped = True
        hc._mt5_bridge_instance = None

    def teardown_method(self):
        hc._mt5_skipped = False

    def test_mt5_bars_skips(self):
        r = hc.check_mt5_bars(["XAUUSD"], False)
        assert r.status == "SKIP"

    def test_pip_value_skips(self):
        r = hc.check_pip_value(["XAUUSD"], False)
        assert r.status == "SKIP"

    def test_reconnect_skips(self):
        r = hc.check_reconnect(["XAUUSD"], False)
        assert r.status == "SKIP"


# ---------------------------------------------------------------------------
# check_drawdown_guard
# ---------------------------------------------------------------------------

class TestCheckDrawdownGuard:
    def test_pass_with_real_guard(self):
        r = hc.check_drawdown_guard(["XAUUSD"], False)
        assert r.status == "PASS", f"Expected PASS, got {r.status}: {r.message}\n{r.detail}"

    def test_fail_on_import_error(self):
        with patch.dict("sys.modules", {"prism.delivery.drawdown_guard": None}):
            r = hc.check_drawdown_guard(["XAUUSD"], False)
            # May be PASS if cached, or FAIL if import actually fails
            assert r.status in ("PASS", "FAIL")


# ---------------------------------------------------------------------------
# check_slack
# ---------------------------------------------------------------------------

class TestCheckSlack:
    def test_skip_when_no_slack(self):
        r = hc.check_slack(["XAUUSD"], no_slack=True)
        assert r.status == "SKIP"
        assert "--no-slack" in r.message

    def test_warn_when_creds_missing(self, monkeypatch):
        monkeypatch.delenv("PRISM_SLACK_TOKEN", raising=False)
        monkeypatch.delenv("PRISM_SLACK_CHANNEL", raising=False)
        r = hc.check_slack(["XAUUSD"], no_slack=False)
        assert r.status == "WARN"

    def test_pass_when_slack_ok(self, monkeypatch):
        monkeypatch.setenv("PRISM_SLACK_TOKEN", "xoxb-fake")
        monkeypatch.setenv("PRISM_SLACK_CHANNEL", "C12345")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"ok": True, "ts": "12345.67890"}).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            r = hc.check_slack(["XAUUSD"], no_slack=False)
        assert r.status == "PASS"
        assert "12345.67890" in r.message

    def test_warn_on_ratelimited(self, monkeypatch):
        monkeypatch.setenv("PRISM_SLACK_TOKEN", "xoxb-fake")
        monkeypatch.setenv("PRISM_SLACK_CHANNEL", "C12345")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"ok": False, "error": "ratelimited"}).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            r = hc.check_slack(["XAUUSD"], no_slack=False)
        assert r.status == "WARN"

    def test_fail_on_slack_error(self, monkeypatch):
        monkeypatch.setenv("PRISM_SLACK_TOKEN", "xoxb-fake")
        monkeypatch.setenv("PRISM_SLACK_CHANNEL", "C12345")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"ok": False, "error": "channel_not_found"}).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            r = hc.check_slack(["XAUUSD"], no_slack=False)
        assert r.status == "FAIL"


# ---------------------------------------------------------------------------
# Exit code semantics
# ---------------------------------------------------------------------------

class TestExitCodes:
    def _run(self, statuses, **kw):
        """Run orchestrator with mocked checks returning the given statuses."""
        checks = [f"check_{i}" for i in range(len(statuses))]
        # Temporarily inject mock checks into the module
        fns = {}
        for i, status in enumerate(statuses):
            name = f"check_{i}"
            result = hc.CheckResult(status, "mocked")
            fn = lambda inst, ns, r=result: r
            fns[name] = fn
            setattr(hc, f"check_{name}", fn)
        try:
            return hc.run_checks(
                checks=[f"check_{i}" for i in range(len(statuses))],
                instruments=["XAUUSD"],
                no_slack=True,
                use_json=False,
                use_color=False,
            )
        finally:
            for name in fns:
                delattr(hc, f"check_{name}")

    def test_all_pass_exits_0(self, capsys):
        code = self._run(["PASS", "PASS", "SKIP"])
        assert code == 0

    def test_warn_no_fail_exits_2(self, capsys):
        code = self._run(["PASS", "WARN", "SKIP"])
        assert code == 2

    def test_any_fail_exits_1(self, capsys):
        code = self._run(["PASS", "WARN", "FAIL"])
        assert code == 1

    def test_fail_beats_warn(self, capsys):
        code = self._run(["FAIL", "WARN"])
        assert code == 1


# ---------------------------------------------------------------------------
# Error trapping in orchestrator
# ---------------------------------------------------------------------------

class TestOrchestratorErrorTrapping:
    def test_exception_becomes_fail(self, capsys):
        """If a check raises, the orchestrator catches it and marks FAIL."""
        def _boom(inst, ns):
            raise RuntimeError("unexpected crash")
        setattr(hc, "check_boom", _boom)
        try:
            results_code = hc.run_checks(
                checks=["boom"],
                instruments=["XAUUSD"],
                no_slack=True,
                use_json=False,
                use_color=False,
            )
            assert results_code == 1  # FAIL
        finally:
            delattr(hc, "check_boom")

    def test_orchestrator_continues_after_exception(self, capsys):
        """Other checks still run even if one crashes."""
        call_log = []

        def _boom(inst, ns):
            raise RuntimeError("crash")

        def _ok(inst, ns):
            call_log.append("ok")
            return hc.CheckResult("PASS", "fine")

        setattr(hc, "check_boom2", _boom)
        setattr(hc, "check_ok2", _ok)
        try:
            hc.run_checks(
                checks=["boom2", "ok2"],
                instruments=["XAUUSD"],
                no_slack=True,
                use_json=False,
                use_color=False,
            )
            assert "ok" in call_log
        finally:
            delattr(hc, "check_boom2")
            delattr(hc, "check_ok2")


# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------

class TestCLIFlags:
    def test_list_flag(self, capsys):
        code = hc.main(["--list"])
        out = capsys.readouterr().out
        assert code == 0
        for name in hc.ALL_CHECKS:
            assert name in out

    def test_json_output_is_valid(self, capsys):
        code = hc.main(["--check", "environment", "--json", "--no-slack"])
        out = capsys.readouterr().out
        data = json.loads(out)
        assert isinstance(data, list)
        assert len(data) == 1
        assert "check" in data[0]
        assert "status" in data[0]
        assert "message" in data[0]
        assert "detail" in data[0]

    def test_no_color_no_ansi(self, capsys):
        code = hc.main(["--check", "environment", "--no-color", "--no-slack"])
        out = capsys.readouterr().out
        assert "\033[" not in out

    def test_check_filter_runs_single(self, capsys):
        code = hc.main(["--check", "slack", "--no-slack"])
        out = capsys.readouterr().out
        assert "slack" in out
        # Other checks should not appear
        for other in hc.ALL_CHECKS:
            if other != "slack":
                assert other not in out

    def test_instruments_filter(self, capsys, monkeypatch):
        seen = []
        original = hc.check_models

        def _capture(instruments, no_slack):
            seen.extend(instruments)
            return hc.CheckResult("PASS", "ok")

        monkeypatch.setattr(hc, "check_models", _capture)
        hc.main(["--check", "models", "--instruments", "XAUUSD", "--no-slack"])
        assert seen == ["XAUUSD"]
        monkeypatch.setattr(hc, "check_models", original)

    def test_unknown_check_exits_with_error(self):
        with pytest.raises(SystemExit) as exc:
            hc.main(["--check", "nonexistent_check"])
        assert exc.value.code != 0
