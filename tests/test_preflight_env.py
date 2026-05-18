"""Tests for scripts/preflight_env.py — env preflight check."""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

# Make project root importable so `scripts.preflight_env` resolves.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import preflight_env  # noqa: E402


# ---------------------------------------------------------------------
# parse_env_example
# ---------------------------------------------------------------------


def _write_env_example(tmp_path: Path, body: str) -> Path:
    path = tmp_path / ".env.example"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_parse_env_example_skips_full_line_comments(tmp_path: Path) -> None:
    path = _write_env_example(
        tmp_path,
        """
        # PRISM_SIGNAL_AUDIT_ENABLED=1
        PRISM_INSTRUMENTS=XAUUSD,EURUSD
        """,
    )
    decls = preflight_env.parse_env_example(path)
    assert "PRISM_SIGNAL_AUDIT_ENABLED" not in decls
    assert decls == {"PRISM_INSTRUMENTS": "XAUUSD,EURUSD"}


def test_parse_env_example_strips_inline_comments(tmp_path: Path) -> None:
    path = _write_env_example(
        tmp_path,
        """
        PRISM_RISK_PCT=0.01   # 1% per trade
        """,
    )
    decls = preflight_env.parse_env_example(path)
    assert decls == {"PRISM_RISK_PCT": "0.01"}


def test_parse_env_example_handles_blank_lines_and_no_equals(tmp_path: Path) -> None:
    path = _write_env_example(
        tmp_path,
        """

        PRISM_INSTRUMENTS=XAUUSD

        not_an_assignment_line
        PRISM_EXECUTION_MODE=NOTIFY
        """,
    )
    decls = preflight_env.parse_env_example(path)
    assert decls == {
        "PRISM_INSTRUMENTS": "XAUUSD",
        "PRISM_EXECUTION_MODE": "NOTIFY",
    }


def test_parse_env_example_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "does_not_exist.env"
    assert preflight_env.parse_env_example(path) == {}


# ---------------------------------------------------------------------
# redact
# ---------------------------------------------------------------------


def test_redact_short_values_passthrough() -> None:
    assert preflight_env.redact("abc") == "abc"
    assert preflight_env.redact("12345678") == "12345678"


def test_redact_long_values_masked() -> None:
    out = preflight_env.redact("xoxb-very-long-secret-token-here")
    assert out.startswith("xoxb")
    assert out.endswith("re")
    assert "…" in out


# ---------------------------------------------------------------------
# CLI behaviour
# ---------------------------------------------------------------------


def test_cli_check_single_key_present(monkeypatch, capsys) -> None:
    monkeypatch.setenv("PRISM_SIGNAL_AUDIT_ENABLED", "1")
    rc = preflight_env.main(["--check", "PRISM_SIGNAL_AUDIT_ENABLED"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK PRISM_SIGNAL_AUDIT_ENABLED" in out


def test_cli_check_single_key_missing(monkeypatch, capsys) -> None:
    monkeypatch.delenv("PRISM_SIGNAL_AUDIT_ENABLED", raising=False)
    rc = preflight_env.main(["--check", "PRISM_SIGNAL_AUDIT_ENABLED"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "MISSING PRISM_SIGNAL_AUDIT_ENABLED" in out


def test_cli_fails_when_critical_missing(tmp_path: Path, monkeypatch, capsys) -> None:
    env_example = _write_env_example(
        tmp_path,
        """
        PRISM_SIGNAL_AUDIT_ENABLED=1
        PRISM_INSTRUMENTS=XAUUSD,EURUSD
        PRISM_EXECUTION_MODE=NOTIFY
        PRISM_SLACK_TOKEN=xoxb-test
        PRISM_STATE_DIR=state
        """,
    )
    for key in ("PRISM_SIGNAL_AUDIT_ENABLED", "PRISM_INSTRUMENTS",
                "PRISM_EXECUTION_MODE", "PRISM_SLACK_TOKEN", "PRISM_STATE_DIR"):
        monkeypatch.delenv(key, raising=False)

    rc = preflight_env.main(["--env-example", str(env_example), "--quiet"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL critical key missing: PRISM_SIGNAL_AUDIT_ENABLED" in out


def test_cli_warn_only_never_fails(tmp_path: Path, monkeypatch, capsys) -> None:
    env_example = _write_env_example(
        tmp_path,
        """
        PRISM_SIGNAL_AUDIT_ENABLED=1
        """,
    )
    monkeypatch.delenv("PRISM_SIGNAL_AUDIT_ENABLED", raising=False)
    rc = preflight_env.main(
        ["--env-example", str(env_example), "--warn-only", "--quiet"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "FAIL" in out  # still reports it


def test_cli_passes_when_all_critical_present(tmp_path: Path, monkeypatch, capsys) -> None:
    env_example = _write_env_example(
        tmp_path,
        """
        PRISM_SIGNAL_AUDIT_ENABLED=1
        PRISM_INSTRUMENTS=XAUUSD,EURUSD
        PRISM_EXECUTION_MODE=NOTIFY
        PRISM_SLACK_TOKEN=xoxb-test
        PRISM_STATE_DIR=state
        """,
    )
    monkeypatch.setenv("PRISM_SIGNAL_AUDIT_ENABLED", "1")
    monkeypatch.setenv("PRISM_INSTRUMENTS", "XAUUSD,EURUSD")
    monkeypatch.setenv("PRISM_EXECUTION_MODE", "NOTIFY")
    monkeypatch.setenv("PRISM_SLACK_TOKEN", "xoxb-real-secret-here")
    monkeypatch.setenv("PRISM_STATE_DIR", "state")

    rc = preflight_env.main(["--env-example", str(env_example)])
    out = capsys.readouterr().out
    assert rc == 0
    # Secret should be redacted in the output.
    assert "xoxb-real-secret-here" not in out
    assert "[redacted]" in out
