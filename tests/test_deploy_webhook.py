"""Tests for scripts/deploy_webhook.py."""
from __future__ import annotations

import hashlib
import hmac
import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_module():
    """Load scripts/deploy_webhook.py as a module despite being a script."""
    path = _ROOT / "scripts" / "deploy_webhook.py"
    spec = importlib.util.spec_from_file_location("prism_deploy_webhook", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


dw = _load_module()


# ---------------------------------------------------------------------------
# verify_signature
# ---------------------------------------------------------------------------
def test_verify_signature_accepts_valid():
    secret = "supersecret"
    body = b'{"hello":"world"}'
    sig = "sha256=" + hmac.new(
        secret.encode(), body, hashlib.sha256
    ).hexdigest()
    assert dw.verify_signature(secret, body, sig) is True


def test_verify_signature_rejects_tampered_body():
    secret = "supersecret"
    body = b'{"hello":"world"}'
    sig = "sha256=" + hmac.new(
        secret.encode(), b'{"hello":"evil"}', hashlib.sha256
    ).hexdigest()
    assert dw.verify_signature(secret, body, sig) is False


def test_verify_signature_rejects_missing_or_bad_format():
    assert dw.verify_signature("s", b"x", "") is False
    assert dw.verify_signature("s", b"x", "md5=abc") is False
    assert dw.verify_signature("s", b"x", "sha256=") is False


# ---------------------------------------------------------------------------
# deploy
# ---------------------------------------------------------------------------
def test_deploy_full_success_path():
    pull = MagicMock(return_value=0)
    pip = MagicMock(return_value=0)
    restart = MagicMock(return_value=0)
    slack = MagicMock(return_value=True)
    out = dw.deploy(
        sha="abcdef1234567890", message="feat: ship it",
        pull_fn=pull, pip_fn=pip, restart_fn=restart, slack_fn=slack,
    )
    assert out == {"pull_rc": 0, "pip_rc": 0, "restart_rc": 0}
    pull.assert_called_once()
    pip.assert_called_once()
    restart.assert_called_once()
    msg = slack.call_args[0][0]
    assert "abcdef1" in msg
    assert "feat: ship it" in msg


def test_deploy_skips_pip_and_restart_when_pull_fails():
    pull = MagicMock(return_value=1)
    pip = MagicMock()
    restart = MagicMock()
    slack = MagicMock()
    out = dw.deploy(pull_fn=pull, pip_fn=pip, restart_fn=restart, slack_fn=slack)
    assert out["pull_rc"] == 1
    pip.assert_not_called()
    restart.assert_not_called()
    slack.assert_called_once()
    assert "git pull failed" in slack.call_args[0][0]


def test_deploy_pip_failure_still_restarts():
    pull = MagicMock(return_value=0)
    pip = MagicMock(return_value=1)
    restart = MagicMock(return_value=0)
    slack = MagicMock()
    out = dw.deploy(pull_fn=pull, pip_fn=pip, restart_fn=restart, slack_fn=slack)
    assert out["pip_rc"] == 1
    restart.assert_called_once()
    # Two slack calls: pip warning + the deploy confirmation.
    assert slack.call_count == 2


# ---------------------------------------------------------------------------
# create_app refuses to start without secret
# ---------------------------------------------------------------------------
def test_create_app_requires_secret(monkeypatch):
    monkeypatch.delenv("PRISM_DEPLOY_SECRET", raising=False)
    pytest.importorskip("flask")
    with pytest.raises(RuntimeError, match="PRISM_DEPLOY_SECRET"):
        dw.create_app(secret=None)
