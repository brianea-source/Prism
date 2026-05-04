"""GitHub push webhook → auto-deploy on the PRISM VPS.

Listens on ``PRISM_DEPLOY_PORT`` (default ``9000``). When GitHub posts a
push event matching the configured branch (default ``main``), validates
the HMAC signature against ``PRISM_DEPLOY_SECRET``, then runs:

  1. ``git pull origin main``
  2. ``pip install -r requirements.txt -q``
  3. ``schtasks /end /tn PRISM-Runner`` and ``schtasks /run /tn PRISM-Runner``
  4. Posts a Slack confirmation: short SHA + commit subject.

Designed to run as the ``PRISM-DeployWebhook`` scheduled task on Windows.
Flask is the bare-minimum HTTP layer; we avoid pulling in a full WSGI
server because this only ever speaks to GitHub's webhook IP block.

Security:
  - ``PRISM_DEPLOY_SECRET`` is REQUIRED at startup. The server refuses to
    start without it (returning RC=2) so no unauthenticated fork-and-deploy
    attack is possible.
  - HMAC verification uses ``hmac.compare_digest`` (constant-time).
  - Only push events on the configured branch trigger deploy. Anything
    else is acknowledged with 204 and ignored.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("prism.deploy_webhook")


DEFAULT_PORT = 9000
DEFAULT_BRANCH = "main"
DEFAULT_LOG_PATH = "logs/deploy.log"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _configure_logging() -> Path:
    path = Path(os.environ.get("PRISM_DEPLOY_LOG", DEFAULT_LOG_PATH))
    path.parent.mkdir(parents=True, exist_ok=True)
    has_file_handler = any(
        isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path.resolve()
        for h in logger.handlers
    )
    if not has_file_handler:
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        ))
        logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    return path


# ---------------------------------------------------------------------------
# HMAC verification
# ---------------------------------------------------------------------------
def verify_signature(secret: str, body: bytes, header_value: str) -> bool:
    """Validate the ``X-Hub-Signature-256`` header GitHub sends.

    Header is ``sha256=<hex>``. We compute HMAC over the raw request body.
    """
    if not header_value or "=" not in header_value:
        return False
    algo, _, sig = header_value.partition("=")
    if algo != "sha256" or not sig:
        return False
    expected = hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, sig)


# ---------------------------------------------------------------------------
# Deploy steps (each isolated so tests can patch them)
# ---------------------------------------------------------------------------
def git_pull(branch: str = DEFAULT_BRANCH) -> int:
    return subprocess.run(
        ["git", "pull", "origin", branch],
        capture_output=True, text=True, timeout=300, check=False,
    ).returncode


def pip_install() -> int:
    return subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
        capture_output=True, text=True, timeout=600, check=False,
    ).returncode


def restart_runner(task_name: str = "PRISM-Runner") -> int:
    subprocess.run(
        ["schtasks", "/end", "/tn", task_name],
        capture_output=True, text=True, timeout=30, check=False,
    )
    return subprocess.run(
        ["schtasks", "/run", "/tn", task_name],
        capture_output=True, text=True, timeout=30, check=False,
    ).returncode


def post_slack(text: str) -> bool:
    try:
        from prism.delivery.slack_notifier import SlackNotifier
    except Exception as exc:  # pragma: no cover
        logger.error("Could not import SlackNotifier: %s", exc)
        return False
    try:
        notifier = SlackNotifier()
        if not notifier.client:
            logger.info("Slack not configured; would have posted: %s", text)
            return False
        return notifier.send_alert(text) is not None
    except Exception as exc:  # pragma: no cover
        logger.error("Slack post failed: %s", exc)
        return False


def deploy(
    *,
    branch: str = DEFAULT_BRANCH,
    sha: str = "",
    message: str = "",
    pull_fn=git_pull,
    pip_fn=pip_install,
    restart_fn=restart_runner,
    slack_fn=post_slack,
) -> dict:
    """Full deploy workflow. Each step's RC is captured but failures
    after pull still attempt the runner restart so we don't leave the
    VPS sitting on stale dependencies forever."""
    result = {"pull_rc": None, "pip_rc": None, "restart_rc": None}
    result["pull_rc"] = pull_fn(branch)
    if result["pull_rc"] != 0:
        slack_fn(f"⚠️ PRISM auto-deploy: git pull failed (rc={result['pull_rc']}). Skipping deploy.")
        return result

    result["pip_rc"] = pip_fn()
    if result["pip_rc"] != 0:
        slack_fn(f"⚠️ PRISM auto-deploy: pip install failed (rc={result['pip_rc']}).")
        # Still try the restart — running stale code is better than a
        # half-restarted runner. Caller logs the partial failure.

    result["restart_rc"] = restart_fn()
    short_sha = (sha or "")[:7] or "unknown"
    subject = (message or "").splitlines()[0] if message else ""
    slack_fn(
        f"🚀 PRISM auto-deployed from {branch} (`{short_sha}` — "
        f"{subject or 'no commit message'}). Runner restarted."
    )
    return result


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
def create_app(secret: Optional[str] = None, branch: str = DEFAULT_BRANCH):
    """Build the Flask app. Importing Flask is deferred so the test suite
    can exercise the helpers without requiring Flask installed."""
    from flask import Flask, request, jsonify  # type: ignore

    secret = secret or os.environ.get("PRISM_DEPLOY_SECRET", "")
    if not secret:
        raise RuntimeError(
            "PRISM_DEPLOY_SECRET is required — refusing to start without it"
        )

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def _health():
        return jsonify({"ok": True})

    @app.route("/webhook", methods=["POST"])
    def _hook():
        body = request.get_data() or b""
        sig_header = request.headers.get("X-Hub-Signature-256", "")
        if not verify_signature(secret, body, sig_header):
            logger.warning("webhook: signature mismatch")
            return ("forbidden", 403)

        event = request.headers.get("X-GitHub-Event", "")
        if event == "ping":
            return jsonify({"pong": True})
        if event != "push":
            return ("", 204)

        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return ("bad json", 400)

        ref = payload.get("ref", "")
        if ref != f"refs/heads/{branch}":
            logger.info("webhook: ignoring push to %s (want %s)", ref, branch)
            return ("", 204)

        head = payload.get("head_commit") or {}
        sha = head.get("id", "")
        message = head.get("message", "")
        logger.info("webhook: deploying %s (%s)", sha[:7], message.splitlines()[0] if message else "")
        result = deploy(branch=branch, sha=sha, message=message)
        return jsonify({"deployed": True, "result": result})

    return app


def main(argv=None) -> int:
    _configure_logging()
    secret = os.environ.get("PRISM_DEPLOY_SECRET", "")
    if not secret:
        logger.error("PRISM_DEPLOY_SECRET is required — refusing to start")
        return 2
    port = int(os.environ.get("PRISM_DEPLOY_PORT", DEFAULT_PORT))
    branch = os.environ.get("PRISM_DEPLOY_BRANCH", DEFAULT_BRANCH)
    app = create_app(secret=secret, branch=branch)
    logger.info("deploy webhook listening on 0.0.0.0:%d branch=%s", port, branch)
    app.run(host="0.0.0.0", port=port)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
