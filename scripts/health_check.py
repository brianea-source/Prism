"""
PRISM Pre-Live Health Check
============================
Single-command smoke test that exercises every guard end-to-end
against a real MT5 demo account before going live.

Usage:
    python scripts/health_check.py
    python scripts/health_check.py --check environment
    python scripts/health_check.py --instruments XAUUSD --no-slack --json
    python scripts/health_check.py --list

Exit codes:
    0  — all checks PASS
    1  — at least one check FAIL
    2  — at least one WARN, no FAIL
"""
from __future__ import annotations

# Ensure the project root is on sys.path so `prism` is importable when
# the script is run directly (e.g. `python scripts/health_check.py`).
import sys as _sys
import pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
import json
import os
import platform
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class CheckResult:
    status: str  # PASS | FAIL | WARN | SKIP
    message: str
    detail: str = field(default="")


_ANSI = {
    "PASS": "\033[32m",
    "FAIL": "\033[31m",
    "WARN": "\033[33m",
    "SKIP": "\033[34m",
    "RESET": "\033[0m",
}


def _badge(status: str, use_color: bool) -> str:
    if use_color:
        return f"{_ANSI[status]}[{status}]{_ANSI['RESET']}"
    return f"[{status}]"


def _fmt(name: str, result: CheckResult, use_color: bool) -> str:
    pad = " " * (4 - len(result.status))
    badge = _badge(result.status, use_color)
    line = f"{badge}{pad}{name} — {result.message}"
    if result.detail:
        for dl in result.detail.strip().splitlines():
            line += f"\n        {dl}"
    return line


ALL_CHECKS = [
    "environment",
    "state_dir",
    "models",
    "session_clock",
    "inflight_persistence",
    "mt5_connect",
    "mt5_bars",
    "pip_value",
    "reconnect",
    "drawdown_guard",
    "slack",
]

_mt5_bridge_instance = None
_mt5_skipped = False

_REQUIRED_VARS = [
    "PRISM_SLACK_TOKEN",
    "PRISM_SLACK_CHANNEL",
    "MT5_LOGIN",
    "MT5_SERVER",
    "MT5_PASSWORD",
    "TIINGO_API_KEY",
    "FRED_API_KEY",
]
_OPTIONAL_VARS = [
    "PRISM_STATE_DIR",
    "PRISM_RISK_PCT",
    "PRISM_MAX_DAILY_LOSS_PCT",
    "PRISM_SUN_OPEN_SKIP_MIN",
]


def check_environment(instruments: list, no_slack: bool) -> CheckResult:
    missing_req = [v for v in _REQUIRED_VARS if not os.environ.get(v)]
    missing_opt = [v for v in _OPTIONAL_VARS if not os.environ.get(v)]
    if missing_req:
        return CheckResult("FAIL", f"missing required env vars: {', '.join(missing_req)}", "Set these in your .env file and re-run.")
    if missing_opt:
        return CheckResult("WARN", f"optional env vars not set: {', '.join(missing_opt)}", "Runner will use defaults; set them for full control.")
    return CheckResult("PASS", "all required env vars set")


def check_state_dir(instruments: list, no_slack: bool) -> CheckResult:
    state_path = Path(os.environ.get("PRISM_STATE_DIR", "state"))
    if not state_path.exists():
        return CheckResult("WARN", f"state dir '{state_path}' does not exist — will be created by runner")
    probe = state_path / "_health_check_probe.json"
    try:
        probe.write_text(json.dumps({"probe": True}))
        data = json.loads(probe.read_text())
        probe.unlink()
        if data.get("probe") is True:
            return CheckResult("PASS", f"state dir '{state_path}' is writable and JSON-capable")
        return CheckResult("FAIL", f"state dir round-trip returned unexpected data: {data}")
    except Exception as exc:
        return CheckResult("FAIL", f"state dir I/O error: {exc}", traceback.format_exc())


def check_models(instruments: list, no_slack: bool) -> CheckResult:
    model_dir = Path(os.environ.get("PRISM_MODEL_DIR", "models"))
    if not model_dir.exists():
        return CheckResult("FAIL", f"model directory '{model_dir}' not found")
    missing = []
    for inst in instruments:
        for layer in range(4):
            f = model_dir / f"{inst}_layer{layer}.pkl"
            if not f.exists():
                missing.append(str(f))
    if missing:
        return CheckResult("FAIL", f"{len(missing)} model file(s) missing", "\n".join(missing))
    total = len(instruments) * 4
    return CheckResult("PASS", f"all {total} model files present ({len(instruments)} instruments × 4 layers)")


def check_session_clock(instruments: list, no_slack: bool) -> CheckResult:
    try:
        from prism.delivery.session_filter import is_kill_zone, is_sunday_open_gap
    except ImportError as exc:
        return CheckResult("FAIL", f"cannot import session_filter: {exc}", traceback.format_exc())
    dt = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
    try:
        kz = is_kill_zone(dt)
        sg = is_sunday_open_gap(dt)
    except Exception as exc:
        return CheckResult("FAIL", f"session_filter raised: {exc}", traceback.format_exc())
    if not isinstance(kz, bool):
        return CheckResult("FAIL", f"is_kill_zone() returned {type(kz).__name__}, expected bool")
    if not isinstance(sg, bool):
        return CheckResult("FAIL", f"is_sunday_open_gap() returned {type(sg).__name__}, expected bool")
    return CheckResult("PASS", f"session predicates OK (2024-01-15 09:00 UTC: kill_zone={kz}, sunday_gap={sg})")


def check_inflight_persistence(instruments: list, no_slack: bool) -> CheckResult:
    # The runner persists a module-level dict _last_signal_key via _persist_inflight_keys
    try:
        import prism.delivery.runner as _rm
        persist_fn = getattr(_rm, "_persist_inflight_keys", None)
        load_fn = getattr(_rm, "_load_inflight_keys", None)
        if persist_fn is None or load_fn is None:
            return CheckResult("WARN", "inflight persistence functions not found in runner module", "Expected _persist_inflight_keys and _load_inflight_keys in prism.delivery.runner")
    except ImportError as exc:
        return CheckResult("FAIL", f"cannot import runner: {exc}", traceback.format_exc())

    with tempfile.TemporaryDirectory() as tmpdir:
        test_key = ("XAUUSD", "BUY", "2024-01-15 00:00:00")
        try:
            # Inject a test key into the runner's module-level dict and persist it
            original = dict(_rm._last_signal_key)
            _rm._last_signal_key["_hc_test"] = test_key
            persist_fn(Path(tmpdir))
            loaded = load_fn(Path(tmpdir))
            _rm._last_signal_key.clear()
            _rm._last_signal_key.update(original)
            if "_hc_test" not in loaded:
                return CheckResult("FAIL", "persisted key not found after load", f"Loaded: {loaded}")
            return CheckResult("PASS", "in-flight key survives persist → load round-trip")
        except Exception as exc:
            return CheckResult("FAIL", f"inflight persistence error: {exc}", traceback.format_exc())


def _should_skip_mt5() -> bool:
    return platform.system() == "Darwin" or not os.environ.get("MT5_LOGIN")


def check_mt5_connect(instruments: list, no_slack: bool) -> CheckResult:
    global _mt5_bridge_instance, _mt5_skipped
    if _should_skip_mt5():
        reason = "macOS detected (MT5 requires Windows)" if platform.system() == "Darwin" else "MT5_LOGIN not set"
        _mt5_skipped = True
        return CheckResult("SKIP", reason)
    try:
        from prism.execution.mt5_bridge import MT5Bridge
    except ImportError as exc:
        _mt5_skipped = True
        return CheckResult("FAIL", f"cannot import MT5Bridge: {exc}", traceback.format_exc())
    try:
        bridge = MT5Bridge()
        ok = bridge.connect()
        if ok:
            _mt5_bridge_instance = bridge
            return CheckResult("PASS", "MT5 connected successfully")
        _mt5_skipped = True
        return CheckResult("FAIL", "MT5Bridge.connect() returned False")
    except Exception as exc:
        _mt5_skipped = True
        return CheckResult("FAIL", f"MT5Bridge.connect() raised: {exc}", traceback.format_exc())


def check_mt5_bars(instruments: list, no_slack: bool) -> CheckResult:
    if _mt5_skipped or _mt5_bridge_instance is None:
        return CheckResult("SKIP", "mt5_connect skipped or failed")
    bridge = _mt5_bridge_instance
    failures = []
    for inst in instruments:
        for tf in ("H4", "H1", "M5"):
            try:
                bars = bridge.fetch_bars(inst, tf, 100)
                if bars is None or len(bars) == 0:
                    failures.append(f"{inst}/{tf}: empty result")
            except Exception as exc:
                failures.append(f"{inst}/{tf}: {exc}")
    if failures:
        return CheckResult("FAIL", f"{len(failures)} bar fetch(es) failed", "\n".join(failures))
    return CheckResult("PASS", f"H4/H1/M5 bars fetched for {', '.join(instruments)}")


def check_pip_value(instruments: list, no_slack: bool) -> CheckResult:
    if _mt5_skipped or _mt5_bridge_instance is None:
        return CheckResult("SKIP", "mt5_connect skipped or failed")
    try:
        from prism.execution.mt5_bridge import APPROX_PIP_VALUE_PER_LOT
    except ImportError:
        APPROX_PIP_VALUE_PER_LOT = {}
    bridge = _mt5_bridge_instance
    warnings = []
    failures = []
    for inst in instruments:
        try:
            live = bridge._pip_value_per_lot(inst)
            approx = APPROX_PIP_VALUE_PER_LOT.get(inst)
            if approx is not None and live == approx:
                warnings.append(f"{inst}: live pip value == approx fallback ({live})")
        except Exception as exc:
            failures.append(f"{inst}: {exc}")
    if failures:
        return CheckResult("FAIL", f"pip value errors: {'; '.join(failures)}")
    if warnings:
        return CheckResult("WARN", "pip value matches fallback for some instruments", "\n".join(warnings))
    return CheckResult("PASS", f"live pip values differ from fallback for {', '.join(instruments)}")


def check_reconnect(instruments: list, no_slack: bool) -> CheckResult:
    if _mt5_skipped or _mt5_bridge_instance is None:
        return CheckResult("SKIP", "mt5_connect skipped or failed")
    bridge = _mt5_bridge_instance
    try:
        hb = bridge._heartbeat_ok()
        if not isinstance(hb, bool):
            return CheckResult("FAIL", f"_heartbeat_ok() returned {type(hb).__name__}, expected bool")
        bridge.ensure_connected()
        event = None
        if hasattr(bridge, "pop_reconnect_event"):
            event = bridge.pop_reconnect_event()
        if event is not None:
            return CheckResult("FAIL", f"spurious reconnect event on healthy bridge: {event}")
        return CheckResult("PASS", "heartbeat OK, ensure_connected idempotent, no spurious event")
    except Exception as exc:
        return CheckResult("FAIL", f"reconnect check error: {exc}", traceback.format_exc())


def check_drawdown_guard(instruments: list, no_slack: bool) -> CheckResult:
    try:
        from prism.delivery.drawdown_guard import DrawdownGuard
    except ImportError as exc:
        return CheckResult("FAIL", f"cannot import DrawdownGuard: {exc}", traceback.format_exc())

    class _MockBridge:
        def __init__(self, balance=10000.0):
            self._balance = balance
        def get_account_balance(self):
            return self._balance
        def deals_since_utc_midnight(self, *a, **kw):
            return []

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            guard = DrawdownGuard(
                bridge=_MockBridge(10000.0),
                state_dir=tmpdir,
                max_daily_loss_pct=0.01,
                max_daily_loss_usd=None,
            )
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            guard.refresh(now)
            if guard.is_tripped:
                return CheckResult("FAIL", "DrawdownGuard.is_tripped True before any loss")
            guard.record_manual(pnl_usd=-200.0, now=now)
            if not guard.is_tripped:
                return CheckResult("FAIL", "DrawdownGuard not tripped after 2% loss (threshold 1%)")
            state_files = list(Path(tmpdir).glob("*.json"))
            if not state_files:
                return CheckResult("WARN", "guard tripped correctly but no state file found")
            return CheckResult("PASS", "drawdown guard trips at 2% loss (threshold 1%), state persisted")
        except Exception as exc:
            return CheckResult("FAIL", f"drawdown guard error: {exc}", traceback.format_exc())


def check_slack(instruments: list, no_slack: bool) -> CheckResult:
    if no_slack:
        return CheckResult("SKIP", "--no-slack flag set")
    token = os.environ.get("PRISM_SLACK_TOKEN", "")
    channel = os.environ.get("PRISM_SLACK_CHANNEL", "")
    if not token or not channel:
        return CheckResult("WARN", "PRISM_SLACK_TOKEN or PRISM_SLACK_CHANNEL not set; skipping probe")
    import urllib.request, urllib.error
    text = f"PRISM health-check probe {datetime.now(timezone.utc).isoformat()}"
    payload = json.dumps({"channel": channel, "text": text}).encode()
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
        if body.get("ok"):
            return CheckResult("PASS", f"Slack probe posted (ts={body.get('ts','?')})")
        error = body.get("error", "unknown")
        if error == "ratelimited":
            return CheckResult("WARN", "Slack returned rate_limited — bot alive but throttled")
        return CheckResult("FAIL", f"Slack API error: {error}", json.dumps(body, indent=2))
    except Exception as exc:
        return CheckResult("FAIL", f"Slack probe error: {exc}", traceback.format_exc())


def _validate_check_registry() -> None:
    this = sys.modules[__name__]
    missing = [c for c in ALL_CHECKS if not callable(getattr(this, f"check_{c}", None))]
    if missing:
        raise RuntimeError(f"ALL_CHECKS names with no implementation: {missing}")


_validate_check_registry()


def run_checks(checks, instruments, no_slack, use_json, use_color):
    this = sys.modules[__name__]
    results = []
    for name in checks:
        fn = getattr(this, f"check_{name}", None)
        if fn is None:
            results.append((name, CheckResult("FAIL", f"no check_{name}() function found")))
            continue
        try:
            result = fn(instruments, no_slack)
        except Exception as exc:
            result = CheckResult("FAIL", f"check raised unexpected exception: {exc}", traceback.format_exc())
        results.append((name, result))

    if use_json:
        print(json.dumps([{"check": n, "status": r.status, "message": r.message, "detail": r.detail} for n, r in results], indent=2))
    else:
        for name, result in results:
            print(_fmt(name, result, use_color))

    statuses = {r.status for _, r in results}
    if "FAIL" in statuses:
        return 1
    if "WARN" in statuses:
        return 2
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="PRISM pre-live health check", epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", metavar="NAME")
    parser.add_argument("--instruments", default="XAUUSD,EURUSD,GBPUSD")
    parser.add_argument("--no-slack", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        for name in ALL_CHECKS:
            print(name)
        return 0

    checks = ALL_CHECKS
    if args.check:
        if args.check not in ALL_CHECKS:
            parser.error(f"Unknown check '{args.check}'. Valid: {', '.join(ALL_CHECKS)}")
        checks = [args.check]

    instruments = [i.strip() for i in args.instruments.split(",") if i.strip()]
    use_color = not args.no_color and sys.stdout.isatty()
    if args.json:
        use_color = False

    return run_checks(checks=checks, instruments=instruments, no_slack=args.no_slack, use_json=args.json, use_color=use_color)


if __name__ == "__main__":
    sys.exit(main())
