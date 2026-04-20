"""
PRISM Signal Runner -- main loop.
Runs continuously, checks session + generates signals + delivers to Slack.

Usage:
    PRISM_SLACK_TOKEN=xoxb-... TIINGO_API_KEY=... FRED_API_KEY=... \\
        python prism/delivery/runner.py

Environment variables:
    PRISM_SLACK_TOKEN     -- Brian Corp PRISM Signals bot token
    PRISM_SLACK_CHANNEL   -- default: #prism-signals
    PRISM_EXECUTION_MODE  -- CONFIRM (default) | AUTO | NOTIFY
    TIINGO_API_KEY        -- for live news + data
    FRED_API_KEY          -- for macro features
    MT5_LOGIN             -- Exness account login
    MT5_SERVER            -- e.g. Exness-MT5Real
    MT5_PASSWORD          -- Exness account password
    PRISM_INSTRUMENTS     -- comma-separated, default: XAUUSD,EURUSD,GBPUSD
    PRISM_SCAN_INTERVAL   -- seconds between scans, default: 60
"""
import logging
import os
import signal as signal_module
import time
from datetime import datetime, timezone
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful shutdown flag — set by SIGTERM / SIGINT handlers
# ---------------------------------------------------------------------------
_shutdown = False


def _handle_sigterm(signum, frame):
    """Set the shutdown flag so the main loop exits cleanly."""
    global _shutdown
    logger.info("SIGTERM received — PRISM shutting down gracefully")
    _shutdown = True


# ---------------------------------------------------------------------------
# Daily brief tracking
# ---------------------------------------------------------------------------
_last_brief_date = None


def _maybe_send_daily_brief(notifier, stats_accumulator: dict, now: datetime) -> None:
    """Fire send_daily_brief once per day at 22:00 UTC, then reset stats."""
    global _last_brief_date
    if now.hour == 22 and _last_brief_date != now.date():
        notifier.send_daily_brief(stats_accumulator)
        _last_brief_date = now.date()
        stats_accumulator.clear()


# ---------------------------------------------------------------------------
# Bridge factory
# ---------------------------------------------------------------------------

def _build_bridge(execution_mode: str):
    """Instantiate the appropriate MT5 bridge based on env credentials."""
    from prism.execution.mt5_bridge import MT5Bridge, MockMT5Bridge

    mt5_login = os.environ.get("MT5_LOGIN")
    if mt5_login:
        bridge = MT5Bridge(mode=execution_mode)
        connected = bridge.connect()
        if connected:
            logger.info("MT5Bridge connected in %s mode", execution_mode)
            return bridge
        logger.error("MT5 connection failed -- falling back to MockMT5Bridge")

    logger.info("Using MockMT5Bridge (demo mode) in %s mode", execution_mode)
    bridge = MockMT5Bridge(mode=execution_mode)
    bridge.connect()
    return bridge


# ---------------------------------------------------------------------------
# Cache path resolution
# ---------------------------------------------------------------------------

def _resolve_cache_paths(instrument: str, timeframe: str) -> list:
    """
    Resolve the parquet cache paths for an instrument + timeframe.

    Cache files are written by TiingoClient.get_ohlcv as
        tiingo_{ticker}_{timeframe}_{start}_{end}.parquet
    where ticker = INSTRUMENT_MAP[symbol] (e.g. XAUUSD -> GLD). We must look up
    the mapped ticker instead of globbing on the MT5 symbol name, otherwise
    XAUUSD never matches its GLD-named cache file. We also glob case-
    insensitively so FX pairs resolve whether the ticker is stored upper- or
    lowercase (the mapping flipped case across PRs).
    """
    from pathlib import Path
    from prism.data.tiingo import INSTRUMENT_MAP

    cache_dir = Path("data/raw")
    ticker = INSTRUMENT_MAP.get(instrument, instrument)

    candidates = {ticker, ticker.lower(), ticker.upper()}
    paths: list = []
    for t in candidates:
        paths.extend(cache_dir.glob(f"tiingo_{t}_{timeframe}_*.parquet"))
    return sorted(set(paths))


# ---------------------------------------------------------------------------
# Per-instrument scan
# ---------------------------------------------------------------------------

# Shown on every signal while the runner is feeding H4 bars into the M5 slot.
# Flip off once Phase 4 wires real live bars from MT5.
_DEMO_WARNING = (
    "H4 bars are being aliased as H1/M5. FVG break-retest is disabled. "
    "Signals are directional only — verify manually before confirming."
)


def _scan_instrument(
    instrument: str,
    notifier,
    bridge,
    now: datetime,
    stats: dict = None,
    approvers: Optional[set] = None,
) -> None:
    """Scan one instrument, generate a signal if conditions are met, deliver to Slack."""
    import pandas as pd

    from prism.signal.generator import SignalGenerator
    from prism.delivery.confirm_handler import PollConfirmHandler, ConfirmationResult

    # -- Data loading (Phase 4 will pull live bars from MT5) --
    h4_paths = _resolve_cache_paths(instrument, "4hour")
    d1_paths = _resolve_cache_paths(instrument, "daily")

    if not h4_paths and not d1_paths:
        logger.warning("%s: No cached data found -- skipping", instrument)
        return

    base_df = pd.read_parquet(h4_paths[0] if h4_paths else d1_paths[0])

    if stats is None:
        stats = {}

    # Phase 4 will replace these with real H1/M5 bars from MT5 bridge.
    # Until then, H4 data is aliased — FVG break-retest logic disabled.
    h4_df = base_df
    h1_df = base_df       # NOTE: aliased to H4 — NOT real H1 data
    entry_df = base_df    # NOTE: aliased to H4 — NOT real M5 data
    logger.warning("%s: %s", instrument, _DEMO_WARNING)

    # persist_fvg=False because entry_df is H4 aliased data, not real M5;
    # retest validation is meaningless on the alias.
    gen = SignalGenerator(instrument, persist_fvg=False)
    signal = gen.generate(h4_df, h1_df, entry_df)

    if signal is None:
        logger.info("%s: No signal this scan", instrument)
        return

    logger.info(
        "%s: Signal generated — %s confidence=%.2f id=%s",
        instrument, signal.direction, signal.confidence,
        getattr(signal, "signal_id", "n/a"),
    )
    stats["signals_fired"] = stats.get("signals_fired", 0) + 1

    # -- Delivery --
    # DEMO MODE warning rides on every signal while we're on aliased bars so
    # Brian sees the banner in Slack, not just the server log.
    ts = notifier.send_signal(
        signal,
        mode=bridge.mode,
        use_buttons=False,
        demo_warning=_DEMO_WARNING,
    )
    if not ts:
        logger.error("%s: Failed to send signal to Slack", instrument)
        return

    if bridge.mode == "CONFIRM":
        # Reuse the notifier's WebClient instead of spinning a fresh one per
        # scan. Same token, same rate-limit pool.
        handler = PollConfirmHandler(
            notifier.client,
            notifier.channel,
            ts,
            approvers=approvers,
        )
        # Honour PRISM_CONFIRM_TIMEOUT_SEC set on the notifier so the Slack
        # context block ("auto-expires in N min") and the actual wait stay
        # in sync.
        result = handler.wait(timeout_sec=notifier.confirm_timeout_sec)

        if result == ConfirmationResult.CONFIRMED:
            notifier.update_signal_status(ts, "CONFIRMED", signal)
            stats["confirmed"] = stats.get("confirmed", 0) + 1
            exec_result = bridge.submit_order(signal)
            if exec_result.success:
                notifier.update_signal_status(ts, "EXECUTED", signal)
                stats["executed"] = stats.get("executed", 0) + 1
                logger.info("%s: Executed — ticket=%s", instrument, exec_result.ticket)
            else:
                notifier.update_signal_status(ts, "FAILED", signal)
                logger.error("%s: Execution failed — %s", instrument, exec_result.error)

        elif result == ConfirmationResult.SKIPPED:
            notifier.update_signal_status(ts, "SKIPPED", signal)
            stats["skipped"] = stats.get("skipped", 0) + 1

        else:  # EXPIRED
            notifier.update_signal_status(ts, "EXPIRED", signal)
            stats["expired"] = stats.get("expired", 0) + 1

    elif bridge.mode == "AUTO":
        exec_result = bridge.execute_signal(signal)
        status = "EXECUTED" if exec_result.success else "FAILED"
        notifier.update_signal_status(ts, status, signal)
        if exec_result.success:
            logger.info("%s: Auto-executed — ticket=%s", instrument, exec_result.ticket)
            stats["executed"] = stats.get("executed", 0) + 1
        else:
            logger.error("%s: Auto-execution failed — %s", instrument, exec_result.error)

    # NOTIFY mode: no execution, message already sent above


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run() -> None:
    """Main signal runner loop."""
    # Register signal handlers before any blocking I/O
    signal_module.signal(signal_module.SIGTERM, _handle_sigterm)
    signal_module.signal(signal_module.SIGINT, _handle_sigterm)

    # Read env at run() time, not import time, so test fixtures work correctly
    instruments = os.environ.get("PRISM_INSTRUMENTS", "XAUUSD,EURUSD,GBPUSD").split(",")
    scan_interval = int(os.environ.get("PRISM_SCAN_INTERVAL", "60"))
    execution_mode = os.environ.get("PRISM_EXECUTION_MODE", "CONFIRM")
    # Slack user IDs allowed to approve a signal via reaction. If unset, ANY
    # reactor in the channel can approve — safe for demo, dangerous in prod.
    approvers_raw = os.environ.get("PRISM_APPROVERS", "")
    approvers = {u.strip() for u in approvers_raw.split(",") if u.strip()} or None
    if not approvers:
        logger.warning(
            "PRISM_APPROVERS not set — ANY reactor in the channel can approve "
            "trades. Set PRISM_APPROVERS=U01...,U02... to restrict."
        )

    from prism.delivery.slack_notifier import SlackNotifier
    from prism.delivery.session_filter import is_kill_zone, session_label

    notifier = SlackNotifier()
    bridge = _build_bridge(execution_mode)

    # Stats accumulator — cleared after each daily brief
    stats: dict = {}

    logger.info(
        "PRISM runner started | instruments=%s | mode=%s | scan_interval=%ds | approvers=%s",
        instruments, execution_mode, scan_interval,
        "any" if not approvers else sorted(approvers),
    )

    while not _shutdown:
        now = datetime.now(timezone.utc)

        # Fire daily brief at 22:00 UTC
        _maybe_send_daily_brief(notifier, stats, now)

        if not is_kill_zone(now):
            logger.debug("Off kill zone (%s) -- sleeping %ds", session_label(now), scan_interval)
            time.sleep(scan_interval)
            continue

        logger.info("Kill zone active: %s -- scanning %s", session_label(now), instruments)

        for instrument in instruments:
            if _shutdown:
                break
            try:
                _scan_instrument(
                    instrument, notifier, bridge, now, stats,
                    approvers=approvers,
                )
            except Exception as exc:
                logger.error("Error scanning %s: %s", instrument, exc, exc_info=True)

        time.sleep(scan_interval)

    logger.info("PRISM runner stopped")


if __name__ == "__main__":
    run()
