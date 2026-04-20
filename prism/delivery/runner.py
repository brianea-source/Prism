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
import time
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

INSTRUMENTS = os.environ.get("PRISM_INSTRUMENTS", "XAUUSD,EURUSD,GBPUSD").split(",")
SCAN_INTERVAL = int(os.environ.get("PRISM_SCAN_INTERVAL", "60"))
EXECUTION_MODE = os.environ.get("PRISM_EXECUTION_MODE", "CONFIRM")


def _build_bridge():
    """Instantiate the appropriate MT5 bridge based on env credentials."""
    from prism.execution.mt5_bridge import MT5Bridge, MockMT5Bridge

    mt5_login = os.environ.get("MT5_LOGIN")
    if mt5_login:
        bridge = MT5Bridge(mode=EXECUTION_MODE)
        connected = bridge.connect()
        if connected:
            logger.info("MT5Bridge connected in %s mode", EXECUTION_MODE)
            return bridge
        logger.error("MT5 connection failed -- falling back to MockMT5Bridge")

    logger.info("Using MockMT5Bridge (demo mode) in %s mode", EXECUTION_MODE)
    bridge = MockMT5Bridge(mode=EXECUTION_MODE)
    bridge.connect()
    return bridge


def _scan_instrument(instrument: str, notifier, bridge, now: datetime) -> None:
    """Scan one instrument, generate a signal if conditions are met, deliver to Slack."""
    import pandas as pd
    from pathlib import Path

    from prism.signal.generator import SignalGenerator
    from prism.delivery.confirm_handler import PollConfirmHandler, ConfirmationResult
    from slack_sdk import WebClient

    # -- Data loading (Phase 4 will pull live bars from MT5) --
    cache_dir = Path("data/raw")
    h4_paths = sorted(cache_dir.glob(f"tiingo_*{instrument.lower()}*4hour*.parquet"))
    d1_paths = sorted(cache_dir.glob(f"tiingo_*{instrument.lower()}*daily*.parquet"))

    if not h4_paths and not d1_paths:
        logger.warning("%s: No cached data found -- skipping", instrument)
        return

    base_df = pd.read_parquet(h4_paths[0] if h4_paths else d1_paths[0])
    h4_df = base_df
    h1_df = base_df   # Phase 4: replace with real H1 from MT5
    entry_df = base_df  # Phase 4: replace with real M5 bars

    gen = SignalGenerator(instrument)
    signal = gen.generate(h4_df, h1_df, entry_df)

    if signal is None:
        logger.info("%s: No signal this scan", instrument)
        return

    logger.info(
        "%s: Signal generated — %s confidence=%.2f",
        instrument, signal.direction, signal.confidence,
    )

    # -- Delivery --
    ts = notifier.send_signal(signal, mode=bridge.mode)
    if not ts:
        logger.error("%s: Failed to send signal to Slack", instrument)
        return

    if bridge.mode == "CONFIRM":
        slack_client = WebClient(token=os.environ.get("PRISM_SLACK_TOKEN", ""))
        handler = PollConfirmHandler(slack_client, notifier.channel, ts)
        result = handler.wait(timeout_sec=300)

        if result == ConfirmationResult.CONFIRMED:
            notifier.update_signal_status(ts, "CONFIRMED", signal)
            exec_result = bridge.execute_signal(signal)
            if exec_result.success:
                notifier.update_signal_status(ts, "EXECUTED", signal)
                logger.info("%s: Executed — ticket=%s", instrument, exec_result.ticket)
            else:
                notifier.update_signal_status(ts, "FAILED", signal)
                logger.error("%s: Execution failed — %s", instrument, exec_result.error)

        elif result == ConfirmationResult.SKIPPED:
            notifier.update_signal_status(ts, "SKIPPED", signal)

        else:  # EXPIRED
            notifier.update_signal_status(ts, "EXPIRED", signal)

    elif bridge.mode == "AUTO":
        exec_result = bridge.execute_signal(signal)
        status = "EXECUTED" if exec_result.success else "FAILED"
        notifier.update_signal_status(ts, status, signal)
        if exec_result.success:
            logger.info("%s: Auto-executed — ticket=%s", instrument, exec_result.ticket)
        else:
            logger.error("%s: Auto-execution failed — %s", instrument, exec_result.error)

    # NOTIFY mode: no execution, message already sent above


def run() -> None:
    """Main signal runner loop."""
    from prism.delivery.slack_notifier import SlackNotifier
    from prism.delivery.session_filter import is_kill_zone, session_label

    notifier = SlackNotifier()
    bridge = _build_bridge()

    logger.info(
        "PRISM runner started | instruments=%s | mode=%s | scan_interval=%ds",
        INSTRUMENTS, EXECUTION_MODE, SCAN_INTERVAL,
    )

    while True:
        now = datetime.now(timezone.utc)

        if not is_kill_zone(now):
            logger.debug("Off kill zone (%s) -- sleeping %ds", session_label(now), SCAN_INTERVAL)
            time.sleep(SCAN_INTERVAL)
            continue

        logger.info("Kill zone active: %s -- scanning %s", session_label(now), INSTRUMENTS)

        for instrument in INSTRUMENTS:
            try:
                _scan_instrument(instrument, notifier, bridge, now)
            except Exception as exc:
                logger.error("Error scanning %s: %s", instrument, exc, exc_info=True)

        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    run()
