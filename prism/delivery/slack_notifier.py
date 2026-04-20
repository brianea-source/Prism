"""
PRISM Slack Signal Delivery
Sends formatted trade signals to #prism-signals in Brian Corp workspace.
Supports CONFIRM mode (interactive approve/skip) and NOTIFY mode (alert only).

Signal format:
  XAUUSD  LONG  MEDIUM
  -----
  Entry (M5):    2,385.50
  SL:            2,371.20  (14.3 pips)
  TP1:           2,400.50  RR 1:1.0  (50% close)
  TP2:           2,428.40  RR 1:3.0  (50% close)
  ICC Phase:     CONTINUATION
  FVG Zone:      H4 2,378.40-2,382.60 (retest)
  Regime:        RISK_OFF - News: NEUTRAL
  Session:       London Kill Zone (08:30 UTC)
  Conf: 0.74  |  PRISM v2.0
"""
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from prism.execution.mt5_bridge import SignalPacket

logger = logging.getLogger(__name__)

SLACK_TOKEN = os.environ.get("PRISM_SLACK_TOKEN", "")
SLACK_CHANNEL = os.environ.get("PRISM_SLACK_CHANNEL", "#prism-signals")
CONFIRM_TIMEOUT_SEC = int(os.environ.get("PRISM_CONFIRM_TIMEOUT_SEC", "300"))  # 5 min

# Pip sizes per instrument
PIP_SIZE = {"XAUUSD": 0.01, "EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01}

CONFIDENCE_EMOJI = {"HIGH": ":large_green_circle:", "MEDIUM": ":large_yellow_circle:", "LOW": ":red_circle:"}
DIRECTION_EMOJI = {"LONG": ":chart_with_upwards_trend:", "SHORT": ":chart_with_downwards_trend:", "NEUTRAL": ":arrow_right:"}
REGIME_EMOJI = {"RISK_OFF": ":warning:", "RISK_ON": ":white_check_mark:", "NEUTRAL": ":white_circle:"}


class SlackNotifier:
    """Delivers PRISM trade signals to Slack with optional confirmation flow."""

    def __init__(self, token: Optional[str] = None, channel: Optional[str] = None):
        self.token = token or SLACK_TOKEN
        self.channel = channel or SLACK_CHANNEL
        self.client = WebClient(token=self.token) if self.token else None

    def _pip_size(self, instrument: str) -> float:
        return PIP_SIZE.get(instrument, 0.0001)

    def _format_pips(self, instrument: str, price_diff: float) -> str:
        pip = self._pip_size(instrument)
        pips = abs(price_diff) / pip
        return f"{pips:.1f} pips"

    def _calc_rr(self, entry: float, sl: float, tp: float) -> float:
        sl_dist = abs(entry - sl)
        tp_dist = abs(tp - entry)
        return round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0.0

    def _format_signal_blocks(self, signal: SignalPacket) -> list:
        """Build Slack Block Kit payload for a signal."""
        pip = self._pip_size(signal.instrument)
        sl_pips = abs(signal.entry - signal.sl) / pip
        tp1_pips = abs(signal.tp1 - signal.entry) / pip
        tp2_pips = abs(signal.tp2 - signal.entry) / pip
        rr1 = self._calc_rr(signal.entry, signal.sl, signal.tp1)
        rr2 = self._calc_rr(signal.entry, signal.sl, signal.tp2)

        conf_emoji = CONFIDENCE_EMOJI.get(signal.confidence_level, ":white_circle:")
        dir_emoji = DIRECTION_EMOJI.get(signal.direction, ":arrow_right:")
        regime_emoji = REGIME_EMOJI.get(signal.regime, ":white_circle:")

        # ICC phase info from fvg_zone
        fvg = signal.fvg_zone or {}
        fvg_str = ""
        if fvg:
            tf = fvg.get("timeframe", "H4")
            bottom = fvg.get("bottom", 0)
            top = fvg.get("top", 0)
            mitigated = "retest :white_check_mark:" if fvg.get("partially_mitigated") else "fresh zone :new:"
            fvg_str = f"{tf} {bottom:.2f}-{top:.2f} ({mitigated})"

        # Session context
        now_utc = datetime.now(timezone.utc)
        hour_utc = now_utc.hour
        if 7 <= hour_utc < 11:
            session_str = f"London Kill Zone ({now_utc.strftime('%H:%M')} UTC)"
        elif 13 <= hour_utc < 17:
            session_str = f"NY Kill Zone ({now_utc.strftime('%H:%M')} UTC)"
        else:
            session_str = f"Off-session ({now_utc.strftime('%H:%M')} UTC)"

        header = f"{signal.instrument}  {signal.direction}  {conf_emoji} {signal.confidence_level}"
        signal_time = signal.signal_time[:19].replace("T", " ") + " UTC"

        body_lines = [
            f"*{dir_emoji} {header}*",
            "------------------------------",
            f":round_pushpin: *Entry (M5/M15):*  `{signal.entry:.5f}`",
            f":octagonal_sign: *SL:*              `{signal.sl:.5f}`  ({sl_pips:.1f} pips)",
            f":dart: *TP1:*             `{signal.tp1:.5f}`  RR 1:{rr1}  _(50% close)_",
            f":dart: *TP2:*             `{signal.tp2:.5f}`  RR 1:{rr2}  _(50% close)_",
            "",
            f":bar_chart: *ICC Phase:*       CONTINUATION",
        ]

        if fvg_str:
            body_lines.append(f":package: *FVG Zone:*        {fvg_str}")

        body_lines += [
            f"{regime_emoji} *Regime:*          {signal.regime} - News: {signal.news_bias}",
            f":clock1: *Session:*         {session_str}",
            "",
            f"_Conf: {signal.confidence:.2f} - Mag: ~{signal.magnitude_pips:.0f} pips - {signal.model_version}_",
            f"_Signal time: {signal_time}_",
        ]

        body = "\n".join(body_lines)

        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": body},
            },
            {"type": "divider"},
        ]
        return blocks

    def _format_confirm_blocks(self, signal: SignalPacket, message_ts: str) -> list:
        """Add CONFIRM/SKIP action buttons below the signal."""
        blocks = self._format_signal_blocks(signal)
        blocks.append({
            "type": "actions",
            "block_id": f"prism_confirm_{signal.signal_time}",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "CONFIRM"},
                    "style": "primary",
                    "action_id": "prism_confirm",
                    "value": message_ts,
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "SKIP"},
                    "style": "danger",
                    "action_id": "prism_skip",
                    "value": message_ts,
                },
            ],
        })
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f":timer_clock: Auto-expires in {CONFIRM_TIMEOUT_SEC // 60} minutes if no action taken",
                }
            ],
        })
        return blocks

    def send_signal(self, signal: SignalPacket, mode: str = "CONFIRM") -> Optional[str]:
        """
        Send signal to Slack.

        mode: "CONFIRM" -- with approve/skip buttons
              "NOTIFY"  -- formatted alert only, no buttons

        Returns message_ts (used to update the message on confirmation/expiry).
        """
        if not self.client:
            logger.warning("No Slack token configured -- signal not sent")
            return None

        try:
            if mode == "CONFIRM":
                blocks = self._format_confirm_blocks(signal, "pending")
            else:
                blocks = self._format_signal_blocks(signal)

            resp = self.client.chat_postMessage(
                channel=self.channel,
                text=f"PRISM Signal: {signal.instrument} {signal.direction} -- {signal.confidence_level}",
                blocks=blocks,
            )
            ts = resp["ts"]
            logger.info(f"Signal sent to Slack: {signal.instrument} {signal.direction} ts={ts}")

            # Update message with correct ts stamped into confirm button value
            if mode == "CONFIRM":
                updated_blocks = self._format_confirm_blocks(signal, ts)
                self.client.chat_update(
                    channel=self.channel,
                    ts=ts,
                    blocks=updated_blocks,
                    text=f"PRISM Signal: {signal.instrument} {signal.direction}",
                )

            return ts

        except SlackApiError as e:
            logger.error(f"Slack API error: {e.response['error']}")
            return None

    def update_signal_status(self, ts: str, status: str, signal: SignalPacket) -> None:
        """
        Update a previously sent signal message with its outcome.
        status: "CONFIRMED" | "SKIPPED" | "EXPIRED" | "EXECUTED" | "FAILED"
        """
        if not self.client:
            return

        status_map = {
            "CONFIRMED": ":white_check_mark: *CONFIRMED* -- Sending to MT5...",
            "SKIPPED": ":x: *SKIPPED* -- Signal dismissed",
            "EXPIRED": ":alarm_clock: *EXPIRED* -- No action taken in 5 minutes",
            "EXECUTED": ":rocket: *EXECUTED* -- Order placed on MT5",
            "FAILED": ":rotating_light: *FAILED* -- MT5 execution error",
        }
        status_text = status_map.get(status, status)

        try:
            blocks = self._format_signal_blocks(signal)
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": status_text},
            })
            self.client.chat_update(
                channel=self.channel,
                ts=ts,
                blocks=blocks,
                text=f"PRISM Signal {status}: {signal.instrument} {signal.direction}",
            )
        except SlackApiError as e:
            logger.error(f"Slack update error: {e.response['error']}")

    def send_daily_brief(self, stats: dict) -> None:
        """Send daily PnL summary to #prism-signals."""
        if not self.client:
            return

        lines = [
            "*:bar_chart: PRISM Daily Brief*",
            f"Date: {stats.get('date', 'today')}",
            f"Signals fired: {stats.get('signals_fired', 0)}",
            (
                f"Confirmed: {stats.get('confirmed', 0)}  |  "
                f"Skipped: {stats.get('skipped', 0)}  |  "
                f"Expired: {stats.get('expired', 0)}"
            ),
            (
                f"Executed: {stats.get('executed', 0)}  |  "
                f"Wins: {stats.get('wins', 0)}  |  "
                f"Losses: {stats.get('losses', 0)}"
            ),
            f"Win rate: {stats.get('win_rate', 0):.1%}  |  Net R: {stats.get('net_r', 0):+.2f}R",
        ]

        try:
            self.client.chat_postMessage(
                channel=self.channel,
                text="\n".join(lines),
                blocks=[
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": "\n".join(lines)},
                    }
                ],
            )
        except SlackApiError as e:
            logger.error(f"Slack daily brief error: {e.response['error']}")
