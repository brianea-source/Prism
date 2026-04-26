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

from prism.execution.mt5_bridge import PIP_SIZE, SignalPacket
from prism.delivery.session_filter import session_label

logger = logging.getLogger(__name__)

CONFIDENCE_EMOJI = {"HIGH": ":large_green_circle:", "MEDIUM": ":large_yellow_circle:", "LOW": ":red_circle:"}
DIRECTION_EMOJI = {"LONG": ":chart_with_upwards_trend:", "SHORT": ":chart_with_downwards_trend:", "NEUTRAL": ":arrow_right:"}
REGIME_EMOJI = {"RISK_OFF": ":warning:", "RISK_ON": ":white_check_mark:", "NEUTRAL": ":white_circle:"}


class SlackNotifier:
    """Delivers PRISM trade signals to Slack with optional confirmation flow."""

    def __init__(self, token: Optional[str] = None, channel: Optional[str] = None):
        # Env reads happen at instantiation time (not import time) so test
        # fixtures can set env vars before constructing a SlackNotifier.
        self.token = token or os.environ.get("PRISM_SLACK_TOKEN", "")
        self.channel = channel or os.environ.get("PRISM_SLACK_CHANNEL", "#prism-signals")
        self.confirm_timeout_sec = int(os.environ.get("PRISM_CONFIRM_TIMEOUT_SEC", "300"))
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

    def _format_signal_blocks(
        self, signal: SignalPacket, demo_warning: Optional[str] = None
    ) -> list:
        """
        Build Slack Block Kit payload for a signal.

        demo_warning: optional human-readable string rendered as a highlighted
        warning block at the TOP of the message. Used when the runner is
        feeding aliased data (e.g. H4 bars as M5) so Brian can't miss the
        context while reviewing an approval.
        """
        pip = self._pip_size(signal.instrument)
        sl_pips = abs(signal.entry - signal.sl) / pip
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

        # Session context — single source of truth via session_filter
        now_utc = datetime.now(timezone.utc)
        session_str = session_label(now_utc)

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

        # HTF Bias section (Phase 5)
        htf = getattr(signal, "htf_bias", None) or {}
        if htf:
            bias_1h = htf.get("bias_1h", "N/A")
            bias_4h = htf.get("bias_4h", "N/A")
            aligned = htf.get("aligned", False)
            allowed_dir = htf.get("allowed_direction")

            # Swing sequence enrichment is sourced from the htf_bias dict
            # itself (populated by SignalGenerator). Falling back to
            # signal.htf_result is a no-op since SignalPacket does not
            # carry the full HTFBiasResult object.
            swing_seq_1h_list = htf.get("swing_seq_1h") or []
            swing_seq_4h_list = htf.get("swing_seq_4h") or []
            swing_seq_1h = (
                f" ({' → '.join(swing_seq_1h_list)})"
                if len(swing_seq_1h_list) >= 3
                else ""
            )
            swing_seq_4h = (
                f" ({' → '.join(swing_seq_4h_list)})"
                if len(swing_seq_4h_list) >= 3
                else ""
            )

            if aligned and allowed_dir:
                align_str = f":white_check_mark: {allowed_dir} only"
            else:
                align_str = ":x: RANGING/misaligned"
            body_lines += [
                "",
                ":chart_with_upwards_trend: *HTF Bias*",
                f"  1H: {bias_1h}{swing_seq_1h}",
                f"  4H: {bias_4h}{swing_seq_4h}",
                f"  Alignment: {align_str}",
            ]

        # Show a short, human-scannable signal_id so Slack audit ↔ MT5 comment
        # ↔ server logs reconcile without copy-pasting a full UUID.
        short_id = (getattr(signal, "signal_id", "") or "")[:8] or "n/a"

        body_lines += [
            "",
            f"{regime_emoji} *Regime:*          {signal.regime} - News: {signal.news_bias}",
            f":clock1: *Session:*         {session_str}",
            "",
            f"_Conf: {signal.confidence:.2f} - Mag: ~{signal.magnitude_pips:.0f} pips - {signal.model_version}_",
            f"_Signal: `#{short_id}`  -  {signal_time}_",
        ]

        body = "\n".join(body_lines)

        blocks: list = []
        if demo_warning:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f":warning: *DEMO MODE* — {demo_warning}",
                },
            })
            blocks.append({"type": "divider"})

        blocks.extend([
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": body},
            },
            {"type": "divider"},
        ])

        # Approximate sizing warning — prepended so it leads the card.
        # Set on the signal by submit_order() when PRISM_ALLOW_APPROX_PIP_VALUE=1
        # and MT5 symbol_info was unavailable (source="approximation-forced").
        if getattr(signal, "approximate_sizing", False):
            blocks.insert(0, {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": (
                    ":warning: *Approximate lot sizing* — MT5 symbol_info unavailable. "
                    "Position size may be ~10-20% off. "
                    "Set PRISM_ALLOW_APPROX_PIP_VALUE=0 to reject instead."
                )}],
            })

        return blocks

    def _format_confirm_blocks(
        self,
        signal: SignalPacket,
        message_ts: str,
        use_buttons: bool = False,
        demo_warning: Optional[str] = None,
    ) -> list:
        """
        Build confirmation blocks below the signal.

        use_buttons=False (default) → Reaction instructions context block
                                       (Phase 3 poll mode). PollConfirmHandler
                                       watches for ✅/❌ reactions; buttons
                                       would do nothing in that flow.
        use_buttons=True             → Slack action buttons (Phase 4 webhook).

        demo_warning: forwarded to _format_signal_blocks; surfaces a big warning
        block at the top when the runner is scanning on aliased data.
        """
        blocks = self._format_signal_blocks(signal, demo_warning=demo_warning)
        timeout_min = self.confirm_timeout_sec // 60

        if use_buttons:
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
                        "text": f":timer_clock: Auto-expires in {timeout_min} minutes if no action taken",
                    }
                ],
            })
        else:
            # Poll mode: instruct the approver to use emoji reactions.
            # PollConfirmHandler polls for ✅/❌ on this message.
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            f":timer_clock: React :white_check_mark: to confirm "
                            f"or :x: to skip (auto-expires in {timeout_min} minutes)"
                        ),
                    }
                ],
            })

        return blocks

    def send_signal(
        self,
        signal: SignalPacket,
        mode: str = "CONFIRM",
        use_buttons: bool = False,
        demo_warning: Optional[str] = None,
    ) -> Optional[str]:
        """
        Send signal to Slack.

        mode: "CONFIRM" -- with confirmation prompt (buttons or reaction hint)
              "NOTIFY"  -- formatted alert only, no confirmation prompt

        use_buttons: relevant only when mode="CONFIRM".
            False (default) -- Phase 3 poll mode: show reaction instructions.
                               PollConfirmHandler monitors emoji reactions.
            True            -- Phase 4 webhook mode: render action buttons.

        demo_warning: when set, prepends a highlighted ":warning: DEMO MODE"
            section to the message. Use this whenever the runner is feeding
            aliased or simulated data so Brian cannot mistake it for a live
            M5/M15-driven signal.

        Returns message_ts (used to update the message on confirmation/expiry).
        """
        if not self.client:
            logger.warning("No Slack token configured -- signal not sent")
            return None

        try:
            if mode == "CONFIRM":
                blocks = self._format_confirm_blocks(
                    signal, "pending",
                    use_buttons=use_buttons,
                    demo_warning=demo_warning,
                )
            else:
                blocks = self._format_signal_blocks(signal, demo_warning=demo_warning)

            resp = self.client.chat_postMessage(
                channel=self.channel,
                text=f"PRISM Signal: {signal.instrument} {signal.direction} -- {signal.confidence_level}",
                blocks=blocks,
            )
            ts = resp["ts"]
            logger.info(f"Signal sent to Slack: {signal.instrument} {signal.direction} ts={ts}")

            # Poll-mode blocks don't reference message_ts, so the post-then-
            # update round-trip is pure overhead. Only webhook mode needs the
            # second call because the button action values encode the real ts.
            if mode == "CONFIRM" and use_buttons:
                updated_blocks = self._format_confirm_blocks(
                    signal, ts,
                    use_buttons=use_buttons,
                    demo_warning=demo_warning,
                )
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

    def send_alert(self, text: str) -> Optional[str]:
        """
        Send a plain alert to the signals channel. Used by the drawdown
        guard (and future operational alerts) where a single line of text
        is more appropriate than a full Block Kit signal card.

        Returns the Slack message ts on success, None otherwise (no client,
        or an API error). Callers that need one-shot semantics — e.g. the
        drawdown guard's "halt" notification — must track their own
        already-sent bit; this method doesn't dedupe.
        """
        if not self.client:
            logger.info("Slack alert (no client): %s", text)
            return None
        try:
            resp = self.client.chat_postMessage(
                channel=self.channel,
                text=text,
                blocks=[
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": text},
                    }
                ],
            )
            return resp.get("ts")
        except SlackApiError as e:
            logger.error(f"Slack alert error: {e.response['error']}")
            return None
