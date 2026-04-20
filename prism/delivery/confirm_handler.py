"""
PRISM Confirmation Handler
Listens for Slack button interactions (CONFIRM/SKIP) and routes to MT5Bridge.

In production: this runs as a small Flask webhook that Slack sends button events to.
For Phase 3 (pre-webhook): uses a polling approach checking Slack message reactions.
Slack Interactivity requires a public endpoint (ngrok or VPS URL).
"""
import logging
import time as time_module
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class ConfirmationResult:
    CONFIRMED = "CONFIRMED"
    SKIPPED = "SKIPPED"
    EXPIRED = "EXPIRED"


class PollConfirmHandler:
    """
    Polls Slack for check-mark or x reactions on the signal message.
    Fallback approach when Slack Interactivity endpoint is not set up yet.

    Usage:
        handler = PollConfirmHandler(slack_client, channel, message_ts)
        result = handler.wait(timeout_sec=300)
        # result: "CONFIRMED", "SKIPPED", or "EXPIRED"
    """

    CONFIRM_REACTION = "white_check_mark"   # checkmark emoji
    SKIP_REACTION = "x"                     # x emoji

    def __init__(self, client, channel: str, message_ts: str):
        self.client = client
        self.channel = channel
        self.message_ts = message_ts

    def wait(self, timeout_sec: int = 300, poll_interval_sec: int = 10) -> str:
        """
        Block until confirm or skip reaction is added, or until timeout.
        Returns a ConfirmationResult constant.
        """
        deadline = time_module.time() + timeout_sec
        logger.info(
            f"Waiting for confirmation on ts={self.message_ts} (timeout={timeout_sec}s)"
        )

        while time_module.time() < deadline:
            try:
                resp = self.client.reactions_get(
                    channel=self.channel,
                    timestamp=self.message_ts,
                )
                reactions = resp.get("message", {}).get("reactions", [])
                for r in reactions:
                    if r["name"] == self.CONFIRM_REACTION:
                        logger.info("Signal CONFIRMED via reaction")
                        return ConfirmationResult.CONFIRMED
                    if r["name"] == self.SKIP_REACTION:
                        logger.info("Signal SKIPPED via reaction")
                        return ConfirmationResult.SKIPPED
            except Exception as exc:
                logger.warning(f"Reaction poll error: {exc}")

            time_module.sleep(poll_interval_sec)

        logger.info("Signal EXPIRED -- no action in timeout window")
        return ConfirmationResult.EXPIRED


class WebhookConfirmHandler:
    """
    Production approach: Slack sends button clicks to a webhook endpoint.

    Set Slack app Interactivity URL to:
        https://<your-vps>/prism/slack/interactions

    Handler extracts action_id and routes accordingly.
    """

    @staticmethod
    def parse_interaction(payload: dict) -> Tuple[str, str]:
        """
        Parse Slack interaction payload.

        Returns:
            (action_id, message_ts) tuple.
            action_id: "prism_confirm" | "prism_skip" | "" if unrecognised
        """
        actions = payload.get("actions", [])
        if not actions:
            return "", ""
        action = actions[0]
        action_id = action.get("action_id", "")
        value = action.get("value", "")
        return action_id, value
