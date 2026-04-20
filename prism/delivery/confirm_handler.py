"""
PRISM Confirmation Handler
Listens for Slack button interactions (CONFIRM/SKIP) and routes to MT5Bridge.

In production: this runs as a small Flask webhook that Slack sends button events to.
For Phase 3 (pre-webhook): uses a polling approach checking Slack message reactions.
Slack Interactivity requires a public endpoint (ngrok or VPS URL).
"""
import logging
import time as time_module
from typing import Iterable, Optional, Set, Tuple

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
        handler = PollConfirmHandler(
            slack_client, channel, message_ts,
            approvers={"U012ABC", "U345DEF"},  # optional allow-list
        )
        result = handler.wait(timeout_sec=300)
        # result: "CONFIRMED", "SKIPPED", or "EXPIRED"

    Authorisation:
        If ``approvers`` is provided and non-empty, only reactions from users in
        that set count. Reactions from anyone else (including Slack bots that
        auto-react, or channel members who aren't on the trading desk) are
        ignored. If ``approvers`` is None or empty, any reactor is accepted —
        keep the allow-list set in production.
    """

    CONFIRM_REACTION = "white_check_mark"   # checkmark emoji
    SKIP_REACTION = "x"                     # x emoji

    def __init__(
        self,
        client,
        channel: str,
        message_ts: str,
        approvers: Optional[Iterable[str]] = None,
    ):
        self.client = client
        self.channel = channel
        self.message_ts = message_ts
        # Normalise to a set of non-empty user IDs; treat empty iterable as None
        # so callers can pass ``os.environ.get("PRISM_APPROVERS","").split(",")``
        # without having to filter beforehand.
        self.approvers: Optional[Set[str]] = None
        if approvers:
            filtered = {u.strip() for u in approvers if u and u.strip()}
            self.approvers = filtered or None

    def _reaction_from_approver(self, reaction: dict) -> bool:
        """Return True if this reaction counts per the approver allow-list."""
        if not self.approvers:
            return True
        users = reaction.get("users") or []
        return any(u in self.approvers for u in users)

    def wait(self, timeout_sec: int = 300, poll_interval_sec: int = 10) -> str:
        """
        Block until an authorised confirm/skip reaction is added, or timeout.
        Returns a ConfirmationResult constant.
        """
        deadline = time_module.time() + timeout_sec
        logger.info(
            "Waiting for confirmation on ts=%s (timeout=%ds, approvers=%s)",
            self.message_ts,
            timeout_sec,
            "any" if not self.approvers else sorted(self.approvers),
        )

        while time_module.time() < deadline:
            try:
                resp = self.client.reactions_get(
                    channel=self.channel,
                    timestamp=self.message_ts,
                    full=True,  # ensure ``users`` list is populated
                )
                reactions = resp.get("message", {}).get("reactions", [])
                for r in reactions:
                    if r["name"] == self.CONFIRM_REACTION and self._reaction_from_approver(r):
                        logger.info("Signal CONFIRMED via reaction")
                        return ConfirmationResult.CONFIRMED
                    if r["name"] == self.SKIP_REACTION and self._reaction_from_approver(r):
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
