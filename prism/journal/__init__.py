"""PRISM trade journal — GitHub Issues backend.

Each fired signal is mirrored as a GitHub issue. Lifecycle events
(fill, TP1, close) post comments and update labels so the issue
reflects the current state of the trade.

The journal is intentionally side-band: the runner records signals
through ``signal_audit`` regardless. ``github_issues`` is layered on
top and gated by ``PRISM_GITHUB_JOURNAL_ENABLED`` so a GitHub outage
never affects signal delivery.
"""

from prism.journal import github_issues

__all__ = ["github_issues"]
