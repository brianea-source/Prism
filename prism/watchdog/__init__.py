"""PRISM self-sufficiency package.

Holds the watchdog, drift monitor, and daily digest processes that run on
the production VPS alongside the live runner. Each module is independently
schedulable (via ``schtasks`` on Windows) and degrades gracefully when its
external dependencies — Slack, audit log, scheduled-task service — are
unavailable.

Components:
  - :mod:`prism.watchdog.watchdog` — runner liveness loop + auto-restart
  - :mod:`prism.watchdog.drift_monitor` — daily model-drift gate + auto-retrain
  - :mod:`prism.watchdog.daily_digest` — 08:00 UTC Slack health report
"""

__all__ = ("watchdog", "drift_monitor", "daily_digest")
