"""
prism/data/cache.py
Tiny JSON file cache with TTL for upstream data feeds (CFTC COT, CNN F&G, ...).

Why this exists
---------------
Upstream feeds break in two annoying ways:
  1. Endpoint URL changes (CFTC reorganized in early 2026).
  2. Bot-mitigation flips (CNN F&G returns 418 to bare User-Agents).

Both fail intermittently. A short on-disk cache lets PRISM:
  * Avoid hammering upstream when fresh data is on disk.
  * Serve a graceful stale-but-usable value during transient outages
    (better than the silent neutral fallback that masked the 4-week
    signal blackout in May 2026).

Design
------
* JSON on disk — human-readable, easy to inspect from the VPS.
* Atomic writes via tempfile + os.replace (rename) — no half-written
  files if the process is killed mid-write.
* Read returns ``None`` for ``missing | corrupt | expired`` so callers
  treat all three identically (refetch).
* ``read_json_cache_stale`` exposes the same payload regardless of TTL
  for the "network down, return last-known-good" path.

Public API
----------
* ``read_json_cache(path, ttl_sec)`` -> ``dict | list | None``
* ``read_json_cache_stale(path)`` -> ``dict | list | None``
* ``write_json_cache(path, data)`` -> ``None``
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _path(path: str | os.PathLike[str]) -> Path:
    return Path(path)


def read_json_cache(
    path: str | os.PathLike[str],
    ttl_sec: float,
) -> Any | None:
    """Return cached JSON value if fresh, else ``None``.

    Returns ``None`` if the file is missing, corrupt, or older than
    ``ttl_sec`` seconds. Callers treat all three the same: refetch.

    ``ttl_sec <= 0`` is treated as "always expired" — useful in tests.
    """
    p = _path(path)
    if not p.exists():
        return None
    try:
        age = time.time() - p.stat().st_mtime
    except OSError:
        return None
    if ttl_sec <= 0 or age > ttl_sec:
        return None
    return _read_payload(p)


def read_json_cache_stale(path: str | os.PathLike[str]) -> Any | None:
    """Return cached value regardless of age. Used for graceful fallback.

    Returns ``None`` only if the file is missing or unreadable.
    """
    p = _path(path)
    if not p.exists():
        return None
    return _read_payload(p)


def write_json_cache(
    path: str | os.PathLike[str],
    data: Any,
) -> None:
    """Atomically write ``data`` as JSON to ``path``.

    Creates parent directories as needed. Uses tempfile + os.replace so
    a crash mid-write cannot leave a half-written cache file behind
    (which ``read_json_cache`` would then misclassify as corrupt and
    silently refetch on every call).
    """
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # NamedTemporaryFile in the same dir → os.replace is atomic on the
    # same filesystem.
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{p.name}.",
        suffix=".tmp",
        dir=str(p.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, separators=(",", ":"), default=str)
        os.replace(tmp_name, p)
    except Exception:
        # Best-effort cleanup; never raise from cleanup path.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _read_payload(p: Path) -> Any | None:
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("cache: dropping corrupt/unreadable file %s: %s", p, exc)
        return None
