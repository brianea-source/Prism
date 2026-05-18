"""Tests for prism.data.cache — TTL'd atomic JSON cache."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from prism.data.cache import (
    read_json_cache,
    read_json_cache_stale,
    write_json_cache,
)


def test_write_then_read_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "feed.json"
    payload = {"fear_greed": 42, "asof": "2026-05-18"}

    write_json_cache(path, payload)
    got = read_json_cache(path, ttl_sec=3600)

    assert got == payload


def test_read_returns_none_when_missing(tmp_path: Path) -> None:
    assert read_json_cache(tmp_path / "missing.json", ttl_sec=3600) is None


def test_read_returns_none_when_expired(tmp_path: Path) -> None:
    path = tmp_path / "expired.json"
    write_json_cache(path, [1, 2, 3])

    # Backdate mtime 10 minutes; ttl 60s → expired.
    past = time.time() - 600
    os.utime(path, (past, past))

    assert read_json_cache(path, ttl_sec=60) is None


def test_read_returns_none_when_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.json"
    path.write_text("{not valid json", encoding="utf-8")

    assert read_json_cache(path, ttl_sec=3600) is None
    # And stale read also tolerates it (returns None, doesn't raise).
    assert read_json_cache_stale(path) is None


def test_stale_read_ignores_ttl(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    write_json_cache(path, {"a": 1})
    past = time.time() - 99_999
    os.utime(path, (past, past))

    # Fresh read says no.
    assert read_json_cache(path, ttl_sec=60) is None
    # Stale read says yes — that's the graceful-fallback contract.
    assert read_json_cache_stale(path) == {"a": 1}


def test_write_is_atomic_no_temp_left_behind(tmp_path: Path) -> None:
    path = tmp_path / "atomic.json"
    write_json_cache(path, {"x": 1})

    # Only the final file should be present — tempfiles get renamed.
    leftovers = [p.name for p in tmp_path.iterdir() if p.name != path.name]
    assert leftovers == [], f"unexpected temp files: {leftovers}"


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "deeply" / "nested" / "dir" / "feed.json"
    write_json_cache(path, [1, 2, 3])

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == [1, 2, 3]


def test_zero_ttl_treated_as_always_expired(tmp_path: Path) -> None:
    path = tmp_path / "now.json"
    write_json_cache(path, {"ok": True})

    # ttl_sec=0 → always refetch. Useful in tests; documents intent.
    assert read_json_cache(path, ttl_sec=0) is None


@pytest.mark.parametrize(
    "payload",
    [
        {"k": "v"},
        [1, 2, 3, 4],
        [{"a": 1}, {"b": 2}],
        {"nested": {"deep": [1, 2, {"x": "y"}]}},
    ],
)
def test_roundtrip_preserves_structure(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "shape.json"
    write_json_cache(path, payload)
    assert read_json_cache(path, ttl_sec=60) == payload
