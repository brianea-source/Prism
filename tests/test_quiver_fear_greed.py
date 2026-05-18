"""Tests for prism.data.quiver.QuiverClient.get_fear_greed.

Mocks HTTP at the requests.get layer. Verifies:
  * Browser-fingerprint headers are sent (User-Agent + Referer + Origin).
  * Successful payload → canonical (date, fear_greed) schema.
  * Cache hit short-circuits HTTP.
  * 418 then 200 retry path works.
  * Stale cache served when network fails.
  * Empty / malformed payload doesn't blow up.
"""

from __future__ import annotations

import os
import time as _t
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

import prism.data.quiver as quiver_mod
from prism.data.quiver import QuiverClient

_FNG_PAYLOAD = {
    "fear_and_greed_historical": {
        "data": [
            {"x": 1715817600000, "y": 64.2, "rating": "greed"},  # 2024-05-16
            {"x": 1715904000000, "y": 58.5, "rating": "greed"},  # 2024-05-17
            {"x": 1715990400000, "y": 52.1, "rating": "neutral"},  # 2024-05-18
        ]
    }
}


def _mock_response(status_code: int, json_payload=None):
    m = MagicMock()
    m.status_code = status_code
    m.url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    m.json.return_value = json_payload or {}
    return m


@pytest.fixture(autouse=True)
def _isolate_cache_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quiver_mod, "JSON_CACHE_DIR", tmp_path / "state" / "cache")
    monkeypatch.setattr(quiver_mod, "CACHE_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(quiver_mod.time, "sleep", lambda _s: None)


def test_successful_fetch_parses_canonical_schema() -> None:
    with patch.object(
        quiver_mod.requests,
        "get",
        return_value=_mock_response(200, _FNG_PAYLOAD),
    ):
        df = QuiverClient().get_fear_greed()

    assert list(df.columns) == ["date", "fear_greed"]
    assert len(df) == 3
    assert df["fear_greed"].iloc[0] == pytest.approx(64.2)
    assert df["date"].is_monotonic_increasing


def test_browser_headers_are_sent() -> None:
    with patch.object(
        quiver_mod.requests,
        "get",
        return_value=_mock_response(200, _FNG_PAYLOAD),
    ) as mock_get:
        QuiverClient().get_fear_greed()

    _, kwargs = mock_get.call_args
    headers = kwargs["headers"]
    # CNN's 418 was a header check; verify all three triggers are set.
    assert "Mozilla" in headers["User-Agent"]
    assert "Safari" in headers["User-Agent"]
    assert headers["Referer"].startswith("https://edition.cnn.com")
    assert headers["Origin"] == "https://edition.cnn.com"


def test_418_then_200_retry_succeeds() -> None:
    responses = [
        _mock_response(418),
        _mock_response(200, _FNG_PAYLOAD),
    ]
    with patch.object(quiver_mod.requests, "get", side_effect=responses) as mock_get:
        df = QuiverClient().get_fear_greed()

    assert mock_get.call_count == 2
    assert len(df) == 3


def test_cache_hit_skips_http(tmp_path: Path) -> None:
    cache_path = tmp_path / "state" / "cache" / "fear_greed.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    quiver_mod.write_json_cache(cache_path, _FNG_PAYLOAD)

    with patch.object(quiver_mod.requests, "get") as mock_get:
        df = QuiverClient().get_fear_greed()

    mock_get.assert_not_called()
    assert len(df) == 3


def test_network_failure_falls_back_to_stale_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "state" / "cache" / "fear_greed.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    quiver_mod.write_json_cache(cache_path, _FNG_PAYLOAD)
    past = _t.time() - quiver_mod.FNG_CACHE_TTL_SEC - 100
    os.utime(cache_path, (past, past))

    with patch.object(
        quiver_mod.requests,
        "get",
        return_value=_mock_response(418),
    ):
        df = QuiverClient().get_fear_greed()

    assert len(df) == 3  # served from stale cache


def test_persistent_418_no_cache_returns_empty(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("ERROR", logger="prism.data.quiver"):
        with patch.object(
            quiver_mod.requests,
            "get",
            return_value=_mock_response(418),
        ):
            df = QuiverClient().get_fear_greed()

    assert df.empty
    assert list(df.columns) == ["date", "fear_greed"]
    # The error log includes the status so the next outage is debuggable.
    assert any("418" in r.message for r in caplog.records)


def test_malformed_payload_returns_empty() -> None:
    with patch.object(
        quiver_mod.requests,
        "get",
        return_value=_mock_response(200, {"unexpected": "shape"}),
    ):
        df = QuiverClient().get_fear_greed()
    assert df.empty


def test_module_level_helper_delegates() -> None:
    with patch.object(
        quiver_mod.requests,
        "get",
        return_value=_mock_response(200, _FNG_PAYLOAD),
    ):
        df = quiver_mod.get_fear_greed()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
