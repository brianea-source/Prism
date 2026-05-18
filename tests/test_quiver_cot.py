"""Tests for prism.data.quiver.QuiverClient.get_cot_report.

Mocks HTTP at the requests.get layer. Verifies:
  * Socrata JSON → canonical (date, net_speculative, net_commercial) schema.
  * Cache hit short-circuits HTTP.
  * Cache miss writes a fresh cache on success.
  * HTTP failure with stale cache → returns stale cache.
  * HTTP failure with no cache → empty DataFrame.
  * Unmapped symbol → empty DataFrame without HTTP.
  * Retry semantics (one retry on non-200).
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

_SOCRATA_EURUSD_ROW = {
    "market_and_exchange_names": "EURO FX - CHICAGO MERCANTILE EXCHANGE",
    "report_date_as_yyyy_mm_dd": "2026-05-12T00:00:00.000",
    "noncomm_positions_long_all": "224002",
    "noncomm_positions_short_all": "183802",
    "comm_positions_long_all": "485382",
    "comm_positions_short_all": "564436",
}

_SOCRATA_EURUSD_OLDER = {
    "market_and_exchange_names": "EURO FX - CHICAGO MERCANTILE EXCHANGE",
    "report_date_as_yyyy_mm_dd": "2026-05-05T00:00:00.000",
    "noncomm_positions_long_all": "210000",
    "noncomm_positions_short_all": "190000",
    "comm_positions_long_all": "480000",
    "comm_positions_short_all": "560000",
}


def _mock_response(status_code: int, json_payload=None, raise_on_json: bool = False):
    m = MagicMock()
    m.status_code = status_code
    m.url = "https://publicreporting.cftc.gov/resource/6dca-aqww.json?mocked"
    if raise_on_json:
        m.json.side_effect = ValueError("not json")
    else:
        m.json.return_value = json_payload or []
    return m


@pytest.fixture(autouse=True)
def _isolate_cache_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect quiver's cache dirs to tmp_path. Sleep is patched out."""
    monkeypatch.setattr(quiver_mod, "JSON_CACHE_DIR", tmp_path / "state" / "cache")
    monkeypatch.setattr(quiver_mod, "CACHE_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(quiver_mod.time, "sleep", lambda _s: None)


def test_unmapped_symbol_returns_empty_without_http() -> None:
    with patch.object(quiver_mod.requests, "get") as mock_get:
        df = QuiverClient().get_cot_report("BTCUSD")
    assert df.empty
    assert list(df.columns) == ["date", "net_speculative", "net_commercial"]
    mock_get.assert_not_called()


def test_successful_fetch_parses_canonical_schema() -> None:
    rows = [_SOCRATA_EURUSD_ROW, _SOCRATA_EURUSD_OLDER]
    with patch.object(
        quiver_mod.requests, "get", return_value=_mock_response(200, rows)
    ):
        df = QuiverClient().get_cot_report("EURUSD")

    assert list(df.columns) == ["date", "net_speculative", "net_commercial"]
    assert len(df) == 2
    # Sorted ascending by date.
    assert df["date"].is_monotonic_increasing
    # Net speculative for the May 12 row: 224002 - 183802 = 40200
    latest = df.iloc[-1]
    assert latest["net_speculative"] == pytest.approx(40200)
    # Net commercial: 485382 - 564436 = -79054
    assert latest["net_commercial"] == pytest.approx(-79054)


def test_cache_hit_skips_http(tmp_path: Path) -> None:
    # Pre-seed cache.
    cache_path = tmp_path / "state" / "cache" / "cot_EURUSD.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    quiver_mod.write_json_cache(cache_path, [_SOCRATA_EURUSD_ROW])

    with patch.object(quiver_mod.requests, "get") as mock_get:
        df = QuiverClient().get_cot_report("EURUSD")

    mock_get.assert_not_called()
    assert len(df) == 1
    assert df.iloc[0]["net_speculative"] == pytest.approx(40200)


def test_cache_written_after_successful_fetch(tmp_path: Path) -> None:
    cache_path = tmp_path / "state" / "cache" / "cot_EURUSD.json"
    assert not cache_path.exists()

    with patch.object(
        quiver_mod.requests,
        "get",
        return_value=_mock_response(200, [_SOCRATA_EURUSD_ROW]),
    ):
        QuiverClient().get_cot_report("EURUSD")

    assert cache_path.exists()


def test_http_failure_falls_back_to_stale_cache(tmp_path: Path) -> None:
    # Seed cache, then backdate it past TTL.
    cache_path = tmp_path / "state" / "cache" / "cot_EURUSD.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    quiver_mod.write_json_cache(cache_path, [_SOCRATA_EURUSD_ROW])
    past = _t.time() - quiver_mod.COT_CACHE_TTL_SEC - 100
    os.utime(cache_path, (past, past))

    with patch.object(
        quiver_mod.requests,
        "get",
        return_value=_mock_response(404),
    ):
        df = QuiverClient().get_cot_report("EURUSD")

    # Came from stale cache, not the failed HTTP.
    assert len(df) == 1
    assert df.iloc[0]["net_speculative"] == pytest.approx(40200)


def test_http_failure_no_cache_returns_empty() -> None:
    with patch.object(
        quiver_mod.requests,
        "get",
        return_value=_mock_response(404),
    ):
        df = QuiverClient().get_cot_report("EURUSD")

    assert df.empty
    assert list(df.columns) == ["date", "net_speculative", "net_commercial"]


def test_request_exception_no_cache_returns_empty() -> None:
    with patch.object(
        quiver_mod.requests,
        "get",
        side_effect=quiver_mod.requests.ConnectionError("dns boom"),
    ):
        df = QuiverClient().get_cot_report("EURUSD")
    assert df.empty


def test_retry_on_non_200_then_success() -> None:
    responses = [
        _mock_response(503),
        _mock_response(200, [_SOCRATA_EURUSD_ROW]),
    ]
    with patch.object(quiver_mod.requests, "get", side_effect=responses) as mock_get:
        df = QuiverClient().get_cot_report("EURUSD")

    assert mock_get.call_count == 2
    assert len(df) == 1


def test_two_attempts_then_give_up_logs_error(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("ERROR", logger="prism.data.quiver"):
        with patch.object(
            quiver_mod.requests,
            "get",
            return_value=_mock_response(500),
        ) as mock_get:
            QuiverClient().get_cot_report("EURUSD")

    # initial + 2 backoff retries = 3 attempts
    assert mock_get.call_count == 3
    assert any("giving up" in r.message for r in caplog.records)


def test_empty_socrata_response_returns_empty() -> None:
    with patch.object(
        quiver_mod.requests,
        "get",
        return_value=_mock_response(200, []),
    ):
        df = QuiverClient().get_cot_report("EURUSD")
    assert df.empty


def test_unparseable_json_returns_empty() -> None:
    with patch.object(
        quiver_mod.requests,
        "get",
        return_value=_mock_response(200, raise_on_json=True),
    ):
        df = QuiverClient().get_cot_report("EURUSD")
    assert df.empty


def test_module_level_helper_delegates() -> None:
    with patch.object(
        quiver_mod.requests,
        "get",
        return_value=_mock_response(200, [_SOCRATA_EURUSD_ROW]),
    ):
        df = quiver_mod.get_cot_report("EURUSD")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
