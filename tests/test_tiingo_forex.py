"""Test that forex pairs are routed to /tiingo/fx/ and produce a 6-column
OHLCV frame with volume=0 (the FX endpoint omits volume)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from prism.data.tiingo import FOREX_TICKERS, TiingoClient


# Sample payload that mirrors what /tiingo/fx/{ticker}/prices returns:
# date, ticker, open, high, low, close (no volume).
SAMPLE_FX_PAYLOAD = [
    {
        "date": "2024-01-02T00:00:00.000Z",
        "ticker": "eurusd",
        "open": 1.1042,
        "high": 1.1058,
        "low": 1.1031,
        "close": 1.1049,
    },
    {
        "date": "2024-01-02T01:00:00.000Z",
        "ticker": "eurusd",
        "open": 1.1049,
        "high": 1.1062,
        "low": 1.1045,
        "close": 1.1055,
    },
    {
        "date": "2024-01-02T02:00:00.000Z",
        "ticker": "eurusd",
        "open": 1.1055,
        "high": 1.1068,
        "low": 1.1050,
        "close": 1.1063,
    },
]


def _make_response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


@pytest.fixture
def fx_client(tmp_path, monkeypatch):
    # Redirect cache dir so we don't pollute / collide with real cached data.
    from prism.data import tiingo as tiingo_mod

    monkeypatch.setattr(tiingo_mod, "CACHE_DIR", tmp_path)
    # Skip the rate-limit sleep in tests.
    monkeypatch.setattr(tiingo_mod.TiingoClient, "RATE_LIMIT_DELAY", 0)
    client = TiingoClient(api_key="test-key")
    # Re-create cache dir under tmp_path because __init__ already mkdir'd the
    # module-level CACHE_DIR before we patched it.
    tmp_path.mkdir(parents=True, exist_ok=True)
    return client


def test_forex_tickers_constant_contains_majors():
    assert FOREX_TICKERS == {"EURUSD", "GBPUSD", "USDJPY"}


def test_eurusd_routes_to_fx_endpoint(fx_client, monkeypatch):
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return _make_response(SAMPLE_FX_PAYLOAD)

    monkeypatch.setattr(fx_client.session, "get", fake_get)

    df = fx_client.get_ohlcv("EURUSD", "2024-01-01", "2024-01-05", "1hour")

    # Routing assertion: hit /tiingo/fx/EURUSD/prices, NOT /iex/.
    assert captured["url"].endswith("/tiingo/fx/EURUSD/prices"), captured["url"]
    assert "/iex/" not in captured["url"]
    assert captured["params"]["resampleFreq"] == "1Hour"
    assert captured["params"]["startDate"] == "2024-01-01"
    assert captured["params"]["endDate"] == "2024-01-05"

    # Schema assertion: 6 columns, volume synthesized to 0.
    assert set(df.columns) == {"datetime", "open", "high", "low", "close", "volume"}
    assert len(df) == 3
    assert (df["volume"] == 0).all()
    # Sanity: EUR/USD prices around 1.10 (per task spec).
    assert df["close"].between(1.0, 1.2).all()


def test_gbpusd_and_usdjpy_also_route_to_fx(fx_client, monkeypatch):
    seen_urls = []

    def fake_get(url, params=None, timeout=None):
        seen_urls.append(url)
        return _make_response(SAMPLE_FX_PAYLOAD)

    monkeypatch.setattr(fx_client.session, "get", fake_get)

    fx_client.get_ohlcv("GBPUSD", "2024-01-01", "2024-01-02", "1hour")
    fx_client.get_ohlcv("USDJPY", "2024-01-01", "2024-01-02", "15min")

    assert any(u.endswith("/tiingo/fx/GBPUSD/prices") for u in seen_urls)
    assert any(u.endswith("/tiingo/fx/USDJPY/prices") for u in seen_urls)
    assert not any("/iex/" in u for u in seen_urls)


def test_xauusd_still_uses_iex_via_gld(fx_client, monkeypatch):
    """XAUUSD must NOT be routed to /fx/ — it stays on the GLD equity proxy."""
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        # GLD comes from IEX with volume present.
        return _make_response([
            {
                "date": "2024-01-02T15:30:00.000Z",
                "open": 190.0,
                "high": 191.0,
                "low": 189.5,
                "close": 190.5,
                "volume": 12345,
            }
        ])

    monkeypatch.setattr(fx_client.session, "get", fake_get)

    df = fx_client.get_ohlcv("XAUUSD", "2024-01-01", "2024-01-05", "1hour")
    assert "/iex/GLD/prices" in captured["url"]
    assert "/tiingo/fx/" not in captured["url"]
    # IEX response carried volume — should be preserved (not zeroed).
    assert df["volume"].iloc[0] == 12345


def test_forex_cache_round_trip(fx_client, monkeypatch):
    """Second call should hit the cache and not re-fetch."""
    call_count = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        call_count["n"] += 1
        return _make_response(SAMPLE_FX_PAYLOAD)

    monkeypatch.setattr(fx_client.session, "get", fake_get)

    df1 = fx_client.get_ohlcv("EURUSD", "2024-01-01", "2024-01-05", "1hour")
    df2 = fx_client.get_ohlcv("EURUSD", "2024-01-01", "2024-01-05", "1hour")

    assert call_count["n"] == 1, "second call should be cached"
    assert list(df1.columns) == list(df2.columns)
    assert len(df2) == 3
    assert (df2["volume"] == 0).all()
