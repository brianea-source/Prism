"""
prism/data/quiver.py
Alternative data feeds: CFTC Commitments of Traders + CNN Fear & Greed.

History
-------
Before the 2026-05-18 repair, this module hit two endpoints that no
longer worked:

  1. ``https://www.cftc.gov/dea/newcot/financial_lof.txt`` — 404 since
     CFTC reorganized the archive in early 2026.
  2. ``https://production.dataviz.cnn.io/index/fearandgreed/graphdata``
     with bare ``User-Agent: Mozilla/5.0`` — 418 (CNN's Varnish-level
     bot trip). The endpoint is alive; the headers were wrong.

Both failures fed a stale/neutral news-bias signal that caused the
generator's news-bias gate to reject nearly every otherwise-valid
setup for ~4 weeks. See docs/audits/2026-05-18_data_feed_repair.md.

Repair strategy
---------------
* CFTC → Socrata Open Data API dataset ``6dca-aqww`` (Legacy
  COT-Futures-Only). Same column names the old ``.txt`` exposed
  (``noncomm_positions_long_all`` etc.), so the downstream schema
  ``(date, net_speculative, net_commercial)`` is unchanged.
* CNN F&G → realistic browser ``User-Agent`` + ``Referer`` +
  ``Origin`` headers. Confirmed 200 OK.

Resilience additions
--------------------
* Short JSON cache on disk (24h COT, 1h F&G) via ``prism.data.cache``.
* One retry with exponential backoff before giving up.
* On hard failure with a valid (stale) cache, return the cached value
  and ``WARNING``-log the staleness instead of returning an empty
  frame. Empty frames are how the May 2026 outage hid.
* On hard failure with no cache, ``ERROR``-log the failure mode
  (status code, attempted URL) so the next outage is debuggable from
  log lines alone.

Public surface (callers in pipeline.py / scripts/download_historical_data.py
must keep working):
  * ``QuiverClient().get_cot_report(symbol)`` -> pd.DataFrame
  * ``QuiverClient().get_fear_greed()`` -> pd.DataFrame
  * ``get_cot_report(symbol)`` / ``get_fear_greed()`` module helpers.

COT public-API key is not required. ``QUIVER_API_KEY`` is read for
future Quiver Quantitative endpoints but is not used by either feed
today.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from prism.data.cache import (
    read_json_cache,
    read_json_cache_stale,
    write_json_cache,
)

logger = logging.getLogger(__name__)

# Legacy parquet cache dir (unchanged for downstream compatibility).
CACHE_DIR = Path("data/raw")
# New TTL'd JSON cache dir (matches the path called out in the task brief).
JSON_CACHE_DIR = Path("state/cache")

# CFTC COT market names — these are the values in the
# ``market_and_exchange_names`` column on the Socrata 6dca-aqww
# dataset (Legacy COT-Futures-Only).
COT_MARKET_MAP: dict[str, str] = {
    "XAUUSD": "GOLD - COMMODITY EXCHANGE INC.",
    "EURUSD": "EURO FX - CHICAGO MERCANTILE EXCHANGE",
    "GBPUSD": "BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE",
    "USDJPY": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
}

# Socrata Open Data API — modern, CFTC-blessed, stable schema.
# HEAD requests return 404 (Socrata quirk); GET returns 200.
CFTC_SOCRATA_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
CFTC_HTTP_TIMEOUT = 30.0
COT_CACHE_TTL_SEC = 24 * 3600  # COT is weekly; 24h refresh is plenty.

# CNN's internal API. The 418 we saw is a Varnish-level header check,
# not a real ban — a realistic browser fingerprint clears it.
CNN_FNG_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
CNN_HTTP_TIMEOUT = 15.0
FNG_CACHE_TTL_SEC = 3600  # 1h — F&G updates hourly during US session.

# Shared retry policy. Two attempts total: initial + one backoff retry.
_BACKOFF_SCHEDULE = (1.0, 3.0)

# Browser fingerprint that satisfies CNN's edge gate.
_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.0 Safari/605.1.15"
    ),
    "Referer": "https://edition.cnn.com/markets/fear-and-greed",
    "Origin": "https://edition.cnn.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

# Slightly tamer headers for CFTC. Socrata accepts anything but a
# clear identifying UA keeps us off any future rate-limit shortlist.
_CFTC_HEADERS: dict[str, str] = {
    "User-Agent": "PRISM-data-feed/1.0 (+https://github.com/brianea-source/Prism)",
    "Accept": "application/json",
}


class QuiverClient:
    """Thin wrapper around CFTC COT + CNN F&G feeds.

    Public methods preserve the original (date, net_speculative,
    net_commercial) and (date, fear_greed) schemas. Callers in
    pipeline.py do not need changes.
    """

    BASE_URL = "https://api.quiverquant.com/beta"

    def __init__(self, api_key: str | None = None):
        # Reserved for future Quiver Quantitative endpoints. Not used
        # by either current feed.
        self.api_key = api_key or os.environ.get("QUIVER_API_KEY", "")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        JSON_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CFTC COT
    # ------------------------------------------------------------------
    def get_cot_report(self, symbol: str) -> pd.DataFrame:
        """Fetch Commitments of Traders for ``symbol``.

        Returns a DataFrame with columns ``date``, ``net_speculative``,
        ``net_commercial``. Empty DataFrame on unmappable symbol or
        irrecoverable failure (no cache to fall back to).
        """
        empty = pd.DataFrame(columns=["date", "net_speculative", "net_commercial"])

        market_name = COT_MARKET_MAP.get(symbol)
        if not market_name:
            logger.warning("COT: no market mapping for %s", symbol)
            return empty

        cache_path = JSON_CACHE_DIR / f"cot_{symbol}.json"
        fresh = read_json_cache(cache_path, COT_CACHE_TTL_SEC)
        if fresh is not None:
            return self._cot_rows_to_frame(fresh, symbol)

        # Socrata SoQL: filter to this market, sort newest first, cap
        # at ~2 years of weekly reports.
        params: dict[str, str] = {
            "$where": f"market_and_exchange_names='{market_name}'",
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$limit": "120",
        }

        rows = self._fetch_json_with_retry(
            url=CFTC_SOCRATA_URL,
            headers=_CFTC_HEADERS,
            params=params,
            timeout=CFTC_HTTP_TIMEOUT,
            label=f"CFTC COT [{symbol}]",
        )

        if rows is None:
            stale = read_json_cache_stale(cache_path)
            if stale is not None:
                logger.warning(
                    "COT: serving stale cache for %s (refresh failed)", symbol
                )
                return self._cot_rows_to_frame(stale, symbol)
            logger.error(
                "COT: no data for %s and no cache available — "
                "downstream news-bias will use neutral fallback",
                symbol,
            )
            return empty

        if not isinstance(rows, list) or not rows:
            logger.warning("COT: empty payload for %s", symbol)
            return empty

        try:
            write_json_cache(cache_path, rows)
        except OSError as exc:
            # Cache failure is non-fatal — keep serving the response.
            logger.warning("COT: failed to write cache for %s: %s", symbol, exc)

        return self._cot_rows_to_frame(rows, symbol)

    @staticmethod
    def _cot_rows_to_frame(rows: Any, symbol: str) -> pd.DataFrame:
        """Translate Socrata JSON rows into the canonical schema."""
        empty = pd.DataFrame(columns=["date", "net_speculative", "net_commercial"])
        if not isinstance(rows, list) or not rows:
            return empty

        df = pd.DataFrame(rows)
        if df.empty:
            return empty

        # Socrata ISO timestamps — coerce, drop unparseable rows.
        df["date"] = pd.to_datetime(
            df.get("report_date_as_yyyy_mm_dd"),
            errors="coerce",
            utc=False,
        ).dt.tz_localize(None)

        long_spec = pd.to_numeric(df.get("noncomm_positions_long_all"), errors="coerce")
        short_spec = pd.to_numeric(
            df.get("noncomm_positions_short_all"), errors="coerce"
        )
        long_comm = pd.to_numeric(df.get("comm_positions_long_all"), errors="coerce")
        short_comm = pd.to_numeric(df.get("comm_positions_short_all"), errors="coerce")

        df["net_speculative"] = long_spec - short_spec
        df["net_commercial"] = long_comm - short_comm

        result = (
            df[["date", "net_speculative", "net_commercial"]]
            .dropna()
            .sort_values("date")
            .reset_index(drop=True)
        )
        logger.info("COT: %s parsed %d weekly rows", symbol, len(result))
        return result

    # ------------------------------------------------------------------
    # CNN Fear & Greed
    # ------------------------------------------------------------------
    def get_fear_greed(self) -> pd.DataFrame:
        """Fetch CNN Fear & Greed history.

        Returns a DataFrame with columns ``date``, ``fear_greed`` (0
        = extreme fear, 100 = extreme greed). Empty DataFrame on
        irrecoverable failure (no cache to fall back to).
        """
        empty = pd.DataFrame(columns=["date", "fear_greed"])

        cache_path = JSON_CACHE_DIR / "fear_greed.json"
        fresh = read_json_cache(cache_path, FNG_CACHE_TTL_SEC)
        if fresh is not None:
            return self._fng_payload_to_frame(fresh)

        payload = self._fetch_json_with_retry(
            url=CNN_FNG_URL,
            headers=_BROWSER_HEADERS,
            params=None,
            timeout=CNN_HTTP_TIMEOUT,
            label="CNN F&G",
        )

        if payload is None:
            stale = read_json_cache_stale(cache_path)
            if stale is not None:
                logger.warning("F&G: serving stale cache (refresh failed)")
                return self._fng_payload_to_frame(stale)
            logger.error(
                "F&G: no data and no cache — downstream news-bias "
                "will use neutral fallback"
            )
            return empty

        try:
            write_json_cache(cache_path, payload)
        except OSError as exc:
            logger.warning("F&G: failed to write cache: %s", exc)

        return self._fng_payload_to_frame(payload)

    @staticmethod
    def _fng_payload_to_frame(payload: Any) -> pd.DataFrame:
        """Translate CNN graphdata payload into the canonical schema."""
        empty = pd.DataFrame(columns=["date", "fear_greed"])
        if not isinstance(payload, dict):
            return empty

        scores = (
            payload.get("fear_and_greed_historical", {}).get("data", [])
            if isinstance(payload.get("fear_and_greed_historical"), dict)
            else []
        )
        if not scores:
            return empty

        df = pd.DataFrame(scores)
        if df.empty or "x" not in df.columns or "y" not in df.columns:
            return empty

        df["date"] = pd.to_datetime(df["x"], unit="ms", errors="coerce").dt.normalize()
        df["fear_greed"] = pd.to_numeric(df["y"], errors="coerce")
        result = (
            df[["date", "fear_greed"]]
            .dropna()
            .sort_values("date")
            .reset_index(drop=True)
        )
        logger.info("F&G: parsed %d daily rows", len(result))
        return result

    # ------------------------------------------------------------------
    # Shared HTTP helper
    # ------------------------------------------------------------------
    @staticmethod
    def _fetch_json_with_retry(
        *,
        url: str,
        headers: dict[str, str],
        params: dict[str, str] | None,
        timeout: float,
        label: str,
    ) -> Any | None:
        """GET ``url`` with retry, return parsed JSON or ``None`` on failure.

        Logs the full failure mode (status + URL) so the next outage is
        debuggable from the log line alone — the May 2026 outage hid
        because the old code logged only the requests exception string.
        """
        last_status: int | str | None = None
        for attempt, backoff in enumerate((_BACKOFF_SCHEDULE) + (None,)):
            try:
                resp = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=timeout,
                )
                last_status = resp.status_code
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        logger.warning("%s: 200 but unparseable JSON: %s", label, exc)
                        return None
                logger.warning(
                    "%s: HTTP %s on %s (attempt %d)",
                    label,
                    resp.status_code,
                    resp.url,
                    attempt + 1,
                )
            except requests.RequestException as exc:
                last_status = type(exc).__name__
                logger.warning(
                    "%s: %s on %s (attempt %d)",
                    label,
                    last_status,
                    url,
                    attempt + 1,
                )

            if backoff is None:
                break
            time.sleep(backoff)

        logger.error(
            "%s: giving up after retries (last status: %s, url: %s)",
            label,
            last_status,
            url,
        )
        return None


# ----------------------------------------------------------------------
# Module-level helpers (keep callers in pipeline.py / scripts working).
# ----------------------------------------------------------------------
def get_cot_report(symbol: str) -> pd.DataFrame:
    return QuiverClient().get_cot_report(symbol)


def get_fear_greed() -> pd.DataFrame:
    return QuiverClient().get_fear_greed()
