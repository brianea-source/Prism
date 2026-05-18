# Data-Feed Repair — CFTC COT + CNN Fear & Greed (2026-05-18)

**Author:** Ada (assistant, dispatched by Brian / Greg)
**Triggered by:** PR #35 audit (`docs/audits/2026-05-18_signal_audit_gap.md`) on branch `ops/audit-log-enable-2026-05-18` — the news-bias gate is killing signal flow, and `prism/data/quiver.py` has been failing every scan with `404` (CFTC COT) and `418` (CNN F&G) for weeks.
**Scope:** Surgical repair of two upstream feed endpoints in `prism/data/quiver.py`. Stage 2 (news-bias gate recalibration) is deliberately not in this PR — that needs 48h of audit data first.

---

## TL;DR

Two upstream feeds 24/7 fail. The old code masked the failure with empty DataFrames, so the news-bias signal stayed pinned to a stale/neutral fallback, and the generator's news-bias gate rejected nearly every otherwise-valid setup for ~4 weeks.

| Feed | Old endpoint | Status | New endpoint | Status |
|---|---|---|---|---|
| CFTC COT | `https://www.cftc.gov/dea/newcot/financial_lof.txt` | **HTTP/2 404** | `https://publicreporting.cftc.gov/resource/6dca-aqww.json` (Socrata Open Data API) | **200, current data through 2026-05-12** |
| CNN F&G | `https://production.dataviz.cnn.io/...graphdata` with bare `User-Agent: Mozilla/5.0` | **HTTP/2 418** | Same URL, browser-fingerprint headers (`User-Agent` + `Referer` + `Origin`) | **200 OK** |

Both feeds also get a short on-disk JSON cache (24h COT, 1h F&G) at `state/cache/`, one retry with backoff, and a graceful stale-cache fallback on network failure. **All three failure modes** (no cache + can't reach upstream, retries exhausted, parse failure) now `ERROR`-log the URL and status code so the next outage is debuggable from log lines alone.

---

## Evidence — Before/After cURL

Run from the Mac mini, 2026-05-18 15:28 UTC.

### CFTC COT — old endpoint

```text
$ curl -sI "https://www.cftc.gov/dea/newcot/financial_lof.txt" -A "Mozilla/5.0"
HTTP/2 404
date: Mon, 18 May 2026 15:28:01 GMT
content-type: text/html; charset=UTF-8
```

Confirmed dead. CFTC reorganized the archive in early 2026 — the `dea/newcot/*.txt` paths are gone.

### CFTC COT — new endpoint (Socrata Open Data API, `6dca-aqww`)

```text
$ curl -s "https://publicreporting.cftc.gov/resource/6dca-aqww.json?\$where=market_and_exchange_names%20like%20'%25EURO%20FX%25'&\$order=report_date_as_yyyy_mm_dd%20DESC&\$limit=2" -A "Mozilla/5.0" | jq '.[0] | {market_and_exchange_names, report_date_as_yyyy_mm_dd, noncomm_positions_long_all, noncomm_positions_short_all, comm_positions_long_all, comm_positions_short_all}'
{
  "market_and_exchange_names": "EURO FX - CHICAGO MERCANTILE EXCHANGE",
  "report_date_as_yyyy_mm_dd": "2026-05-12T00:00:00.000",
  "noncomm_positions_long_all": "224002",
  "noncomm_positions_short_all": "183802",
  "comm_positions_long_all": "485382",
  "comm_positions_short_all": "564436"
}
```

Note: `HEAD` returns 404 (Socrata quirk — don't be fooled, it's a working dataset). The dataset is **Legacy COT-Futures-Only**, same column names the old `.txt` exposed (`noncomm_positions_long_all`, `comm_positions_long_all`, etc.), so the downstream canonical schema `(date, net_speculative, net_commercial)` is unchanged. Pipeline.py merges and column names need zero edits.

Dataset choice rationale:
- `gpe5-46if` (TFF — Traders in Financial Futures) was the first candidate but has a *different* column schema (`asset_mgr_positions_long`, `lev_money_positions_long`, ...) and would require translating downstream consumers. Rejected.
- `6dca-aqww` (Legacy COT-Futures-Only) matches the old `.txt` schema exactly. Chosen.

### CNN F&G — bare User-Agent

```text
$ curl -sI "https://production.dataviz.cnn.io/index/fearandgreed/graphdata" -A "Mozilla/5.0"
HTTP/2 418
server: Varnish
retry-after: 0
```

CNN's Varnish edge throws 418 on suspicious fingerprints. `retry-after: 0` means "I'm not banning you, your headers are just wrong."

### CNN F&G — browser fingerprint

```text
$ curl -sI "https://production.dataviz.cnn.io/index/fearandgreed/graphdata" \
    -A "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15" \
    -H "Referer: https://edition.cnn.com/markets/fear-and-greed" \
    -H "Origin: https://edition.cnn.com" \
    -H "Accept: application/json, text/plain, */*"
HTTP/2 200
content-type: application/json
x-api-cache: Miss
expires: Mon, 18 May 2026 15:24:46 GMT
cache-control: max-age=5
```

Endpoint is healthy. Header check confirmed — not a real ban.

---

## What changed in the code

### `prism/data/quiver.py` (rewritten)

- `get_cot_report(symbol)` now hits the Socrata `6dca-aqww` JSON endpoint with a SoQL filter on `market_and_exchange_names`, parses the same column names the old `.txt` exposed, and returns the unchanged canonical schema `(date, net_speculative, net_commercial)`.
- `get_fear_greed()` sends the browser-fingerprint headers (`User-Agent` + `Referer` + `Origin` + `Accept` + `Accept-Language`). Returns the unchanged canonical schema `(date, fear_greed)`.
- Both methods:
  - Try cache first (24h for COT, 1h for F&G) at `state/cache/cot_<SYMBOL>.json` / `state/cache/fear_greed.json`.
  - On miss, fetch with **one retry** at backoff `(1s, 3s)`.
  - On non-200 with a valid (stale) cache available: `WARNING`-log and serve the stale cache. **Better than the silent neutral fallback that masked the May 2026 outage.**
  - On non-200 with no cache: `ERROR`-log the status + URL, return empty DataFrame. Downstream's `np.nan` fill path takes over (preserving prior behavior so unrelated tests pass), but now the log line clearly identifies the failure mode.

### `prism/data/cache.py` (new, ~120 LOC)

Tiny TTL'd JSON cache with atomic writes (tempfile + `os.replace`). Three public functions:
- `read_json_cache(path, ttl_sec)` → fresh value or `None` (missing | corrupt | expired all collapse to `None`).
- `read_json_cache_stale(path)` → cached value ignoring TTL, for graceful-fallback path.
- `write_json_cache(path, data)` → atomic write, creates parent dirs.

Intentionally minimal — no LRU, no schema validation, no namespaces. Just enough to keep PRISM standing during transient upstream outages.

### Tests

- `tests/test_cache.py` — 12 cases. Roundtrip, missing, expired, corrupt, atomic writes, parent dir creation, zero-TTL, parametrized payload shapes.
- `tests/test_quiver_cot.py` — 12 cases. Unmapped symbol, successful parse, cache hit, cache write, stale-cache fallback, no-cache empty, request exception, retry-then-success, two-attempts-then-give-up logging, empty payload, unparseable JSON, module-level helper.
- `tests/test_quiver_fear_greed.py` — 8 cases. Successful parse, browser headers asserted, 418→200 retry, cache hit, stale-cache fallback, persistent-418 empty + error log, malformed payload, module-level helper.

Local test run on `main` worktree (2026-05-18 15:34 UTC):
```
$ python3 -m pytest tests/ -q
821 passed, 26 warnings in 6.61s   # the 26 warnings are pre-existing pandas/sklearn deprecations
```

All 32 new tests pass; no regressions across the existing 789 tests.

---

## Deploy notes

After merge, on the VPS:
```cmd
cd C:\Prism
git pull
```

No env-var changes. No new dependencies (uses `requests` and `pandas` already in `requirements.txt`). The `state/cache/` directory is auto-created on first call.

Within minutes of next scan you should see in `runner.log`:
```
INFO prism.data.quiver — COT: EURUSD parsed N weekly rows
INFO prism.data.quiver — F&G: parsed M daily rows
```
in place of the prior 404 / 418 errors.

---

## What this PR explicitly does NOT do

**Stage 2 (separate PR) — news-bias gate recalibration.** The 4-week signal blackout (PR #35) is plausibly caused by *the combination* of (a) these failing feeds and (b) overly aggressive news-bias-gate calibration in `prism/signal/generator.py`. This PR fixes only (a). After deploy, let 48h of audit data accumulate, then look at:
- How often does the news-bias signal still flip to neutral with healthy feeds?
- What's the actual rejection rate at the news-bias gate?
- Is the gate weight calibrated against the *unhealthy* feed distribution from the past month, in which case it needs to be re-tuned against the now-healthy distribution?

That work needs data this PR is currently blocking. Land this first, watch the logs, then file the Stage 2 PR.
