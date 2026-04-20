#!/usr/bin/env python3
"""
scripts/download_historical_data.py
------------------------------------
Download initial training data for PRISM Phase 0.

Downloads OHLCV price data for all Phase 1 instruments plus
FRED macro features and saves everything to data/raw/.

Usage
-----
    python scripts/download_historical_data.py \\
        --start 2018-01-01 \\
        --end   2026-04-19

    # Or with specific instruments:
    python scripts/download_historical_data.py \\
        --start 2018-01-01 --end 2026-04-19 \\
        --instruments EURUSD GBPUSD

    # Dry run (show what would be fetched):
    python scripts/download_historical_data.py --start 2020-01-01 --dry-run

Environment variables required
-------------------------------
    TIINGO_API_KEY  — Tiingo API key (price data)
    FRED_API_KEY    — FRED API key (macro data, optional)
    QUIVER_API_KEY  — Quiver Quantitative key (optional)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---- ensure repo root is on the path -----------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("download_historical")

# Phase 1 instruments (Tiingo mapped)
PHASE_1_INSTRUMENTS = ["EURUSD", "GBPUSD", "XAUUSD"]
PHASE_2_INSTRUMENTS = ["USDJPY"]  # future

INTRADAY_FREQ = "1hour"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download PRISM Phase 0 historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--instruments",
        nargs="+",
        default=PHASE_1_INSTRUMENTS,
        help=f"Instrument codes to download (default: {PHASE_1_INSTRUMENTS})",
    )
    parser.add_argument(
        "--freq",
        default=INTRADAY_FREQ,
        help=f"Intraday frequency (default: {INTRADAY_FREQ})",
    )
    parser.add_argument(
        "--skip-macro",
        action="store_true",
        help="Skip FRED macro data download",
    )
    parser.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="Skip Tiingo news sentiment download",
    )
    parser.add_argument(
        "--skip-cot",
        action="store_true",
        help="Skip COT/Quiver data download",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without making API calls",
    )
    return parser.parse_args()


def download_price_data(instruments: list[str], start: str, end: str, freq: str, dry_run: bool) -> None:
    """Download daily and intraday OHLCV for each instrument."""
    from prism.data.tiingo import get_ohlcv, get_intraday

    for symbol in instruments:
        logger.info("── %s: daily OHLCV %s → %s", symbol, start, end)
        if not dry_run:
            try:
                df = get_ohlcv(symbol, start, end, timeframe="daily")
                logger.info("   ✓ Daily: %d rows", len(df))
            except Exception as exc:
                logger.error("   ✗ Daily OHLCV failed for %s: %s", symbol, exc)

        logger.info("── %s: intraday OHLCV (%s) %s → %s", symbol, freq, start, end)
        if not dry_run:
            try:
                df = get_intraday(symbol, start, end, freq=freq)
                logger.info("   ✓ Intraday: %d rows", len(df))
            except Exception as exc:
                logger.error("   ✗ Intraday OHLCV failed for %s: %s", symbol, exc)


def download_sentiment(instruments: list[str], start: str, end: str, dry_run: bool) -> None:
    """Download Tiingo news sentiment for each instrument."""
    from prism.data.tiingo import get_news_sentiment

    for symbol in instruments:
        logger.info("── %s: news sentiment %s → %s", symbol, start, end)
        if not dry_run:
            try:
                df = get_news_sentiment(symbol, start, end)
                logger.info("   ✓ Sentiment: %d daily rows", len(df))
            except Exception as exc:
                logger.error("   ✗ Sentiment failed for %s: %s", symbol, exc)


def download_macro(start: str, end: str, dry_run: bool) -> None:
    """Download FRED macro features."""
    from prism.data.fred import get_macro_features

    logger.info("── FRED macro features %s → %s", start, end)
    if not dry_run:
        try:
            df = get_macro_features(start, end)
            logger.info("   ✓ Macro: %d rows × %d cols", df.shape[0], df.shape[1])
        except Exception as exc:
            logger.error("   ✗ Macro features failed: %s", exc)


def download_cot(instruments: list[str], dry_run: bool) -> None:
    """Download COT commitment of traders data."""
    from prism.data.quiver import get_cot_report, get_fear_greed

    for symbol in instruments:
        logger.info("── %s: COT report", symbol)
        if not dry_run:
            try:
                df = get_cot_report(symbol)
                logger.info("   ✓ COT: %d rows", len(df))
            except Exception as exc:
                logger.error("   ✗ COT failed for %s: %s", symbol, exc)

    logger.info("── Fear & Greed index")
    if not dry_run:
        try:
            df = get_fear_greed()
            logger.info("   ✓ Fear & Greed: %d rows", len(df))
        except Exception as exc:
            logger.error("   ✗ Fear & Greed failed: %s", exc)


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("PRISM Historical Data Download")
    logger.info("  Start:       %s", args.start)
    logger.info("  End:         %s", args.end)
    logger.info("  Instruments: %s", args.instruments)
    logger.info("  Freq:        %s", args.freq)
    logger.info("  Dry run:     %s", args.dry_run)
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("[DRY RUN] No API calls will be made")

    # Price data
    download_price_data(args.instruments, args.start, args.end, args.freq, args.dry_run)

    # Sentiment
    if not args.skip_sentiment:
        download_sentiment(args.instruments, args.start, args.end, args.dry_run)

    # Macro
    if not args.skip_macro:
        download_macro(args.start, args.end, args.dry_run)

    # COT
    if not args.skip_cot:
        download_cot(args.instruments, args.dry_run)

    logger.info("=" * 60)
    logger.info("Download complete.  Data saved to data/raw/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
