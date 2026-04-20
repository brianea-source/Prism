#!/usr/bin/env python3
"""
Download historical OHLCV + macro data for PRISM training.
Usage: python scripts/download_historical_data.py --start 2018-01-01 --end 2026-04-19
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INSTRUMENTS = ["EURUSD", "GBPUSD", "XAUUSD"]
TIMEFRAMES = ["D1", "H4", "H1", "M15"]

def main():
    parser = argparse.ArgumentParser(description="Download PRISM historical data")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-04-19")
    parser.add_argument("--instruments", nargs="+", default=INSTRUMENTS)
    parser.add_argument("--timeframes", nargs="+", default=TIMEFRAMES)
    args = parser.parse_args()

    logger.info(f"Starting download: {args.start} to {args.end}")

    from prism.data.tiingo import TiingoClient
    client = TiingoClient()
    tf_map = {"D1": "daily", "H4": "4hour", "H1": "1hour", "M15": "15min"}
    for symbol in args.instruments:
        for tf in args.timeframes:
            logger.info(f"Downloading {symbol} {tf}...")
            try:
                df = client.get_ohlcv(symbol, args.start, args.end, tf_map[tf])
                logger.info(f"  OK {symbol} {tf}: {len(df)} bars")
            except Exception as e:
                logger.warning(f"  FAIL {symbol} {tf}: {e}")

    from prism.data.quiver import QuiverClient
    qclient = QuiverClient()
    for symbol in args.instruments:
        try:
            cot = qclient.get_cot_report(symbol)
            logger.info(f"  OK COT {symbol}: {len(cot)} weeks")
        except Exception as e:
            logger.warning(f"  FAIL COT {symbol}: {e}")

    try:
        fg = qclient.get_fear_greed()
        logger.info(f"  OK Fear & Greed: {len(fg)} days")
    except Exception as e:
        logger.warning(f"  FAIL Fear & Greed: {e}")

    from prism.data.fred import FREDClient
    try:
        macro = FREDClient().get_macro_features(args.start, args.end)
        logger.info(f"  OK FRED macro: {len(macro)} rows")
    except Exception as e:
        logger.warning(f"  FAIL FRED: {e}")

    logger.info("Download complete. Data in data/raw/")

if __name__ == "__main__":
    main()
