#!/usr/bin/env python3
"""
prism/model/retrain.py
CLI retraining script for PRISM models.

Usage:
    python prism/model/retrain.py --instrument EURUSD
    python prism/model/retrain.py --instrument EURUSD --start 2022-01-01 --end 2025-01-01
"""
import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prism.model.train import PRISMTrainer, MODELS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("prism.retrain")


def main() -> None:
    parser = argparse.ArgumentParser(description="PRISM model retraining CLI")
    parser.add_argument(
        "--instrument", "-i", default="EURUSD",
        help="Trading instrument symbol (default: EURUSD)"
    )
    parser.add_argument(
        "--timeframe", "-t", default="H1",
        help="Timeframe string: H1, H4, D1, M15 (default: H1)"
    )
    parser.add_argument(
        "--start", default=None,
        help="Training start date YYYY-MM-DD (default: 3 years ago)"
    )
    parser.add_argument(
        "--end", default=None,
        help="Training end date YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path for JSON report (default: models/retrain_report_<instrument>.json)"
    )
    args = parser.parse_args()

    today = datetime.utcnow().date()
    end_date = args.end or str(today)
    start_date = args.start or str(today - timedelta(days=3 * 365))

    logger.info(f"Retraining {args.instrument} | {start_date} → {end_date}")

    trainer = PRISMTrainer(instrument=args.instrument, timeframe=args.timeframe)

    try:
        results = trainer.train_all_layers(start_date, end_date)
    except Exception as exc:
        logger.error(f"Training failed: {exc}", exc_info=True)
        sys.exit(1)

    # Build report
    report = {
        "instrument": args.instrument,
        "timeframe": args.timeframe,
        "train_period": {"start": start_date, "end": end_date},
        "retrain_timestamp": datetime.utcnow().isoformat() + "Z",
        "layers": [asdict(r) for r in results],
        "overfit_warnings": [
            {"layer": r.layer, "train": r.train_accuracy, "test": r.test_accuracy,
             "delta": round(r.train_accuracy - r.test_accuracy, 4)}
            for r in results if r.overfit_flag
        ],
    }

    out_path = args.output or str(MODELS_DIR / f"retrain_report_{args.instrument}.json")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved → {out_path}")

    # Warn on overfit
    if report["overfit_warnings"]:
        logger.warning("⚠  OVERFIT DETECTED in the following layers:")
        for w in report["overfit_warnings"]:
            logger.warning(
                f"   {w['layer']}: train={w['train']:.4f} test={w['test']:.4f} "
                f"delta={w['delta']:.4f} (threshold=0.15)"
            )
        print("\n⚠  OVERFIT WARNING — see log for details. Consider:")
        print("   • Reducing n_estimators or max_depth")
        print("   • Adding more training data")
        print("   • Increasing regularisation (min_child_weight, lambda)")
    else:
        logger.info("✓ No overfitting detected across all layers.")

    print(f"\n✓ Retraining complete. Report: {out_path}")


if __name__ == "__main__":
    main()
