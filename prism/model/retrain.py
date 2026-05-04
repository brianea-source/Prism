#!/usr/bin/env python3
"""
prism/model/retrain.py
CLI retraining script for PRISM models.

Usage:
    python prism/model/retrain.py --instrument EURUSD
    python prism/model/retrain.py --instrument EURUSD --start 2022-01-01 --end 2025-01-01
    python prism/model/retrain.py --instrument EURUSD --walkforward \
        --phase7a-sidecar models/historical_state_EURUSD.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prism.model.predict import write_manifest
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
    parser.add_argument(
        "--walkforward", action="store_true",
        help=(
            "After training, run the Phase 7.A walk-forward harness "
            "(baseline vs. Phase 7.A-enriched feature set) and write "
            "models/phase7a_walkforward_<instrument>.json. Requires "
            "--phase7a-sidecar."
        ),
    )
    parser.add_argument(
        "--phase7a-sidecar", default=None,
        help=(
            "Path to the historical state parquet sidecar produced by "
            "prism/data/historical_state.py. When supplied, the training "
            "feature matrix is enriched with the five Phase 7.A ICT "
            "feature columns. Required when --walkforward is set."
        ),
    )
    args = parser.parse_args()

    if args.walkforward and not args.phase7a_sidecar:
        parser.error("--walkforward requires --phase7a-sidecar")

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

    # Phase 7.A: lock the env-derived OB threshold into the model
    # artifact sidecar so predict.py can detect train/live drift.
    # PHASE_7A_SCOPE.md §2.4. Always written — even when not running
    # walk-forward — so legacy retrains start producing manifests
    # the moment this PR ships.
    ob_max_dist_pips = float(os.environ.get("PRISM_OB_MAX_DISTANCE_PIPS", "30.0"))
    # The feature_cols sidecar is written inside trainer.train_all_layers
    # (see prism.model.train.write_feature_cols). Surface its path in the
    # manifest so ops can audit the trained schema without grepping the
    # joblibs directory by hand.
    from prism.model.train import feature_cols_path as _fc_path
    fc_sidecar = str(_fc_path(args.instrument))
    write_manifest(
        args.instrument,
        ob_max_distance_pips=ob_max_dist_pips,
        phase7a_features_active=bool(args.phase7a_sidecar),
        extra={
            "retrain_report": out_path,
            "feature_cols_sidecar": fc_sidecar,
            "n_features": len(results[0].feature_importance) if results else 0,
        },
    )

    if args.walkforward:
        try:
            _run_walkforward(args, ob_max_dist_pips=ob_max_dist_pips)
        except Exception as exc:
            logger.error("Walk-forward harness failed: %s", exc, exc_info=True)
            # Don't kill the retrain — the manifest + joblibs are
            # already on disk. Surface the failure to ops via exit code
            # so CI alerts.
            print(f"\n⚠  Walk-forward failed: {exc}", file=sys.stderr)
            sys.exit(2)

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


def _run_walkforward(args: argparse.Namespace, *, ob_max_dist_pips: float) -> None:
    """Drive the Phase 7.A walk-forward harness using the freshly-trained
    pipeline as the baseline and the sidecar-enriched pipeline as the
    candidate. Writes ``models/phase7a_walkforward_<instrument>.json``.

    Lazy-imported so the legacy retrain path doesn't pay the lightgbm/
    walkforward import cost when ``--walkforward`` is off.
    """
    from prism.data.feature_engineering import ICTFeatureEngineer, PHASE_7A_FEATURE_COLUMNS
    from prism.data.pipeline import PRISMFeaturePipeline
    from prism.model.walkforward import (
        acceptance_decision, run_walkforward, write_walkforward_artifact,
    )

    today = datetime.utcnow().date()
    end_date = args.end or str(today)
    start_date = args.start or str(today - timedelta(days=3 * 365))

    logger.info("Walk-forward: building baseline feature matrix")
    baseline_pipeline = PRISMFeaturePipeline(args.instrument, args.timeframe)
    baseline_df = baseline_pipeline.build_features(start_date, end_date)
    baseline_features = list(baseline_pipeline._feature_cols)

    logger.info(
        "Walk-forward: building Phase 7.A feature matrix from sidecar %s",
        args.phase7a_sidecar,
    )
    candidate_pipeline = PRISMFeaturePipeline(
        args.instrument, args.timeframe,
        phase7a_sidecar_path=args.phase7a_sidecar,
        phase7a_ob_max_distance_pips=ob_max_dist_pips,
    )
    candidate_df = candidate_pipeline.build_features(start_date, end_date)
    candidate_features = list(candidate_pipeline._feature_cols)

    logger.info("Walk-forward: running baseline harness")
    baseline_result = run_walkforward(
        baseline_df,
        feature_cols=baseline_features,
        instrument=args.instrument,
        feature_set="baseline",
    )
    logger.info(
        "Walk-forward baseline: %d folds, median F1=%.4f, Sharpe=%.4f, MaxDD=%.4f",
        baseline_result.n_folds, baseline_result.median_f1,
        baseline_result.median_sharpe, baseline_result.median_max_drawdown,
    )

    logger.info("Walk-forward: running candidate (Phase 7.A) harness")
    candidate_result = run_walkforward(
        candidate_df,
        feature_cols=candidate_features,
        instrument=args.instrument,
        feature_set="phase7a",
    )
    logger.info(
        "Walk-forward candidate: %d folds, median F1=%.4f, Sharpe=%.4f, MaxDD=%.4f",
        candidate_result.n_folds, candidate_result.median_f1,
        candidate_result.median_sharpe, candidate_result.median_max_drawdown,
    )

    decision = acceptance_decision(baseline_result, candidate_result)
    artifact_path = MODELS_DIR / f"phase7a_walkforward_{args.instrument}.json"
    write_walkforward_artifact(
        artifact_path,
        baseline=baseline_result,
        candidate=candidate_result,
        decision=decision,
    )
    logger.info("Walk-forward artifact saved → %s", artifact_path)
    print(f"\n✓ Walk-forward complete. Decision: {decision['rationale']}")
    print(f"  Artifact: {artifact_path}")
    if not decision["passed"]:
        print(
            "\n⚠  Phase 7.A acceptance gates 1-3 did NOT pass. "
            "Inspect the artifact for per-gate breakdown."
        )


if __name__ == "__main__":
    main()
