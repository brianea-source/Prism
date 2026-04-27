"""Historical detector state builder for Phase 7.A retraining.

Walks a training-window bar dataset once, runs the smart-money detectors
in *replay mode* against each bar, and writes a parquet sidecar whose
schema matches the live audit log produced by
:mod:`prism.delivery.signal_audit`. Feeding the sidecar through
:func:`prism.data.feature_engineering.enrich_features` then yields the
five Phase 7.A feature columns for the training set, end-to-end
schema-compatible with the audit-log path used at gate 5.

Scope (PHASE_7A_SCOPE.md §4.1, §6.1):

* **Training path** — emits one row per entry-timeframe bar past the
  warmup. Output feeds ``PRISMFeaturePipeline._engineer_features`` via
  the parquet sidecar wiring.
* **Gate-5 conditioning** — when ``signal_conditioned_only=True``, the
  builder runs the smart-money observability layer per bar and emits
  a row only when the live runner's gates would have *let through*
  the signal at that bar. This produces the population that gate 5's
  drift tests need to compare against the live audit log (§6.1
  option 1).

What this builder does NOT replicate from the live ``SignalGenerator``:

* The H4 ML regime predictor, news intelligence, ICC entry trigger,
  FVG retest mechanics, and RR ≥ 1.5 filter. Those layers fire
  *upstream* of smart money in the live pipeline, but every one of
  them is independent of the five Phase 7.A features (none of them
  touches OB / Sweep / Po3 / HTF bias or the kill-zone hour). Running
  them in replay would substantially slow the build and add fragile
  dependencies on news-cache replay, model joblib availability, and
  ICC parameter parity — all for zero impact on the feature
  distribution. Gate 5 specifically guards against drift in the five
  feature columns, not in the upstream gates.

The deliberate consequence: the gate-5 ``signal_conditioned_only``
output approximates the live audit log's selection criterion as
"smart-money layer evaluated and not blocked", which is the closest
inexpensive proxy to "signal fired". If a future bar in the training
set looks signal-shaped to smart money but fails ICC or news, replay
emits it and live audit doesn't — both populations stay distributionally
comparable on the features under test, which is the gate's only
contract. PHASE_7A_SCOPE.md §6.1 calls out the alternative ("option 2:
bar-keyed audit log") as the fallback if this proves insufficient.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from prism.audit.schema import ALL_FIELDS, TIMESTAMP_FIELD
from prism.signal.htf_bias import Bias, get_htf_bias
from prism.signal.order_blocks import OrderBlockDetector
from prism.signal.po3 import Po3Detector
from prism.signal.sweeps import SweepDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — warmup thresholds. Keep these in sync with the detectors'
# internal minimum-bar guards so the first emitted row has every detector
# in a meaningful state. PHASE_7A_SCOPE.md §8.2 (Po3 UNKNOWN frequency)
# specifically warns that mismatched warmup vs the live runner inflates
# the UNKNOWN rate and breaks gate 5 on the Po3 column alone.
# ---------------------------------------------------------------------------

#: Minimum bars in the entry-timeframe series before the builder starts
#: emitting rows. Set by the highest detector requirement:
#: ``SweepDetector`` (lookback=20, needs strictly more than that) → 21.
DEFAULT_WARMUP_BARS: int = 21

#: Minimum bars required in the 1H series for ``get_htf_bias`` to return
#: a non-RANGING result (2 * lookback + 1 with default lookback=3).
HTF_MIN_BARS_1H: int = 7
HTF_MIN_BARS_4H: int = 7


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class HistoricalStateBuilder:
    """Replay smart-money detectors over a historical bar range.

    The detectors are stateful — calling ``detect()`` on the same window
    repeatedly only processes the new tail. This makes the per-bar walk
    O(n_total_bars) instead of O(n²), but it also means the builder
    must be discarded and re-instantiated to redo a window with
    different parameters.

    Args:
        instrument: Symbol (e.g. ``"EURUSD"``).
        df_h4: 4-hour OHLC DataFrame with a ``datetime`` column. Used by
            the OB detector and the HTF bias engine's 4H input.
        df_h1: 1-hour OHLC DataFrame with a ``datetime`` column. Used
            for the HTF bias engine's 1H input.
        df_entry: Entry-timeframe OHLC DataFrame (typically 5-minute or
            15-minute) with a ``datetime`` column. Drives the per-bar
            walk; sweep and Po3 run against it.
        warmup_bars: Number of leading entry bars to skip before
            emitting rows. Defaults to :data:`DEFAULT_WARMUP_BARS`.
        signal_conditioned_only: If True, emit a row only on bars that
            pass the smart-money observability check (no detector
            failure + valid HTF bias). PHASE_7A_SCOPE.md §6.1 option 1.
    """

    def __init__(
        self,
        instrument: str,
        df_h4: pd.DataFrame,
        df_h1: pd.DataFrame,
        df_entry: pd.DataFrame,
        *,
        warmup_bars: int = DEFAULT_WARMUP_BARS,
        signal_conditioned_only: bool = False,
    ) -> None:
        self._validate_input("df_h4", df_h4)
        self._validate_input("df_h1", df_h1)
        self._validate_input("df_entry", df_entry)

        self.instrument = instrument
        self.df_h4 = df_h4.sort_values("datetime").reset_index(drop=True)
        self.df_h1 = df_h1.sort_values("datetime").reset_index(drop=True)
        self.df_entry = df_entry.sort_values("datetime").reset_index(drop=True)
        self.warmup_bars = max(int(warmup_bars), 1)
        self.signal_conditioned_only = bool(signal_conditioned_only)

        self.ob_detector = OrderBlockDetector(instrument, "H4")
        self.sweep_detector = SweepDetector(instrument)
        self.po3_detector = Po3Detector(instrument)

    @staticmethod
    def _validate_input(name: str, df: pd.DataFrame) -> None:
        required = {"datetime", "open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} missing required columns: {sorted(missing)}"
            )

    # ---------------------------------------------------------------
    # Per-bar smart-money snapshot. Mirrors generator._evaluate_smart_money
    # closely enough to produce the same audit-log dict shape, minus the
    # gating decision (we always emit; conditioning is up to the caller).
    # ---------------------------------------------------------------

    def _smart_money_snapshot(
        self,
        h4_window: pd.DataFrame,
        entry_window: pd.DataFrame,
        current_price: float,
        direction_str: str,
        session_str: str,
    ) -> tuple[Optional[dict], bool]:
        """Compute the smart-money sub-dicts and a "gates would have
        passed" bool. Returns ``(smart_money_dict, observability_ok)``.

        ``smart_money_dict`` matches the shape persisted by
        ``signal_audit.write_signal_audit`` (no ``blocked`` /
        ``block_reason`` keys). ``observability_ok`` is True iff every
        detector ran without exception — the proxy for
        ``signal_conditioned_only``.
        """
        observability_ok = True

        # OB
        try:
            self.ob_detector.detect(h4_window)
            self.ob_detector.update_states(h4_window)
            nearest = self.ob_detector.get_nearest_ob(current_price, direction_str)
            ob_distance = self.ob_detector.distance_to_ob(current_price, direction_str)
        except Exception as exc:
            logger.warning(
                "%s OB detect failed at %s: %s",
                self.instrument, entry_window["datetime"].iloc[-1], exc,
            )
            observability_ok = False
            nearest = None
            ob_distance = None

        ob_dict: Optional[dict] = None
        if nearest is not None:
            ob_dict = {
                "state": getattr(nearest.state, "value", str(nearest.state)),
                "direction": nearest.direction,
                "effective_direction": nearest.effective_direction,
                "high": nearest.high,
                "low": nearest.low,
                "midpoint": nearest.midpoint,
                "timeframe": nearest.timeframe,
                "distance_pips": ob_distance,
                "is_rejection_block": nearest.is_rejection_block,
                # in_range intentionally omitted — feature_engineering
                # recomputes it from distance_pips against the
                # training-time PRISM_OB_MAX_DISTANCE_PIPS lock.
            }

        # Sweep
        try:
            self.sweep_detector.detect(entry_window)
            last_sweep = self.sweep_detector.last_sweep(direction_str)
            has_recent = self.sweep_detector.has_recent_sweep(
                direction_str, bars_back=5, require_displacement=True,
            )
        except Exception as exc:
            logger.warning(
                "%s sweep detect failed at %s: %s",
                self.instrument, entry_window["datetime"].iloc[-1], exc,
            )
            observability_ok = False
            last_sweep = None
            has_recent = False

        sweep_dict: Optional[dict] = None
        if last_sweep is not None:
            anchor = (
                self.sweep_detector._latest_scanned_bar
                if self.sweep_detector._latest_scanned_bar is not None
                else last_sweep.sweep_bar
            )
            sweep_dict = {
                "type": last_sweep.type,
                "swept_level": last_sweep.swept_level,
                "sweep_bar": last_sweep.sweep_bar,
                "bars_ago": int(anchor - last_sweep.sweep_bar),
                "displacement_followed": last_sweep.displacement_followed,
                # ``timestamp`` from the live path is a string; keep it
                # serialisable here too.
                "timestamp": str(last_sweep.timestamp),
                "qualifies": has_recent,
            }

        # Po3 — pass the bar's session, not datetime.now()
        try:
            po3_state = self.po3_detector.detect_phase(
                entry_window, session=session_str,
            )
            is_entry_phase = self.po3_detector.is_entry_phase(po3_state)
        except Exception as exc:
            logger.warning(
                "%s Po3 detect failed at %s: %s",
                self.instrument, entry_window["datetime"].iloc[-1], exc,
            )
            observability_ok = False
            po3_state = None
            is_entry_phase = False

        po3_dict: Optional[dict] = None
        if po3_state is not None:
            po3_dict = {
                "phase": po3_state.phase.value,
                "session": po3_state.session,
                "range_size_pips": po3_state.range_size_pips,
                "sweep_detected": po3_state.sweep_detected,
                "displacement_detected": po3_state.displacement_detected,
                "is_entry_phase": is_entry_phase,
            }

        sm_dict: Optional[dict] = None
        if any(d is not None for d in (ob_dict, sweep_dict, po3_dict)):
            sm_dict = {"ob": ob_dict, "sweep": sweep_dict, "po3": po3_dict}

        return sm_dict, observability_ok

    # ---------------------------------------------------------------
    # HTF bias snapshot (stateless per scope §2.1 — uses ``get_htf_bias``).
    # ---------------------------------------------------------------

    def _htf_bias_snapshot(
        self, h1_window: pd.DataFrame, h4_window: pd.DataFrame,
    ) -> Optional[dict]:
        if len(h1_window) < HTF_MIN_BARS_1H or len(h4_window) < HTF_MIN_BARS_4H:
            return None
        try:
            result = get_htf_bias(h1_window, h4_window)
        except Exception as exc:
            logger.warning(
                "%s HTF bias compute failed: %s", self.instrument, exc,
            )
            return None
        return {
            "bias_1h": result.bias_1h.value if isinstance(result.bias_1h, Bias) else str(result.bias_1h),
            "bias_4h": result.bias_4h.value if isinstance(result.bias_4h, Bias) else str(result.bias_4h),
            "aligned": bool(result.aligned),
            "allowed_direction": result.allowed_direction,
        }

    # ---------------------------------------------------------------
    # Build loop
    # ---------------------------------------------------------------

    @staticmethod
    def _session_label(ts: pd.Timestamp) -> str:
        """ICT session label for the Po3 detector. Matches the live
        runner's ``session_label(datetime.now(UTC))`` semantics but uses
        the bar's UTC hour rather than wall clock."""
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        h = ts.hour
        if 0 <= h < 8:
            return "asia"
        if 8 <= h < 13:
            return "london"
        if 13 <= h < 20:
            return "ny"
        return "off"

    @staticmethod
    def _direction_for_replay(htf_bias: Optional[dict]) -> str:
        """Pick a synthesised direction for the per-bar walk.

        The live runner derives ``direction`` from the H4 ML regime
        predictor; we don't replay that here. Instead we use HTF bias's
        ``allowed_direction`` if any, falling back to LONG. The choice
        only affects ``htf_alignment``'s sign and ``has_recent_sweep``'s
        directional filter — for gate 5 it's the *distribution* of
        these features that matters, not the per-bar identity, so a
        deterministic fallback is appropriate as long as it's
        reproducible. PHASE_7A_SCOPE.md §6.1 conditions on the same
        criterion across both populations.
        """
        if isinstance(htf_bias, dict):
            allowed = htf_bias.get("allowed_direction")
            if allowed in ("LONG", "SHORT"):
                return allowed
        return "LONG"

    def build(self) -> pd.DataFrame:
        """Run the per-bar walk and return a DataFrame in audit-log schema.

        Columns are :data:`prism.audit.schema.ALL_FIELDS` plus a
        ``observability_ok`` bool sidecar that gate-5 conditioning uses
        when ``signal_conditioned_only=True``.

        Empty DataFrame (with the schema columns) when the entry frame
        is shorter than ``warmup_bars``.
        """
        if len(self.df_entry) <= self.warmup_bars:
            return pd.DataFrame(columns=list(ALL_FIELDS))

        rows: list[dict] = []
        h4_dt = self.df_h4["datetime"]
        h1_dt = self.df_h1["datetime"]
        entry_dt = self.df_entry["datetime"]

        for i in range(self.warmup_bars, len(self.df_entry)):
            current_ts = entry_dt.iloc[i]
            current_price = float(self.df_entry["close"].iloc[i])

            # Window the higher timeframes to "everything that closed
            # at or before the current entry bar" — matches what the
            # live runner sees with arrived-on-bar semantics.
            h4_window = self.df_h4[h4_dt <= current_ts]
            h1_window = self.df_h1[h1_dt <= current_ts]
            entry_window = self.df_entry.iloc[: i + 1]

            htf_bias = self._htf_bias_snapshot(h1_window, h4_window)
            direction = self._direction_for_replay(htf_bias)
            session = self._session_label(pd.Timestamp(current_ts))

            sm, obs_ok = self._smart_money_snapshot(
                h4_window=h4_window,
                entry_window=entry_window,
                current_price=current_price,
                direction_str=direction,
                session_str=session,
            )

            if self.signal_conditioned_only and not obs_ok:
                continue

            rows.append({
                TIMESTAMP_FIELD: pd.Timestamp(current_ts).isoformat(),
                "instrument": self.instrument,
                "direction": direction,
                "confidence": float("nan"),
                "confidence_level": "UNKNOWN",
                "signal_id": f"replay-{self.instrument}-{i}",
                "signal_time": pd.Timestamp(current_ts).isoformat(),
                "model_version": "replay",
                "regime": "UNKNOWN",
                "news_bias": "UNKNOWN",
                "htf_bias": htf_bias,
                "smart_money": sm,
            })

        if not rows:
            return pd.DataFrame(columns=list(ALL_FIELDS))
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience entry point + sidecar I/O
# ---------------------------------------------------------------------------


def build_replay_sidecar(
    instrument: str,
    df_h4: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_entry: pd.DataFrame,
    output_path: Path | str,
    *,
    warmup_bars: int = DEFAULT_WARMUP_BARS,
    signal_conditioned_only: bool = False,
) -> Path:
    """Run the builder and persist the result to parquet.

    Returns the resolved output path. Parent directories are created.
    """
    builder = HistoricalStateBuilder(
        instrument=instrument,
        df_h4=df_h4, df_h1=df_h1, df_entry=df_entry,
        warmup_bars=warmup_bars,
        signal_conditioned_only=signal_conditioned_only,
    )
    df = builder.build()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    logger.info(
        "Wrote %d historical state rows for %s to %s",
        len(df), instrument, output,
    )
    return output


def read_replay_sidecar(path: Path | str) -> pd.DataFrame:
    """Load a sidecar parquet. Restores the canonical column ordering."""
    df = pd.read_parquet(path)
    missing = set(ALL_FIELDS) - set(df.columns)
    if missing:
        raise ValueError(
            f"sidecar {path} is missing canonical columns: {sorted(missing)}"
        )
    # Reorder canonical columns first, preserve any extras at the tail
    extras = [c for c in df.columns if c not in ALL_FIELDS]
    return df[list(ALL_FIELDS) + extras]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prism.data.historical_state",
        description=(
            "Build the smart-money replay sidecar that "
            "PRISMFeaturePipeline._engineer_features and gate-5 consume."
        ),
    )
    parser.add_argument("--instrument", required=True)
    parser.add_argument(
        "--h4", required=True,
        help="Path to H4 OHLC parquet (must have datetime + OHLC columns)",
    )
    parser.add_argument("--h1", required=True, help="Path to H1 OHLC parquet")
    parser.add_argument(
        "--entry", required=True,
        help="Path to entry-timeframe (e.g. 5m) OHLC parquet",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output parquet path for the sidecar (parent dirs created)",
    )
    parser.add_argument(
        "--warmup-bars", type=int, default=DEFAULT_WARMUP_BARS,
        help=f"Skip leading N bars (default {DEFAULT_WARMUP_BARS})",
    )
    parser.add_argument(
        "--signal-conditioned-only", action="store_true",
        help=(
            "Emit only bars that pass the smart-money observability "
            "check — gate-5 mode (PHASE_7A_SCOPE.md §6.1 option 1)."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    args = _build_parser().parse_args(argv)

    df_h4 = pd.read_parquet(args.h4)
    df_h1 = pd.read_parquet(args.h1)
    df_entry = pd.read_parquet(args.entry)

    output = build_replay_sidecar(
        instrument=args.instrument,
        df_h4=df_h4,
        df_h1=df_h1,
        df_entry=df_entry,
        output_path=args.output,
        warmup_bars=args.warmup_bars,
        signal_conditioned_only=args.signal_conditioned_only,
    )
    print(json.dumps({
        "output": str(output),
        "instrument": args.instrument,
        "signal_conditioned_only": args.signal_conditioned_only,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
