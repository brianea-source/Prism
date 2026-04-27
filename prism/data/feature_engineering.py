"""ICT feature engineering for Phase 7.A retraining.

Implements the five buildable-now ICT features per
``docs/PHASE_7A_SCOPE.md`` §2:

    htf_alignment        — int 0-3        (chi-squared GoF for gate 5)
    kill_zone_strength   — int 0-3        (chi-squared GoF)
    sweep_confirmed      — bool           (Fisher's exact)
    ob_distance_pips     — float / -1     (KS two-sample)
    po3_phase            — categorical    (chi-squared on the 4-cat label)

The two deferred features (``fvg_quality_score``, ``ote_zone``) ship in
Phase 7.B once Phase 8's ``check_ote_zone()`` lands and the ``formed_bar``
df-relative-index bug in ``prism/signal/fvg.py`` is corrected.

Two output flavours of ``po3_phase`` are produced because they serve
different consumers (PHASE_7A_SCOPE.md §2.5 + the carry-in note in
``compare_features``'s docstring):

* the **single 4-category label column** ``po3_phase`` — for gate 5's
  chi-squared test on the joint distribution
* the **four one-hot bool columns** (``po3_accumulation``,
  ``po3_manipulation``, ``po3_distribution``, ``po3_unknown``) — for ML
  ingestion, where the gradient-boosted splitters need a numeric input
  but must NOT see the phases as an ordinal ladder.

Lock-in: ``compute_ob_distance_features`` takes ``ob_max_distance_pips``
explicitly so the training pipeline can pass the sidecar-locked value
and ``predict.py`` can assert at load time that the runtime env matches
what the model was trained against (PRD2 §7 / PHASE_7A_SCOPE.md §2.4
+ §8.1).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from prism.signal.htf_bias import Bias

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 5 ICT feature derivation functions
# ---------------------------------------------------------------------------


def compute_htf_alignment(
    bias_1h: Bias | str | None,
    bias_4h: Bias | str | None,
    direction: str,
) -> int:
    """Score 1H/4H bias agreement with the trade direction (PHASE_7A_SCOPE.md §2.1).

    Per-timeframe scoring against ``direction``:

    * ``+1`` aligned   — ``BULLISH`` for ``LONG`` or ``BEARISH`` for ``SHORT``
    * ``0``  neutral   — ``RANGING`` (or any unrecognised input)
    * ``-1`` against   — opposite of aligned

    Aggregate (4 levels for the model to split on):

    * ``3`` — both timeframes aligned
    * ``2`` — one aligned + one neutral
    * ``1`` — exactly one against (regardless of the other), OR both neutral
    * ``0`` — both against

    Both-neutral mapping to ``1`` is a deliberate choice. PRD2 §7 specifies
    a 4-level encoding but doesn't enumerate the both-neutral case; we
    map it to the same tier as "one against" because neither timeframe is
    actually supportive of the trade. Rating it ``2`` would over-credit
    a directional signal that has no HTF backing whatsoever.

    Args:
        bias_1h: ``Bias`` enum value or its string equivalent
            (``"BULLISH"`` / ``"BEARISH"`` / ``"RANGING"``). ``None`` is
            treated as ``RANGING``.
        bias_4h: Same type as ``bias_1h``.
        direction: ``"LONG"`` or ``"SHORT"``. Case-insensitive; other
            inputs (``"BUY"`` / ``"SELL"``) are normalised.

    Returns:
        Integer in the closed interval ``[0, 3]``.
    """
    direction_norm = (direction or "").upper().strip()
    if direction_norm in ("BUY",):
        direction_norm = "LONG"
    elif direction_norm in ("SELL",):
        direction_norm = "SHORT"
    if direction_norm not in ("LONG", "SHORT"):
        raise ValueError(
            f"compute_htf_alignment: direction must be LONG/SHORT, got {direction!r}"
        )

    def _score(b: Bias | str | None) -> int:
        if b is None:
            return 0
        b_str = b.value if isinstance(b, Bias) else str(b).upper().strip()
        if b_str == "RANGING":
            return 0
        if b_str == "BULLISH":
            return 1 if direction_norm == "LONG" else -1
        if b_str == "BEARISH":
            return 1 if direction_norm == "SHORT" else -1
        return 0  # unknown bias string — be lenient, treat as neutral

    s1 = _score(bias_1h)
    s4 = _score(bias_4h)

    n_aligned = (s1 == 1) + (s4 == 1)
    n_against = (s1 == -1) + (s4 == -1)

    if n_against >= 2:
        return 0
    if n_against == 1:
        return 1
    if n_aligned >= 2:
        return 3
    if n_aligned == 1:
        return 2
    return 1  # both neutral — see docstring


def compute_kill_zone_strength(hour_utc: int) -> int:
    """ICT kill-zone scoring by UTC hour (PHASE_7A_SCOPE.md §2.2).

    Standard ICT kill zones (UTC):

    * Asian (Tokyo)  — 00:00-03:59 UTC
    * London core    — 08:00-09:59 UTC
    * London edges   — 07:00-07:59 UTC, 10:00-10:59 UTC
    * NY core        — 13:00-14:59 UTC
    * NY edges       — 12:00-12:59 UTC, 15:00-15:59 UTC

    Scoring:

    * ``3`` — London or NY core
    * ``2`` — London or NY edge
    * ``1`` — Asian
    * ``0`` — off-session (everything else)

    PHASE_7A_SCOPE.md §2.2 explicitly says NOT to reuse
    ``pipeline._session()`` because the existing 0-4 encoding mixes
    overlap into a separate level — the model can learn whichever proves
    more predictive when both are present.
    """
    h = int(hour_utc) % 24
    if h in (8, 9, 13, 14):
        return 3
    if h in (7, 10, 12, 15):
        return 2
    if h in (0, 1, 2, 3):
        return 1
    return 0


def compute_sweep_confirmed(smart_money: Optional[dict]) -> bool:
    """Did a direction-matching liquidity sweep with displacement complete recently?

    Reads the ``smart_money.sweep`` sub-dict written by
    ``generator._evaluate_smart_money`` (and persisted via
    ``signal_audit.write_signal_audit``). The ``qualifies`` flag is the
    output of ``SweepDetector.has_recent_sweep(direction, bars_back=5,
    require_displacement=True)``, which captures the gate-firing
    condition exactly — see PHASE_7A_SCOPE.md §2.3.
    """
    if not isinstance(smart_money, dict):
        return False
    sweep = smart_money.get("sweep")
    if not isinstance(sweep, dict):
        return False
    return bool(sweep.get("qualifies", False))


def compute_ob_distance_features(
    smart_money: Optional[dict],
    *,
    ob_max_distance_pips: float,
) -> tuple[float, bool]:
    """Distance to the nearest active direction-matching OB (pips) + companion bool.

    Returns ``(ob_distance_pips, ob_in_range)``:

    * ``ob_distance_pips`` — float pips to the OB midpoint, or ``-1.0``
      when no qualifying OB exists. The ``-1`` sentinel gives the
      gradient boosting splitter a clean discontinuity it can isolate
      from the legitimate distance values; encoding ``None`` as ``NaN``
      would be silently coerced or dropped by the pipeline's
      ``fillna(0)`` and merge with the smallest real distances.
    * ``ob_in_range`` — bool, ``True`` iff a qualifying OB exists AND
      its distance is ``<= ob_max_distance_pips``.

    The threshold is passed in (NOT read from env) so the training
    pipeline can lock the value into the model artifact sidecar at
    training time, and ``predict.py`` can assert on load that the
    runtime ``PRISM_OB_MAX_DISTANCE_PIPS`` matches. Recomputing the
    bool from ``smart_money["ob"]["distance_pips"]`` (rather than
    trusting the persisted ``smart_money["ob"]["in_range"]``) is what
    makes the lock-in actually enforce on training data — the audit
    log's pre-computed ``in_range`` reflects whatever the live runner's
    env said at write time, which we don't want bleeding into training
    if an operator changed the env mid-run. PHASE_7A_SCOPE.md §2.4 +
    §8.1.
    """
    if not isinstance(smart_money, dict):
        return -1.0, False
    ob = smart_money.get("ob")
    if not isinstance(ob, dict):
        return -1.0, False
    dist = ob.get("distance_pips")
    if dist is None:
        return -1.0, False
    try:
        d = float(dist)
    except (TypeError, ValueError):
        return -1.0, False
    if d < 0:  # detector contract: distance_pips is non-negative or None
        return -1.0, False
    return d, bool(d <= ob_max_distance_pips)


PO3_PHASES: tuple[str, ...] = (
    "accumulation",
    "manipulation",
    "distribution",
    "unknown",
)
"""Lower-case canonical labels — must match the column-name suffixes used
in :func:`compute_po3_phase_features` so the one-hot output is stable."""


def compute_po3_phase_features(
    smart_money: Optional[dict],
) -> tuple[str, dict[str, bool]]:
    """Po3 phase as a label + 4-bool one-hot (PHASE_7A_SCOPE.md §2.5).

    Returns a tuple ``(label, one_hot_dict)``:

    * ``label`` — one of ``PO3_PHASES`` (always lower-case). When
      ``smart_money["po3"]`` is missing, the label defaults to
      ``"unknown"``.
    * ``one_hot_dict`` — keys ``po3_accumulation`` /
      ``po3_manipulation`` / ``po3_distribution`` / ``po3_unknown``,
      values bool. Exactly one is ``True`` per call.

    Both outputs are emitted because they serve different consumers:
    the label is the gate-5 chi-squared input (one test on the joint
    4-cat distribution per §6.1), the one-hots are the ML feature
    matrix columns. Iterating the four one-hots and running four
    Fisher's exacts is the wrong answer statistically — see the
    :func:`prism.audit.smart_money_export.compare_features` docstring
    (PR #23 N1 carry-in).
    """
    if not isinstance(smart_money, dict):
        label = "unknown"
    else:
        po3 = smart_money.get("po3")
        if not isinstance(po3, dict):
            label = "unknown"
        else:
            raw = po3.get("phase", "UNKNOWN")
            label = str(raw).strip().lower() if raw else "unknown"
            if label not in PO3_PHASES:
                label = "unknown"

    one_hot = {f"po3_{p}": (label == p) for p in PO3_PHASES}
    return label, one_hot


# ---------------------------------------------------------------------------
# DataFrame-level enrich (consumer entry point)
# ---------------------------------------------------------------------------


#: Columns produced by :func:`enrich_features`. Order is stable so
#: callers can reindex against this tuple to assert column presence.
PHASE_7A_FEATURE_COLUMNS: tuple[str, ...] = (
    "htf_alignment",
    "kill_zone_strength",
    "sweep_confirmed",
    "ob_distance_pips",
    "ob_in_range",
    "po3_phase",
    "po3_accumulation",
    "po3_manipulation",
    "po3_distribution",
    "po3_unknown",
)


def _coerce_hour(value: Any) -> Optional[int]:
    """Best-effort hour-of-UTC extraction from a timestamp-like value."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value) % 24
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    if ts is pd.NaT or pd.isna(ts):
        return None
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.hour)


def enrich_features(
    df: pd.DataFrame,
    *,
    ob_max_distance_pips: float,
    timestamp_col: str = "audit_ts",
    htf_bias_col: str = "htf_bias",
    smart_money_col: str = "smart_money",
    direction_col: str = "direction",
) -> pd.DataFrame:
    """Add the 10 Phase 7.A feature columns to ``df`` per row.

    Designed for two input shapes:

    1. **Audit log** loaded by
       :func:`prism.audit.smart_money_export.read_audit_window`.
       ``htf_bias`` and ``smart_money`` are dicts; ``audit_ts`` is the
       timestamp.
    2. **Historical replay sidecar** produced by
       :func:`prism.data.historical_state.build_replay_sidecar`. Same
       column names by construction so the two sides of the gate-5 diff
       are schema-identical.

    Args:
        df: Input frame. Missing input columns are tolerated — the
            corresponding output columns will hold their "no signal"
            sentinels (``htf_alignment=1``, ``kill_zone_strength=0``,
            ``sweep_confirmed=False``, ``ob_distance_pips=-1.0``,
            ``po3_phase="unknown"``).
        ob_max_distance_pips: Threshold for the ``ob_in_range`` bool.
            Should equal the value locked into the model artifact
            sidecar at training time. PHASE_7A_SCOPE.md §2.4.
        timestamp_col: Column holding the bar's UTC timestamp.
        htf_bias_col: Column holding the ``htf_bias`` dict.
        smart_money_col: Column holding the ``smart_money`` dict.
        direction_col: Column holding ``"LONG"``/``"SHORT"``.

    Returns:
        A copy of ``df`` with :data:`PHASE_7A_FEATURE_COLUMNS` appended.
        Existing columns of the same name are overwritten.
    """
    out = df.copy()

    htf_alignment: list[int] = []
    kill_zone: list[int] = []
    sweep_conf: list[bool] = []
    ob_dist: list[float] = []
    ob_range: list[bool] = []
    po3_label: list[str] = []
    po3_oh = {f"po3_{p}": [] for p in PO3_PHASES}

    # itertuples is meaningfully faster than iterrows for ~26k-row
    # historical builds and ~1k-row audit windows alike.
    for row in out.itertuples(index=False):
        rd = row._asdict()  # type: ignore[attr-defined]
        bias_dict = rd.get(htf_bias_col)
        sm_dict = rd.get(smart_money_col)
        direction = rd.get(direction_col, "")
        ts = rd.get(timestamp_col)

        b1 = b4 = None
        if isinstance(bias_dict, dict):
            b1 = bias_dict.get("bias_1h")
            b4 = bias_dict.get("bias_4h")

        try:
            htf_alignment.append(compute_htf_alignment(b1, b4, direction or "LONG"))
        except ValueError:
            # Unknown direction — stamp the neutral row score (1) and
            # let downstream split decide what to do with it.
            htf_alignment.append(1)

        hour = _coerce_hour(ts)
        kill_zone.append(compute_kill_zone_strength(hour) if hour is not None else 0)

        sweep_conf.append(compute_sweep_confirmed(sm_dict))

        d, in_r = compute_ob_distance_features(
            sm_dict, ob_max_distance_pips=ob_max_distance_pips
        )
        ob_dist.append(d)
        ob_range.append(in_r)

        label, oh = compute_po3_phase_features(sm_dict)
        po3_label.append(label)
        for col, val in oh.items():
            po3_oh[col].append(val)

    out["htf_alignment"] = htf_alignment
    out["kill_zone_strength"] = kill_zone
    out["sweep_confirmed"] = sweep_conf
    out["ob_distance_pips"] = ob_dist
    out["ob_in_range"] = ob_range
    out["po3_phase"] = po3_label
    for col, vals in po3_oh.items():
        out[col] = vals

    return out


# ---------------------------------------------------------------------------
# Class wrapper (matches scope §7's "ICTFeatureEngineer" naming)
# ---------------------------------------------------------------------------


class ICTFeatureEngineer:
    """Stateful wrapper around :func:`enrich_features`.

    Holds the ``ob_max_distance_pips`` threshold so the same engineer
    instance can enrich both the training set and the gate-5 live audit
    frame without the threshold drifting between calls. The threshold
    is exposed as :attr:`ob_max_distance_pips` for the
    ``retrain.py`` sidecar writer to read at save time.

    Usage::

        engineer = ICTFeatureEngineer.from_env()
        train_features = engineer.enrich(train_df)
        # ...later, in predict.py / Phase 7.A walk-forward:
        live_features = engineer.enrich(audit_df)
    """

    def __init__(self, ob_max_distance_pips: float = 30.0) -> None:
        if ob_max_distance_pips <= 0:
            raise ValueError(
                f"ob_max_distance_pips must be > 0, got {ob_max_distance_pips}"
            )
        self.ob_max_distance_pips = float(ob_max_distance_pips)

    @classmethod
    def from_env(cls) -> "ICTFeatureEngineer":
        """Construct from ``PRISM_OB_MAX_DISTANCE_PIPS`` (default 30.0).

        For training: prefer this constructor at the entry point of
        ``retrain.py``, then write the value into the model sidecar
        immediately. For predict: load the sidecar value first, then
        construct with that explicit value (NOT from env) so a runtime
        env mismatch is detected, not silently overridden.
        """
        raw = os.environ.get("PRISM_OB_MAX_DISTANCE_PIPS", "30.0")
        try:
            value = float(raw)
        except ValueError:
            logger.warning(
                "PRISM_OB_MAX_DISTANCE_PIPS=%r could not be parsed as float, "
                "using default 30.0",
                raw,
            )
            value = 30.0
        return cls(ob_max_distance_pips=value)

    def enrich(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Apply :func:`enrich_features` with the locked threshold."""
        return enrich_features(
            df,
            ob_max_distance_pips=self.ob_max_distance_pips,
            **kwargs,
        )

    def feature_columns(self) -> tuple[str, ...]:
        """Stable column ordering for downstream X-matrix construction.

        Excludes ``po3_phase`` (the string label is for gate-5, not for
        ML feature ingestion — splitters can't consume strings without
        a separate encoder). The four one-hot bool columns ARE included.
        """
        return tuple(c for c in PHASE_7A_FEATURE_COLUMNS if c != "po3_phase")
