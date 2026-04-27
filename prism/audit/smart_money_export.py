"""Smart-money audit log consumer (Phase 7.A Track B).

Reads JSONL audit logs produced by :mod:`prism.delivery.signal_audit`,
emits them as parquet for downstream analysis, summarises distributional
properties, and runs the gate-5 distribution-drift tests against an
arbitrary "historical" DataFrame.

Architectural note — why drift machinery lives here, not in
``prism/data/feature_engineering.py``:

The Phase 7.A implementation PR (PHASE_7A_SCOPE.md §7) will introduce
``feature_engineering.py`` with the 5 ICT feature derivation functions
(``compute_htf_alignment``, ``compute_kill_zone_strength``, etc.). At
that point the gate-5 pipeline becomes:

    audit_log → feature_engineering.enrich → smart_money_export.compare_features
    historical_state.parquet → feature_engineering.enrich → ↑

Track B owns the comparison side of that flow — single-feature drift
tests, Bonferroni correction, the ≤1/5 majority-vote pass rule from
PHASE_7A_SCOPE.md §6.1. The feature derivation step is Phase 7.A's
responsibility and lands later. By making :func:`compare_features`
accept any two DataFrames with named columns, this module is plug-in
compatible with whatever ``feature_engineering`` ends up emitting.

The chicken-and-egg with ``historical_state.py`` is intentional: ship
the diff *machinery* now (statistical tests, Bonferroni, pass rule),
plug in the historical side once Phase 7.A's builder produces it.

CLI
---
::

    python -m prism.audit.smart_money_export summary
    python -m prism.audit.smart_money_export export --output audit.parquet
    python -m prism.audit.smart_money_export diff \\
        --live live.parquet --historical hist.parquet \\
        --features htf_alignment:int_ordinal,sweep_confirmed:bool

See ``--help`` on each subcommand for full options.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from prism.audit.schema import (
    ALL_FIELDS,
    AUDIT_FIELDS,
    AuditSchemaError,
    SMART_MONEY_SUBKEYS,
    TIMESTAMP_FIELD,
    validate_record,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def _read_audit_jsonl(path: Path) -> pd.DataFrame:
    """Read a single audit JSONL file into a DataFrame.

    Records that fail schema validation are logged at WARNING and
    skipped — this is intentional forward-compatibility for the case
    where a writer schema bump lands before a reader pin update.
    """
    if not path.exists():
        raise FileNotFoundError(f"audit log not found: {path}")

    records = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
                validate_record(record, strict=False)
                records.append(record)
            except (json.JSONDecodeError, AuditSchemaError) as exc:
                logger.warning(
                    "Skipping malformed audit record at %s:%d: %s",
                    path, line_no, exc,
                )
                skipped += 1

    if skipped:
        logger.info(
            "Read %d records from %s (%d skipped)",
            len(records), path, skipped,
        )

    df = pd.DataFrame(records, columns=list(ALL_FIELDS))
    if not df.empty:
        df[TIMESTAMP_FIELD] = pd.to_datetime(df[TIMESTAMP_FIELD], utc=True)
    return df


def read_audit_window(
    state_dir: Path | str,
    *,
    instrument: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """Load all audit JSONL files in a state directory matching the filter.

    Args:
        state_dir: The PRISM state dir (the same one ``PRISM_STATE_DIR``
            points at). Audit logs live under
            ``<state_dir>/signal_audit/<instrument>/YYYY-MM-DD.jsonl``.
        instrument: If set, only load this instrument's logs. Otherwise
            globs all instruments.
        start_date: Inclusive lower bound on the file date (UTC).
        end_date: Inclusive upper bound on the file date (UTC).

    Returns:
        A single concatenated DataFrame. Empty DataFrame (with the
        canonical columns) when no matching logs exist — operators
        running ``summary`` against a fresh state dir get a clean empty
        result rather than an exception.
    """
    base = Path(state_dir) / "signal_audit"
    if not base.exists():
        return pd.DataFrame(columns=list(ALL_FIELDS))

    if instrument is not None:
        instruments: Iterable[str] = [instrument]
    else:
        instruments = sorted(d.name for d in base.iterdir() if d.is_dir())

    frames: list[pd.DataFrame] = []
    for inst in instruments:
        inst_dir = base / inst
        if not inst_dir.exists():
            continue
        for jsonl_path in sorted(inst_dir.glob("*.jsonl")):
            try:
                file_date = date.fromisoformat(jsonl_path.stem)
            except ValueError:
                logger.warning(
                    "Unexpected audit filename (skipping): %s", jsonl_path,
                )
                continue
            if start_date is not None and file_date < start_date:
                continue
            if end_date is not None and file_date > end_date:
                continue
            frames.append(_read_audit_jsonl(jsonl_path))

    if not frames:
        return pd.DataFrame(columns=list(ALL_FIELDS))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------

def to_parquet(df: pd.DataFrame, output_path: Path | str) -> Path:
    """Write the audit DataFrame to parquet.

    Nested dict columns (``htf_bias``, ``smart_money``) are serialized
    as struct/map types via pyarrow's natural pandas conversion. The
    parent directory is created if missing — operators don't need to
    pre-create the output dir.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    return output


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> dict:
    """Operator-friendly summary of an audit DataFrame.

    Reports total signal count, breakdown by instrument and direction,
    smart-money sub-feature presence rates (% of signals where each of
    ``ob`` / ``sweep`` / ``po3`` was non-None), confidence quantiles,
    and the date range covered. Empty DataFrame returns a stub with
    zero counts so the CLI doesn't crash on a fresh state dir.
    """
    if df.empty:
        return {
            "total_signals": 0,
            "by_instrument": {},
            "by_direction": {},
            "smart_money_presence": {sub: 0.0 for sub in SMART_MONEY_SUBKEYS},
            "confidence": None,
            "date_range": None,
        }

    presence = {}
    for sub in SMART_MONEY_SUBKEYS:
        is_present = df["smart_money"].apply(
            lambda sm: isinstance(sm, dict) and sm.get(sub) is not None
        )
        presence[sub] = round(float(is_present.mean()), 4)

    # PR #22 review N3: coerce confidence to numeric defensively. If a
    # malformed JSONL line slips through with confidence as a string,
    # df["confidence"].mean() raises TypeError. ``errors="coerce"``
    # turns the bad rows into NaN and drops them — summary stats stay
    # representative of the well-formed records.
    confidence_numeric = pd.to_numeric(df["confidence"], errors="coerce").dropna()
    if confidence_numeric.empty:
        confidence = None
    else:
        confidence = {
            "mean": float(confidence_numeric.mean()),
            "median": float(confidence_numeric.median()),
            "p25": float(confidence_numeric.quantile(0.25)),
            "p75": float(confidence_numeric.quantile(0.75)),
            "min": float(confidence_numeric.min()),
            "max": float(confidence_numeric.max()),
        }

    ts = pd.to_datetime(df[TIMESTAMP_FIELD], utc=True)
    date_range = [ts.min().isoformat(), ts.max().isoformat()]

    return {
        "total_signals": int(len(df)),
        "by_instrument": df["instrument"].value_counts().to_dict(),
        "by_direction": df["direction"].value_counts().to_dict(),
        "smart_money_presence": presence,
        "confidence": confidence,
        "date_range": date_range,
    }


# ---------------------------------------------------------------------------
# Drift comparison (gate-5 machinery)
# ---------------------------------------------------------------------------

#: Feature types accepted by :func:`compare_feature`. Matches the
#: per-feature dispatch in PHASE_7A_SCOPE.md §6.1.
FEATURE_TYPES: tuple[str, ...] = (
    "int_ordinal",   # chi-squared GoF
    "bool",          # Fisher's exact
    "categorical",   # chi-squared GoF
    "continuous",    # KS two-sample
)


def _chi_squared_gof(
    live: pd.Series,
    historical: pd.Series,
    *,
    strict_novel_bins: bool = False,
    **_,
) -> tuple[float, float, str]:
    """Chi-squared goodness-of-fit: live observed vs historical expected.

    Treats the historical sample's proportions as the hypothesized
    distribution, scales to live's sample size, and runs chi-squared
    GoF. This is the right framing for our setup (years of historical
    bars vs. weeks of live audits) — historical is treated as ground
    truth, deviations get attributed to live or to builder bugs.

    Args:
        live: Observed sample (live audit log).
        historical: Reference sample (historical replay builder).
        strict_novel_bins: If True, raise when ``live`` contains values
            in bins absent from ``historical``. Default False — emits a
            WARNING instead and proceeds with the surviving bins, which
            biases the gate low. PR #22 review item B2.

    Returns:
        ``(statistic, p_value, test_name)``.

    Raises:
        ValueError: when both samples are empty, when no historical bins
            have non-zero proportion, when fewer than 2 bins survive the
            historical mask (degenerate chi-squared with df=0 — see B1),
            or when ``strict_novel_bins=True`` and live has novel bins.
    """
    from scipy.stats import chisquare

    live_clean = live.dropna()
    hist_clean = historical.dropna()
    if live_clean.empty or hist_clean.empty:
        raise ValueError("chi-squared GoF requires non-empty samples")

    bins = sorted(set(live_clean.unique()) | set(hist_clean.unique()))
    live_counts = live_clean.value_counts().reindex(bins, fill_value=0).astype(float)
    hist_counts = hist_clean.value_counts().reindex(bins, fill_value=0).astype(float)

    n_live = float(live_counts.sum())
    n_hist = float(hist_counts.sum())
    expected = (hist_counts / n_hist) * n_live

    # Drop bins where the historical proportion is zero — chi-squared
    # is undefined when E[i] = 0. mask=False precisely on bins absent
    # from historical.
    mask = expected > 0
    if not mask.any():
        raise ValueError("no bins with non-zero historical proportion")

    # PR #22 review B2: detect live observations in bins absent from
    # historical. These get silently excluded by the mask below — a
    # real risk when, e.g., a model bump introduces a new label
    # post-training. Warn by default; strict mode raises.
    novel_mask = (~mask) & (live_counts > 0)
    if novel_mask.any():
        novel_bins = list(live_counts.index[novel_mask])
        novel_obs = int(live_counts[novel_mask].sum())
        msg = (
            f"chi-squared GoF: live has {novel_obs} observations in "
            f"{len(novel_bins)} bin(s) absent from historical "
            f"({novel_bins!r}). These will be excluded — gate-5 result "
            f"will be biased low."
        )
        if strict_novel_bins:
            raise ValueError(msg)
        logger.warning(msg)

    live_kept = live_counts[mask].values
    expected_kept = expected[mask].values

    # PR #22 review B1: chi-squared GoF needs ≥2 surviving bins. With
    # k=1, df = k - 1 = 0 — scipy's chisquare returns either nan
    # (silent false negative) or a degenerate stat that always rejects.
    # Refuse rather than emit a misleading p-value. Common in practice
    # when historical is an all-same-value sample (stuck detector).
    if len(live_kept) < 2:
        raise ValueError(
            "chi-squared GoF requires ≥2 non-zero bins in historical; "
            "got an all-same-value historical sample (degenerate test, "
            "df=0)"
        )

    expected_kept = expected_kept * (live_kept.sum() / expected_kept.sum())

    statistic, p_value = chisquare(live_kept, expected_kept)
    return float(statistic), float(p_value), "chi_squared_gof"


def _fishers_exact(
    live: pd.Series, historical: pd.Series, **_,
) -> tuple[float, float, str]:
    """Fisher's exact on a 2x2 contingency. Rows = sample, cols = bool.

    Accepts and ignores extra kwargs (e.g. ``strict_novel_bins``) so the
    dispatch table in :func:`compare_feature` can pass kwargs uniformly
    without per-test branching. A 2x2 contingency has no novel-bin
    failure mode by construction.
    """
    from scipy.stats import fisher_exact

    def _coerce(s: pd.Series) -> pd.Series:
        cleaned = s.dropna()
        if cleaned.dtype == bool:
            return cleaned
        return cleaned.astype(bool)

    l = _coerce(live)
    h = _coerce(historical)
    if l.empty or h.empty:
        raise ValueError("Fisher's exact requires non-empty samples")

    table = [
        [int((~l).sum()), int(l.sum())],
        [int((~h).sum()), int(h.sum())],
    ]
    odds_ratio, p_value = fisher_exact(table)
    return float(odds_ratio), float(p_value), "fishers_exact"


def _ks_two_sample(
    live: pd.Series, historical: pd.Series, **_,
) -> tuple[float, float, str]:
    """Kolmogorov-Smirnov two-sample on continuous data.

    Accepts and ignores extra kwargs (e.g. ``strict_novel_bins``) so the
    dispatch table in :func:`compare_feature` can pass kwargs uniformly.
    KS is on continuous distributions and has no notion of "bins" — the
    novel-bin concept simply doesn't apply.
    """
    from scipy.stats import ks_2samp

    live_clean = live.dropna().astype(float).values
    hist_clean = historical.dropna().astype(float).values
    if len(live_clean) == 0 or len(hist_clean) == 0:
        raise ValueError("KS test requires non-empty samples")

    statistic, p_value = ks_2samp(live_clean, hist_clean)
    return float(statistic), float(p_value), "ks_two_sample"


_TEST_DISPATCH = {
    "int_ordinal": _chi_squared_gof,
    "categorical": _chi_squared_gof,
    "bool": _fishers_exact,
    "continuous": _ks_two_sample,
}


def compare_feature(
    live: pd.Series,
    historical: pd.Series,
    *,
    feature_type: str,
    alpha: float = 0.01,
    strict_novel_bins: bool = False,
) -> dict:
    """Run the gate-5 drift test for one feature.

    Args:
        live: Feature values from the live audit log.
        historical: Feature values from the historical replay builder.
        feature_type: One of :data:`FEATURE_TYPES`.
        alpha: Per-test significance threshold. Default 0.01 matches
            the Bonferroni-corrected per-feature α from
            PHASE_7A_SCOPE.md §6.1.
        strict_novel_bins: If True, raise when a chi-squared test sees
            live observations in bins absent from historical. Default
            False — emits a WARNING. Only meaningful for
            ``feature_type`` of ``int_ordinal`` or ``categorical``;
            ignored by Fisher's exact and KS. PR #22 review item B2.

    Returns:
        A dict with keys ``feature_type``, ``test``, ``statistic``,
        ``p_value``, ``alpha``, ``reject``, ``n_live``, ``n_historical``.
    """
    if feature_type not in _TEST_DISPATCH:
        raise ValueError(
            f"unknown feature_type {feature_type!r}; "
            f"expected one of {FEATURE_TYPES}"
        )

    test_fn = _TEST_DISPATCH[feature_type]
    statistic, p_value, test_name = test_fn(
        live, historical, strict_novel_bins=strict_novel_bins,
    )

    return {
        "feature_type": feature_type,
        "test": test_name,
        "statistic": statistic,
        "p_value": p_value,
        "alpha": alpha,
        "reject": bool(p_value < alpha),
        "n_live": int(live.dropna().shape[0]),
        "n_historical": int(historical.dropna().shape[0]),
    }


def compare_features(
    live_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    feature_specs: Sequence[tuple[str, str]],
    *,
    family_alpha: float = 0.05,
    max_rejections_for_pass: int = 1,
    strict_novel_bins: bool = False,
) -> dict:
    """Run the gate-5 drift gate across multiple features.

    Implements the protocol in PHASE_7A_SCOPE.md §6.1:

    1. Bonferroni-correct the family-wise α: per-feature α =
       ``family_alpha / len(feature_specs)``.
    2. Run the appropriate test per feature (dispatched by type).
    3. Pass if at most ``max_rejections_for_pass`` features reject —
       defaults to 1, matching the "≤1 of 5" relaxation rule.

    Args:
        live_df: DataFrame with one column per spec, sourced from the
            live audit log (after :func:`feature_engineering.enrich`
            once Phase 7.A ships).
        historical_df: Same column shape, sourced from the historical
            replay builder.
        feature_specs: List of ``(column_name, feature_type)`` tuples.
        family_alpha: Family-wise α (default 0.05).
        max_rejections_for_pass: Pass threshold (default 1, per §6.1).
        strict_novel_bins: Forwarded to chi-squared tests. See
            :func:`compare_feature`.

    Returns:
        Aggregate result dict with per-feature results and overall
        ``passed`` boolean.

    Notes:
        **po3_phase encoding (cross-reference with Phase 7.A impl PR).**
        ``feature_engineering.enrich`` exposes ``po3_phase`` two ways:

        * a single 4-category label column ``po3_phase``
          (values: ``accumulation``, ``manipulation``, ``distribution``,
          ``unknown``);
        * four one-hot bool columns ``po3_accumulation``,
          ``po3_manipulation``, ``po3_distribution``, ``po3_unknown``.

        For gate-5, pass the **single label column** as
        ``("po3_phase", "categorical")`` — one chi-squared GoF on the
        joint 4-cat distribution per PHASE_7A_SCOPE.md §6.1. Iterating
        the four one-hot columns and running four separate Fisher's
        exacts double-counts the family α and is the wrong answer
        statistically; the one-hot columns exist for downstream ML
        feature ingestion, not gate-5.
    """
    n = len(feature_specs)
    if n == 0:
        raise ValueError("feature_specs must be non-empty")

    per_feature_alpha = family_alpha / n

    per_feature_results = []
    rejections = 0
    for col, ftype in feature_specs:
        if col not in live_df.columns:
            raise KeyError(f"live_df missing column {col!r}")
        if col not in historical_df.columns:
            raise KeyError(f"historical_df missing column {col!r}")
        result = compare_feature(
            live_df[col], historical_df[col],
            feature_type=ftype,
            alpha=per_feature_alpha,
            strict_novel_bins=strict_novel_bins,
        )
        result["feature"] = col
        per_feature_results.append(result)
        if result["reject"]:
            rejections += 1

    return {
        "n_features": n,
        "family_alpha": family_alpha,
        "per_feature_alpha": per_feature_alpha,
        "max_rejections_for_pass": max_rejections_for_pass,
        "rejections": rejections,
        "passed": bool(rejections <= max_rejections_for_pass),
        "per_feature": per_feature_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_feature_specs(raw: str) -> list[tuple[str, str]]:
    """Parse ``col1:type1,col2:type2`` from the --features arg."""
    specs = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise argparse.ArgumentTypeError(
                f"invalid feature spec {piece!r}; expected col:type"
            )
        col, ftype = piece.split(":", 1)
        col, ftype = col.strip(), ftype.strip()
        if ftype not in FEATURE_TYPES:
            raise argparse.ArgumentTypeError(
                f"unknown feature type {ftype!r} for column {col!r}; "
                f"expected one of {FEATURE_TYPES}"
            )
        specs.append((col, ftype))
    if not specs:
        raise argparse.ArgumentTypeError("--features must be non-empty")
    return specs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m prism.audit.smart_money_export",
        description=(
            "Read, summarise, and drift-compare PRISM signal audit logs "
            "(Phase 6.F output → Phase 7.A gate-5 input)."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # `summary` — print summary stats over a window of the audit log
    p_summary = sub.add_parser("summary", help="Print summary stats as JSON")
    p_summary.add_argument(
        "--state-dir", default="state",
        help="Path to PRISM_STATE_DIR (default: state)",
    )
    p_summary.add_argument("--instrument", help="Filter to one instrument")
    p_summary.add_argument(
        "--start-date", type=date.fromisoformat,
        help="Inclusive YYYY-MM-DD lower bound",
    )
    p_summary.add_argument(
        "--end-date", type=date.fromisoformat,
        help="Inclusive YYYY-MM-DD upper bound",
    )

    # `export` — write window to parquet for downstream tooling
    p_export = sub.add_parser("export", help="Export window to parquet")
    p_export.add_argument("--state-dir", default="state")
    p_export.add_argument("--instrument")
    p_export.add_argument("--start-date", type=date.fromisoformat)
    p_export.add_argument("--end-date", type=date.fromisoformat)
    p_export.add_argument(
        "--output", required=True,
        help="Output parquet path (parent dirs created if missing)",
    )

    # `diff` — gate-5 drift gate between two parquets
    p_diff = sub.add_parser("diff", help="Run gate-5 drift gate")
    p_diff.add_argument("--live", required=True, help="Path to live parquet")
    p_diff.add_argument(
        "--historical", required=True, help="Path to historical parquet",
    )
    p_diff.add_argument(
        "--features", required=True, type=_parse_feature_specs,
        help="Comma-separated col:type pairs (types: %s)" % ", ".join(FEATURE_TYPES),
    )
    p_diff.add_argument("--family-alpha", type=float, default=0.05)
    p_diff.add_argument("--max-rejections", type=int, default=1)
    p_diff.add_argument(
        "--strict-novel-bins", action="store_true",
        help=(
            "Raise (instead of warn) when chi-squared tests see live "
            "observations in bins absent from historical. Recommended "
            "for production gate-5 runs once the historical builder "
            "is stable."
        ),
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    args = _build_parser().parse_args(argv)

    if args.cmd == "summary":
        df = read_audit_window(
            args.state_dir,
            instrument=args.instrument,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        print(json.dumps(summarize(df), indent=2, default=str))
        return 0

    if args.cmd == "export":
        df = read_audit_window(
            args.state_dir,
            instrument=args.instrument,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        path = to_parquet(df, args.output)
        print(f"wrote {len(df)} records to {path}")
        return 0

    if args.cmd == "diff":
        live_df = pd.read_parquet(args.live)
        hist_df = pd.read_parquet(args.historical)
        result = compare_features(
            live_df, hist_df, args.features,
            family_alpha=args.family_alpha,
            max_rejections_for_pass=args.max_rejections,
            strict_novel_bins=args.strict_novel_bins,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0 if result["passed"] else 1

    return 2  # unreachable; argparse enforces required subcommand


if __name__ == "__main__":
    sys.exit(main())
