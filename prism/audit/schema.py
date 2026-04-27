"""Canonical schema for the per-signal audit log.

Single source of truth for the JSONL record shape produced by
:mod:`prism.delivery.signal_audit` and consumed by
:mod:`prism.audit.smart_money_export`. The Phase 7.A historical replay
builder (``prism/data/historical_state.py``, future) MUST emit the same
top-level schema so gate-5 drift comparisons are comparing like-for-like.

Why this lives in ``prism.audit`` rather than ``prism.delivery``:
``prism.delivery`` writes one half of the contract; ``prism.audit`` and
``prism.data`` (future) read it. The schema doesn't belong to any single
consumer — it belongs to the audit subsystem as a whole.

Schema invariant — DO NOT change without a coupled change to:
  1. ``prism/delivery/signal_audit.AUDIT_FIELDS`` (re-exported from here)
  2. The historical replay builder when it lands
  3. Any downstream parquet schema that joins live + historical
A schema change here without (1)+(2) silently breaks gate 5.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Top-level fields
# ---------------------------------------------------------------------------

#: Audit timestamp (UTC ISO8601). Pinned to the runner's scan time, not
#: wall-clock at write — see signal_audit.write_signal_audit's docstring.
TIMESTAMP_FIELD = "audit_ts"

#: Subset of ``SignalPacket`` carried into the audit record. The whitelist
#: is explicit (not ``asdict`` + filter) so SignalPacket additions don't
#: silently leak unrelated fields. PR #21 review item: this is 11 fields,
#: not the 7 the original spec listed — historical builder must match.
AUDIT_FIELDS: tuple[str, ...] = (
    "instrument",
    "direction",
    "confidence",
    "confidence_level",
    "signal_id",
    "signal_time",
    "model_version",
    "regime",
    "news_bias",
    "htf_bias",       # nested dict, see HTF_BIAS_SUBKEYS
    "smart_money",    # nested dict, see SMART_MONEY_SUBKEYS
)

#: All top-level keys in the audit JSONL record (12 total: 1 timestamp +
#: 11 SignalPacket fields). This is the contract the historical replay
#: builder must match column-for-column at the top level.
ALL_FIELDS: tuple[str, ...] = (TIMESTAMP_FIELD, *AUDIT_FIELDS)


# ---------------------------------------------------------------------------
# Nested dict sub-schemas
# ---------------------------------------------------------------------------

#: Keys inside ``htf_bias`` (Phase 5). Set by HTFBiasResult; absent when
#: the HTF gate is disabled or RANGING is treated as no-bias.
HTF_BIAS_SUBKEYS: tuple[str, ...] = (
    "bias_1h",
    "bias_4h",
    "aligned",
    "allowed_direction",
)

#: Top-level keys inside ``smart_money`` (Phase 6.D). Each is either a
#: dict with detector-specific shape or ``None`` when the detector found
#: nothing this scan.
SMART_MONEY_SUBKEYS: tuple[str, ...] = (
    "ob",        # OrderBlock dict or None
    "sweep",     # LiquiditySweep dict or None
    "po3",       # Po3State dict or None
)

#: Keys inside ``smart_money.ob`` when present.
SMART_MONEY_OB_SUBKEYS: tuple[str, ...] = (
    "state",
    "direction",
    "high",
    "low",
    "midpoint",
    "timeframe",
    "distance_pips",
    "effective_direction",
)

#: Keys inside ``smart_money.sweep`` when present.
SMART_MONEY_SWEEP_SUBKEYS: tuple[str, ...] = (
    "type",
    "swept_level",
    "sweep_bar",
    "bars_ago",
    "displacement_followed",
)

#: Keys inside ``smart_money.po3`` when present.
SMART_MONEY_PO3_SUBKEYS: tuple[str, ...] = (
    "phase",
    "session",
    "range_size_pips",
    "sweep_detected",
    "displacement_detected",
    "is_entry_phase",
)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

class AuditSchemaError(ValueError):
    """Raised when an audit record violates the canonical schema."""


def validate_record(record: dict, *, strict: bool = False) -> None:
    """Verify ``record`` matches the canonical audit schema.

    Args:
        record: A parsed JSONL line from the audit log.
        strict: If True, raise on extra top-level keys. If False, allow
            extras (forward-compatible reads — useful when the writer
            ships a schema bump before the reader's pin is updated).

    Raises:
        AuditSchemaError: when a required top-level key is missing or
            ``record`` is not a dict.
    """
    if not isinstance(record, dict):
        raise AuditSchemaError(
            f"audit record must be a dict, got {type(record).__name__}"
        )

    missing = [k for k in ALL_FIELDS if k not in record]
    if missing:
        raise AuditSchemaError(
            f"audit record missing required fields: {missing}"
        )

    if strict:
        extras = [k for k in record if k not in ALL_FIELDS]
        if extras:
            raise AuditSchemaError(
                f"audit record has unexpected fields (strict mode): {extras}"
            )
