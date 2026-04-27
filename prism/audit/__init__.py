"""PRISM audit package — consumers of structured signal audit logs.

The audit log is written by ``prism.delivery.signal_audit`` (Phase 6.F).
Modules in this package consume it: parquet export, summary stats,
distribution-drift comparison against the historical replay builder.

Schema source of truth lives in :mod:`prism.audit.schema` so writer,
reader, and future historical-replay builder all reference the same
field whitelist.
"""
