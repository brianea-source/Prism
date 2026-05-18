#!/usr/bin/env python3
"""
PRISM Runner Env Preflight
==========================

Runs once at runner startup. Lists missing-but-expected env keys from
`.env.example` that are NOT present in the live process environment.
Fails loud to runner.log so a silently-masked setting (the 2026-05-18
audit-log gap incident) cannot recur unnoticed.

Usage:
    python scripts/preflight_env.py                 # exits 0 on clean, 1 on missing
    python scripts/preflight_env.py --warn-only     # always exit 0 but log warnings
    python scripts/preflight_env.py --check KEY     # check just one key, machine-friendly

Output is one line per finding, prefixed `[preflight]` so it's grep-able
in runner.log. Redacts values longer than 8 chars by default.

Critical keys (PRISM_SIGNAL_AUDIT_ENABLED, PRISM_INSTRUMENTS,
PRISM_EXECUTION_MODE, PRISM_SLACK_TOKEN) cause non-zero exit when missing
unless --warn-only is passed.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

CRITICAL_KEYS = {
    "PRISM_SIGNAL_AUDIT_ENABLED",
    "PRISM_INSTRUMENTS",
    "PRISM_EXECUTION_MODE",
    "PRISM_SLACK_TOKEN",
    "PRISM_STATE_DIR",
}

SECRET_KEY_PATTERN = re.compile(r"TOKEN|SECRET|KEY|PASSWORD|PAT", re.IGNORECASE)


def parse_env_example(path: Path) -> dict[str, str]:
    """Return {KEY: default_value} for every uncommented `KEY=value` line.

    Lines that start with `#` are documentation and intentionally ignored.
    A commented sample line (`# PRISM_FOO=bar`) is NOT considered a
    declaration; .env.example must commit to *real* defaults if a key is
    intended to be present in every deployment.
    """
    decls: dict[str, str] = {}
    if not path.exists():
        return decls
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        # Strip inline `  # comment`
        line = re.sub(r"\s+#.*$", "", line)
        key, _, val = line.partition("=")
        key = key.strip()
        if key:
            decls[key] = val.strip()
    return decls


def redact(value: str) -> str:
    if len(value) <= 8:
        return value
    return value[:4] + "…" + value[-2:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PRISM env preflight check.")
    parser.add_argument(
        "--env-example",
        type=Path,
        default=Path(__file__).resolve().parent.parent / ".env.example",
        help="Path to .env.example (declarative source of truth).",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Always exit 0, even if critical keys are missing.",
    )
    parser.add_argument(
        "--check",
        metavar="KEY",
        help="Check a single env key and exit 0 (present) or 1 (missing).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print findings; suppress the OK header.",
    )
    args = parser.parse_args(argv)

    if args.check:
        present = bool(os.environ.get(args.check, "").strip())
        if present:
            print(f"[preflight] OK {args.check}")
            return 0
        print(f"[preflight] MISSING {args.check}")
        return 1

    decls = parse_env_example(args.env_example)
    if not decls:
        print(
            f"[preflight] WARN .env.example empty or unreadable at "
            f"{args.env_example}; skipping check.",
            file=sys.stderr,
        )
        return 0

    missing_critical: list[str] = []
    missing_optional: list[str] = []
    present_lines: list[str] = []

    for key, default in sorted(decls.items()):
        live = os.environ.get(key, None)
        if live is None or live == "":
            if key in CRITICAL_KEYS:
                missing_critical.append(key)
            else:
                missing_optional.append(key)
            continue
        # Don't echo secrets to the log.
        shown = "[redacted]" if SECRET_KEY_PATTERN.search(key) else redact(live)
        present_lines.append(f"[preflight] SET {key}={shown}")

    if not args.quiet:
        print(
            f"[preflight] checked {len(decls)} keys from {args.env_example.name}: "
            f"{len(present_lines)} set, {len(missing_critical)} critical missing, "
            f"{len(missing_optional)} optional missing"
        )

    for line in present_lines:
        print(line)
    for key in missing_critical:
        print(f"[preflight] FAIL critical key missing: {key}")
    for key in missing_optional:
        print(f"[preflight] WARN optional key missing: {key}")

    if missing_critical and not args.warn_only:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
