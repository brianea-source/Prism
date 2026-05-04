"""GitHub Issues backend for the PRISM trade journal (Task 0.4).

Lifecycle
---------
* ``on_signal_fired(packet)``  — create the issue, write the body, label it.
* ``on_trade_filled(signal_id, ticket)`` — comment with the MT5 ticket and
  swap ``phase:pending`` → ``phase:open``.
* ``on_tp1_hit(signal_id, ticket)``     — comment, swap to ``phase:tp1``.
* ``on_trade_closed(signal_id, ticket, pnl)`` — comment with realized PnL,
  close the issue, swap to ``phase:closed`` + the matching outcome label.

State
-----
A ``signal_id → issue_number`` map is persisted under
``state/journal_map.json`` so the runner can reconcile fills/TP1/close
events even after a restart. The map is the canonical lookup; the
``find_open_issue(signal_id)`` helper falls back to the GitHub API
search index only when the map is missing an entry (e.g. a journal
created before Task 0.4 shipped, or after a state-dir wipe).

Failure semantics
-----------------
**Every public function in this module catches its own exceptions.**
The journal is a side-channel: a 500 from GitHub, an expired token, or
a corrupt state file MUST NOT take down signal delivery. All errors
are logged at ERROR with a traceback and the function returns ``None``.

Auth
----
Reads ``GH_TOKEN`` from the environment when set. Otherwise it relies
on system ``gh auth`` credentials. The REST fallback is used when
``gh`` is not on ``PATH`` (CI containers, minimal Linux installs).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from prism.execution.mt5_bridge import SignalPacket

logger = logging.getLogger(__name__)

# Resolved at import time so the runner pays the lookup once. ``None`` means
# "fall back to REST"; the value is the absolute path to the gh binary.
_GH_BIN: Optional[str] = shutil.which("gh")

# Repo passed to gh as ``--repo``. Defaults to the production repo so a
# fresh checkout that hasn't run ``gh repo set-default`` still resolves.
_DEFAULT_REPO = "brianea-source/Prism"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def on_signal_fired(packet: "SignalPacket") -> Optional[int]:
    """Create a GitHub issue for ``packet``. Returns the issue number, or
    ``None`` on failure (logged, not raised)."""
    try:
        signal_id = getattr(packet, "signal_id", None)
        if not signal_id:
            logger.error("github_journal: packet has no signal_id, skipping")
            return None

        # Guard against duplicates — a runner restart that re-fires the same
        # in-flight key would otherwise create a second issue.
        existing = _load_map().get(signal_id)
        if existing is not None:
            logger.info(
                "github_journal: issue #%s already exists for signal %s",
                existing, signal_id,
            )
            return existing

        title = _build_title(packet)
        body = _render_body(packet)
        labels = _labels_for_signal(packet)

        number = _gh_issue_create(title=title, body=body, labels=labels)
        if number is None:
            return None

        _store_mapping(signal_id, number)
        logger.info(
            "github_journal: created issue #%s for signal %s (%s %s)",
            number, signal_id, packet.instrument, packet.direction,
        )
        return number
    except Exception as exc:  # noqa: BLE001 — journal must not propagate
        logger.error(
            "github_journal: on_signal_fired failed for %s: %s",
            getattr(packet, "signal_id", "?"), exc, exc_info=True,
        )
        return None


def on_trade_filled(signal_id: str, ticket: int) -> Optional[int]:
    """Comment that the trade filled and swap the lifecycle label."""
    try:
        number = _resolve_issue(signal_id)
        if number is None:
            logger.warning("github_journal: no issue for signal_id=%s on fill", signal_id)
            return None

        body = (
            f"✅ **Filled** — MT5 ticket `{ticket}` at "
            f"{_now_iso()}.\n\n_Phase: pending → open._"
        )
        if not _gh_issue_comment(number, body):
            return None
        _gh_issue_swap_label(number, remove="phase:pending", add="phase:open")
        return number
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "github_journal: on_trade_filled failed for %s: %s",
            signal_id, exc, exc_info=True,
        )
        return None


def on_tp1_hit(signal_id: str, ticket: int) -> Optional[int]:
    """Comment that TP1 hit and swap the lifecycle label."""
    try:
        number = _resolve_issue(signal_id)
        if number is None:
            logger.warning("github_journal: no issue for signal_id=%s on tp1", signal_id)
            return None

        body = (
            f"🎯 **TP1 hit** on ticket `{ticket}` at {_now_iso()}.\n\n"
            f"_Phase: open → tp1. Partial taken; remainder runs to TP2._"
        )
        if not _gh_issue_comment(number, body):
            return None
        _gh_issue_swap_label(number, remove="phase:open", add="phase:tp1")
        return number
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "github_journal: on_tp1_hit failed for %s: %s",
            signal_id, exc, exc_info=True,
        )
        return None


def on_trade_closed(
    signal_id: str,
    ticket: int,
    pnl: float,
    *,
    risk_usd: Optional[float] = None,
) -> Optional[int]:
    """Comment with realized PnL, close the issue, label outcome.

    ``risk_usd`` (optional) lets us classify breakeven within ±0.1R when
    the caller knows the trade's risk in USD. Without it, breakeven means
    |pnl| < $1.
    """
    try:
        number = _resolve_issue(signal_id)
        if number is None:
            logger.warning("github_journal: no issue for signal_id=%s on close", signal_id)
            return None

        outcome = _classify_outcome(pnl, risk_usd=risk_usd)
        sign = "+" if pnl >= 0 else "-"
        body = (
            f"🏁 **Closed** — ticket `{ticket}` at {_now_iso()}.\n\n"
            f"Realized PnL: **{sign}${abs(pnl):,.2f}**  →  outcome: `{outcome}`."
        )
        if not _gh_issue_comment(number, body):
            return None

        # Phase label (remove whatever phase we were in; tp1 is the most
        # likely predecessor, but a position closed before TP1 still came
        # from phase:open).
        for prev in ("phase:tp1", "phase:open", "phase:pending"):
            _gh_issue_swap_label(number, remove=prev, add="phase:closed")

        _gh_issue_add_label(number, f"outcome:{outcome}")
        _gh_issue_close(number)
        return number
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "github_journal: on_trade_closed failed for %s: %s",
            signal_id, exc, exc_info=True,
        )
        return None


def find_open_issue(signal_id: str) -> Optional[int]:
    """Resolve ``signal_id → issue number`` via local map first, then via
    ``gh issue list --search`` as a fallback. Returns ``None`` on miss."""
    cached = _load_map().get(signal_id)
    if cached is not None:
        return cached

    if _GH_BIN is None:
        return None
    try:
        # Search by signal_id in the body. We scope to all states because a
        # closed-and-reopened journal entry should still resolve.
        out = _run_gh([
            "issue", "list",
            "--repo", _repo(),
            "--state", "all",
            "--search", f"signal_id:{signal_id}",
            "--json", "number",
            "--limit", "5",
        ])
        if not out:
            return None
        rows = json.loads(out)
        if not rows:
            return None
        # Take the lowest number — oldest issue, in case of accidental dupes.
        number = min(int(r["number"]) for r in rows if "number" in r)
        # Cache it so the next call is fast.
        _store_mapping(signal_id, number)
        return number
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "github_journal: find_open_issue search failed for %s: %s",
            signal_id, exc, exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# State file (signal_id → issue number)
# ---------------------------------------------------------------------------

def _state_dir() -> Path:
    return Path(os.environ.get("PRISM_STATE_DIR", "state"))


def _map_path() -> Path:
    return _state_dir() / "journal_map.json"


def _load_map() -> Dict[str, int]:
    path = _map_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
        if not isinstance(raw, dict):
            return {}
        # Coerce values to int — JSON lets numerics drift to floats if
        # someone hand-edits the file. Issue numbers are always integers.
        return {str(k): int(v) for k, v in raw.items()}
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "github_journal: corrupt journal_map at %s (%s) — starting fresh",
            path, exc,
        )
        return {}


def _store_mapping(signal_id: str, issue_number: int) -> None:
    path = _map_path()
    try:
        data = _load_map()
        data[signal_id] = int(issue_number)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "github_journal: failed to persist journal_map: %s", exc,
        )


def _resolve_issue(signal_id: str) -> Optional[int]:
    """Map first, search fallback. Used by every lifecycle hook."""
    return find_open_issue(signal_id)


# ---------------------------------------------------------------------------
# Body / title rendering
# ---------------------------------------------------------------------------

def _build_title(packet: "SignalPacket") -> str:
    return (
        f"[trade] {packet.instrument} {packet.direction} "
        f"@ {_fmt_price(packet.entry, packet.instrument)}"
    )


def _render_body(packet: "SignalPacket") -> str:
    """Markdown body. Stable line order — tests assert on substrings."""
    lines = []
    lines.append("## Signal")
    lines.append("")
    lines.append(f"- **signal_id:** `{packet.signal_id}`")
    lines.append(f"- **instrument:** {packet.instrument}")
    lines.append(f"- **direction:** {packet.direction}")
    lines.append(f"- **session:** {_session_for(packet)}")
    lines.append(f"- **signal_time:** {packet.signal_time}")
    lines.append(f"- **model_version:** {getattr(packet, 'model_version', 'unknown')}")
    lines.append("")
    lines.append("## Levels")
    lines.append("")
    lines.append(f"- **entry:** {_fmt_price(packet.entry, packet.instrument)}")
    lines.append(f"- **stop loss:** {_fmt_price(packet.sl, packet.instrument)}")
    lines.append(f"- **TP1:** {_fmt_price(packet.tp1, packet.instrument)}")
    lines.append(f"- **TP2:** {_fmt_price(packet.tp2, packet.instrument)}")
    lines.append(f"- **R:R:** {packet.rr_ratio:.2f}")
    lines.append(f"- **magnitude (pips):** {packet.magnitude_pips:.1f}")
    lines.append("")

    lines.append("## Context")
    lines.append("")
    lines.append(f"- **regime:** {packet.regime}")
    lines.append(f"- **news bias:** {packet.news_bias}")
    lines.append(
        f"- **confidence:** {packet.confidence:.2f} "
        f"({packet.confidence_level})"
    )
    lines.append("")

    htf = getattr(packet, "htf_bias", None)
    if htf:
        lines.append("## HTF Bias")
        lines.append("")
        for k in ("bias_4h", "bias_1h", "aligned", "allowed_direction"):
            if k in htf:
                lines.append(f"- **{k}:** {htf[k]}")
        lines.append("")

    sm = getattr(packet, "smart_money", None)
    if sm:
        lines.append("## Smart Money Confluence")
        lines.append("")
        ob = sm.get("ob") if isinstance(sm, dict) else None
        if ob:
            lines.append(
                f"- **OB:** {ob.get('direction', '?')} "
                f"@ {ob.get('midpoint', '?')} "
                f"({ob.get('distance_pips', '?')} pips, "
                f"{ob.get('timeframe', '?')})"
            )
        sweep = sm.get("sweep") if isinstance(sm, dict) else None
        if sweep:
            lines.append(
                f"- **sweep:** {sweep.get('type', '?')} of "
                f"{sweep.get('swept_level', '?')} "
                f"({sweep.get('bars_ago', '?')} bars ago, "
                f"displacement={sweep.get('displacement_followed', '?')})"
            )
        po3 = sm.get("po3") if isinstance(sm, dict) else None
        if po3:
            lines.append(
                f"- **Po3:** {po3.get('phase', '?')} ({po3.get('session', '?')}, "
                f"range={po3.get('range_size_pips', '?')} pips, "
                f"entry_phase={po3.get('is_entry_phase', '?')})"
            )
        lines.append("")

    fvg = getattr(packet, "fvg_zone", None)
    if fvg:
        lines.append("## FVG Zone")
        lines.append("")
        lines.append(f"```json\n{json.dumps(fvg, default=str, indent=2)}\n```")
        lines.append("")

    if getattr(packet, "approximate_sizing", False):
        lines.append(
            "> ⚠️ **Approximate sizing** — lot size used the retail-pip-value "
            "approximation (`PRISM_ALLOW_APPROX_PIP_VALUE=1`)."
        )
        lines.append("")

    lines.append("---")
    lines.append("_Auto-created by PRISM `prism.journal.github_issues`._")
    return "\n".join(lines)


def _labels_for_signal(packet: "SignalPacket") -> list:
    labels = ["phase:pending"]
    inst = (packet.instrument or "").upper()
    if inst:
        labels.append(f"inst:{inst}")
    direction = (packet.direction or "").upper()
    if direction in {"LONG", "SHORT"}:
        labels.append(f"dir:{direction}")
    session = _session_for(packet)
    if session != "unknown":
        labels.append(f"session:{session.lower()}")
    quality = _quality_for(packet)
    if quality:
        labels.append(f"quality:{quality}")
    mode = _mode_for(packet)
    if mode:
        labels.append(f"mode:{mode}")
    return labels


def _session_for(packet: "SignalPacket") -> str:
    """Resolve the session label. Prefer an explicit ``session`` attribute
    if the packet carries one (future-proofing), else infer from
    ``signal_time`` via ``session_filter``."""
    explicit = getattr(packet, "session", None)
    if explicit:
        s = str(explicit).lower()
        if s.startswith("london"):
            return "London"
        if s.startswith("ny") or "new york" in s:
            return "NY"
        if s.startswith("asia"):
            return "Asia"

    ts = getattr(packet, "signal_time", None)
    if ts:
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            return "unknown"
        from prism.delivery.session_filter import Session, get_current_session
        s = get_current_session(dt)
        if s == Session.LONDON:
            return "London"
        if s == Session.NEW_YORK:
            return "NY"
        if s == Session.ASIAN:
            return "Asia"
    return "unknown"


def _quality_for(packet: "SignalPacket") -> Optional[str]:
    """Map confidence_level → A/B/C bucket. Falls back to numeric confidence."""
    level = (getattr(packet, "confidence_level", None) or "").upper()
    if level in {"HIGH", "A"}:
        return "A"
    if level in {"MEDIUM", "B"}:
        return "B"
    if level in {"LOW", "C"}:
        return "C"
    conf = getattr(packet, "confidence", None)
    if conf is None:
        return None
    if conf >= 0.75:
        return "A"
    if conf >= 0.55:
        return "B"
    return "C"


def _mode_for(packet: "SignalPacket") -> Optional[str]:
    """The packet itself doesn't carry mode — the bridge does. We infer
    from the runner-level env so the issue gets a useful label even when
    nobody plumbs the value through. Tests can override via attribute."""
    explicit = getattr(packet, "mode", None)
    if explicit:
        return str(explicit).lower()
    env = os.environ.get("PRISM_EXECUTION_MODE", "").strip().lower()
    if env in {"notify", "confirm", "auto"}:
        return env
    return None


def _classify_outcome(pnl: float, *, risk_usd: Optional[float]) -> str:
    if risk_usd and abs(pnl) < 0.1 * abs(risk_usd):
        return "breakeven"
    if abs(pnl) < 1.0:
        return "breakeven"
    return "win" if pnl > 0 else "loss"


def _fmt_price(p: float, instrument: str) -> str:
    """5 dp for FX, 2 dp for gold. Good enough for issue rendering."""
    if instrument and instrument.upper().startswith("XAU"):
        return f"{p:,.2f}"
    return f"{p:.5f}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# gh CLI / REST shim
# ---------------------------------------------------------------------------

def _repo() -> str:
    return os.environ.get("PRISM_GITHUB_REPO", _DEFAULT_REPO)


def _run_gh(args: list, *, input_text: Optional[str] = None) -> Optional[str]:
    """Invoke ``gh`` and return stdout. Returns ``None`` on non-zero exit
    or when gh is missing. Never raises."""
    if _GH_BIN is None:
        logger.warning("github_journal: gh CLI not available")
        return None

    env = os.environ.copy()
    # GH_TOKEN takes precedence; if unset, gh falls back to its keyring.
    cmd = [_GH_BIN, *args]
    try:
        result = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            env=env,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.error("github_journal: gh invocation failed: %s", exc)
        return None

    if result.returncode != 0:
        logger.error(
            "github_journal: gh %s exited %s — stderr=%s",
            " ".join(args), result.returncode, (result.stderr or "").strip(),
        )
        return None
    return result.stdout


def _gh_issue_create(*, title: str, body: str, labels: list) -> Optional[int]:
    """Create the issue and return its number. Uses ``--body-file -`` so the
    body can be arbitrarily long and contain shell metacharacters."""
    if _GH_BIN is None:
        return _rest_issue_create(title=title, body=body, labels=labels)

    args = [
        "issue", "create",
        "--repo", _repo(),
        "--title", title,
        "--body-file", "-",
    ]
    for label in labels:
        args.extend(["--label", label])

    out = _run_gh(args, input_text=body)
    if not out:
        return None
    # `gh issue create` prints the URL on stdout. Parse the trailing number.
    url = out.strip().splitlines()[-1].strip()
    try:
        return int(url.rsplit("/", 1)[-1])
    except (ValueError, IndexError):
        logger.error("github_journal: could not parse issue number from %r", url)
        return None


def _gh_issue_comment(number: int, body: str) -> bool:
    if _GH_BIN is None:
        return _rest_issue_comment(number, body)
    out = _run_gh([
        "issue", "comment", str(number),
        "--repo", _repo(),
        "--body-file", "-",
    ], input_text=body)
    return out is not None


def _gh_issue_add_label(number: int, label: str) -> bool:
    if _GH_BIN is None:
        return _rest_issue_edit(number, add_labels=[label])
    out = _run_gh([
        "issue", "edit", str(number),
        "--repo", _repo(),
        "--add-label", label,
    ])
    return out is not None


def _gh_issue_swap_label(number: int, *, remove: str, add: str) -> bool:
    if _GH_BIN is None:
        return _rest_issue_edit(number, add_labels=[add], remove_labels=[remove])
    # `gh issue edit` accepts both flags in a single call.
    out = _run_gh([
        "issue", "edit", str(number),
        "--repo", _repo(),
        "--remove-label", remove,
        "--add-label", add,
    ])
    return out is not None


def _gh_issue_close(number: int) -> bool:
    if _GH_BIN is None:
        return _rest_issue_close(number)
    out = _run_gh([
        "issue", "close", str(number),
        "--repo", _repo(),
    ])
    return out is not None


# ---------------------------------------------------------------------------
# REST fallback (no gh CLI available)
# ---------------------------------------------------------------------------

def _rest_headers() -> Optional[Dict[str, str]]:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.warning(
            "github_journal: no gh CLI and no GH_TOKEN — cannot reach GitHub"
        )
        return None
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "prism-journal/0.1",
    }


def _rest_post(path: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    headers = _rest_headers()
    if headers is None:
        return None
    try:
        import requests  # local import — runner shouldn't pay this cost on import
    except ImportError:
        logger.error("github_journal: REST fallback needs `requests` installed")
        return None
    url = f"https://api.github.com/repos/{_repo()}{path}"
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=20)
    except requests.RequestException as exc:
        logger.error("github_journal: REST POST %s failed: %s", path, exc)
        return None
    if r.status_code >= 300:
        logger.error("github_journal: REST POST %s → %s %s", path, r.status_code, r.text[:200])
        return None
    return r.json()


def _rest_patch(path: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    headers = _rest_headers()
    if headers is None:
        return None
    try:
        import requests
    except ImportError:
        logger.error("github_journal: REST fallback needs `requests` installed")
        return None
    url = f"https://api.github.com/repos/{_repo()}{path}"
    try:
        r = requests.patch(url, json=payload, headers=headers, timeout=20)
    except requests.RequestException as exc:
        logger.error("github_journal: REST PATCH %s failed: %s", path, exc)
        return None
    if r.status_code >= 300:
        logger.error("github_journal: REST PATCH %s → %s %s", path, r.status_code, r.text[:200])
        return None
    return r.json()


def _rest_issue_create(*, title: str, body: str, labels: list) -> Optional[int]:
    data = _rest_post("/issues", {"title": title, "body": body, "labels": labels})
    if data is None or "number" not in data:
        return None
    return int(data["number"])


def _rest_issue_comment(number: int, body: str) -> bool:
    return _rest_post(f"/issues/{number}/comments", {"body": body}) is not None


def _rest_issue_edit(
    number: int,
    *,
    add_labels: Optional[list] = None,
    remove_labels: Optional[list] = None,
) -> bool:
    """Compute the resulting label set and PATCH it. GitHub's REST API
    doesn't support add/remove deltas atomically, but read-modify-write
    against a single issue is fine for our throughput (one fired signal at
    a time)."""
    headers = _rest_headers()
    if headers is None:
        return False
    try:
        import requests
    except ImportError:
        return False
    url = f"https://api.github.com/repos/{_repo()}/issues/{number}"
    try:
        r = requests.get(url, headers=headers, timeout=20)
    except requests.RequestException as exc:
        logger.error("github_journal: REST GET %s failed: %s", url, exc)
        return False
    if r.status_code >= 300:
        logger.error("github_journal: REST GET %s → %s", url, r.status_code)
        return False
    issue = r.json()
    current = {lbl["name"] for lbl in issue.get("labels", [])}
    if add_labels:
        current.update(add_labels)
    if remove_labels:
        current.difference_update(remove_labels)
    data = _rest_patch(f"/issues/{number}", {"labels": sorted(current)})
    return data is not None


def _rest_issue_close(number: int) -> bool:
    return _rest_patch(f"/issues/{number}", {"state": "closed"}) is not None


# ---------------------------------------------------------------------------
# Test seam — let tests force the gh path on/off without monkey-patching
# shutil.which globally.
# ---------------------------------------------------------------------------

def _set_gh_bin_for_tests(path: Optional[str]) -> None:
    """Tests only. Override the resolved gh binary path."""
    global _GH_BIN
    _GH_BIN = path
