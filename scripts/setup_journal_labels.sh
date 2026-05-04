#!/usr/bin/env bash
# scripts/setup_journal_labels.sh
#
# Bootstrap the GitHub label vocabulary used by the PRISM trade journal
# (Task 0.4). Idempotent — uses `gh label create --force` so re-running
# is safe and just refreshes color/description on existing labels.
#
# Requires: `gh` CLI authenticated against this repo (or GH_TOKEN exported).

set -euo pipefail

if ! command -v gh >/dev/null 2>&1; then
  echo "error: gh CLI not found on PATH" >&2
  exit 1
fi

create() {
  local name="$1"
  local color="$2"
  local desc="$3"
  echo "→ ${name}"
  gh label create "${name}" --color "${color}" --description "${desc}" --force >/dev/null
}

# --- Instruments ------------------------------------------------------------
create "inst:XAUUSD" "FFD700" "Gold (XAU/USD)"
create "inst:EURUSD" "4169E1" "Euro / US Dollar"
create "inst:GBPUSD" "228B22" "British Pound / US Dollar"

# --- Direction --------------------------------------------------------------
create "dir:LONG"  "00FF00" "Long position"
create "dir:SHORT" "FF4500" "Short position"

# --- Session ----------------------------------------------------------------
create "session:london" "1E90FF" "London kill zone"
create "session:ny"     "4682B4" "New York kill zone"
create "session:asia"   "87CEEB" "Asian session"

# --- Lifecycle phase --------------------------------------------------------
create "phase:pending"  "C0C0C0" "Signal fired, no fill yet"
create "phase:open"     "F4A300" "Position open in MT5"
create "phase:tp1"      "9ACD32" "TP1 hit, partial taken"
create "phase:closed"   "808080" "Position fully closed"
create "phase:reviewed" "6A5ACD" "Post-trade review complete"

# --- Outcome ----------------------------------------------------------------
create "outcome:win"        "2ECC71" "Net positive PnL"
create "outcome:loss"       "E74C3C" "Net negative PnL"
create "outcome:breakeven"  "95A5A6" "Flat PnL (within 0.1R)"
create "outcome:cancelled"  "BDC3C7" "Cancelled before fill"

# --- Mode -------------------------------------------------------------------
create "mode:notify"  "ECECEC" "NOTIFY mode (no execution)"
create "mode:confirm" "F1C40F" "CONFIRM mode (operator approval)"
create "mode:auto"    "8E44AD" "AUTO mode (auto-execute)"

# --- Quality ----------------------------------------------------------------
create "quality:A" "27AE60" "Textbook setup"
create "quality:B" "F39C12" "Workable setup"
create "quality:C" "C0392B" "Marginal setup"

echo "✓ all labels synced"
