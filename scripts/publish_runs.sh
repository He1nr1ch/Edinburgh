#!/usr/bin/env bash
#
# publish_runs.sh - Copy the desk's run results to Colossus for the positions page
# Usage: ./scripts/publish_runs.sh
#
# The list runs happen on the dev box; the page on Colossus reads decisions
# from the worker's data folder there. This copies the saved run states and
# the sized books across (additively: nothing on Colossus is deleted).
set -euo pipefail

COLOSSUS="heinrich@192.168.0.98"
SRC="${TRADINGAGENTS_RESULTS_DIR:-$HOME/.tradingagents/logs}/backtest/"
DEST="edinburgh/data/logs/backtest/"

ssh "$COLOSSUS" "mkdir -p $DEST"
rsync -az --info=stats1 \
    --include='*/' --include='full_states_log_*.json' --include='sizing/*.csv' --exclude='*' \
    "$SRC" "$COLOSSUS:$DEST"
echo "==> Published run states and books to $COLOSSUS:$DEST"
