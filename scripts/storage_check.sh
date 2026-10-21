#!/usr/bin/env bash
# Storage guard. User limits: target <= 8 GB, 10 GB HARD CAP (warn user immediately). Exit 2 if over budget.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SOFT_GB=8; HARD_GB=10
total_kb=$(du -sk "$ROOT" 2>/dev/null | cut -f1)
total_gb=$(awk "BEGIN{printf \"%.2f\", $total_kb/1048576}")
echo "[storage] project total: ${total_gb} GB (budget ${SOFT_GB} GB, hard cap ${HARD_GB} GB)"
du -sh "$ROOT"/* "$ROOT"/.tmp 2>/dev/null | sort -h | tail -8 | sed 's/^/[storage]   /'
if awk "BEGIN{exit !($total_kb > $HARD_GB*1048576)}"; then echo "[storage] !!! OVER HARD CAP ${HARD_GB} GB — STOP AND WARN USER"; exit 3; fi
if awk "BEGIN{exit !($total_kb > 9*1048576)}"; then echo "[storage] !!! above 9 GB — within 1 GB of HARD CAP, WARN USER"; exit 2; fi
if awk "BEGIN{exit !($total_kb > $SOFT_GB*1048576)}"; then echo "[storage] !   over target ${SOFT_GB} GB — keep outputs minimal"; exit 1; fi
exit 0
