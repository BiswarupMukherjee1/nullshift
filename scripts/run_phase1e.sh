#!/usr/bin/env bash
# Phase 1b follow-ups, queued behind phase1d (GPU) and the MXR-LA-v1 build (network):
#   E05 real Maxar arena (all methods + shiftavg, shiftavg+geoP, radP, geoP+radP) -> E02 with radP (LEVIR energy).
# Launch with: scripts/launch.sh phase1e scripts/run_phase1e.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH=src; PY=nullshift-venv/bin/python
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
echo "[phase1e] waiting for phase1d + mxr $(date)"
while systemctl --user is-active -q ns-phase1d || systemctl --user is-active -q ns-mxr; do sleep 60; done
echo "[phase1e] start $(date)"
E03=experiments/E03_nullpair_dose/runs/20260928-1842_dinov3L_main
$PY experiments/E05_maxar_realpairs/run.py --e03-run $E03 --extra-methods shiftavg,shiftavg+geoP,radP,geoP+radP,blur+geoP,classic+geoP \
    || echo "[phase1e] E05 FAILED"
scripts/storage_check.sh | head -1
$PY experiments/E02_selectivity/run.py --eps 1.0 --n 192 --tag dinov3L_test192_radP || echo "[phase1e] E02-radP FAILED"
scripts/storage_check.sh
echo "[phase1e] done $(date)"
