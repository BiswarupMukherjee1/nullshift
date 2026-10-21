#!/usr/bin/env bash
# Phase-1 completion (after two session-kills): resume E03 in place from its checkpoint -> E04 -> E03b -> E03c (two-sided).
# Launch with: scripts/launch.sh phase1d scripts/run_phase1d.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH=src; PY=nullshift-venv/bin/python; EPS=1.0
E03=experiments/E03_nullpair_dose/runs/20260928-1842_dinov3L_main
CAL=experiments/E03_nullpair_dose/runs/20260928-0410_dinov3L_main
echo "[phase1d] start $(date)"
if [ ! -f "$E03/metrics.json" ]; then
  $PY experiments/E03_nullpair_dose/run.py --eps $EPS --reuse-calib $E03 --run-dir $E03 --resume || echo "[phase1d] E03 FAILED"
fi
scripts/storage_check.sh | head -1
[ -f "$E03/metrics.json" ] && { $PY experiments/E04_perturbed_pair/run.py --eps $EPS --e03-run "$E03" || echo "[phase1d] E04 FAILED"; }
scripts/storage_check.sh | head -1
[ -f "$E03/metrics.json" ] && { $PY experiments/E03b_bootstrap_ci/run.py --eps $EPS --e03-run "$E03" || echo "[phase1d] E03b FAILED"; }
$PY experiments/E03_nullpair_dose/run.py --eps $EPS --reuse-calib $CAL --two-sided --tag dinov3L_twosided \
    --families G1_shift,G2_field,G3_parallax,C1_compound,R3_haze,R4_colour,R5_jpeg --doses 1,3,5 --n-null 256 \
    || echo "[phase1d] E03c two-sided FAILED"
scripts/storage_check.sh
echo "[phase1d] done $(date)"
