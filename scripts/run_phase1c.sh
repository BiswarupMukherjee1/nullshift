#!/usr/bin/env bash
# Resume Phase 1 after the session loss: E03 (reusing val calibration + clean test of the interrupted run) -> E04 -> E03b.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH=src; PY=nullshift-venv/bin/python; EPS=1.0
CAL=experiments/E03_nullpair_dose/runs/20260928-0410_dinov3L_main
echo "[phase1c] start $(date)"
$PY experiments/E03_nullpair_dose/run.py --eps $EPS --reuse-calib $CAL || echo "[phase1c] E03 FAILED"
scripts/storage_check.sh | head -1
E03=$(ls -d experiments/E03_nullpair_dose/runs/*_dinov3L_main | tail -1)
echo "[phase1c] E03 run = $E03"
$PY experiments/E04_perturbed_pair/run.py --eps $EPS --e03-run "$E03" || echo "[phase1c] E04 FAILED"
scripts/storage_check.sh | head -1
$PY experiments/E03b_bootstrap_ci/run.py --eps $EPS --e03-run "$E03" || echo "[phase1c] E03b FAILED"
scripts/storage_check.sh
echo "[phase1c] done $(date)"
