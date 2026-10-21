#!/usr/bin/env bash
# Phase-1 chain: E02 -> E03 -> E04 (E00, E01, E01b already run). eps* = 1.0 from E01 (val).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH=src; PY=nullshift-venv/bin/python; EPS=1.0
echo "[phase1] start $(date)"
if [ "${SKIP_E02:-0}" != "1" ]; then $PY experiments/E02_selectivity/run.py --eps $EPS || echo "[phase1] E02 FAILED"; fi
scripts/storage_check.sh | head -1
$PY experiments/E03_nullpair_dose/run.py --eps $EPS || echo "[phase1] E03 FAILED"
scripts/storage_check.sh | head -1
E03=$(ls -d experiments/E03_nullpair_dose/runs/*_dinov3L_main | tail -1)
$PY experiments/E04_perturbed_pair/run.py --eps $EPS --e03-run "$E03" || echo "[phase1] E04 FAILED"
scripts/storage_check.sh
echo "[phase1] done $(date)"
