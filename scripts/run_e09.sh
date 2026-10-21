#!/usr/bin/env bash
# E09: second real event (MXR-TX-v1), frozen design from LA. Waits for the TX build and E08. Idempotent.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH=src XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"; PY=nullshift-venv/bin/python
E03=experiments/E03_nullpair_dose/runs/20260928-1842_dinov3L_main
done_run() { ls -d $1 2>/dev/null | while read d; do [ -f "$d/metrics.json" ] && echo "$d"; done | tail -1; }
while systemctl --user is-active -q ns-mxrtx || systemctl --user is-active -q ns-e08; do sleep 20; done
echo "[e09] start $(date +%H:%M)"
M="raw,geoP,localmin,localmin+geoPms,c2f,c2f+radP,classic"
[ -z "$(done_run 'experiments/E09_second_event/runs/*_mxrtx_L')" ] && $PY experiments/E06_coarse_to_fine/run.py --experiment E09_second_event \
   --dataset MXR-TX-v1 --e03-run $E03 --methods $M --batch 8 --tag e09_mxrtx_L
echo "[e09] L done $(date +%H:%M)"
[ -z "$(done_run 'experiments/E09_second_event/runs/*_mxrtx_B')" ] && $PY experiments/E06_coarse_to_fine/run.py --experiment E09_second_event \
   --dataset MXR-TX-v1 --e03-run $E03 --encoder dinov3_vitb16_web --methods $M --batch 8 --tag e09_mxrtx_B
echo "[e09] done $(date +%H:%M)"
