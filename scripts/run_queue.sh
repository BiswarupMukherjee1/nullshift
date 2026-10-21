#!/usr/bin/env bash
# IDEMPOTENT experiment queue (Phase 1b). Safe to relaunch after any reboot/kill:
#   finished steps (metrics.json present) are skipped; checkpointed sweeps (E04, E03c) resume in place.
# Launch: loginctl enable-linger $USER; scripts/launch.sh queue scripts/run_queue.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH=src; PY=nullshift-venv/bin/python; EPS=1.0
E03=experiments/E03_nullpair_dose/runs/20260928-1842_dinov3L_main
CAL=experiments/E03_nullpair_dose/runs/20260928-0410_dinov3L_main
done_run() { ls -d $1 2>/dev/null | while read d; do [ -f "$d/metrics.json" ] && echo "$d"; done | tail -1; }
open_run() { ls -d $1 2>/dev/null | while read d; do [ -f "$d/metrics.json" ] || echo "$d"; done | tail -1; }
echo "[queue] start $(date)"

# 1. E05 real Maxar arena (~15 min)
if [ -z "$(done_run 'experiments/E05_maxar_realpairs/runs/*_dinov3L_mxrla')" ]; then
  $PY experiments/E05_maxar_realpairs/run.py --e03-run $E03 \
      --extra-methods shiftavg,shiftavg+geoP,radP,geoP+radP,blur+geoP,classic+geoP || echo "[queue] E05 FAILED"
fi
# 2. E04 perturbed real pairs (checkpointed)
if [ -z "$(done_run 'experiments/E04_perturbed_pair/runs/*_dinov3L_main')" ]; then
  R=$(open_run 'experiments/E04_perturbed_pair/runs/*_dinov3L_main')
  if [ -n "$R" ]; then $PY experiments/E04_perturbed_pair/run.py --eps $EPS --e03-run $E03 --run-dir "$R" --resume || echo "[queue] E04 FAILED"
  else $PY experiments/E04_perturbed_pair/run.py --eps $EPS --e03-run $E03 || echo "[queue] E04 FAILED"; fi
fi
# 3. E03b bootstrap CIs (~20 min)
[ -z "$(done_run 'experiments/E03b_bootstrap_ci/runs/*')" ] && { $PY experiments/E03b_bootstrap_ci/run.py --eps $EPS --e03-run $E03 || echo "[queue] E03b FAILED"; }
# 4. E02 with radP (energy selectivity)
[ -z "$(done_run 'experiments/E02_selectivity/runs/*_radP')" ] && { $PY experiments/E02_selectivity/run.py --eps $EPS --n 192 --tag dinov3L_test192_radP || echo "[queue] E02-radP FAILED"; }
# 5. E03c two-sided null pairs (checkpointed)
if [ -z "$(done_run 'experiments/E03_nullpair_dose/runs/*_twosided')" ]; then
  R=$(open_run 'experiments/E03_nullpair_dose/runs/*_twosided')
  ARGS="--eps $EPS --reuse-calib $CAL --two-sided --families G1_shift,G2_field,G3_parallax,C1_compound,R3_haze,R4_colour,R5_jpeg --doses 1,3,5 --n-null 256"
  if [ -n "$R" ]; then $PY experiments/E03_nullpair_dose/run.py $ARGS --run-dir "$R" --resume || echo "[queue] E03c FAILED"
  else $PY experiments/E03_nullpair_dose/run.py $ARGS --tag dinov3L_twosided || echo "[queue] E03c FAILED"; fi
fi
scripts/storage_check.sh
echo "[queue] done $(date)"
