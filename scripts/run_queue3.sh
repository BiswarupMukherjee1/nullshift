#!/usr/bin/env bash
# Single prioritised, idempotent queue (supersedes run_queue.sh / run_queue2.sh). The lab machine is POWERED OFF
# after logout (orderly shutdowns observed 12:29, 15:44, 19:35 on 29 Sep) -> jobs only progress while it is up;
# relaunch after every boot:  loginctl enable-linger $USER; scripts/launch.sh queue3 scripts/run_queue3.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH=src; PY=nullshift-venv/bin/python; EPS=1.0
E03=experiments/E03_nullpair_dose/runs/20260928-1842_dinov3L_main
CAL=experiments/E03_nullpair_dose/runs/20260928-0410_dinov3L_main
done_run() { ls -d $1 2>/dev/null | while read d; do [ -f "$d/metrics.json" ] && echo "$d"; done | tail -1; }
open_run() { ls -d $1 2>/dev/null | while read d; do [ -f "$d/metrics.json" ] || echo "$d"; done | tail -1; }
step() { echo "[queue3] $1 $(date +%H:%M)"; }
step start
# 1. E07 cost/design ablation, ViT-L sat (~25 min)
[ -z "$(done_run 'experiments/E07_cost_ablation/runs/*_dinov3_vitl16_sat_mxrla')" ] && { step E07; $PY experiments/E07_cost_ablation/run.py || step "E07 FAILED"; }
# 2. E06 on 2nd backbone (~7 min)
[ -z "$(done_run 'experiments/E06_coarse_to_fine/runs/*_vitb16web')" ] && { step E06-B; $PY experiments/E06_coarse_to_fine/run.py --e03-run $E03 --encoder dinov3_vitb16_web \
    --methods raw,geoP,geoPms,localmin,localmin1,localmin+geoP,localmin+geoPms,blur+geoP,classic --tag dinov3B_web_mxrla_c2f_vitb16web || step "E06-B FAILED"; }
# 3. E07 on 2nd backbone (~15 min)
[ -z "$(done_run 'experiments/E07_cost_ablation/runs/*_dinov3_vitb16_web_mxrla')" ] && { step E07-B; $PY experiments/E07_cost_ablation/run.py --encoder dinov3_vitb16_web || step "E07-B FAILED"; }
# 4. E03b bootstrap CIs (checkpointed per batch)
if [ -z "$(done_run 'experiments/E03b_bootstrap_ci/runs/*')" ]; then
  R=$(open_run 'experiments/E03b_bootstrap_ci/runs/*'); step E03b
  if [ -n "$R" ]; then $PY experiments/E03b_bootstrap_ci/run.py --eps $EPS --e03-run $E03 --run-dir "$R" || step "E03b FAILED"
  else $PY experiments/E03b_bootstrap_ci/run.py --eps $EPS --e03-run $E03 || step "E03b FAILED"; fi
fi
# 5. E02 with radP (~25 min)
[ -z "$(done_run 'experiments/E02_selectivity/runs/*_radP')" ] && { step E02-radP; $PY experiments/E02_selectivity/run.py --eps $EPS --n 192 --tag dinov3L_test192_radP || step "E02-radP FAILED"; }
# 6. E03c two-sided null pairs (checkpointed, ~60 min)
if [ -z "$(done_run 'experiments/E03_nullpair_dose/runs/*_twosided')" ]; then
  R=$(open_run 'experiments/E03_nullpair_dose/runs/*_twosided'); step E03c
  ARGS="--eps $EPS --reuse-calib $CAL --two-sided --families G1_shift,G2_field,G3_parallax,C1_compound,R3_haze,R4_colour,R5_jpeg --doses 1,3,5 --n-null 256"
  if [ -n "$R" ]; then $PY experiments/E03_nullpair_dose/run.py $ARGS --run-dir "$R" --resume || step "E03c FAILED"
  else $PY experiments/E03_nullpair_dose/run.py $ARGS --tag dinov3L_twosided || step "E03c FAILED"; fi
fi
scripts/storage_check.sh
step done
