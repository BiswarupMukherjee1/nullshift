#!/usr/bin/env bash
# Queue 2 (idempotent): after queue 1 -> E07 cost ablation (ViT-L sat) -> E06 + E07 on 2nd backbone (DINOv3 ViT-B web).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH=src XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"; PY=nullshift-venv/bin/python
E03=experiments/E03_nullpair_dose/runs/20260928-1842_dinov3L_main
done_run() { ls -d $1 2>/dev/null | while read d; do [ -f "$d/metrics.json" ] && echo "$d"; done | tail -1; }
while systemctl --user is-active -q ns-queue; do sleep 30; done
echo "[queue2] start $(date)"
[ -z "$(done_run 'experiments/E07_cost_ablation/runs/*_dinov3_vitl16_sat_mxrla')" ] && { $PY experiments/E07_cost_ablation/run.py || echo "[queue2] E07 FAILED"; }
[ -z "$(done_run 'experiments/E06_coarse_to_fine/runs/*_vitb16web')" ] && { $PY experiments/E06_coarse_to_fine/run.py --e03-run $E03 --encoder dinov3_vitb16_web \
    --methods raw,geoP,geoPms,localmin,localmin1,localmin+geoP,localmin+geoPms,blur+geoP,classic --tag dinov3B_web_mxrla_c2f_vitb16web || echo "[queue2] E06-B FAILED"; }
[ -z "$(done_run 'experiments/E07_cost_ablation/runs/*_dinov3_vitb16_web_mxrla')" ] && { $PY experiments/E07_cost_ablation/run.py --encoder dinov3_vitb16_web || echo "[queue2] E07-B FAILED"; }
scripts/storage_check.sh
echo "[queue2] done $(date)"
