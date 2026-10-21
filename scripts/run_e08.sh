#!/usr/bin/env bash
# E08: radiometric-robust coarse-to-fine (c2f = 3x3 token search + 16 shift probes r8; +radP / +rad). Idempotent.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
export PYTHONPATH=src; PY=nullshift-venv/bin/python
E03=experiments/E03_nullpair_dose/runs/20260928-1842_dinov3L_main
done_run() { ls -d $1 2>/dev/null | while read d; do [ -f "$d/metrics.json" ] && echo "$d"; done | tail -1; }
echo "[e08] start $(date +%H:%M)"
[ -z "$(done_run 'experiments/E08_c2f_radiometric/runs/*_mxrla_L')" ] && $PY experiments/E06_coarse_to_fine/run.py --experiment E08_c2f_radiometric \
   --e03-run $E03 --methods raw,localmin,localmin+geoPms,c2f,c2f+radP,c2f+rad,geoP+radP --tag e08_mxrla_L
echo "[e08] a-L done $(date +%H:%M)"
[ -z "$(done_run 'experiments/E08_c2f_radiometric/runs/*_mxrla_B')" ] && $PY experiments/E06_coarse_to_fine/run.py --experiment E08_c2f_radiometric \
   --e03-run $E03 --encoder dinov3_vitb16_web --methods raw,localmin,localmin+geoPms,c2f,c2f+radP,geoP+radP --tag e08_mxrla_B
echo "[e08] a-B done $(date +%H:%M)"
[ -z "$(done_run 'experiments/E08_c2f_radiometric/runs/*_levir_perturbed')" ] && $PY experiments/E04_perturbed_pair/run.py --experiment E08_c2f_radiometric \
   --eps 1.0 --e03-run $E03 --methods raw,localmin,localmin+geoPms,c2f,c2f+radP,c2f+rad,geoP+rad \
   --families G1_shift,G2_field,G3_parallax,C1_compound,R3_haze,R5_jpeg,R6_noise --doses 3,5 --n 256 --batch 8 --tag e08_levir_perturbed
echo "[e08] done $(date +%H:%M)"
