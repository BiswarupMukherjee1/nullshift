#!/usr/bin/env bash
# Reproduce every experiment in order (one GPU with >= 20 GB; ~10-12 h in total).
# Each step writes experiments/<id>/runs/<timestamp>_<tag>/; later steps read the E03 run for calibration/thresholds.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=${PY:-python}
EPS=1.0                                   # finite-difference step selected by E01 (px)

# ---- data and encoders
scripts/download_levir.sh                 # LEVIR-CD val + test -> data/raw/LEVIR-CD
$PY scripts/fetch_encoder.py dinov3_vitl16_sat
$PY scripts/fetch_encoder.py dinov3_vitb16_web
$PY scripts/create_nsb.py                 # NSB-v1 null-pair recipes (seeded; no images stored)
$PY scripts/fit_subspaces.py              # global nuisance subspaces fitted on LEVIR val
mkdir -p data/raw/maxar
cp Data_Created/MXR-LA-v1/fire_perimeters.geojson data/raw/maxar/la_fire_perimeters_2025.geojson
cp Data_Created/MXR-TX-v1/fire_perimeters.geojson data/raw/maxar/tx_fire_perimeters_2024.geojson
$PY scripts/scan_maxar_event.py WildFires-LosAngeles-Jan-2025
$PY scripts/scan_maxar_event.py SmokeHouseCreek-Wildfires-Texas-Mar24
$PY scripts/create_maxar_la.py            # MXR-LA-v1
$PY scripts/create_maxar_event.py --event SmokeHouseCreek-Wildfires-Texas-Mar24 --out MXR-TX-v1 \
    --perimeters tx_fire_perimeters_2024.geojson --epsg 32614 --pre-end 2024-02-25 --post-start 2024-02-29 \
    --change-pre-min 2023-06-01 --change-pre-end 2023-12-31 --caps 100,100,60,120 --seed 20260930

# ---- synthetic null pairs (LEVIR-CD)
$PY experiments/E00_sanity/run.py
$PY experiments/E01_linearity/run.py
$PY experiments/E01b_probe_subspace/run.py
$PY experiments/E02_selectivity/run.py --eps $EPS
$PY experiments/E02_selectivity/run.py --eps $EPS --n 192 --tag dinov3L_test192_radP
$PY experiments/E03_nullpair_dose/run.py --eps $EPS
E03=$(ls -d experiments/E03_nullpair_dose/runs/*_dinov3L_main | tail -1)
$PY experiments/E03_nullpair_dose/run.py --eps $EPS --reuse-calib "$E03" --two-sided --tag dinov3L_twosided \
    --families G1_shift,G2_field,G3_parallax,C1_compound,R3_haze,R4_colour,R5_jpeg --doses 1,3,5 --n-null 256
$PY experiments/E03b_bootstrap_ci/run.py --eps $EPS --e03-run "$E03"
$PY experiments/E04_perturbed_pair/run.py --eps $EPS --e03-run "$E03"

# ---- real acquisition pairs (Maxar Open Data)
$PY experiments/E05_maxar_realpairs/run.py --e03-run "$E03" \
    --extra-methods shiftavg,shiftavg+geoP,radP,geoP+radP,blur+geoP,classic+geoP
$PY experiments/E06_coarse_to_fine/run.py --e03-run "$E03"
$PY experiments/E06_coarse_to_fine/run.py --e03-run "$E03" --encoder dinov3_vitb16_web \
    --methods raw,geoP,geoPms,localmin,localmin1,localmin+geoP,localmin+geoPms,blur+geoP,classic --tag dinov3B_web_mxrla_c2f_vitb16web
$PY experiments/E07_cost_ablation/run.py
$PY experiments/E07_cost_ablation/run.py --encoder dinov3_vitb16_web
$PY experiments/E06_coarse_to_fine/run.py --experiment E08_c2f_radiometric --e03-run "$E03" \
    --methods raw,localmin,localmin+geoPms,c2f,c2f+radP,c2f+rad,geoP+radP --tag e08_mxrla_L
$PY experiments/E06_coarse_to_fine/run.py --experiment E08_c2f_radiometric --e03-run "$E03" --encoder dinov3_vitb16_web \
    --methods raw,localmin,localmin+geoPms,c2f,c2f+radP,geoP+radP --tag e08_mxrla_B
$PY experiments/E04_perturbed_pair/run.py --experiment E08_c2f_radiometric --eps $EPS --e03-run "$E03" \
    --methods raw,localmin,localmin+geoPms,c2f,c2f+radP,c2f+rad,geoP+rad \
    --families G1_shift,G2_field,G3_parallax,C1_compound,R3_haze,R5_jpeg,R6_noise --doses 3,5 --n 256 --batch 8 --tag e08_levir_perturbed
M="raw,geoP,localmin,localmin+geoPms,c2f,c2f+radP,classic"
$PY experiments/E06_coarse_to_fine/run.py --experiment E09_second_event --dataset MXR-TX-v1 --e03-run "$E03" --methods $M --batch 8 --tag e09_mxrtx_L
$PY experiments/E06_coarse_to_fine/run.py --experiment E09_second_event --dataset MXR-TX-v1 --e03-run "$E03" --encoder dinov3_vitb16_web \
    --methods $M --batch 8 --tag e09_mxrtx_B
# paired quadkey-cluster bootstraps (B = 400) on the real-pair runs
P6="localmin+geoPms:localmin,localmin+geoPms:localmin1,localmin+geoP:localmin1,geoPms:geoP,localmin+geoPms:raw"
for r in $(ls -d experiments/E06_coarse_to_fine/runs/*_mxrla_c2f*); do $PY scripts/paired_bootstrap.py "$r" MXR-LA-v1 "$P6"; done
P9="c2f:localmin,c2f+radP:localmin,localmin+geoPms:localmin,c2f:raw,localmin:raw,geoP:raw"
for t in e09_mxrtx_L e09_mxrtx_B; do
  r=$(ls -d experiments/E09_second_event/runs/*_$t | tail -1)
  $PY scripts/paired_bootstrap.py "$r" MXR-TX-v1 "$P9"
  $PY scripts/paired_bootstrap.py "$r" MXR-TX-v1 "$P9" 400 null_long
done

# ---- tables and figures
$PY scripts/summarize_null_pairs.py
$PY scripts/collect_results.py
$PY scripts/make_figures.py
