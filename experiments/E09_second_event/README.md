# E09 — Second real event: SmokeHouse Creek wildfires, Texas panhandle (Feb 2024) — pristine held-out test
**Why.** All real-data evidence so far comes from one event (LA, Jan 2025). A second event with different landscape
(grassland + small towns), season, sensors (mostly WorldView-2 / GeoEye-1) and acquisition geometry — including a
same-day pair taken 25 s apart at different off-nadir angles — tests whether the coarse-to-fine result generalises.
**Data.** `Data_Created/MXR-TX-v1`, built with the *identical* protocol as MXR-LA-v1 (`scripts/create_maxar_event.py`):
null_pre (≤ 14 d apart, before 26 Feb 2024), null_post (1 vs 4 Mar 2024, outside perimeters + 300 m), null_long and
change (Jun–Dec 2023 vs Mar 2024, outside / inside NIFC perimeters Smokehouse Creek, Windy Deuce, Grape Vine Creek).
**Frozen design (decided on LA before this dataset existed):** c2f = 3×3 token search + 16 shift probes (1, 4 px),
rank 8 (E07 recommendation); also localmin+geoPms (E06), localmin, geoP, raw, classic; both backbones.
**Pre-registered success criterion (written 29 Sep 22:55, before building MXR-TX-v1):** on MXR-TX-v1, c2f beats
localmin on patch AUROC *and* NCR with paired quadkey-bootstrap 95 % CIs excluding 0, on at least one backbone, and does
not lose on the other.

## Findings (runs `*_e09_mxrtx_L`, `*_e09_mxrtx_B`; 298 pairs, 120 change from 33 quadkeys) — pre-registered criterion met formally, replication partial
Patch AUROC, change vs all near-null pairs (vs `null_long`, the fair same-gap comparison, in brackets):

| method | ViT-L sat | ViT-B web |
|---|---|---|
| raw | 0.918 (0.791) | 0.927 (0.806) |
| geoP (±2 px probes, no search) | 0.926 (0.804) | **0.935 (0.826)** |
| localmin (3×3 search) | 0.932 (0.817) | 0.929 (0.810) |
| localmin+geoPms (40 probes) | **0.936** (0.824) | 0.935 (0.823) |
| **c2f** (16 probes, E07 setting) | 0.935 (0.824) | 0.934 (0.819) |
| c2f+radP | 0.936 (0.825) | 0.935 (0.821) |

Paired quadkey bootstrap, **c2f − localmin**: ViT-L all-null ΔAUROC **+0.003 [+0.001, +0.007]**, ΔNCR **−0.073 [−0.079, −0.057]**;
null_long ΔAUROC +0.007 [−0.001, +0.015], ΔNCR **−0.028 [−0.036, −0.019]**; ViT-B all-null +0.005 [−0.001, +0.010],
ΔNCR **−0.065**; null_long +0.009 [−0.005, +0.024], ΔNCR **−0.052**. c2f − raw: ViT-L +0.018 [+0.010, +0.028];
ViT-B +0.006 [+0.001, +0.012].
- **Pre-registered criterion:** met (ViT-L, all nulls: both CIs exclude 0; ViT-B does not lose).
- **Robust replication:** the nuisance-ratio reduction (NCR) is significant in all 4 settings; the AUROC gain over the
  strongest baseline is positive in all 4 but significant only in 1 → effect on ranking is small on this event.
- **Why the pattern differs from LA — measured:** phase-correlation displacement on TX near-null pairs is small and
  mostly global (null_pre: 90 % confident peaks, median 1.5 px; null_long median 3.6 px) vs LA (≤ 13 % confident,
  median 3.3–8.8 px, up to 22 px). With small displacement, token search adds little (ViT-B localmin ≈ raw) and the
  per-patch projection carries the gain (geoP best on ViT-B). **The component that helps tracks the measured
  displacement scale** — consistent with the mechanism, and a useful rule for practitioners.
- Absolute AUROC vs `null_long` is only ~0.82: autumn→winter rangeland changes are hard to distinguish from burned grass
  for every method (season is a nuisance none of the operators model).
Measurements: `misregistration_TX_phasecorr.csv` (this folder), `../E06_coarse_to_fine/misregistration_LA_phasecorr.csv`.

