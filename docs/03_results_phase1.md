# 03 — Phase-1 results and interpretation

Setting: frozen DINOv3 ViT-L/16 SAT-493M (patch tokens, final block, L2-normalised; 16×16 tokens per 256² tile),
LEVIR-CD val (calibration/selection) and test (evaluation), NSB-v1 null pairs. No training anywhere.
Per-experiment details: `experiments/E*/README.md`; machine-generated tables: `results/PHASE1_TABLES.md`.

## 1. Hypothesis scorecard

| # | Hypothesis (proposal) | Verdict | Evidence |
|---|---|---|---|
| H1 | first-order J·d explains the feature displacement for ≤ 2 px | **FAILED** (kill met) | rel. error 0.805 at 1 px (kill 0.6); response grows as √d; block 12 worse (1.006) — E01 |
| H1-fallback | multi-shift per-patch basis captures misregistration | **PASS** | probe subspace r8: 89 % of held-out 1-px energy removed, 7 % of true change (val) — E01b |
| H2 | per-patch subspace removes misregistration energy but not true change | **PASS (geoP)** | test: 88.9 % vs 6.9 % (12.9×); tangent geo 37 % vs 0.7 % — E02 |
| H6 | global subspace + NFAD gate fails for misregistration | **PASS, with a different mechanism** | global shift subspace removes more true change (35 %) than misregistration (23 %) — entanglement; the gate only backs off for parallax (φ 0.22, γ 0.44), not for shifts/fields (φ ≈ 0.65, γ ≈ 0.99) — E02 |
| H3 | FCR falls on shifted null pairs, clean F1 drop ≤ 1 pt | see §3 | clean: geoP **+0.6 pt** patch-F1, +0.35 pt AUROC (no cost) — E03 |
| H5 | NFAD global subspace transfers to RS radiometry | see §3 | energy level: **entangled** (removes 30 % of true change vs 29–49 % of radiometric nuisance) — E02 |
| H4 | real parallax (S2Looking) | not run (Phase 2) | — |

## 2. What the frozen satellite foundation model is sensitive to (E00, E01, fit report)
- Mean-square feature displacement of val interventions (fit doses): gain 0.004 ≪ gamma 0.022 ≪ shift 0.5–2 px 0.064 <
  blur 0.144 < haze 0.165 < JPEG 0.180 < noise 0.303. DINOv3-sat is essentially **gain-invariant** and most sensitive to
  **noise, JPEG and blur** (texture-destroying nuisances), then to misregistration.
- The response to sub-pixel translation is **rough (≈ √d)**, not linear: ‖Δz‖ = 0.089 / 0.156 / 0.198 / 0.273 / 0.408 at
  0.25 / 0.5 / 1 / 2 / 4 px. A quarter-pixel shift already moves a token by 45 % of what a full pixel does.
- Real LEVIR residuals are large: unchanged ≈ 1.02, changed ≈ 1.26 (median cosine-residual norm) — multi-year Google
  Earth pairs differ a lot even where nothing was built. A 1-px misregistration (≈ 0.2) is ~5× below that.

## 3. Clean accuracy (E03, 2048 test tiles, thresholds from val)

| method | patch-F1 | patch-AUROC | pixel-F1 | FPR on real unchanged patches |
|---|---|---|---|---|
| raw | 0.409 | 0.873 | 0.315 | 0.126 |
| geo (tangent) | 0.412 | 0.874 | 0.317 | 0.121 |
| **geoP** | **0.415** | **0.877** | **0.320** | 0.119 |
| rad | 0.390 | 0.870 | 0.295 | 0.087 |
| geoP+rad | 0.397 | 0.875 | 0.302 | 0.095 |
| shiftV (global) | 0.254 | 0.769 | 0.189 | 0.206 |
| classic (phasecorr + RRN) | 0.412 | 0.875 | 0.318 | 0.117 |
| **classic+ns** | 0.401 | **0.877** | 0.309 | 0.090 |

geoP is a free lunch on clean data (+0.6 pt F1); removing the radiometric subspace lowers F1 at the val threshold
(recall 0.67 → 0.51) although AUROC is unchanged — the threshold transfers less well once 56 directions are removed.

## 4. Null-pair dose-response (E03, LEVIR test512, 11 families × 5 doses)
*What was measured:* for each method, how large the change score is on pairs where nothing changed (null pairs),
relative to its score on real changes (NCR), and how many null patches cross a threshold that catches 99 % of real
changes on clean val (FCR_R99).

| mean over doses | raw | geo | **geoP** | geoP+rad | phasecorr | blur | classic | **classic+ns** |
|---|---|---|---|---|---|---|---|---|
| NCR geometric (G1–G3) | 0.140 | 0.125 | 0.086 | 0.094 | 0.100 | 0.095 | 0.101 | **0.079** |
| NCR radiometric (R1–R7) | 0.226 | 0.220 | 0.202 | 0.188 | 0.226 | 0.196 | 0.178 | **0.151** |
| FCR_R99 geometric | 0.00093 | 0.00046 | **0.00017** | 0.00023 | 0.00084 | 0.00080 | 0.00099 | 0.00026 |
| FCR_R99 radiometric | 0.037 | 0.037 | 0.036 | 0.015 | 0.037 | 0.027 | 0.026 | **0.009** |

How to read it:
- **geoP is the best single operator for geometric nuisance at a high-recall threshold** (5.4× fewer false changes than
  raw) and reduces the nuisance ratio by 39 %, at no clean cost.
- **Per family, the simple specialist wins its own case:** phase correlation on a pure global shift (G1: 0.056 vs geoP
  0.079), Gaussian blur on smooth fields and parallax (G2: 0.128 vs 0.141; G3: 0.032 vs 0.039), pixel normalisation on
  global gain/gamma/colour. NullShift's value is in the **combination**: `classic+ns` is best on average and on the
  compound nuisance C1 (0.230 vs raw 0.312).
- **Radiometry:** the NFAD-style global subspace helps only where pixel normalisation cannot (noise: FCR_R99 0.144 →
  0.045). For global radiometry it is weaker than histogram matching / PIF normalisation.
- Absolute FCRs are tiny and AUROCs saturated (≥ 0.98): on LEVIR the synthetic nuisance never competes with 5–14-year
  real change. This is the dataset's regime — hence the real Maxar arena (E05).

## 5. Perturbed real pairs (E04) and two-sided null pairs (E03c)
- E04 (LEVIR, nuisance on t2 of real pairs): geometric nuisance barely affects real-change detection on LEVIR (≤ 1 pt);
  radiometric nuisance does — the radiometric subspace protects under noise (0.740 → 0.831); token search is the most
  fragile under strong noise (0.692).
- E03c (operator built from an independently perturbed copy): geoP keeps −32 % geometric NCR (vs −39 % one-sided) →
  the synthetic gain is not mainly "home turf".

## 6. Real arena, Los Angeles (E05, E06, E07, E08) — MXR-LA-v1, 578 pairs
- Real near-null residuals are ~5× larger relative to real change than the synthetic doses (raw NCR 0.73 vs 0.14).
- Measured misregistration between real acquisitions: median 3.3 px (pre-fire pairs) to 8–9 px (post-fire / long),
  up to 22 px at 0.5 m, spatially varying (only ~10 % of pairs have a confident global phase-correlation peak).
- Projection alone (geoP) helps modestly (+1.5 pt AUROC); plain 3×3 token search helps more (+5.5 pt); the
  **coarse-to-fine** combination is best on both backbones and significantly better than search alone
  (ViT-L +0.9 pt, ViT-B +1.1 pt AUROC; NCR −0.08). 16 probes suffice (E07). Per-patch radiometric probes add a little
  on real pairs; the global NFAD subspace does not (E08).

## 7. Second real event, Texas (E09) — MXR-TX-v1, 298 pairs, pristine (design frozen before the data existed)
- Nuisance ratio reduced significantly in all 4 (backbone × negatives) settings; AUROC gain over the strongest baseline
  positive in all 4, significant in 1 (pre-registered criterion formally met).
- Measured misregistration is small and mostly global (median 1.5 px same-day pairs, 3.6 px long-gap) → search adds
  little, per-patch projection carries the gain.

## 8. Decision table for practitioners (derived from E03–E09)
| situation (measure first: phase-correlation peak + shift on a few near-null pairs) | use | avoid |
|---|---|---|
| pairs already well co-registered (< 1 px), multi-year change (LEVIR-like) | geoP+rad (no search) | token search (−0.8 pt clean AUROC) |
| small, mostly global residual misregistration (1–4 px; TX-like) | geoP or c2f; bounded phase correlation first | global subspaces |
| multi-pixel, spatially varying misregistration (3–20 px; LA-like, off-nadir mix) | **c2f (16 probes, rank 8, 3×3; 5×5 if > 16 px)** (+radP) | projection-only, global subspaces |
| strong sensor noise / JPEG differences between sensors | add the global radiometric subspace (c2f+rad / geoP+rad) | token search alone |
| global gain / gamma / colour differences | pixel-space PIF normalisation first (classic), then NullShift | feature-space global subspace alone |

## 9. Design decisions taken during the project (and the evidence behind each)
| decision | reason | evidence |
|---|---|---|
| tangent → probe subspace | features not linear in shift (√d) | E01 kill criterion, E01b |
| rank 8 from 24 probes (synthetic), 16 probes / rank 8 + 3×3 search (real) | knee of removal vs true-change loss; cheapest within 0.5 pt on dev tiles | E01b, E07 |
| add token search (coarse-to-fine) | real misregistration measured at 3–22 px | E05 measurement, E06 |
| per-patch radiometry (radP) instead of global (rad) | global subspace entangled with real change | E02-radP, E05, E08 |
| per-method thresholds on LEVIR val; NCR / AUROC as primary on real data | projection shrinks scores; thresholds do not transfer across sensors | E03, E05 (FCR at LEVIR τ) |
| dev/heldout split by quadkey hash; pre-registered second event | limit post-hoc design choices | E06, E07, E09 |
| fair baselines (bounded + gated phase correlation, PIF mean/std RRN) | first versions were strawmen on real data | E03 calibration diagnostics |
| real arenas from Maxar Open Data + NIFC perimeters | no manual labels; different sensors; answers "LEVIR hides nuisance" | MXR-LA-v1, MXR-TX-v1 |
