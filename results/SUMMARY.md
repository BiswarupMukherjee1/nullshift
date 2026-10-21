# NullShift — results summary (consolidated 30 Sep 2026, 00:00)

Plain-language background: `docs/00_GUIDE.md`. Detailed results & design decisions: `docs/03_results_phase1.md`.
Per-experiment details: `experiments/E*/README.md`. Company relevance: `docs/06_industry_relevance.md`.
No training anywhere; encoders frozen (DINOv3 ViT-L/16 SAT-493M = "ViT-L sat"; DINOv3 ViT-B/16 LVD-1689M = "ViT-B web").

## 1. The answer in five lines
1. The proposal's first-order (tangent) operator **fails**: satellite-FM tokens respond to sub-pixel shift like √d (E01).
2. A **per-patch subspace of the reference's own responses to small shifts** (geoP) works; **one global subspace (NFAD)
   does not** — it removes real change with the nuisance. Shown on synthetic energy (E02), synthetic null pairs
   (E03, E03c) and real pairs (E05, E08); also holds for radiometry (radP vs rad, E02-radP, E08).
3. **Real** misregistration between Maxar acquisitions is **multi-pixel** (LA median 3–9 px, up to 22 px) → the
   **coarse-to-fine** operator (3×3 token search + per-patch probe subspace) is best on real pairs, on **two backbones**,
   and a cheap setting (16 probes) keeps the gain (E06, E07, E08).
4. On a **second, pristine event** (Texas) the nuisance reduction replicates everywhere, the ranking gain over the
   strongest baseline only partly — because TX misregistration is small (1.5–3.6 px), where projection alone carries the
   gain (E09). **Which component helps tracks the measured displacement scale.**
5. **No single operator dominates:** token search costs accuracy on already co-registered pairs and is fragile to
   strong sensor noise; the global radiometric subspace fixes noise on synthetic data but not on real pairs (E04, E08).

## 2. Headline numbers
| setting | raw | best simple baseline | NullShift (best / recommended) | significance |
|---|---|---|---|---|
| Synthetic null pairs, LEVIR (NCR, geometric; lower better) | 0.140 | phasecorr 0.100 / blur 0.095 | geoP 0.086; classic+ns **0.079** | E03b scene bootstrap: sig. for fields, compound, noise |
| Two-sided null pairs (NCR geometric) | 0.151 | blur 0.116 | geoP **0.102** | − |
| **Real LA pairs, ViT-L** (patch AUROC, heldout tiles) | 0.835 | localmin 0.900 | localmin+geoPms **0.908**; c2f 0.904 | vs localmin +0.009 [+0.004, +0.014] |
| **Real LA pairs, ViT-B** (heldout) | 0.930 | localmin 0.959 | localmin+geoPms **0.967**; c2f 0.965 | vs localmin +0.011 [+0.006, +0.017] |
| **Real TX pairs, ViT-L** (all nulls) | 0.918 | localmin 0.932 | c2f 0.935; localmin+geoPms **0.936** | c2f vs localmin +0.003 [+0.001, +0.007]; NCR −0.073 |
| **Real TX pairs, ViT-B** (all nulls) | 0.927 | localmin 0.929 | geoP **0.935**; c2f 0.934 | c2f vs localmin +0.005 [−0.001, +0.010] ns; NCR −0.065 sig. |
| Real LA, NCR (ViT-L, all) | 0.731 | localmin 0.685 | localmin+geoPms **0.606** | −0.079 [−0.092, −0.068] vs localmin |

## 3. Scorecard
| hypothesis / claim | verdict |
|---|---|
| H1 first-order linearity (proposal) | ✘ failed (rel. error 0.805 > 0.6) → pivot to probe subspace |
| H2 per-patch operator is selective | ✔ 12–13× (synthetic); ✔ real pairs |
| H3 geometric false changes ↓ at no clean cost | ✔ synthetic (−39 %, two-sided −32 %); ✔ real (NCR −8 … −17 %) |
| H4 real misregistration | ✔ LA (sig. vs strongest baseline, 2 backbones); ~ TX (NCR sig., AUROC gain sig. in 1/4) |
| H5 NFAD radiometric transfer | ✘ entangled; hurts on real pairs; per-patch radP more selective |
| H6 NFAD gate fails for local nuisance | ✔ via entanglement; gate only engages for parallax |
| backbone generality | ✔ ViT-L sat and ViT-B web (web backbone better overall, surprise) |
| cost | ✔ 16 probes (17 passes/reference) keep ≈ all of the gain |
| robustness to strong sensor noise | ✘ for search-based variants (fix: add global radiometric subspace, synthetic only) |

## 4. Created datasets (no manual labels)
NSB-v1 (synthetic null pairs on LEVIR, 26.6 k seeded recipes, 8 MB) · MXR-LA-v1 (578 real Maxar pairs, LA 2025) ·
MXR-TX-v1 (298 real Maxar pairs, Texas 2024). Datasheets in `Data_Created/*/DATASHEET.md`.

## 5. Status
All planned Phase-1/1b experiments done (E00–E09 incl. E01b, E03b, E03c, E06-B, E07-B, E08). Storage 7.8 / 8 GB.
