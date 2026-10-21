# E04 — Perturbed real pairs (A, T(B)) (H3 second arm)
**Question.** When the t2 acquisition carries a nuisance, does each method still detect the REAL change, at the
clean-val threshold from E03? (FAF-CD-style stress test, but with physically parameterised doses.)
**Protocol.** test512, doses {1, 3, 5} of every family + clean; patch-F1/pixel-F1/AUROC; ΔF1 vs the same tiles clean.
## Findings (run `20260929-1819_dinov3L_main`, test512, doses 1/3/5; resumed once from checkpoint)
Patch AUROC at the strongest dose (clean 512-tile subset: raw 0.868, geoP 0.872, geoP+rad 0.868, classic 0.872):

| method | G1 shift 4 px | G2 field 4 px | G3 parallax 8 px | R3 haze | R5 JPEG q15 | R6 noise σ .08 | C1 | mean ΔAUROC (all fam., doses) |
|---|---|---|---|---|---|---|---|---|
| raw | 0.858 | 0.857 | 0.814 | 0.802 | 0.830 | 0.740 | 0.885 | −0.0147 |
| geoP | **0.862** | **0.861** | 0.818 | 0.804 | 0.830 | 0.737 | **0.890** | −0.0149 |
| geoP+rad | 0.859 | 0.850 | **0.827** | 0.834 | **0.845** | **0.831** | 0.888 | **−0.0085** |
| rad | 0.854 | 0.844 | 0.823 | 0.830 | 0.842 | 0.827 | 0.882 | −0.0087 |
| classic+ns | 0.861 | 0.854 | 0.827 | **0.862** | 0.829 | 0.803 | 0.878 | −0.0099 |
| blur | 0.864 | 0.860 | 0.813 | 0.821 | 0.836 | 0.831 | 0.876 | −0.0105 |
| localmin | 0.846 | 0.843 | 0.803 | 0.773 | 0.805 | **0.692** | 0.880 | −0.0169 |

- On LEVIR, geometric nuisance on t2 barely changes real-change detection (≤ 1 pt) — same regime effect as E03.
- Radiometric nuisance matters: the radiometric subspace is what protects real-change detection under noise/JPEG
  (R6: 0.740 → 0.831); classical PIF normalisation wins on haze/colour; combinations degrade least on average.
- **New weakness:** `localmin` (3×3 search) is the most fragile method under strong radiometric nuisance (noise: 0.692 <
  raw 0.740) — the search lets a noisy patch pick a spurious match. Our best real-data operator (localmin+geoPms, E06)
  likely inherits this → it should be combined with radiometric removal (Phase 2: localmin+geoPms+rad / +radP).

