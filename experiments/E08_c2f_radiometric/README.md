# E08 — Radiometric-robust coarse-to-fine (c2f, c2f+radP, c2f+rad)
**Why.** E04 showed 3×3 token search is the method most hurt by strong radiometric nuisance (noise: 0.692 vs raw
0.740), so the best real-data operator inherits that weakness; E02-radP showed per-patch radiometric probes are far more
selective than NFAD's global subspace. **Operators** (cheap setting from E07): `c2f` = 3×3 token search + per-patch
basis of 16 shift probes (1, 4 px), rank 8; `c2f+radP` = same search, basis of 16 shift + 9 radiometric probes, rank 12;
`c2f+rad` = same search, joint projection with the global NFAD subspace (56 dims).

## Findings — real LA pairs (MXR-LA-v1, 578 pairs; paired quadkey bootstrap vs localmin, B = 400)
| | ViT-L sat: AUROC | ΔAUROC vs localmin | ΔNCR | ViT-B web: AUROC | ΔAUROC vs localmin | ΔNCR |
|---|---|---|---|---|---|---|
| localmin | 0.896 | – | – | 0.943 | – | – |
| c2f | 0.902 | +0.005 [+0.001, +0.009] | −0.060 [−0.069, −0.051] | 0.951 | +0.009 [+0.004, +0.014] | −0.057 [−0.062, −0.050] |
| **c2f+radP** | **0.905** | **+0.008 [+0.004, +0.013]** | **−0.067** [−0.074, −0.058] | **0.955** | **+0.012 [+0.006, +0.019]** | −0.056 [−0.062, −0.045] |
| c2f+rad (global) | 0.898 | +0.002 [−0.007, +0.011] ns | −0.050 | – | – | – |
- c2f+radP vs c2f: ViT-L +0.003 [−0.002, +0.010] (NCR −0.007, sig.); ViT-B +0.003 [+0.001, +0.007] → radiometric
  probes help a little on real pairs; the global NFAD subspace does not (third independent confirmation of locality).
- Cost: c2f+radP = 25 extra reference passes (16 shift + 9 radiometric) vs 40 for localmin+geoPms, which is still
  0.4 pt better on ViT-L (quality/cost trade-off).

## Findings — synthetic perturbations of real LEVIR pairs (A, T(B)), 256 tiles, doses 3/5 (run `*_e08_levir_perturbed`)
| patch AUROC | clean | G1 4 px | G3 8 px | haze .3 | JPEG q15 | noise σ .08 | mean ΔAUROC |
|---|---|---|---|---|---|---|---|
| raw | 0.876 | 0.867 | 0.834 | 0.809 | 0.846 | 0.747 | −0.025 |
| localmin | 0.863 | 0.856 | 0.823 | 0.779 | 0.824 | 0.704 | −0.030 |
| localmin+geoPms | 0.862 | 0.854 | 0.825 | 0.776 | 0.821 | 0.691 | −0.030 |
| c2f | 0.868 | 0.860 | 0.829 | 0.786 | 0.830 | 0.706 | −0.029 |
| c2f+radP | 0.872 | 0.864 | 0.831 | 0.803 | 0.837 | 0.713 | −0.028 |
| **c2f+rad** | 0.882 | 0.875 | 0.852 | 0.840 | 0.856 | **0.819** | −0.017 |
| **geoP+rad** (no search) | **0.884** | **0.876** | **0.854** | **0.845** | **0.861** | **0.827** | **−0.016** |
- **Per-patch radiometric probes do not fix the search's noise fragility** (0.713): strong independent sensor noise is
  high-dimensional, not a low-rank per-patch direction. The global NFAD subspace does (0.819) — but it was fitted on
  these very synthetic families (home turf), and on *real* LA pairs c2f+rad was not significant.
- **Token search costs accuracy when there is no misregistration** (LEVIR is co-registered: clean 0.868 vs raw 0.876),
  while it gains +6 pt on real Maxar pairs → the operator must match the regime.
- **Conclusion: no single operator dominates.** Recommendation (see docs/03 §decision table): multi-pixel real
  misregistration → c2f(+radP); strong sensor-noise/JPEG differences → add the global radiometric subspace; already
  co-registered pairs → geoP+rad (no search).
