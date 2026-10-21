# E03b — Scene-bootstrap confidence intervals for E03 (FCR at τ_R99)
**Why.** E03 pools histograms (exact point estimates, no uncertainty). Here per-tile FCR_R99 is recomputed for key
methods × 13 conditions on the same seeds/thresholds; 95 % CIs by resampling the 128 test *scenes*; paired differences.
Checkpointed per batch (reboot-safe; resumed once).

## Findings (run `20260929-1903_dinov3L_main`)
- Significant FCR reductions vs raw (CI excludes 0): C1 compound d3 (geoP −0.0007 [−0.0011, −0.0004]; geoP+rad −0.0023
  [−0.0048, −0.0007]; classic+ns −0.0027), G2 field 2 px (all NullShift variants ≈ −0.0004 [−0.0006, −0.0003]),
  noise (geoP+rad −0.019 [−0.028, −0.011]; classic+ns −0.021; geoP −0.003), blur (−0.0011 to −0.0017),
  colour (classic+ns / geoP+rad ≈ −0.007).
- At mid doses of G1 (0.5–1 px) and G3 every method's FCR_R99 is ≈ 0 → on LEVIR the synthetic geometric nuisance never
  reaches a 99 %-recall threshold; the benchmark cannot discriminate there (regime limitation, see E05/E06).
- JPEG q50 and haze: no method differs significantly from raw at this operating point.
