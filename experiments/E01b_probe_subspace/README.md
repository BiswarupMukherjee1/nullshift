# E01b — Per-patch probe (secant) subspace: the pre-registered fallback after H1 failed
**Question.** If the shift response is not linear, can the span of *actual* feature responses to K probe shifts of the
reference, per patch, still capture misregistration while sparing true change?
**Operator `geoP`.** D(p) = [f(shift(I₁, d_j)) − f(I₁)]_{j=1..K}(p), K = 24 probes (8 directions × {0.5, 1, 2} px,
rings offset by 22.5°); U_r(p) = top-r left singular vectors; r' = r − U Uᵀ r. Training-free, closed-form; cost K extra
forwards of the reference only (shared across all nuisance conditions).
**Protocol.** LEVIR val256, held-out random-direction shifts (incl. 1.5 and 3 px between rings, 0.25 and 4 px outside),
held-out smooth fields; true-change (≥ 50 %) and real-unchanged patches of real pairs (basis from t1 = A).

## Findings (run `20260928-0251_dinov3L_L-1_probe8x3`)
| operator | shift 0.25 | shift 1 | shift 4 | field 1 px | true change | real unchanged | selectivity (1 px) |
|---|---|---|---|---|---|---|---|
| tangent r2 (proposal) | 0.20 | 0.37 | 0.25 | 0.23 | 0.007 | 0.021 | 53× |
| multi-scale tangent r6 | 0.45 | 0.47 | 0.43 | 0.35 | 0.025 | 0.055 | 19× |
| probe r4 | 0.59 | 0.81 | 0.66 | 0.48 | 0.048 | 0.088 | 17× |
| **probe r8 (selected)** | **0.75** | **0.89** | **0.72** | **0.56** | **0.072** | 0.122 | **12×** |
| probe r24 | 0.90 | 0.97 | 0.78 | 0.66 | 0.121 | 0.190 | 8× |
(values = fraction of residual energy removed; chance = r/1024)

- The probe subspace captures 2.4× more 1-px misregistration energy than the proposal's tangent, generalises between and
  beyond the probe rings, and still spares > 92 % of true-change energy.
- Spatially varying fields are harder (56 % at r8): ViT tokens attend globally, so a patch's feature also moves with
  displacements *elsewhere* in the tile — a per-patch local basis cannot represent that (a genuine limitation).
- Rank trades removal vs. selectivity monotonically; **r = 8** chosen on val as the knee (removal ≥ 0.89 at 1 px with
  true-change loss < 10 %). All later experiments report both `geo` (tangent) and `geoP` (probe r8).
