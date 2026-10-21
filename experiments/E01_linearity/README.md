# E01 — Linearity of patch features in sub-pixel shift (H1)
**Question.** Does the first-order model f(shift(I,d)) ≈ f(I) + J·d hold for |d| ≤ 2 px on DINOv3-sat patch tokens?
**Hypothesis H1.** relative error ‖Δz − J d‖/‖Δz‖ is small at 1 px. **Kill:** > 0.6 at 1 px for the best ε.
**Protocol.** LEVIR **val** 256 tiles (subset_val256_fit), J by central differences with ε ∈ {0.25, 0.5, 1, 2} px,
d ∈ {0.25 … 4} px along x, y, diagonal; metrics pooled over patches and by gradient-energy tercile (flat/mid/edge).
Also: energy left after the geo projection (what the operator cannot remove) and the ε-consistency diagnostic from the
external review (principal-angle cos² between span(J_ε) and span(J_1)). ε* (min mean rel. error over d ≤ 1) is used
by all later experiments — chosen on val, never on test.
## Findings — **H1 FAILS (kill criterion met)**
Runs: `20260928-0245_dinov3L_val256` (last block), `20260928-0258_dinov3L_block12_val256` (block 12).

| | last block | block 12 |
|---|---|---|
| ε\* (val) | 1.0 | 0.5 |
| rel. error ‖Δz − J d‖/‖Δz‖ at 1 px | **0.805** (kill > 0.6) | **1.006** |
| energy left after rank-2 tangent projection at 1 px | 0.56 | 0.77 |

- **The response to sub-pixel shift is rough, not differentiable:** ‖Δz‖ = 0.089, 0.156, 0.198, 0.273, 0.408 at
  d = 0.25, 0.5, 1, 2, 4 px — growth ≈ √d, not ∝ d. The ε-diagnostic proposed by the external review confirms it:
  ‖J_ε‖/‖J_1‖ = 2.12, 1.44, 1, 0.76 for ε = 0.25, 0.5, 1, 2 — exactly the ε^−½ law of a √d response (2.0, 1.41, 1, 0.71);
  span(J_ε) also rotates with ε (cos² principal angle vs ε=1: 0.26, 0.50, –, 0.51).
- Best fit always occurs at d = ε (the tangent is really a secant). No difference between flat/mid/edge patches, so it is
  not an edge effect.
- Why: LEVIR tiles carry strong near-Nyquist content; even in pixel space a first-order Taylor model needs
  d ≪ 1/(2π k_max) ≈ 0.3 px (the reason Lucas–Kanade works on pyramids). The frozen ViT inherits and amplifies it
  (Azulay & Weiss 2019). Block 12 is *less* linear → the "intermediate layer" fallback fails.
- **Scale:** 1-px misregistration gives ‖r‖ ≈ 0.20, whereas real *unchanged* LEVIR patches give 1.01 and real changes
  1.26 — on this dataset nuisance-induced false alarms are small relative to real temporal differences.
- Decision: pre-registered fallback → multi-shift per-patch subspace → **E01b**.

