# E05 — Real arena: real destruction vs real acquisition nuisance (Maxar LA, MXR-LA-v1)
**Why.** Answers the two external criticisms of Phase 1 (LEVIR hides nuisance; null pairs are the operator's home turf).
**Question.** On two *different* real satellite acquisitions, does removing nuisance directions (geoP, radP, their
combination) make real destruction more separable from real instrument differences than raw residuals, classical
preprocessing (bounded phase correlation + PIF normalisation) and shift averaging?
**Protocol.** MXR-LA-v1 (see its DATASHEET). All parameters frozen from LEVIR val (ε, probe rank 8, radP rank 4,
subspaces, thresholds) → no tuning on this data. Metrics: patch- and window-level AUROC(change vs near-null),
TPR at 5 % real false-change rate, NCR, FCR at LEVIR-calibrated thresholds (transfer), null score vs off-nadir
difference, quadkey-bootstrap 95 % CIs of AUROC differences vs raw.
**Success criterion (set before running).** geoP (or geoP+radP) improves window-AUROC over raw with a CI excluding 0,
and is not worse than `classic` and `shiftavg`.
## Findings (run `20260929-1812_dinov3L_mxrla`, 578 real pairs: 200 change / 378 near-null) — success criterion only partly met

| method | patch AUROC | window AUROC | TPR@5 % FCR (patch) | NCR (lower better) | FCR at LEVIR τ_R99 |
|---|---|---|---|---|---|
| raw | 0.841 | 0.921 | 0.468 | 0.731 | 0.239 |
| geo (proposal tangent) | 0.846 | 0.922 | 0.472 | 0.714 | 0.225 |
| **geoP** | 0.856 | 0.925 | 0.481 | 0.675 | 0.212 |
| geoP+radP | 0.859 | **0.930** | **0.506** | 0.668 | – |
| blur+geoP | **0.863** | 0.929 | 0.486 | **0.661** | – |
| classic+geoP | 0.860 | 0.929 | 0.487 | 0.671 | – |
| shiftavg+geoP | 0.857 | 0.925 | 0.484 | 0.671 | – |
| radP | 0.847 | 0.927 | 0.489 | 0.711 | – |
| rad (NFAD global) | 0.839 | 0.911 | 0.432 | 0.747 | 0.317 |
| classic (phasecorr+PIF) | 0.845 | 0.925 | 0.472 | 0.727 | 0.251 |
| shiftavg | 0.845 | 0.922 | 0.472 | 0.721 | – |
| blur | 0.847 | 0.925 | 0.474 | 0.719 | 0.204 |
| **localmin (3×3 token search)** | **0.896** | **0.943** | **0.631** | 0.685 | **0.129** |

Window-AUROC difference vs raw, 95 % quadkey-bootstrap CI: geoP +0.004 [−0.000, +0.008] (P = 0.97);
shiftavg+geoP +0.004 [+0.001, +0.007]; localmin **+0.022 [+0.008, +0.041]**; rad **−0.010 [−0.020, −0.004]**;
classic +0.004 [−0.002, +0.010]; classic+ns −0.001 [−0.009, +0.005].

1. **Real acquisition nuisance is large:** NCR(raw) = 0.73 on real near-null pairs vs 0.14 for LEVIR synthetic doses.
   The external critique was right: LEVIR/NSB understate the problem.
2. **The per-patch-from-the-reference principle transfers to real data:** geoP, radP and their combinations improve every
   metric over raw at every off-nadir difference (NCR 0.700→0.640, 0.738→0.684, 0.804→0.763 for Δoff-nadir 0–5°,
   5–10°, 10–20°); the NFAD-style *global* radiometric subspace significantly **hurts** (as E02 predicted).
3. **But gains are modest** (+1.5–2.2 pt patch AUROC), and the 3×3 token-search baseline is clearly best.
   Measurement (`.tmp`→ docs/03): on real near-null pairs with a confident global peak, displacement is 3.3 px (pre)
   to 8–9 px (post) median, up to 22 px, uncorrelated with off-nadir (r = 0.08) → **real misregistration is
   multi-pixel**, beyond geoP's ±2 px probes. → coarse-to-fine operator, **E06**.
4. Transfer of LEVIR thresholds: at τ_R99, 24 % of real near-null patches fire for raw vs 13 % for localmin — the
   operating point does not transfer across sensors, as expected.
Confound check: change windows lie in burn areas; only 8 null windows lie inside perimeters, 98 within 2 km
(landscape-matched subset analysed in E06).

