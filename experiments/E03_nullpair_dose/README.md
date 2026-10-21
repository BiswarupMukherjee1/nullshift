# E03 — Null-pair dose-response of the false-change rate (H3, H5) — headline experiment
**Question.** How fast do false changes grow with each physical nuisance, and how much does each method suppress them
at an operating point fixed on clean data?
**Protocol.** (1) per-method threshold τ* = argmax patch-F1 on clean LEVIR val (1024 tiles); (2) clean test
(2048 tiles): patch/pixel F1, AUROC; (3) NSB-v1 null pairs (test512 × 11 families × 5 doses): FCR = fraction of
patches ≥ τ*, AUROC(true-change test patches vs null patches). All 16 methods (NullShift operators + baselines).
**Hypotheses.** H3: geo cuts FCR on G1–G3 substantially with clean patch-F1 drop ≤ 1 pt (kill > 3 pt).
H5: rad cuts FCR on R1–R7. Expected: phasecorr wins on G1 (pure global shift) but not on G2/G3; histmatch/rrn win on
global R1/R2; the interesting cells are G2, G3, R3, R4, R5, R7, C1 and classic+ns vs classic.
## Findings (run `20260928-1842_dinov3L_main`; null stage resumed from checkpoint after two logout kills)
Calibration + clean test reused from `20260928-0410_dinov3L_main` (identical code/config for stages 1–2).
**Clean test (2048 tiles):** geoP patch-F1 0.415 vs raw 0.409, AUROC 0.877 vs 0.873 → no clean cost (H3 clean part ✔).

**Nuisance-to-change ratio NCR** (median null score / median true-change score; mean over doses 1–5; lower = better):

| method | G1 shift | G2 field | G3 parallax | C1 compound | mean geo | mean radiometric |
|---|---|---|---|---|---|---|
| raw | 0.182 | 0.195 | 0.044 | 0.312 | 0.140 | 0.226 |
| geo (proposal tangent) | 0.157 | 0.177 | 0.042 | 0.298 | 0.125 | 0.220 |
| **geoP** | 0.079 | 0.141 | 0.039 | 0.258 | 0.086 | 0.202 |
| geoP+rad | 0.088 | 0.153 | 0.043 | 0.270 | 0.094 | 0.188 |
| rad | 0.194 | 0.206 | 0.046 | 0.322 | 0.149 | 0.208 |
| phasecorr (bounded, gated) | **0.056** | 0.200 | 0.044 | 0.317 | 0.100 | 0.226 |
| blur σ = 1 px | 0.126 | **0.128** | **0.032** | 0.281 | 0.095 | 0.196 |
| histmatch | 0.185 | 0.198 | 0.049 | 0.269 | 0.144 | 0.177 |
| classic (phasecorr + PIF RRN) | 0.056 | 0.201 | 0.045 | 0.271 | 0.101 | 0.178 |
| **classic+ns** (classic → geoP+rad) | **0.039** | 0.155 | 0.043 | **0.230** | **0.079** | **0.151** |

**FCR at τ_R99** (99 % recall of real change on clean val), mean over doses: geometric families raw 0.00093 → geoP
0.00017 (5.4× fewer false changes); radiometric raw 0.037 → geoP+rad 0.015, classic+ns 0.009; noise (R6) raw 0.144 →
geoP+rad 0.049, classic+ns 0.041. At τ* and τ_R90 geometric FCR is ≈ 0 for every method (LEVIR regime).
AUROC(change vs null) ≥ 0.98 for all methods/families — synthetic nuisance on LEVIR is always separable from
multi-year real change, so AUROC is saturated; NCR and FCR_R99 carry the signal.

**Verdicts.**
- H3 (geometric): **PASS in the null-pair setting** — geoP reduces geometric nuisance 39 % (NCR) / 5.4× (FCR_R99) at no
  clean cost. The proposal's tangent (geo) gives only 11 %.
- **Not the best single tool everywhere:** bounded phase correlation wins on pure global shift (as predicted); a plain
  Gaussian blur wins on smooth fields and parallax — a strong, cheap baseline that must be reported.
- H5 (radiometric, NFAD transfer): **weak** — rad reduces radiometric NCR only 8 % (0.226 → 0.208); pixel-space
  normalisation (histmatch, PIF) is better for global radiometry; rad helps where pixel normalisation cannot reach
  (noise, JPEG, blur).
- **Best overall: classic+ns** (NullShift *after* classical preprocessing): best mean NCR for geometric *and*
  radiometric families, best on G1 and C1 → the operator complements preprocessing rather than replacing it.
- Caveat: LEVIR's multi-year pairs make nuisance small relative to real change → the decisive test is E05 (real pairs).


## E03c — two-sided null pairs (run `20260929-2111_dinov3L_twosided`; 256 tiles, 7 families, doses 1/3/5)
Answers the external critique "the operator is built from the image that gets perturbed": here t1 = T_a(I) and
t2 = T_b(I) with independent draws, so the operator never sees the image that became t2.
| mean over doses | raw | geo | **geoP** | geoP+rad | phasecorr | blur | classic | **classic+ns** |
|---|---|---|---|---|---|---|---|---|
| NCR geometric | 0.151 | 0.135 | **0.102** | 0.114 | 0.125 | 0.116 | 0.125 | 0.103 |
| NCR radiometric (R3–R5) | 0.199 | 0.193 | 0.180 | 0.181 | 0.199 | 0.199 | 0.141 | **0.132** |
| FCR_R99 geometric | 0.0048 | 0.0033 | **0.0018** | 0.0018 | 0.0048 | 0.0046 | 0.0055 | 0.0020 |
→ geoP keeps most of its benefit (−32 % geometric NCR vs −39 % one-sided; 2.7× fewer false changes at τ_R99): the
gain was **not** mainly home-turf. classic+ns remains the best combination overall.
