# E06 — Coarse-to-fine operator on real pairs (two backbones) (motivated by a measurement, validated on held-out tiles)
**Why.** E05 showed the per-patch principle transfers to real data but gains were modest and plain 3×3 token search
(`localmin`) was best. Measurement on real near-null Maxar pairs (confident phase-correlation peaks only, ~10 %):
displacement median 3.3 px (pre-fire pairs) / 8–9 px (post, long) at 0.5 m, up to 22 px, uncorrelated with off-nadir
(r = 0.08) → real misregistration is **multi-pixel and spatially varying**; geoP's probes only cover ±2 px.
**Operator `localmin+geoPms` (coarse-to-fine).** For each t2 patch p: search the 3×3 token neighbourhood q of the t1
map (absorbs up to 16 px), and score min_q ‖P⊥_{U(q)} (z2(p) − z1(q))‖ where U(q) is the per-patch probe subspace of the
matched t1 patch built from 40 shifted copies (8 directions × {0.5, 1, 2, 4, 8} px, rank 16). Training-free.
**Anti-overfitting.** Tiles split by quadkey hash into dev / heldout before any E06 number was seen.

## Findings (run `20260929-1825_dinov3L_mxrla_c2f`) — ✔ significant gain over the strongest baseline, consistent on both splits
| heldout (37 change / 110 null) | patch AUROC | window AUROC | TPR@5 % FCR | NCR |
|---|---|---|---|---|
| raw | 0.835 | 0.913 | 0.439 | 0.738 |
| geoP (±2 px) | 0.849 | 0.915 | 0.470 | 0.682 |
| geoPms (±8 px, no search) | 0.881 | 0.930 | 0.581 | 0.637 |
| localmin (symmetric 3×3 search) | 0.900 | **0.955** | **0.670** | 0.696 |
| localmin1 (one-directional search) | 0.892 | 0.954 | 0.635 | 0.693 |
| localmin+geoP | 0.900 | 0.953 | 0.657 | 0.648 |
| **localmin+geoPms** | **0.908** | 0.952 | 0.664 | **0.622** |
Dev split: same ordering (localmin+geoPms 0.904 vs localmin 0.895 patch AUROC). Every off-nadir bin: lowest NCR.

**Paired quadkey bootstrap (all tiles, B = 400, patch level)** — `runs/*/posthoc_paired_bootstrap.txt`:
| comparison | ΔAUROC [95 % CI] | ΔNCR [95 % CI] |
|---|---|---|
| localmin+geoPms − localmin | **+0.009 [+0.004, +0.014]** | **−0.079 [−0.092, −0.068]** |
| localmin+geoPms − localmin1 (projection's own contribution) | +0.016 [+0.009, +0.022] | −0.076 [−0.088, −0.065] |
| geoPms − geoP (wider probes) | +0.030 [+0.020, +0.040] | −0.051 [−0.069, −0.037] |
| localmin+geoPms − raw | +0.064 [+0.045, +0.085] | −0.125 [−0.158, −0.098] |

**Landscape-matched control** (nulls within 2 km of the perimeters, n = 98): every method loses ≈ 2.5 pt; ranking
unchanged (localmin+geoPms 0.882 > localmin 0.877 > geoPms 0.859 > raw 0.815).
**Caveats.** Change windows come from only 9 quadkeys → cluster bootstrap CIs are optimistic; window-level AUROC is a tie
with localmin; the design was chosen after E05 exposed all tiles (mitigated, not removed, by the physical motivation and
the held-out split); cost is 40 extra reference passes (→ E07).

## Second backbone — DINOv3 ViT-B/16 **web** (run `20260929-2044_dinov3B_web_mxrla_c2f_vitb16web`)
| heldout | patch AUROC | window AUROC | TPR@5 % FCR | NCR |
|---|---|---|---|---|
| raw | 0.930 | 0.965 | 0.695 | 0.587 |
| geoP | 0.940 | 0.970 | 0.728 | 0.531 |
| geoPms | 0.955 | 0.977 | 0.779 | 0.483 |
| localmin | 0.959 | 0.975 | 0.777 | 0.547 |
| **localmin+geoPms** | **0.967** | **0.982** | **0.825** | **0.471** |
Paired quadkey bootstrap (all tiles): vs localmin **ΔAUROC +0.011 [+0.006, +0.017], ΔNCR −0.077 [−0.083, −0.069]**;
vs raw +0.034 [+0.020, +0.049]. Window-AUROC vs raw: +0.013 [+0.005, +0.022] (localmin alone: +0.004 [−0.001, +0.010]).
Landscape-matched: 0.952 > geoPms 0.945 > localmin 0.939 > raw 0.916.
→ **The coarse-to-fine gain replicates across backbones** (size and training data differ).
**Unexpected:** the smaller web-trained backbone is much better on this task than the satellite-trained ViT-L
(raw 0.930 vs 0.835; best 0.967 vs 0.908) — backbone choice matters more than the operator. Hypothesis (untested):
DINOv3-sat is more texture/illumination-sensitive (it is most sensitive to noise/JPEG/blur, E01 fit report).
