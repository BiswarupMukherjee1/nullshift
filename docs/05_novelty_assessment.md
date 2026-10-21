# 05 — Is there something novel yet? (honest assessment, updated 29 Sep 2026 evening, after E05/E06)

**Short answer (updated 21:00): yes — a specific, real-data-validated contribution that now replicates on two
backbones at 2.5× lower cost; still modest in effect size (~1 pt AUROC over simple search) and one real event.** It is a credible workshop / letter-level result
(IGARSS / GRSL / EarthVision) if the second backbone (queued) and a second real dataset confirm it; not yet a
main-venue paper.

What is substantial now:
1. **A measured, previously unreported property of frozen satellite FMs:** tokens respond to sub-pixel shift like √d
   (first-order/tangent models fail), and real VHR acquisition pairs are misregistered by *several* pixels in a
   spatially varying way — so the nuisance lives in a curved, multi-scale set of directions *per patch*.
2. **A training-free operator matched to that measurement — coarse-to-fine `localmin+geoPms`** (token search +
   per-patch subspace of the reference's own responses to 40 shifts). On 578 real Maxar pairs it beats every
   baseline including the strongest one (3×3 search): +0.9 pt patch AUROC [+0.4, +1.4], −11 % nuisance ratio
   [−0.092, −0.068 absolute], +6.4 pt over raw; consistent on dev and held-out tiles, every off-nadir bin, and a
   landscape-matched control.
3. **A clean negative result on the NFAD transfer:** a single global nuisance subspace (the most recent training-free
   approach) is entangled with real change in remote sensing and *significantly hurts* on real pairs
   (−1.0 pt window AUROC, CI excludes 0); for geometry it removes more real change than nuisance. Locality is necessary.
4. **Two created benchmarks** (NSB-v1 synthetic null pairs with physical doses; MXR-LA-v1 real near-null vs real
   destruction pairs from open Maxar data, no manual labels) and several evaluation pitfalls (LEVIR's JPEG history;
   unbounded registration and OLS normalisation damaging multi-year pairs).

What is not (yet) strong: effect size over simple search is ~1 pt AUROC; 40 extra encoder passes per reference;
change windows from 9 tiles; one backbone; the E06 design was chosen after E05 exposed all tiles (physically
motivated and held-out-validated, but not a pristine test).

## 1. Prior art map (all re-checked 28–29 Sep 2026)
| work | what it removes / how | where it sits |
|---|---|---|
| Simard, LeCun, Denker 1992–93 — *Tangent distance/prop* | projects a difference off the tangent space of known transforms (pixels, digits) | origin of the idea; first-order, pixel space |
| Bruzzone & Cossu, TGRS 41:2455, 2003; Marchesi, Bovolo, Bruzzone, TIP 19:1877, 2010 | model **registration noise** as a region of the spectral change-vector (magnitude/direction) space and exclude it | **closest RS ancestor**: pixel-spectral CVA, statistical estimate from the pair, 3–4-D |
| Mercier & Inglada 2008 | CD with misregistration errors (pixel statistics) | RS, classical |
| NFAD, arXiv 2608.29112 (Aug 2026) | **one global** nuisance subspace from content-preserving interventions (photometric/background); names geometric interventions as future work | industrial AD, frozen DINOv3 |
| SPARC, arXiv 2608.18585 (Aug 2026) | **per-cell** subspace from k ≤ 8 *aligned normal images*; explicitly requires alignment | industrial AD, frozen encoders |
| SubspaceAD, CVPR 2026 | PCA subspace of normal patch features | industrial AD |
| DRIFT, arXiv 2606.06918 (Jun 2026) | *trained* projection heads splitting robust/fragile subspaces; patch-wise drift | AI-image detection |
| PatchShift (arXiv 2510.13643), DVT (ECCV 2024) | aggregate/denoise over shifted views (averaging / neural fields) | adversarial AD / ViT artifacts |
| SEED, SCDistill, FAF-CD, CPEN, DiffRegCD, "Morphing Through Time" | robustness to shift/illumination by **training** or learned registration | RS CD |

## 2. What is new in NullShift (detail)
1. **The operator `geoP`:** a *per-image, per-patch* subspace built **at test time from the reference image's own
   shifted copies** (secant directions of the shift manifold, K = 24 probes, rank 8), projected out of a bitemporal
   residual of a frozen foundation model. In the design space {global | per-location} × {from calibration data | from
   interventions on the test image}, the cell *per-location × interventions-on-the-test-image* is empty in the works
   above: NFAD is global-intervention, SPARC per-location-calibration (and needs alignment — the nuisance we target),
   Bruzzone et al. is pixel-spectral and statistical. **This is the core novelty claim.**
2. **Measured facts about a satellite foundation model** (useful regardless of the method):
   - DINOv3-sat tokens respond to sub-pixel shift like **√d, not linearly** (tangent/Jacobian models fail; E01);
     ε-scaling of finite differences confirms it exactly (ε^−½).
   - A **single global subspace cannot represent misregistration**: NFAD's construction applied to shifts removes more
     true-change energy (35 %) than misregistration energy (23 %) (E02). NFAD's radiometric subspace is strongly
     entangled with real change in RS (30 % of true-change energy removed).
   - Misregistration energy at 16-px token scale is *spatially spread*, so NFAD's incidence gate does not engage; only
     parallax is spatially concentrated (E02, refines H6).
   - DINOv3-sat is almost gain-invariant; most sensitive to noise, JPEG, blur (fit report).
3. **The null-pair protocol (NSB-v1)** with physically parameterised doses and FCR/NCR/AUROC, stored as seeds —
   incremental over FAF-CD's perturbed-pair arm (low–moderate novelty on its own).
4. **Methodological findings about evaluation:** (a) LEVIR-CD tiles carry a q≈75 JPEG history on the 8×8 grid
   (aligned re-encoding is near-idempotent); (b) on multi-year pairs, unbounded phase correlation and OLS radiometric
   normalisation actively harm CD (median 8-px spurious "registration"; regression dilution halves contrast).
   Small, but reviewers value such pitfalls.

## 3. What is NOT novel (say it first)
- Projecting nuisance directions out of a difference: tangent distance (1992), CVA registration-noise modelling (2003),
  NFAD (2026). The contribution is *where the directions come from* and *that it works on frozen FM tokens for RS*.
- Using shifted copies at test time: PatchShift / DVT average them; we project them. Averaging must be beaten (E05).
- The radiometric part (`rad`) is a direct NFAD transfer and, in our data, **does not work well** (entangled).

## 4. What would make it a paper (and current status)
| requirement | status |
|---|---|
| real-data evidence (not only synthetic null pairs) | ✔ E05/E06 (Maxar LA): significant vs strongest baseline |
| a design matched to the real nuisance scale | ✔ coarse-to-fine (E06), from measured multi-pixel misregistration |
| beats or complements classical preprocessing (phase correlation + PIF) | clean LEVIR: classic+ns best AUROC; synthetic: pending E03; real: E05 |
| beats shift-averaging (same probes, averaged) | ✔ E05: shiftavg ≈ raw; shiftavg+geoP ≈ geoP |
| second backbone — backbone-agnostic claim | ✔ DINOv3 ViT-B/16 web: +1.1 pt [+0.6, +1.7] vs strongest baseline |
| second real dataset (S2Looking, labelled parallax) | not started (fits storage: ~0.8 GB) |
| cost reduction | ✔ E07: 16 probes keep ≈ all of the gain (17 passes/reference) |
| confidence intervals | E03b queued, E05 includes quadkey bootstrap |

**Realistic venues if E05 is positive:** IGARSS 2027 / IEEE GRSL letter / CVPR EarthVision 2027 workshop.
If E05 is negative: an analysis paper ("what frozen satellite FMs see under acquisition nuisance") at workshop level.

## Update 29 Sep 21:00 — what changed the assessment
- Second backbone replicates the coarse-to-fine gain (+1.1 pt AUROC, −0.077 NCR vs the strongest baseline) → the
  "backbone-agnostic" requirement is met (two DINOv3 models differing in size and training data).
- Cost fixed: 16 probes suffice.
- New observation for the paper: a small web-trained FM beats the large satellite FM on real bitemporal pairs.
- Remaining gaps for a stronger paper: second real event/sensor (another Maxar event, split by event; S2Looking),
  radiometric fragility of token search (E04) → coarse-to-fine + radiometric removal, and a non-DINO backbone.

## Update 30 Sep 00:00 — after E08 and E09 (second real event)
- **Second event (Texas, pristine, pre-registered):** criterion formally met; nuisance-ratio reduction replicates in all
  4 settings; the AUROC gain over the strongest baseline is significant in 1 of 4. Measured TX misregistration is small
  (1.5–3.6 px), so search adds little there and projection carries the gain — *mechanistically consistent*, but it
  means the headline "beats the best baseline by ~1 pt" is **event-dependent**.
- **Locality principle confirmed a third time** (radiometry on real pairs: per-patch radP helps, global rad does not).
- **New negative:** token-search variants are fragile to strong sensor noise; per-patch radiometry does not fix it.
**Revised verdict:** the most defensible paper is a *principle + protocol + measurement* paper — "acquisition nuisance in
frozen satellite FMs is local and scale-dependent; per-patch intervention subspaces (not global ones) remove it; here is a
label-free benchmark from open Maxar data and the measured misregistration scale that decides which component helps" —
with coarse-to-fine as the working instance. Workshop/letter level (EarthVision, IGARSS, GRSL) is realistic now;
a journal version needs a non-DINO backbone, a third event/sensor family (e.g. S2Looking), and a labelled change subset.
