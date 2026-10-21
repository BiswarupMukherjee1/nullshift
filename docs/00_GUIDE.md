# 00 — NullShift in plain language (read this first)

## 1. The problem
A change detector compares two satellite images of the same place (t1, t2) and marks what changed. It also "sees"
change when **nothing on the ground changed** but the *instrument* did:
- **geometry:** the images are misaligned by a fraction of a pixel, or buildings lean differently because the
  satellite looked from a different angle (off-nadir *parallax*);
- **radiometry:** different satellite, sun, haze, colour processing, JPEG, blur, noise.
For a company selling *verified* change (Xoople), or evaluating foundation models on precise data (GEOSAT), these
false changes are the product failing.

## 2. The idea (NullShift)
Use a **frozen** satellite foundation model (DINOv3 ViT-L/16 trained on 493 M Maxar tiles). It turns each 16×16-pixel
patch into a 1024-number vector (a *token*). Change score of a patch = length of the difference vector
r = z(t2) − z(t1). NullShift removes, **before** scoring, the part of r that an instrument effect could have produced.
No training, no labels.

- **Geometric operator `geoP` (the main contribution).** Take the t1 image, shift it by 24 small amounts (8
  directions × 0.5/1/2 px) and record how each patch's vector moves. Those 24 "motion vectors" span the directions in
  which *that patch* moves when the image is misaligned. Keep the 8 most important (SVD) and delete them from r.
  A misregistration lives in these 8 directions → removed; a new/destroyed building changes the vector in many other
  directions → survives. Every patch gets its own directions, built from its own content at test time.
- **Radiometric operator `rad` (transfer of NFAD).** One global set of 56 directions learned from how features move
  under gain/gamma/haze/colour/JPEG/noise/blur on unlabeled images; deleted from every patch.
- **`geoP+rad`** = both at once (exact joint projection).
- **`localmin+geoPms` — coarse-to-fine (best on real data).** Real satellite pairs are misaligned by *several* pixels
  (measured in E06), more than geoP's ±2 px probes. So first let each t2 patch look for its best match among the 3×3
  neighbouring t1 patches (absorbs shifts up to one patch = 16 px), then delete the probe directions (shifts up to 8 px)
  of that matched patch from the residual.

## 3. Why the proposal's original operator (`geo`) was replaced
The proposal assumed features move *linearly* with sub-pixel shift (first-order Taylor/“Jacobian”, 2 directions).
Experiment E01 showed they do not: the response grows like √shift (a quarter-pixel shift already moves a token by 45 %
of what a full pixel does). The first-order model explains too little (kill criterion met). The proposal's own backup
plan — use several real shifts instead of a derivative — became `geoP` (E01b).

## 4. How we test it
- **Null pairs (created benchmark NSB-v1).** Pair an image with a perturbed copy of itself: nothing changed, so every
  flagged patch is a false change. 10 physical nuisance families × 5 strengths ("doses") + a compound one. No labelling.
  Stored as recipes (seeds), not images (~8 MB instead of ~5 GB).
- **Real change (LEVIR-CD).** Public building-change pairs, used only to check that real change is still detected and
  to pick thresholds (on the validation split only).
- **Real pairs (created benchmark MXR-LA-v1, Maxar Open Data, LA wildfires 2025).** Two *different* satellite
  acquisitions days apart (real misregistration, real building lean, real colour differences) vs. pre/post images of
  burned areas (real destruction). Answers the two main criticisms: LEVIR's pairs are years apart so nuisance is hidden,
  and null pairs build the operator from the very image that gets perturbed.

## 5. Metrics (what the numbers mean)
| metric | meaning | better |
|---|---|---|
| FCR (false-change rate) | fraction of null-pair patches above the threshold | lower |
| τ*, τ_R90, τ_R99 | thresholds fixed on clean LEVIR val: F1-optimal / catching 90 % / 99 % of real changes | — |
| NCR (nuisance-to-change ratio) | median score on null pairs ÷ median score on real changes; threshold-free | lower |
| AUROC change-vs-null | probability a real-change patch scores above a nuisance-only patch | higher |
| energy removed | share of a residual's squared length deleted by an operator | high on nuisance, low on real change |
| selectivity | energy removed on nuisance ÷ energy removed on real change | higher |

## 6. Experiments (IDs are stable; folders `experiments/E*/`)
| ID | question | status |
|---|---|---|
| E00 | is the pipeline numerically correct, how fast? | ✔ pass |
| E01 | are features linear in sub-pixel shift (proposal's H1)? | ✘ no → pivot |
| E01b | does a per-patch *probe* subspace work instead? | ✔ yes (on val) |
| E02 | which operator removes nuisance energy but not real-change energy? is "local" necessary? | ✔ geoP selective; global subspaces entangled |
| E03 | false-change rate vs nuisance strength (null pairs), + clean accuracy | ✔ geoP −39 % nuisance; classic+ns best overall |
| E03b | confidence intervals for E03 | ✔ significant for fields/compound/noise; geometric FCR ≈ 0 at mid doses |
| E03c | two-sided null pairs (operator built from an independently perturbed copy) | ✔ gain survives (−32 %) |
| E04 | real change + nuisance on t2: is real change still found? | ✔ rad protects under noise; 3×3 search fragile under noise |
| E05 | **real Maxar pairs**: real destruction vs real instrument differences | ✔ per-patch helps modestly; global hurts; 3×3 search best |
| E06 | **coarse-to-fine** (3×3 token search + wide probe subspace), motivated by measured multi-pixel misregistration | ✔ best; significant vs strongest baseline |
| E07 | cost/design ablation (how many probes, rank, search radius) | ✔ 16 probes enough |
| E06/E07-B | same on a second backbone (DINOv3 ViT-B web) | ✔ replicates; web backbone better overall |
| E08 | coarse-to-fine + radiometric probes; robustness to strong noise | ✔ radP helps on real pairs; search stays noise-fragile |
| E09 | **second real event** (Texas 2024), design frozen beforehand | ~ nuisance ↓ everywhere; ranking gain partly; small misregistration there |

## 6b. The one-paragraph conclusion
Removing acquisition nuisance works when it is done **per patch, from the reference image's own reaction to controlled
changes** — never with one global correction. On real satellite pairs, misalignment is several pixels, so the best
operator first lets each patch find its best-matching neighbour and then removes the patch's own shift directions
(coarse-to-fine). How much each part helps depends on how misaligned the data really is — so measure it first
(decision table in `docs/03` §8).

## 7. Design choices (and why)
- **Frozen DINOv3-sat:** trained on Maxar imagery like our real arena; no training keeps the claim clean.
- **Patch-level evaluation:** a token covers 8 m on LEVIR; pixel scores are coarse, so the primary metrics are per patch.
- **Per-method thresholds on val:** removing directions shrinks all scores, so each method gets its own threshold.
- **64-px margin around every tile:** shifts/warps never pull wrapped image content into the evaluated area.
- **Baselines strong on purpose:** phase-correlation registration (bounded, confidence-gated), PIF radiometric
  normalisation, histogram matching, blur, 3×3 neighbourhood matching, shift-averaging; two were strawmen at first
  and were fixed before the headline runs.
- **Storage ≤ 8 GB:** perturbations regenerated from seeds; fp16 weights; only small result files kept.
- **Robust execution:** long jobs via `scripts/launch.sh` (survives logout), checkpoints + resume.

Where to read more: `docs/01` (proposal review), `docs/02` (plan + changes), `docs/03` (results),
`docs/04` (independent critical review), `docs/05` (novelty assessment), `docs/06` (company relevance), `results/` (tables & figures).
