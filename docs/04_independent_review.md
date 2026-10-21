# 04 — Independent critical review of Phase 1 (written as an external reviewer would)

Scope: the code in `src/nullshift/`, the created benchmark `Data_Created/NSB-v1`, experiments E00–E04 (+E01b, E03b)
and their interpretation. The reviewer's question: *which claims would survive a hostile reading, and which would not?*

## A. Process / integrity checks
| Check | Status |
|---|---|
| Test set never used for any selection (ε, rank r, thresholds, subspaces) | ✔ ε* (E01) and r = 8 (E01b) chosen on val; τ*/τ_R90/τ_R99 on clean val; V_rad/V_shift fitted on val |
| Pre-registered kill criteria honoured | ✔ H1 declared FAILED (0.805 > 0.6); pivot followed the proposal's own fallback table |
| Baselines implemented fairly | ⚠→✔ two strawmen caught *before* the headline run (unbounded phase correlation; OLS RRN with regression dilution); first E03 attempt aborted and discarded; both failure modes unit-tested |
| Bugs caught by tests | ✔ 34 unit tests; they caught a phase-correlation failure on smooth scenes and the Nyquist non-composition of FFT shifts; an einsum index bug in the joint projection was caught in review before first run |
| Determinism / numerical floor | ✔ identity null pairs exactly 0; batch-size-1 kernel difference (0.0016) avoided by construction |
| Data creation checked visually and numerically | ✔ previews + pixel dose statistics; found and fixed a non-monotonic JPEG dose caused by LEVIR's own q≈75 JPEG history |
| Storage / location constraints | ✔ ≤ 7 GB, all inside the project; caches redirected (HF, torch, CUDA JIT, matplotlib, pycache, xet disabled) |

## B. Threats to validity (design level)
1. **Null pairs favour reference-based operators ("home turf").** In a null pair the probe basis is computed on the very
   image that is then perturbed; G1 even uses the same Fourier interpolator as the probes. Mitigations in place: G2/G3
   use bicubic warps, C1 adds independent noise to the reference, E04 computes the basis on t1 while the nuisance is on t2.
   Residual risk: gains on G1 are an upper bound. **Real misregistration (S2Looking / multi-sensor pairs) is still untested.**
2. **LEVIR's temporal regime hides nuisance.** Pairs are 5–14 years apart; real *unchanged* residuals (‖r‖ ≈ 1.0) are ~5×
   the residual of a 1-px misregistration (≈ 0.2). At LEVIR-calibrated thresholds nuisance-induced false changes are rare,
   so FCR@τ* is close to 0 for most doses — the dataset's regime, not a property of the operators. The operationally
   relevant regime (short revisit, same constellation — Xoople's case) has much smaller natural residuals and lower
   thresholds; the high-recall τ_R99 and the scale-free NCR/AUROC are proxies for it, not a substitute.
3. **Patch-level labels.** 16×16 tokens (8 m) vs small LEVIR buildings: positive = ≥ 25 % changed pixels, mixed patches
   ignored. Absolute F1 values (≈ 0.4) are therefore not comparable to the literature (supervised pixel-F1 ≈ 0.9); only
   relative comparisons between methods are meaningful.
4. **One backbone, one dataset, one intervention sampling.** Backbone-agnosticism (NFAD's argument) is not shown.
5. **Uncertainty.** E03 pools histograms (exact point estimates, no CIs). E03b adds scene-level bootstrap CIs for the
   headline comparisons only.
6. **Cost.** geoP needs K = 24 extra encoder passes of the reference (25× the reference cost, vs 4 in the proposal).
   Shared across all conditions in a benchmark, but at deployment it is a real cost (≈ 0.3 s per 256² tile on an RTX 3060).
   K was not optimised; smaller K is plausible (E01b shows rank 8 suffices).
7. **fp16 weights.** Local fp16 copy differs from the official fp32 model by a mean feature-residual norm of 0.013 — same
   model for both dates, so no bias between methods, but numbers are not exactly those of the released checkpoint.

## C. Claims — verdicts (after E03, E05, E06)
| claim | verdict | reviewer's comment |
|---|---|---|
| "First-order tangent removes misregistration" (proposal) | **rejected** | √d response; kill criterion met; honest pivot |
| "Per-patch probe subspace is selective" | **supported** (synthetic, E02; real, E05) | 13× selectivity synthetic; on real pairs NCR −8 % |
| "Global (NFAD) subspaces fail in RS" | **supported, strongly** | selectivity < 1 for shifts; significant harm on real pairs |
| "NullShift reduces false changes on null pairs" | **supported but home-turf** | −39 % NCR; specialists (phase corr., blur) win their own families; best as *classic+ns* |
| "Works on real acquisitions" | **supported, modest** | geoP alone +1.5 pt; coarse-to-fine +0.9 pt over the strongest baseline, CI > 0 |
| "Backbone-agnostic" | **supported (2 DINOv3 backbones)** | a non-DINO backbone would make it stronger |
| "Cheap" | **acceptable** | 16 probes keep ≈ all gain (E07); 17 passes/reference |
| "Robust to radiometric nuisance" | **not for search-based variants** | localmin degrades most under noise (E04); combine with rad |

**Would a hostile reviewer accept the headline?** They would accept (a) the measurements, (b) the negative result on
global subspaces, (c) the protocol; they would push on (d) effect size over simple token search, (e) single event and
backbone, (f) E06's post-hoc design. (d) is answered by the paired CI; (e), (f) need the queued runs and a second
real dataset (S2Looking or another Maxar event, split by event).

## C2. Threats added after E04–E07
- The selection of the E07 variant used dev tiles only (rule fixed beforehand) — clean. E06's operator family was
  chosen after E05; held-out split + second backbone reduce but do not remove this.
- E04 shows token search trades radiometric robustness for geometric robustness — must be reported with the headline.
- The web backbone beating the satellite backbone could reflect GSD/resampling mismatch (0.5 m vs SAT-493M's 0.6 m
  Maxar); untested.

## C3. After E08/E09
| claim | verdict |
|---|---|
| "coarse-to-fine beats the best baseline on real data" | **LA: yes (2 backbones); TX: NCR yes, AUROC only 1/4 significant** → state as event-dependent |
| "radiometric probes make c2f robust" | **real LA: small gain; synthetic strong noise: no** |
| "global radiometric subspace is useless" | **on real pairs yes; on synthetic strong noise it is the only fix** (but fitted on those families) |
| "which component helps is predicted by measured displacement" | **supported by 2 events** (LA multi-pixel → search; TX small → projection) — only 2 data points |

## D. Bugs found during the review cycles (all fixed, all now covered by tests)
einsum index bug in joint projection (before first run) · phase-correlation fails on smooth scenes (Hann window) ·
unbounded phase correlation / OLS RRN strawmen (bounded + gated; PIF mean-std) · non-monotonic JPEG dose (grid offset) ·
pandas-3 groupby dropping the key column · two-sided mode running a *change* pair as identity and crashing on a missing
reference · dataclass field order breaking `pipeline` import (caught by a new end-to-end engine test before any GPU job
imported it) · shell `pkill -f` pattern matching its own shell · jobs killed at logout (systemd-run + linger) and by
reboots (checkpoint/resume, idempotent queues) · E03b had no checkpoint → per-batch per-tile checkpoint added · E08: `c2f+rad` missing from the subspace-loading set (KeyError) · CUDA OOM in `probe_basis` with 40 probes (chunked, unit-tested) · prepared-geometry misuse in the TX builder (`tile.intersects(prep)`) · E04 leaderboard column mislabelled after switching to AUROC (renamed) · log-read races after relaunches (verified live state before trusting output).
