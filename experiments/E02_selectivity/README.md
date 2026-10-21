# E02 — Energy selectivity and the locality question (H2, H6)
**Question.** Does each operator remove nuisance energy but spare true-change energy? Is a *per-patch* tangent needed,
or does one global subspace (NFAD-style, fitted on shift interventions) do as well?
**Hypotheses.** H2: geo removes far more energy from misregistration residuals than from true-change residuals
(kill: ratio ≈ 1). H6: NFAD's incidence gate does not rescue a global shift subspace, because misregistration energy
is spatially concentrated on edges (low participation ratio φ → gate backs off exactly where it is needed).
**Protocol.** NSB-v1 test512: null pairs for all families × doses (tangents from B, and T(B) for geo4); real pairs A→B
for true change (≥ 50 % changed patches) and real-unchanged (0 %) patches. Operators: geo, geo4, geo-smooth, shiftV,
shiftV+gate, rad, rad+gate, geo+rad. Metric: pooled removed-energy fraction, selectivity vs true change, chance = rank/C.
## Findings (run `20260928-0309_dinov3L_test512`, test512) — H2 PASS (geoP), global subspaces entangled, H6 nuanced
Fraction of residual energy removed (null pairs at dose idx 3; real pairs):

| set | geo (r2) | geo4 | **geoP (r8)** | geo-smooth | shiftV (r32) | shiftV+gate | rad (r56) | rad+gate | geo+rad | geoP+rad |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 shift 1 px | 0.374 | 0.544 | **0.889** | 0.288 | 0.232 | 0.162 | 0.215 | 0.150 | 0.504 | 0.904 |
| G2 field 1 px RMS | 0.229 | 0.324 | **0.572** | 0.011 | 0.286 | 0.215 | 0.257 | 0.178 | 0.420 | 0.666 |
| G3 parallax 2 px | 0.217 | 0.305 | **0.454** | 0.040 | 0.212 | 0.062 | 0.186 | 0.059 | 0.358 | 0.544 |
| C1 compound | 0.104 | 0.172 | 0.386 | 0.011 | 0.203 | 0.149 | 0.284 | 0.218 | 0.355 | 0.554 |
| R1–R7 radiometric | 0.04–0.09 | – | 0.17–0.35 | ≤0.02 | 0.13–0.34 | 0.08–0.28 | **0.29–0.49** | 0.21–0.40 | 0.35–0.51 | 0.51–0.62 |
| **true change** | 0.007 | 0.014 | **0.069** | 0.001 | **0.352** | 0.237 | **0.300** | 0.236 | 0.306 | 0.357 |
| real unchanged | 0.020 | 0.041 | 0.122 | 0.002 | 0.257 | 0.186 | 0.296 | 0.224 | 0.309 | 0.384 |

1. **H2 PASS for geoP:** 88.9 % of 1-px misregistration energy removed, 12.9× more than true-change energy (val: 12.4×).
   The proposal's tangent (geo) is more selective (50×) but removes only 37 %.
2. **A single global subspace cannot represent misregistration:** shiftV (NFAD construction on shift interventions,
   rank 32) removes **more true-change energy (35 %) than misregistration energy (23 %)** — selectivity < 1. Per-patch
   locality is necessary, not decorative (the scientific core of O3).
3. **NFAD's radiometric transfer is strongly entangled on RS residuals:** rad (rank 56 = 5.5 % of dims) removes 30 % of
   true-change energy vs 29–49 % of radiometric nuisance energy (selectivity 1.0–1.6). NFAD's "nuisance and defect share
   directions" finding reproduces — here even more strongly. Expect weak H5.
4. **H6 nuanced:** at 16-px patch scale misregistration energy is *spatially spread* (φ = 0.69 G1, 0.63 G2; γ ≈ 0.99), so the
   gate does not back off — the global subspace fails through entanglement, not through the gate. Only parallax is
   spatially concentrated (φ = 0.22) and there the gate does back off (γ = 0.44, removal 21 % → 6 %), exactly H6's
   mechanism. Real pairs are spread (φ = 0.95).
5. Real unchanged residuals lose 12 % of energy to geoP vs 6.9 % for true change → LEVIR's "unchanged" pairs contain
   real residual misregistration that geoP recognises (1.8× selectivity) — a first hint of value on real data.
6. geo-smooth (one global LS translation from tangents) is useless for fields (1 %) and weak for shifts (29 %):
   consequence of the non-linear response (E01).


## Addendum — per-patch radiometric probes (run `20260929-2100_dinov3L_test192_radP`, 192 tiles)
| energy removed (dose idx 3) | haze | colour | noise | JPEG | blur | G1 1 px | **true change** | real unchanged |
|---|---|---|---|---|---|---|---|---|
| rad (global, rank 56) | 0.49 | 0.43 | 0.38 | 0.30 | 0.42 | 0.22 | **0.300** | 0.290 |
| **radP** (per-patch, rank 4, 9 probes) | 0.55 | 0.51 | 0.29 | 0.32 | 0.98* | 0.20 | **0.105** | 0.119 |
| geoP+rad | 0.60 | 0.54 | 0.57 | 0.53 | 0.62 | 0.90 | 0.360 | 0.381 |
| **geoP+radP** (joint per-patch, rank 12) | 0.67 | 0.60 | 0.45 | 0.49 | 0.995* | 0.88 | **0.154** | 0.205 |
→ The locality principle also holds for radiometry: a 4-dim per-patch subspace from the reference's own radiometric
responses removes *more* haze/colour energy than NFAD's 56-dim global subspace while removing ~3× *less* real-change
energy (selectivity for haze 5.2× vs 1.6×). \*Home-turf caveat: the radP probes include the tested families (blur
is itself a probe → 98 % removal); real data (E05/E08) is the arbiter.
