# NSB-v1 — NullShift null-pair benchmark (datasheet)

**What it is.** A benchmark of *null pairs*: a real VHR satellite tile `I` paired with a physically parameterised,
content-preserving perturbation of itself `T(I)`. The ground truth is **"no change anywhere"** by construction, so
every patch a change detector flags is a false change. **No human labelling is involved.**
A second arm (*perturbed pairs*) pairs a real bitemporal tile `A` with `T(B)`, reusing LEVIR-CD's existing change
masks only to check that real change is still detected.

**Why it is stored as manifests, not images.** Every pair is `(crop_id, family, dose_idx, seed)` and is regenerated
bit-exactly by `src/nullshift/nuisance.py::apply_nuisance` (deterministic per-sample `torch.Generator`). Storing the
28k perturbed test pairs as PNG would cost ~5 GB; the manifests cost < 3 MB.

## Source imagery
LEVIR-CD (Chen & Shi, *Remote Sensing* 12:1662, 2020): Google Earth, 0.5 m GSD, RGB, 1024×1024, Texas (US) cities,
2002–2018. Mirror: HF `satellite-image-deep-learning/LEVIR-CD` (`test.zip`, `val.zip`; SHA256 in `data/raw/SHA256SUMS.txt`).
Licence: LEVIR-CD is released for academic research use; Google Earth imagery terms apply. **Do not redistribute
imagery**; NSB-v1 manifests reference it by id only.

## Files
| File | Content |
|---|---|
| `crops_val.csv`, `crops_test.csv` | standard non-overlapping 4×4 grid of 256² tiles per scene (1024 val, 2048 test), with changed-pixel fraction |
| `subset_test512.csv` | dose-response subset: 4 tiles per test scene (128 scenes), seed 20260928; `g3_eligible` = ≥ 1 % building pixels |
| `subset_val256_fit.csv` | 4 tiles per val scene (64 scenes): subspace fitting, ε selection (never test) |
| `nullpairs_test512.csv` | null-pair manifest: crop × family × dose (1–5) × seed |
| `fit_val256.csv` | radiometric intervention manifest for subspace fitting (doses 2–4, `fit/`-prefixed seeds ≠ test seeds) |
| `nuisances.yaml` | families, dose grids, units, margin |
| `dose_stats_pixel.csv` | pixel-space strength of each (family, dose): mean |ΔI|, PSNR, fraction of pixels changed > 2 DN |
| `previews/<family>.png` | visual QA: T(I) and |T(I) − I| for 2 tiles × 6 doses |
| `subspaces/*.pt/json` | global subspaces fitted on val (V_rad, V_shift, gate bases) + fit report |

## Generation protocol
- Each 256² tile is served with a **64-px margin** of real scene context (reflect-padded only at scene borders);
  every nuisance acts on the padded tile and the centre 256² is evaluated → no wrap-around from Fourier shifts.
- Base image of a null pair = the **t2 image (B)** of the tile (it contains the labelled buildings needed by G3).
- Dose index 0 is the identity for every family (FCR ≡ 0 up to numerical noise; the noise floor is measured in E00).

| Family | Physical model | Doses (idx 1→5) |
|---|---|---|
| G1_shift | global sub-pixel translation, Fourier phase ramp, random direction | 0.25, 0.5, 1, 2, 4 px |
| G2_field | zero-mean smooth random displacement field (Gaussian-correlated, L = 32 px), bicubic backward warp — *residual after global registration* | RMS 0.25 … 4 px |
| G3_parallax | building-roof displacement ∝ random per-building height (U[0.5,1]) along one view azimuth, footprints from LEVIR B masks, σ = 1 px edge smoothing | max 0.5, 1, 2, 4, 8 px |
| R1_gain | global gain 1 ± δ | δ = 0.02 … 0.30 |
| R2_gamma | tone curve I^γ, γ = e^{±δ} | δ = 0.05 … 0.50 |
| R3_haze | I·t + A(1 − t), t_b = exp(−τ_b), τ_b = τ_g (λ_b/0.55 µm)^−4 (Rayleigh), spatial ramp 0.5–1.5× | τ_g = 0.02 … 0.30 |
| R4_colour | cross-sensor 3×3 band mixing I + δ·U(−1,1) | δ = 0.02 … 0.30 |
| R5_jpeg | JPEG re-compression on an 8×8 grid offset by a random (1–7, 1–7) px from the source grid | quality 90, 75, 50, 30, 15 |
| R6_noise | additive Gaussian sensor noise | σ = 0.005 … 0.08 |
| R7_blur | Gaussian PSF / MTF difference | σ = 0.3 … 2.0 px |
| C1_compound | G2 + R4 + R6 at the same dose index, **plus independent R6 noise on the reference** | idx 1 … 5 |

## Known limitations
- Parallax is a 2-D proxy (no occlusion/disocclusion modelling, no DSM); only *changed* (labelled) buildings move.
- Radiometric nuisances are global or smoothly varying; no cloud shadows, no BRDF/sun-angle effects.
- The t2 image is itself a Google Earth mosaic (already resampled and JPEG-compressed upstream). **Measured:** LEVIR-CD
  tiles carry a JPEG q≈75 history on the original 8×8 grid — grid-aligned re-encoding at q75 changes pixels by only
  0.0009 (mean |Δ|) vs 0.0216 with a 3-px grid offset. R5 therefore uses a random grid offset; aligned re-encoding
  would make the dose non-monotonic (observed in the first NSB build and fixed before any experiment).
- Null pairs favour methods whose tangent is estimated with the same interpolator (G1); G2/G3 use a different
  interpolator (bicubic) on purpose, and real misregistration (S2Looking, Phase 2) is the decisive test.
