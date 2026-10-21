# 02 — Work plan (Phase 1 = initial experiments)

Status legend: ☐ todo · ◐ running · ☑ done · ✗ killed

## 0. Constraints
- All files inside `nullshift_cd/`. venv: `nullshift-venv/`. HF/timm cache: `models/hf_cache/`. Temp: `.tmp/`.
- Storage: target ≤ 8 GB total, 10 GB HARD CAP → `scripts/storage_check.sh` after every data/model step.
- GPU: RTX 3060 12 GB. fp16 autocast for the encoder; operator maths in fp32.

## 1. Data (Phase 1)
| Asset | Source | Size | Use |
|---|---|---|---|
| LEVIR-CD **test** (128 × 1024² pairs, PNG) | HF `satellite-image-deep-learning/LEVIR-CD` `test.zip` | 0.50 GB | evaluation |
| LEVIR-CD **val** (64 × 1024² pairs) | same, `val.zip` | 0.25 GB | thresholds τ\*, radiometric subspace fit |
| DINOv3 ViT-L/16 SAT-493M | HF `timm/vit_large_patch16_dinov3.sat493m` | ~1.2 GB | main encoder |
| (Phase 2) S2Looking test, LEVIR train, DINOv2-B/DINOv3-B-web | — | decided later | E5–E7 |

**Created data (`Data_Created/`)** — the NullShift null-pair benchmark **NSB-v1**:
- `crops_{val,test}.csv`: deterministic 256² crop index (scene, y, x) on the 1024² originals, 32-px margin kept for geometric warps; checksums of source files.
- `nuisances.yaml`: the 10 nuisance families + compound, dose grids, fixed seeds (one seed per (crop, family, dose)).
- `nullpairs_test512.csv`: the 512-crop dose-response subset × family × dose with seeds → every null pair is regenerated bit-exactly on the fly (no need to store ~5 GB of images).
- `previews/`: PNG grid per family × dose for visual QA.
- `subspaces/`: fitted radiometric/global subspaces (KB-sized `.pt`).
- `DATASHEET.md`: provenance, licence, generation procedure.

## 2. Code (`src/nullshift/`)
`paths.py` · `data.py` (LEVIR loader, crops) · `nuisance.py` (all families, deterministic) · `encoder.py` (timm wrapper, patch tokens, L2 norm, layer selection) · `operators.py` (geo-free, geo-smooth, rad-global, nfad-gate, global-shift subspace) · `baselines.py` (phasecorr, localmin3x3, blur, histmatch, rrn) · `metrics.py` (FCR, patch/pixel F1, AUROC, energy selectivity) · `tracking.py` (MLflow + JSON) · `tests/` (unit tests for shifts, projections, metrics).

## 3. Experiments (Phase 1)
Folder convention: `experiments/E{NN}_{slug}/` → `README.md` (question, hypothesis, protocol, status, findings), `config.yaml`, `run.py`, `runs/{YYYYMMDD-HHMM}_{tag}/` (metrics.json, *.csv, *.png, log.txt). MLflow experiment = folder name, run = run tag (`mlops/mlflow.db`, UI: `mlflow ui --backend-store-uri sqlite:///mlops/mlflow.db`).

| ID | Name | Question | Pass / kill |
|---|---|---|---|
| E00 | `E00_sanity` | Pipeline correct? throughput? zero-shot clean baseline on val/test | exact-zero residual for identity; shift(0) = identity; timings logged |
| E01 | `E01_linearity` | H1: does J·d predict f(shift(I,d))−f(I)? ε-consistency of J (issues-file diagnostic) | **kill** if rel. error > 0.6 at 1 px → ✗ **FAILED** (0.805; block 12: 1.006) |
| E01b | `E01b_probe_subspace` | pre-registered fallback: per-patch *probe* (secant) subspace from K = 24 shifted copies, rank r | ☑ r = 8 selected on val: 89 % of held-out 1-px misregistration removed, 7 % of true change |
| E02 | `E02_selectivity` | H2/H6: energy removed on misregistration vs true-change residuals, for per-patch J vs global shift-subspace (± NFAD gate) vs global radiometric V | **kill** if nuisance/change removal ratio ≈ 1 |
| E03 | `E03_nullpair_dose` | H3/H5: FCR vs dose for 10 families + compound, all operators and baselines, per-method τ\* | geo operator must cut FCR on G1–G3; rad on R1–R7 |
| E04 | `E04_perturbed_pair` | H3: clean F1 cost and F1 on (I1, T(I2)) | clean patch-F1 drop ≤ 1 pt (kill > 3 pt) |

Phase 2 (decided after Phase 1 results): E05 S2Looking real parallax · E06 supervised plug-in (Siamese U-Net / light head) · E07 second backbone (DINOv2-B-reg / DINOv3-B) · E08 layer ablation · E09 radiometric per-patch tangents (unifying O2 and O3).

## 4. Operators compared (names used in all tables)
`raw` · `geoP` (**main geometric operator after E01**: per-patch probe subspace, K = 24, rank 8) · `geoP+rad` (joint, NullShift full) · `geo` (per-patch tangent J, rank 2 from I1 — the proposal's operator, kept for comparison) · `geo4` (rank 4, J from I1 and I2) · `geo-smooth` (LS global translation in feature space) · `rad` (NFAD-style global V, rank 32) · `geo+rad` (NullShift full) · `shiftV` (global subspace from shift interventions) · `shiftV+gate` (with NFAD incidence gate) · baselines `phasecorr`, `localmin`, `blur`, `histmatch`, `rrn`.

## 4b. Protocol additions made during Phase 1 (and why)
- **E01b + `geoP`** — H1 failed; the proposal's own fallback ("multi-scale shift basis") generalised to a probe subspace.
- **Operating points τ_R90 / τ_R99 and NCR** — E00/E01 showed real LEVIR residuals (multi-year) are ≈ 5× larger than
  1-px misregistration residuals, so FCR at the F1-optimal τ* is ≈ 0 for most doses; high-recall thresholds and the
  scale-free nuisance-to-change ratio keep the null-pair measurement informative.
- **R5 JPEG grid offset** — LEVIR carries a q≈75 JPEG history; aligned re-encoding was near-idempotent (non-monotonic dose).
- **Phase correlation Hann window** — without it the baseline failed on smooth scenes (unit test).
- **Baselines made fair (found in the first E03 calibration, run aborted and discarded):** on real multi-year LEVIR pairs
  (A–B correlation median 0.34) (i) unbounded phase correlation "registered" already co-registered pairs by a median
  8 px (up to 62 px) → now bounded to ≤ 4 px and gated by phase-correlation peak ≥ 0.15 (real pairs peak ≤ 0.12,
  genuine displacements ≥ 0.2) → 0 % spurious corrections, 100 % of G1/G2/C1 corrected; (ii) OLS-based RRN suffered
  regression dilution (slope 0.48, B contrast halved) → PIF + mean/std normalisation (contrast preserved 0.204 vs 0.205).
  Both failure modes are now unit-tested.

## 4c. Phase 1b (29 Sep 2026) — answers to the external critique
- Critique 1 "LEVIR is the wrong arena" (natural residual ≈ 5× a 1-px misregistration) — **agreed** (it was our own
  E00/E01 finding). → **E05 `E05_maxar_realpairs`** on the created real benchmark **MXR-LA-v1** (Maxar Open Data, LA
  wildfires): real near-null pairs days apart (different satellites/off-nadir/sun) vs real destruction (pre/post inside
  NIFC burn perimeters). No tuning on it.
- Critique 2 "operator built from the image that gets perturbed" — **agreed**. → **E03c** two-sided null pairs
  (t1 = T_a(I), t2 = T_b(I), independent draws) and, decisively, E05 (operator sees only t1; t2 is another acquisition).
- New mandatory baseline from the literature pass: **`shiftavg`** (test-time shift averaging, PatchShift/DVT-style) and
  `shiftavg+geoP`.
- Execution: two E03 runs were killed at logout (logind KillUserProcesses=yes) → `scripts/launch.sh` (systemd-run
  --user + linger), checkpoint/resume.

## 5. Order of work
1. ☑ env + storage check → 2. ☑ download LEVIR test/val + weights → 3. ☑ NSB-v1 creation + previews + datasheet → 4. ☑ code + unit tests (32) → 5. ☑ E00 → 6. ✗ E01 (H1 fails) → 6b. ☑ E01b fallback → 7. ☑ E02 (H2 passes for geoP) → 8. ◐ E03, E04 → 9. ☐ bug review of code + results → 10. ☐ independent self-review (`docs/04_independent_review.md`) → 11. ☐ results summary → decide Phase 2 with user.

## 6. Status (30 Sep 2026, 00:00) — Phase 1 + 1b complete
☑ E00 ☑ E01 (✘ H1) ☑ E01b ☑ E02 (+radP) ☑ E03 ☑ E03b ☑ E03c ☑ E04 ☑ E05 ☑ E06 (+B) ☑ E07 (+B) ☑ E08 ☑ E09 (+B).
**Proposed Phase 2** (decide with user): (1) non-DINO backbone (e.g. a satellite MAE / CLIP-type encoder) for the
generality claim; (2) third real source with labels — S2Looking test shard (~0.44 GB) or another Maxar event split by
event; (3) radiometric robustness of search (e.g. search on radiometrically normalised features, or noise-aware
matching); (4) paper draft (principle + protocol + measurement).
