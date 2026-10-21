# Results by experiment

Every number below comes from a file in [`results/`](../results). Run commands are in
[`scripts/reproduce.sh`](../scripts/reproduce.sh).

**Encoders** (frozen; patch tokens of the last block, L2-normalised, 16 × 16 tokens per 256² tile):
- ViT-L: DINOv3 ViT-L/16 SAT-493M (303 M parameters), the main backbone;
- ViT-B: DINOv3 ViT-B/16 LVD-1689M (web data), the second backbone.

**Selection.** Thresholds and operator settings are chosen on LEVIR-CD val or on the dev tiles of the real-pair
benchmarks only.

---

## Synthetic null pairs (LEVIR-CD)

### E00 — Pipeline sanity
**Question.** Is the pipeline numerically correct?

**Result.**
- Identity null pairs score exactly 0.
- Batch-size effects are below 1e-7.
- The Fourier sub-pixel shift matches integer `roll` to 3e-16.
- The projection removes chance-level energy from random vectors.
- Throughput is ~90 images/s (ViT-L, 256², fp16, 2.4 GB VRAM).
- Zero-shot raw patch AUROC on LEVIR val is 0.880.

### E01 — Is the token response linear in sub-pixel shift?
**Setup.** Finite-difference Jacobian with step ε; relative error ‖Δz − J·d‖ / ‖Δz‖ for shifts of 0.25–4 px; 256 val
tiles. **Pre-set rejection criterion:** relative error > 0.6 at 1 px.

**Result.**
- Rejected: relative error 0.805 at 1 px (best ε = 1 px).
- Token displacement grows like √shift: 0.089 / 0.156 / 0.198 / 0.273 / 0.408 at 0.25 / 0.5 / 1 / 2 / 4 px.
- File: `synthetic/E01_linearity.csv`, `synthetic/E01_token_displacement.csv`.

### E01b — Per-patch probe subspace
**Setup.** 24 probe shifts (8 directions × 0.5 / 1 / 2 px) of the reference; per-patch SVD basis, rank r. Evaluated on
held-out random-direction shifts.

**Result.** At rank 8 the basis removes 89 % of held-out 1-px misregistration energy and 7 % of real-change energy (val).
Rank 8 is the knee of this trade-off.

### E02 — Selectivity: nuisance energy removed vs real-change energy removed
**Result** (test, 512 tiles; `synthetic/E02_energy_removed*.csv`):
- `geoP` removes 88.9 % of misregistration energy vs 6.9 % of real-change energy (12.9×).
- The first-order tangent removes 37 % vs 0.7 %.
- A *global* shift subspace removes 35 % of real-change energy but only 23 % of misregistration energy: it is entangled.
- The NFAD-style incidence gate only backs off for parallax (φ 0.22), not for shifts or smooth warps (φ ≈ 0.65).
- The global radiometric subspace removes 30 % of real-change energy vs 29–49 % of radiometric nuisance, so it is
  entangled as well.
- The per-patch radiometric probes (`radP`) are more selective.

### E03 — Null-pair dose response
**Setup.** 512 test tiles × 11 families × 5 strengths. Metrics: NCR, FCR at the val thresholds (F1-optimal, 90 % and
99 % recall). Files: `synthetic/E03_*.csv`, figure 2.

**Result.**
- `geoP` lowers the geometric NCR by 39 % (0.140 → 0.086) and FCR@R99 5.4× with no clean cost (patch F1 0.409 → 0.415).
- Specialists win their own family: phase correlation on a global shift (NCR 0.056), blur on smooth warps (0.128).
- Classical preprocessing + NullShift (`classic+ns`) is best on average (geometric 0.079, radiometric 0.151) and on the
  compound family (0.230 vs 0.312).
- **Two-sided variant** (operator built from an independently perturbed copy): −32 % geometric NCR instead of −39 %.

### E03b — Tile-bootstrap confidence intervals for E03
**Result** (`synthetic/E03b_bootstrap_FCR_R99.csv`):
- FCR reductions vs raw are significant for the compound family, smooth warps, sensor noise (`geoP+rad` −0.019
  [−0.028, −0.011]), blur and colour.
- At mid geometric strengths every method's FCR@R99 is ≈ 0. On LEVIR the synthetic geometric nuisance never reaches a
  99 %-recall threshold, because multi-year real change is much larger. This is the main limitation of synthetic null
  pairs and the reason for E05–E09.

### E04 — Real LEVIR change with nuisance added to t2
**Result** (`synthetic/E04_perturbed_pairs.csv`).
- Geometric nuisance barely affects detection of real LEVIR change (≤ 1 pt).
- Under strong sensor noise (dose 5) patch AUROC is:

| method | patch AUROC |
|---|---|
| raw | 0.740 |
| `geoP+rad` | 0.831 |
| 3×3 token search | 0.692 (search is fragile under noise) |

---

## Real acquisition pairs (Maxar Open Data)

### E05 — First real-pair evaluation (LA, ViT-L, all tiles)
**Result** (`real/E05_LA_vitl.csv`, all near-null negatives):
- Raw AUROC 0.841, NCR 0.731. Real nuisance is ~5× larger relative to real change than in the synthetic doses.
- Per-patch operators help modestly: `geoP` 0.856, `blur+geoP` 0.863.
- Global subspaces do not help (`rad` 0.839, `shiftV` 0.852).
- 3×3 token search is best (0.896) → this led to measuring the misregistration, which turned out to be multi-pixel.

### E06 — Coarse-to-fine operator, two backbones
**Setup.** 3×3 token search + per-patch probe subspace of the matched patch (40 probes: 8 directions ×
{0.5, 1, 2, 4, 8} px, rank 16). Dev / held-out split by quadkey hash before any E06 result. Files:
`real/E06_LA_*_dev_heldout.csv`, `real/E06_LA_*_bootstrap.txt`, figure 4.

**Result** (held-out): see README §4.4. Paired bootstrap vs token search: ViT-L +0.009 [+0.004, +0.014] AUROC,
ViT-B +0.011 [+0.006, +0.017]; NCR −0.08 on both.

**Landscape-matched control** (near-null windows within 2 km of the perimeters): every method loses ~2.5 pt and the
ranking is unchanged.

### E07 — Cost and design ablation
**Result** (held-out; `real/E07_ablation_*.csv`):

| variant (probes / rank / search radius) | extra passes per pair | ViT-L AUROC | ViT-B AUROC |
|---|---|---|---|
| raw | 0 | 0.835 | 0.930 |
| token search only (radius 1 / 2) | 0 | 0.891 / 0.897 | 0.956 / 0.959 |
| probes only (40 / 16) | 40 | 0.881 | 0.955 |
| c2f 40 / 16 / 1 | 40 | 0.908 | 0.967 |
| **c2f 16 / 8 / 1** | **16** | **0.904** | **0.965** |
| c2f 8 / 4 / 1 | 8 | 0.899 | 0.962 |
| c2f 40 / 16 / 2 | 40 | 0.910 | 0.968 |

→ 16 probes at rank 8 keep almost all of the gain, and a 5×5 search adds ≤ 0.2 pt.

### E08 — Adding radiometric probes; robustness to strong noise
**Result.**
- LA held-out AUROC: per-patch radiometric probes add a little on ViT-L (`c2f` 0.903 → `c2f+radP` 0.910) and nothing on
  ViT-B (0.965 → 0.965). The global radiometric subspace does not help on real pairs (`c2f+rad` 0.902).
- Under strong synthetic noise (LEVIR, dose 5), search-based operators drop (`c2f` 0.706 vs raw 0.747), and adding the
  global radiometric subspace restores them (`c2f+rad` 0.819).
- Files: `real/E08_*`, `synthetic/E08_perturbed_pairs.csv`.

### E09 — Second event (Texas 2024), design frozen beforehand
**Result:** README §4.4 (`real/E09_TX_*`).
- The pre-set criterion is met on ViT-L.
- The NCR reduction vs token search is significant in all four settings (backbone × negatives).
- The AUROC gain is significant in one of four.
- Measured misregistration is small and mostly global (median 1.5 px same-period, 3.6 px long-gap, `real/misregistration_TX.csv`).
  So token search adds little and the per-patch projection carries the gain (ViT-B: `geoP` 0.935 vs token search 0.929).
- Against same-gap negatives (autumn 2023 vs March 2024 rangeland) every method stays near 0.82 AUROC: seasonal change is
  not modelled by any operator.
