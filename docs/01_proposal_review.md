# 01 — Review of the NullShift proposal (28 Sep 2026)

Reviewed: `PROPOSALS_28_09/nullshift_proposal.md` and `PROPOSALS_28_09/proposal_ext_issues.txt`.
Method: every load-bearing citation was re-opened on arXiv/HF; every technical claim was checked against
the actual model/data; open design questions were resolved into concrete protocol decisions (§2).

---

## 1. Citation verification ("trust but verify")

| Claim in proposal | Checked where | Verdict |
|---|---|---|
| NFAD, Cao et al., arXiv 2608.29112 (29 Aug 2026), 91.0 % AUROC AeBAD-S | arXiv abs + HTML full text | **Correct.** Details the proposal omits, and which we must match: features are **L2-normalised**, taken from **intermediate blocks {2,4,6,8}** of DINOv3 ViT-B/16; subspace = top-16 eigvecs of the **uncentred** second moment **per family**, QR-stacked → rank 32; projection `z − VVᵀz` applied to features (≡ projecting the residual, by linearity). |
| NFAD names geometric interventions as future work; *view* domain weakest | NFAD §5.2 / Conclusion | **Correct.** "extending the intervention family to geometric transformations that capture viewpoint changes"; View domain 86.0 % AUROC because "geometric and parallax variation lies outside the photometric and background families". |
| NFAD: nuisance and defect entangled in direction, separated by spatial incidence | NFAD §3, Table 1, gate formula | **Correct.** Gate: per-column participation ratio ρ selects a "global" basis `V_g` (ρ > q25); per-image gate `γ(x)=min{1, φ(x)/τ}`, φ = participation ratio of residual-energy map. We re-implement exactly this (no code released). |
| SCDistill, Qu/Liang/Fu, 2607.19000 | arXiv abs | **Exists** (ECCV 2026, code released). Abstract confirms diffusion perturbation simulation + self-distillation. "DINOv3", "SECOND/HRSCD", "~40 GPU-h" **not verifiable from abstract** → soften in text until full paper is read. |
| FAF-CD, 2606.03114 | arXiv abs | **Exists** (Wang, Makrogiannis, Kambhamettu). Pseudo-change stress tests confirmed; perturbation list/severity 1–5 is from the supplement (not verified). |
| SEED, 2601.07805 | arXiv abs + search snippet | **Exists** (Dong et al.). Misregistration discussion not in abstract; a search snippet indicates the PDF discusses "small global shifts after preprocessing". Keep claim but cite section after reading PDF. |
| DRIFT (from issues file), 2606.06918 | arXiv abs | **Exists and is relevant.** Frozen VFM; *trained* lightweight projection heads split robust/fragile subspaces; patch-wise drift for localisation. Must be cited and differentiated (ours: closed-form, training-free, per-patch *local* tangent subspace). |
| Tomanič et al. benchmark, 2608.28247 | arXiv abs | **Exists**; checkpoints public (repo URL still to be located → E6). |
| DINOv3 ViT-L/16 SAT-493M in timm | HF `timm/vit_large_patch16_dinov3.sat493m` | **Exists, not gated**, 303.1 M params, 256², DINOv3 licence. |
| Hardware "RTX 4000 (20 GB)" | `nvidia-smi` | **Wrong for this machine:** RTX 3060 **12 GB**, 20 CPU cores, 62 GB RAM. ViT-L fp16 at batch 16–32 at 256² fits. |
| Storage "30–40 GB" | user constraint | Superseded: ≤ 9 GB total project folder (user-approved), ≤ 10 GB hard cap. |

Not re-opened (lower priority, keep as "cited, unverified" until read): FAR-Copula-CD (TIP), HyperLUCID, systematic review 18(15):2573 (search confirms it exists on MDPI), Adversarial SCD, SAAF, CPEN, DiffRegCD.

**Additional related work found during review (to cite):**
- Simard, LeCun, Denker (NeurIPS 1992/93) *Tangent Distance / Tangent Prop* — the correct precedent for "project a difference off a transformation's tangent space" (issues file §1 is right).
- Mercier & Inglada (2008) *Change detection with misregistration errors* — classical RS treatment of exactly our nuisance.
- Azulay & Weiss (JMLR 2019), Zhang (BlurPool, ICML 2019) — deep features are not shift-invariant; motivates H1 and its risk.
- Darcet et al. (ICLR 2024) *ViTs need registers* — DINOv3 has 4 register tokens; we only use patch tokens.
- Canty & Nielsen (IRMAD, 2008), Schroeder et al. (PIF normalisation, 2006) — operational incumbent for radiometry.
- "Morphing Through Time" arXiv 2511.07976 — diffusion-based alignment for CD (learned alternative to our geometric operator).
- Zero-OVCD (2608.11663), ChangeDINO (2511.16322), AdaDINO (2608.07982) — DINOv3-based CD; none addresses nuisance attribution.

No paper found (2 targeted searches) that builds a per-patch, finite-difference shift-tangent projection on frozen foundation-model residuals for bitemporal RS CD. Novelty claim for O3 stands **conditional on** explicit positioning vs Tangent Distance and DRIFT.

---

**Added 29 Sep 2026 (second literature pass, after the method changed to a probe subspace):**
- **SPARC**, Han et al., arXiv 2608.18585 (Aug 2026): closed-form *per-cell* subspace projection on frozen patch features,
  built from k ≤ 8 aligned normal images; states that it *requires spatial alignment* → closest in form, complementary in
  target (it cannot handle misregistration; we target exactly that). Must be cited.
- **SubspaceAD** (CVPR 2026): PCA normal subspace of DINOv2 patch features (training-free AD).
- **PatchShift** (arXiv 2510.13643) and **DVT** (Yang et al., ECCV 2024): aggregate/denoise features over *shifted
  views* → the "shift averaging" baseline (`shiftavg`) is now mandatory.
- **Bruzzone & Cossu (TGRS 2003), Marchesi, Bovolo & Bruzzone (TIP 2010)**: registration noise modelled as a region of
  the spectral change-vector space — the direct RS ancestors of the NullShift idea (pixel-spectral, statistical).
Full positioning: `docs/05_novelty_assessment.md`.

## 2. Technical issues in the proposal (and resolutions)

**T1. Position-embedding worry (issues file) is mis-aimed for DINOv3.** DINOv3 uses **RoPE** attached to the fixed token grid; shifting image *content* does not move positional encodings. The real source of non-smoothness is the stride-16 **patchify grid** and the attention/MLP nonlinearities. The diagnostic the issues file proposes (compare J at ε ∈ {0.25, 0.5} vs {1, 2} px) is still exactly right and becomes part of E1 ("ε-consistency").

**T2. Sub-pixel shift via Fourier phase ramp wraps around.** A circular shift of a 256² crop injects wrapped content at the borders → spurious "change" in border tokens. **Fix:** shift a crop with a 32-px margin cut from the 1024² original, then crop the centre 256². Fourier shift also rings at sharp edges; same margin handles the border; interior ringing is part of the "resampling" nuisance and is shared by J-estimation and the null pair (noted as an in-distribution advantage; see T6).

**T3. Resolution of the change map.** ViT/16 at 256² gives a **16×16** token grid (16 px = 8 m at LEVIR's 0.5 m). Many LEVIR buildings are smaller than one token. Zero-shot pixel-F1 will be low regardless of nuisance. **Decision:** report *patch-level* metrics (primary, resolution-matched) and upsampled pixel-level metrics (secondary). FCR on null pairs is defined at patch level (every patch is a true negative).

**T4. Threshold protocol must be per-method.** Projection removes energy everywhere, so a shared threshold is unfair. **Decision:** each method (raw, +geo, +rad, baselines) gets its own threshold τ\* maximising patch-F1 on **clean LEVIR val**; FCR and test F1 are reported at that τ\*. Also report threshold-free **AUROC(null-pair patches vs true-change patches)**, which directly asks "can the score separate instrument change from real change?"

**T5. Free per-patch projection may be too permissive.** Removing span(J(p)) lets every patch pick its own 2-D displacement. A smoother, more constrained variant is natural: fit a **global/low-order displacement field** d(p) by least squares across patches (feature-space Lucas–Kanade), then subtract J(p)d(p). **Added as variant `geo-smooth`**; `geo-free` is the proposal's operator. This also gives a feature-space analogue of the registration baseline.

**T6. Null pairs are J's home turf.** In `I2 = shift(I1, d)` the tangent is estimated on I1 with the same interpolator → optimistic. **Mitigations in protocol:** (a) J estimated with Fourier shift, while displacement-field and parallax null pairs use **bicubic warping** (interpolator mismatch); (b) compound nuisances (shift + radiometric + independent sensor noise on both dates); (c) symmetric J from I1 and I2 (rank 4) as ablation; (d) real misregistration (S2Looking, E5) remains the decisive test.

**T7. Radiometric removal: the classical incumbent will win on global radiometry.** Per-band histogram matching / robust linear regression (PIF/IRMAD-lite) will almost perfectly undo *global* gain/gamma on a null pair. The feature-space subspace can only claim value for (i) spatially varying radiometry (haze gradient), (ii) nuisances that pixel normalisation cannot express (colour-matrix cross-talk, JPEG, MTF blur, noise), and (iii) mixtures where changed pixels bias pixel-space regression. **Baselines added:** `histmatch`, `rrn` (robust per-band linear regression = PIF-style relative radiometric normalisation).

**T8. H6 needs a precise test.** NFAD's V contains no shift directions, so "the gate fails for misregistration" is only meaningful if we also build a **global subspace from shift interventions** (`nfad-geo-global`) and apply NFAD's gate to it. **Decision:** E2 compares energy selectivity of (a) global V from shifts, (b) global V + incidence gate, (c) per-patch J. This is the core evidence that locality is necessary.

**T9. Evaluation of the radiometric subspace must be out-of-sample.** Fit V on **LEVIR val** images with fixed "fit" doses; evaluate on **LEVIR test** at the dose grid; plus a **leave-one-family-out** check (does V fitted without haze remove haze?).

**T10. Nuisance families in the proposal are under-specified ("six families").** Final list (dose grids in `configs/nuisances.yaml`):
geometric — G1 global sub-pixel shift, G2 smooth random displacement field, G3 building-parallax proxy;
radiometric — R1 gain, R2 gamma/tone curve, R3 Rayleigh-weighted haze, R4 cross-sensor 3×3 colour matrix, R5 JPEG, R6 sensor noise, R7 MTF blur.
Plus a compound family C1 (G2 + R4 + R6) for realism.

**T11. Feature layer and normalisation.** NFAD uses L2-normalised intermediate features. **Decision:** main = final-block patch tokens, L2-normalised (cosine geometry); ablation over blocks {6, 12, 18, 24} of ViT-L later. J is always estimated on the *same* feature transform.

**T12. Budget realism.** Proposal's cost statement is right in spirit: 6 forwards/pair. Measured throughput will be recorded in E0. Dose-response uses a fixed 512-crop subset of LEVIR test (all 128 test scenes × 4 crops, fixed seed); clean F1 uses all 2048 test crops.

**T13. Supervised plug-in (E6) needs LEVIR train.** Deferred to phase 2 with its own storage check (the 256² train parquet is 241 MB; check whether it is JPEG-compressed before use).

---

## 3. Assessment of the external-issues file

| Point | Verdict |
|---|---|
| 1. Tangent Distance is the right precedent | **Agree.** Reframe O3 as "tangent-distance on frozen FM tokens, per patch, bitemporal, finite-difference". |
| 2. DRIFT is closest scoop risk | **Agree** (verified). Differentiator: training-free, closed-form, local geometric tangent. |
| 3. Shift-invariance literature, registers | **Agree.** Registers are dropped (patch tokens only); note in methods. |
| 4. Classical normalisation incumbent | **Agree — most important for industry.** Added `histmatch`, `rrn`, `phasecorr` baselines. |
| Position-embedding risk | **Partly wrong** (RoPE, see T1), diagnostic kept. |

## 4. Verdict

The idea is sound, cheap and falsifiable, and the E1 → E2 → E3 go/no-go chain is well ordered. The main weaknesses are (a) the protocol leaves thresholds, resolution and splits undefined (fixed above), (b) null pairs favour the method (mitigated, not removed, until E5), and (c) the classical baselines are missing (added). Proceed with Phase 1 = E0–E4 on LEVIR-CD, then decide on E5/E6 from the results.
