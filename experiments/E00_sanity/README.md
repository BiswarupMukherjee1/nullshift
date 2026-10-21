# E00 — Sanity gate
**Question.** Is the pipeline numerically correct and how fast is it?
**Checks.** encoder facts (registers, RoPE vs abs-pos), residual noise floor across batch compositions (fp16/cuDNN),
Fourier shift identities on real crops, tangent finiteness, projection removes ≈ chance energy of random vectors,
throughput, zero-shot raw quality on clean val256.
**Pass.** noise floor ≪ smallest nuisance effect; shift(0) exact; geo removes ≈ 2/C of random energy.
**Status.** see `runs/` — findings below.
## Findings (run `20260928-0244_dinov3L`) — PASS
- Encoder: DINOv3 ViT-L/16 SAT-493M, 303 M params, 24 blocks, C = 1024, **5 prefix tokens (CLS + 4 registers)**,
  **RoPE (RotaryEmbeddingDinoV3), no absolute position embedding**, SAT normalisation mean (0.430, 0.411, 0.296).
  Weights: local fp16 copy (0.6 GB); vs fp32 weights the cosine-feature residual norm is 0.013 mean (fixed model
  difference, identical for both dates).
- Noise floor: batch 32 vs 8 → max |Δz| 9e-8; repeat call bit-identical; only batch-size-1 differs (‖Δz‖ 0.0016) —
  sweeps never use it. Identity null pairs are exactly zero.
- Fourier shift(0) and integer shift vs roll on real crops: 3e-16. Tangents finite; geo removes 0.00195 of random-vector
  energy = chance 2/1024.
- Throughput ≈ 90 img/s (ViT-L, 256², fp16), peak 2.4 GB VRAM.
- Zero-shot raw (val256): patch AUROC 0.880, best patch-F1 0.379 at **τ\* = 1.22** — a high threshold: real
  5–14-year LEVIR residuals are large (see E01), which motivated the extra high-recall operating points in E03.

