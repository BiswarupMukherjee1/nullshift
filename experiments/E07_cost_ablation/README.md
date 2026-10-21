# E07 — Cost / design ablation of the coarse-to-fine operator (real arena MXR-LA-v1, two backbones)
**Question.** How many probe shifts (K), what rank r and what search radius are needed? (E06 used K = 40, r = 16, 3×3.)
**Protocol.** 40 probe responses computed once per reference; each variant uses a subset → variants differ only in
design. Selection rule fixed before running: on **dev** tiles, the cheapest variant within 0.5 pt patch-AUROC of the
best; report **heldout**. Runs: `20260929-2038_dinov3_vitl16_sat_mxrla`, `20260929-2046_dinov3_vitb16_web_mxrla`.

## Findings — 16 probes are enough (2.5× cheaper than E06), on both backbones
| variant (heldout patch AUROC / NCR) | extra passes | ViT-L sat | ViT-B web |
|---|---|---|---|
| raw | 0 | 0.835 / 0.738 | 0.930 / 0.587 |
| localmin1, 3×3 search | 0 | 0.892 / 0.693 | 0.956 / 0.548 |
| localmin1, 5×5 search | 0 | 0.898 / 0.692 | 0.960 / 0.545 |
| geoP (K 24, r 8, no search) | 24 | 0.849 / 0.682 | 0.940 / 0.531 |
| c2f K 8, r 4, 3×3 | 8 | 0.899 / 0.657 | 0.962 / 0.506 |
| **c2f K 16, r 8, 3×3** | **16** | **0.904 / 0.638** | **0.965 / 0.489** |
| c2f K 24, r 8, 5×5 | 24 | 0.907 / 0.638 | 0.967 / 0.489 |
| c2f K 40, r 16, 3×3 (= E06) | 40 | 0.908 / 0.622 | 0.967 / 0.471 |
| c2f K 40, r 16, 5×5 | 40 | 0.910 / 0.622 | 0.968 / 0.472 |
Selected by the dev rule: ViT-L → K 24/r 8/5×5; ViT-B → K 16/r 8/3×3. Recommendation: **K = 16 (rings 1 and 4 px),
r = 8, 3×3 search** — within ~0.4 pt of the best, 17 encoder passes per reference instead of 41.
- Even K = 8 beats every search-only baseline; larger rank lowers NCR further (more nuisance removed) at flat AUROC.
- A wider search (5×5) adds ~0.3 pt: real displacements up to 22 px sometimes exceed one token.
