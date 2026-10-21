"""E00 — Sanity gate before any experiment.

1. Encoder facts (prefix/register tokens, embed dim, grid, normalisation constants, RoPE or absolute pos-embed).
2. Determinism: f(x) computed alone vs inside different batch compositions -> numerical noise floor of residuals
   (fp16 + cuDNN algorithm choice can make an identity null pair non-zero; that floor must be << nuisance effects).
3. Fourier shift by 0 and by an integer on real crops; J finite; projection removes <= rank/C energy of random vectors.
4. Throughput (images/s) and peak GPU memory for the configured batch size.
5. Zero-shot raw baseline on 256 clean val crops (patch AUROC / best F1) — the reference quality level.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from nullshift import paths  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from nullshift import metrics as M  # noqa: E402
from nullshift import operators as op  # noqa: E402
from nullshift.data import LevirScenes  # noqa: E402
from nullshift.encoder import Encoder  # noqa: E402
from nullshift.nuisance import center_crop, fourier_shift  # noqa: E402
from nullshift.tracking import Run  # noqa: E402

run = Run("E00_sanity", "dinov3L", {"encoder": "dinov3_vitl16_sat", "layer": -1})
enc = Encoder()
mdl = enc.model
facts = {"timm_name": mdl.pretrained_cfg.get("architecture"), "embed_dim": enc.dim, "n_blocks": enc.n_blocks,
         "patch": enc.patch, "num_prefix_tokens": getattr(mdl, "num_prefix_tokens", None),
         "num_reg_tokens": getattr(mdl, "num_reg_tokens", None), "mean": mdl.pretrained_cfg["mean"],
         "std": mdl.pretrained_cfg["std"], "has_abs_pos_embed": getattr(mdl, "pos_embed", None) is not None,
         "rope": type(getattr(mdl, "rope", None)).__name__, "params_M": sum(p.numel() for p in mdl.parameters()) / 1e6}
run.save_json("encoder_facts.json", facts)
run.log(f"encoder facts: {facts}")

fit = pd.read_csv(paths.NSB / "subset_val256_fit.csv")
ds = LevirScenes("val", scenes=sorted(fit.scene.unique())[:16])
rows = fit[fit.scene.isin(ds.scenes)].iloc[:32]
b = ds.batch(rows, "cuda")
x = center_crop(b.b)

# 2. determinism / noise floor
z_all = enc(x, batch=32)
z_b8 = enc(x, batch=8)
z_one = torch.cat([enc(x[i:i + 1]) for i in range(4)])
z_rev = enc(x.flip(0), batch=32).flip(0)
floor = {"max_abs_b32_vs_b8": float((z_all - z_b8).abs().max()),
         "mean_rnorm_b32_vs_b8": float((z_all - z_b8).norm(dim=1).mean()),
         "mean_rnorm_b32_vs_single": float((z_all[:4] - z_one).norm(dim=1).mean()),
         "mean_rnorm_b32_vs_reversed_order": float((z_all - z_rev).norm(dim=1).mean()),
         "repeat_identical_call_max_abs": float((enc(x, batch=32) - z_all).abs().max())}
for k, v in floor.items():
    run.metric(f"noise_floor/{k}", v)
run.log(f"noise floor: {floor}")

# 3. shifts, tangents, projection
s0 = fourier_shift(b.b, torch.zeros(len(rows), 2, device="cuda"))
run.metric("fourier_shift0_maxabs", float((s0 - b.b).abs().max()))
s1 = fourier_shift(b.b, torch.tensor([[0.0, 1.0]], device="cuda").expand(len(rows), 2))
run.metric("fourier_shift_int_vs_roll_center_maxabs",
           float((center_crop(s1) - center_crop(torch.roll(b.b, 1, dims=-1))).abs().max()))
J = op.shift_tangents(enc, b.b[:8], 1.0)
run.metric("J_finite", float(torch.isfinite(J).all()))
run.metric("J_mean_col_norm", float(J.norm(dim=1).mean()))
rnd = torch.randn(8, enc.dim, 16, 16, device="cuda")
rp, _ = op.project_tangent(rnd, J)
run.metric("geo_removed_frac_random_vectors", float(1 - (rp ** 2).sum() / (rnd ** 2).sum()))
run.metric("geo_chance_level", 2 / enc.dim)

# 4. throughput
torch.cuda.reset_peak_memory_stats()
xx = x.repeat(4, 1, 1, 1)[:128]
for bs in (16, 32, 64):
    torch.cuda.synchronize(); t = time.time()
    enc(xx, batch=bs)
    torch.cuda.synchronize()
    run.metric(f"throughput_img_per_s_bs{bs}", 128 / (time.time() - t))
run.metric("peak_gpu_mem_GB", torch.cuda.max_memory_allocated() / 1e9)

# 5. zero-shot raw on clean val (subset)
P, N = M.Hist(), M.Hist()
v = LevirScenes("val", scenes=sorted(fit.scene.unique()))
for i in range(0, len(fit), 32):
    bb = v.batch(fit.iloc[i:i + 32], "cuda")
    s = op.score(enc(center_crop(bb.b)) - enc(center_crop(bb.a)))
    pos, neg = M.patch_masks(M.patch_fraction(center_crop(bb.label)))
    P.add(s, pos); N.add(s, neg)
t, f1 = M.best_f1(P, N)
run.metric("raw_val256_patch_AUROC", M.auroc(P, N))
run.metric("raw_val256_best_patch_F1", f1)
run.metric("raw_val256_tau", float(M.EDGES[t]))
run.metric("n_pos_patches", P.n); run.metric("n_neg_patches", N.n)
run.log(f"metrics: {run.metrics}")
run.close()
