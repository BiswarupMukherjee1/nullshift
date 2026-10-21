"""E01 — Linearity of frozen-encoder patch features in sub-pixel translation (H1) + epsilon-consistency diagnostic.

For clean val images x: dz(d) = f(shift(x, d u)) - f(x) vs first-order prediction J_eps (d u), J_eps from central
differences with step eps. Also: fraction of dz energy left after the geo projection (what the operator cannot remove),
consistency of span(J_eps) across eps (issues-file diagnostic), stratification by patch gradient energy.
Runs on LEVIR val (it selects eps*). Usage: PYTHONPATH=src nullshift-venv/bin/python experiments/E01_linearity/run.py
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from nullshift import paths  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from nullshift import operators as op  # noqa: E402
from nullshift.data import LevirScenes  # noqa: E402
from nullshift.encoder import Encoder  # noqa: E402
from nullshift.metrics import patch_fraction  # noqa: E402
from nullshift.nuisance import center_crop, fourier_shift  # noqa: E402
from nullshift.tracking import Run  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--tag", default="dinov3L_val256")
p.add_argument("--encoder", default="dinov3_vitl16_sat")
p.add_argument("--layer", type=int, default=-1)
p.add_argument("--n", type=int, default=256)
p.add_argument("--batch", type=int, default=16)
a = p.parse_args()

EPS = [0.25, 0.5, 1.0, 2.0]
DS = [0.25, 0.5, 1.0, 2.0, 4.0]
DIRS = {"x": (0.0, 1.0), "y": (1.0, 0.0), "diag": (1 / math.sqrt(2), 1 / math.sqrt(2))}
STRATA = ["flat", "mid", "edge"]
cfg = dict(vars(a), eps=EPS, d=DS, dirs=list(DIRS), split="val", subset="subset_val256_fit.csv",
           kill_criterion="rel_err(J d vs dz) > 0.6 at d=1 px for best eps")
run = Run("E01_linearity", a.tag, cfg)

enc = Encoder(a.encoder, a.layer)
fit = pd.read_csv(paths.NSB / "subset_val256_fit.csv").iloc[:a.n]
ds = LevirScenes("val", scenes=sorted(fit.scene.unique()))


def grad_energy(x):  # (B,3,H,W) centre crop -> (B,16,16) mean squared gradient of luminance
    g = (x * torch.tensor([0.299, 0.587, 0.114], device=x.device).view(1, 3, 1, 1)).sum(1, keepdim=True)
    gy, gx = g[..., 1:, :-1] - g[..., :-1, :-1], g[..., :-1, 1:] - g[..., :-1, :-1]
    e = F.pad(gy ** 2 + gx ** 2, (0, 1, 0, 1))
    return F.avg_pool2d(e, 16)[:, 0]


# pass 0: tercile cuts of gradient energy
ge_all = []
for i in range(0, len(fit), 64):
    b = ds.batch(fit.iloc[i:i + 64], "cuda")
    ge_all.append(grad_energy(center_crop(b.b)).flatten().cpu())
cuts = torch.quantile(torch.cat(ge_all), torch.tensor([1 / 3, 2 / 3])).tolist()
run.log(f"gradient-energy tercile cuts: {cuts}")

acc = {}   # (eps, d, dir, stratum) -> sums


def add(key, **kv):
    s = acc.setdefault(key, {})
    for k, v in kv.items():
        s[k] = s.get(k, 0.0) + float(v)


consist = {e: [0.0, 0.0, 0] for e in EPS}          # sum cos^2 principal angles, sum norm ratio, count
sens = {d: [0.0, 0] for d in DS}                    # mean ||dz|| per patch
change_ref = {"changed": [0.0, 0], "unchanged": [0.0, 0]}

for i in range(0, len(fit), a.batch):
    b = ds.batch(fit.iloc[i:i + a.batch], "cuda")
    x = b.b
    B = x.shape[0]
    z0 = enc(center_crop(x))
    ge = grad_energy(center_crop(x))
    strat = torch.bucketize(ge, torch.tensor(cuts, device=ge.device))    # 0 flat,1 mid,2 edge
    J = {e: op.shift_tangents(enc, x, e) for e in EPS}
    # epsilon consistency vs eps = 1
    Q1 = torch.linalg.qr(op._flat(J[1.0]))[0]
    n1 = op._flat(J[1.0]).norm(dim=(1, 2))
    for e in EPS:
        Qe = torch.linalg.qr(op._flat(J[e]))[0]
        sv = torch.linalg.svdvals(Qe.transpose(1, 2) @ Q1)
        consist[e][0] += float((sv ** 2).mean(1).sum())
        consist[e][1] += float((op._flat(J[e]).norm(dim=(1, 2)) / n1.clamp_min(1e-12)).sum())
        consist[e][2] += sv.shape[0]
    # true-change reference magnitude (clean val pair A->B)
    za = enc(center_crop(b.a))
    rn = (z0 - za).norm(dim=1)
    fr = patch_fraction(center_crop(b.label))
    change_ref["changed"][0] += float(rn[fr >= 0.5].sum()); change_ref["changed"][1] += int((fr >= 0.5).sum())
    change_ref["unchanged"][0] += float(rn[fr == 0].sum()); change_ref["unchanged"][1] += int((fr == 0).sum())
    for dname, (uy, ux) in DIRS.items():
        for d in DS:
            dv = torch.tensor([[d * uy, d * ux]], device=x.device).expand(B, 2)
            dz = enc(center_crop(fourier_shift(x, dv).clamp(0, 1))) - z0
            e_dz = (dz ** 2).sum(1)
            if dname == "x":
                sens[d][0] += float(dz.norm(dim=1).sum()); sens[d][1] += dz.shape[0] * 256
            for e in EPS:
                pred = torch.einsum("bckhw,bk->bchw", J[e], dv)
                e_err = ((dz - pred) ** 2).sum(1)
                cos = F.cosine_similarity(dz, pred, dim=1)
                rp, _ = op.project_tangent(dz, J[e], 1e-3)
                e_res = (rp ** 2).sum(1)
                for si, sname in enumerate(STRATA + ["all"]):
                    m = (strat == si) if sname != "all" else torch.ones_like(strat, dtype=torch.bool)
                    add((e, d, dname, sname), err=e_err[m].sum(), den=e_dz[m].sum(), res=e_res[m].sum(),
                        cos=cos[m].sum(), n=m.sum())
    run.log(f"{i + B}/{len(fit)} crops")

rows = []
for (e, d, dname, s), v in acc.items():
    rows.append({"eps": e, "d_px": d, "dir": dname, "stratum": s, "rel_err": math.sqrt(v["err"] / v["den"]),
                 "mean_cos": v["cos"] / v["n"], "resid_after_geo": v["res"] / v["den"], "n_patches": int(v["n"])})
df = pd.DataFrame(rows).sort_values(["eps", "d_px", "dir", "stratum"])
df.to_csv(run.dir / "linearity_table.csv", index=False)
allrows = df[(df.stratum == "all")].groupby(["eps", "d_px"])[["rel_err", "mean_cos", "resid_after_geo"]].mean().reset_index()
allrows.to_csv(run.dir / "linearity_summary.csv", index=False)

# eps* = argmin mean rel_err over d <= 1 px
small = allrows[allrows.d_px <= 1.0].groupby("eps").rel_err.mean()
eps_star = float(small.idxmin())
rel1 = float(allrows[(allrows.eps == eps_star) & (allrows.d_px == 1.0)].rel_err.iloc[0])
res1 = float(allrows[(allrows.eps == eps_star) & (allrows.d_px == 1.0)].resid_after_geo.iloc[0])
run.metric("eps_star", eps_star)
run.metric("rel_err_at_1px", rel1)
run.metric("resid_after_geo_at_1px", res1)
run.metric("H1_pass", float(rel1 <= 0.6))
for e in EPS:
    run.metric(f"eps_consistency_cos2_eps{e}", consist[e][0] / consist[e][2])
    run.metric(f"eps_norm_ratio_eps{e}", consist[e][1] / consist[e][2])
for d in DS:
    run.metric(f"mean_dz_norm_d{d}", sens[d][0] / sens[d][1])
run.metric("mean_r_norm_true_change", change_ref["changed"][0] / max(change_ref["changed"][1], 1))
run.metric("mean_r_norm_unchanged_real", change_ref["unchanged"][0] / max(change_ref["unchanged"][1], 1))
run.log(f"eps*={eps_star}  rel_err@1px={rel1:.3f}  resid_after_geo@1px={res1:.3f}  H1 {'PASS' if rel1 <= 0.6 else 'FAIL'}")

# ---------------- figures
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for e in EPS:
    s = allrows[allrows.eps == e]
    ax[0].plot(s.d_px, s.rel_err, "o-", label=f"eps={e}")
    ax[1].plot(s.d_px, s.resid_after_geo, "o-", label=f"eps={e}")
ax[0].axhline(0.6, color="k", ls="--", lw=1, label="kill (0.6)")
for axx, t in zip(ax[:2], ["relative error ||dz - J d|| / ||dz||", "energy left after geo projection"]):
    axx.set_xscale("log", base=2); axx.set_xlabel("shift |d| (px)"); axx.set_title(t); axx.legend(fontsize=8)
    axx.grid(alpha=.3)
st = df[(df.eps == eps_star) & (df.stratum != "all")].groupby(["stratum", "d_px"]).rel_err.mean().unstack(0)
st[STRATA].plot(ax=ax[2], marker="o")
ax[2].set_xscale("log", base=2); ax[2].set_title(f"rel. error by patch gradient tercile (eps={eps_star})")
ax[2].grid(alpha=.3)
run.figure(fig, "fig_linearity.png")
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(DS, [sens[d][0] / sens[d][1] for d in DS], "o-", label="null pair: shift only")
ax.axhline(run.metrics["mean_r_norm_true_change"], color="C3", ls="--", label="real pair, changed patches")
ax.axhline(run.metrics["mean_r_norm_unchanged_real"], color="C2", ls=":", label="real pair, unchanged patches")
ax.set_xscale("log", base=2); ax.set_xlabel("shift |d| (px)"); ax.set_ylabel("mean ||r(p)||"); ax.legend(fontsize=8)
ax.set_title("feature sensitivity to misregistration"); ax.grid(alpha=.3)
run.figure(fig, "fig_sensitivity.png")
run.close()
