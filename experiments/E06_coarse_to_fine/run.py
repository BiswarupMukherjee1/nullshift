"""E06 — Coarse-to-fine operator on the real arena, motivated by a MEASUREMENT (not by E05 AUROCs): real Maxar
near-null pairs are misregistered by several pixels (median 3.3 px pre-fire, 8-9 px post-fire, up to 22 px; only
~10 % have a confident global phase-correlation peak -> displacement is spatially varying). geoP's probes cover
+-2 px only. Candidates: localmin1 (3x3 token search), localmin+geoP (search, then probe projection of the matched
neighbour), geoPms (probe rings up to 8 px, rank 16), localmin+geoPms.
Anti-overfitting: MXR-LA-v1 tiles are split by a hash of the quadkey into dev / heldout BEFORE any E06 result is
seen; conclusions are drawn on heldout. (E05 had already exposed all tiles, so the split limits, not removes, leakage.)

Original E05 docstring follows.
E05 — REAL-ARENA test on MXR-LA-v1 (Maxar LA wildfire acquisitions): does each method separate real destruction
(pre vs post inside burn perimeters) from real acquisition nuisance (two different acquisitions days apart)?

Addresses two limitations of synthetic null pairs: (1) LEVIR's multi-year pairs hide nuisance -> here null pairs are days apart;
(2) synthetic null pairs build the operator from the image that is perturbed -> here t1 and t2 are different
acquisitions (satellite, off-nadir, sun, date) and the operator only ever sees t1.
Nothing is tuned here: eps, probe rank, subspaces and LEVIR thresholds are frozen from LEVIR val.
Metrics: patch- and window-level AUROC(change vs null), TPR at 5 % null false-change rate, nuisance-to-change ratio,
FCR of real nulls at the LEVIR-calibrated thresholds, null score vs off-nadir difference, scene-bootstrap CIs.
Usage: PYTHONPATH=src python experiments/E06_coarse_to_fine/run.py --e03-run <E03 run dir> [--encoder dinov3_vitb16_web] [--dataset MXR-TX-v1]
"""
import argparse
import json
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
from PIL import Image  # noqa: E402

from nullshift import metrics as M  # noqa: E402
from nullshift.encoder import Encoder  # noqa: E402
from nullshift.pipeline import ALL_METHODS, Engine  # noqa: E402
from nullshift.tracking import Run  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--tag", default="dinov3L_mxrla_c2f")
p.add_argument("--methods", default="raw,geoP,localmin,localmin1,localmin+geoP,geoPms,localmin+geoPms,blur+geoP,geoP+radP,classic")
p.add_argument("--e03-run", required=True, help="E03 run with LEVIR-val thresholds (for transfer FCR)")
p.add_argument("--eps", type=float, default=1.0)
p.add_argument("--subspaces", default="dinov3_vitl16_sat_L-1_kf8_ks16.pt")
p.add_argument("--encoder", default="dinov3_vitl16_sat")
p.add_argument("--dataset", default="MXR-LA-v1", help="Data_Created/<dataset> with manifest.csv + images/")
p.add_argument("--extra-methods", default="")
p.add_argument("--experiment", default="E06_coarse_to_fine", help="experiment folder to write the run into")
p.add_argument("--batch", type=int, default=16)
p.add_argument("--boot", type=int, default=2000)
a = p.parse_args()
methods = a.methods.split(",")
D = paths.CREATED / a.dataset
man = pd.read_csv(D / "manifest.csv")
tau = json.loads((Path(a.e03_run) / "thresholds.json").read_text())
import hashlib  # noqa: E402
man["split"] = ["heldout" if int(hashlib.sha256(str(q).encode()).hexdigest(), 16) % 2 else "dev" for q in man.quadkey]
run = Run(a.experiment, a.tag, dict(vars(a), methods=methods, n_pairs=len(man),
                                              counts=man.label.value_counts().to_dict()))
enc = Encoder(a.encoder)
NEED_S = {"rad", "geo+rad", "geoP+rad", "shiftV", "shiftV+gate", "rad+gate", "classic+ns", "c2f+rad"}
S = ({k: v.cuda() for k, v in torch.load(paths.NSB / "subspaces" / a.subspaces).items() if k.startswith("V_")}
     if NEED_S & set(methods) else {})
tau = tau if a.encoder == "dinov3_vitl16_sat" else {}      # LEVIR thresholds only valid for the ViT-L/sat encoder
eng = Engine(enc, S, eps=a.eps)


def load(pid, t):
    return torch.from_numpy(np.asarray(Image.open(D / "images" / f"{pid}_{t}.png").convert("RGB"))).permute(2, 0, 1).float() / 255


scores = {m: np.zeros((len(man), 16, 16), np.float32) for m in methods}
for i in range(0, len(man), a.batch):
    rr = man.iloc[i:i + a.batch]
    x1 = torch.stack([load(p_, "t1") for p_ in rr.pair_id]).cuda()
    x2 = torch.stack([load(p_, "t2") for p_ in rr.pair_id]).cuda()
    sc = eng.scores(eng.reference(x1, methods), x2, methods)
    for m in methods:
        scores[m][i:i + len(rr)] = sc[m].cpu().numpy()
    run.log(f"{min(i + a.batch, len(man))}/{len(man)} pairs")
np.savez_compressed(run.dir / "patch_scores.npz", **{m.replace("+", "_plus_"): v for m, v in scores.items()})

lab = man.label.to_numpy()
is_chg = lab == "change"
nulls = ["null_pre", "null_post", "null_long"]


def auroc(pos, neg):
    P, N = M.Hist(), M.Hist()
    P.add(torch.from_numpy(np.asarray(pos, np.float32))); N.add(torch.from_numpy(np.asarray(neg, np.float32)))
    return M.auroc(P, N)


def tpr_at_fcr(pos, neg, fcr=0.05):
    t = np.quantile(neg, 1 - fcr)
    return float((pos > t).mean())


rows = []
for m in methods:
    s = scores[m]
    win_mean = s.reshape(len(man), -1).mean(1)
    win_p90 = np.quantile(s.reshape(len(man), -1), 0.9, axis=1)
    chg_patch = s[is_chg].ravel()
    rec = {"method": m}
    for nt in nulls + ["all_null"]:
        msk = np.isin(lab, nulls) if nt == "all_null" else lab == nt
        if msk.sum() == 0:
            continue
        rec[f"AUROC_patch/{nt}"] = auroc(chg_patch, s[msk].ravel())
        rec[f"AUROC_win_mean/{nt}"] = auroc(win_mean[is_chg], win_mean[msk])
        rec[f"AUROC_win_p90/{nt}"] = auroc(win_p90[is_chg], win_p90[msk])
        rec[f"TPR@5%FCR_win/{nt}"] = tpr_at_fcr(win_mean[is_chg], win_mean[msk])
        rec[f"TPR@5%FCR_patch/{nt}"] = tpr_at_fcr(chg_patch, s[msk].ravel())
        rec[f"NCR/{nt}"] = float(np.median(s[msk]) / np.median(chg_patch))
        if m in tau:
            rec[f"FCR_levirTauStar/{nt}"] = float((s[msk] >= tau[m]["patch_tau"]).mean())
            rec[f"FCR_levirR99/{nt}"] = float((s[msk] >= tau[m]["r99_tau"]).mean())
    rows.append(rec)
tab = pd.DataFrame(rows).set_index("method")
tab.to_csv(run.dir / "table_mxr.csv")
key = ["AUROC_patch/all_null", "AUROC_win_mean/all_null", "TPR@5%FCR_win/all_null", "TPR@5%FCR_patch/all_null",
       "NCR/all_null", "AUROC_patch/null_post", "AUROC_patch/null_pre", "FCR_levirR99/all_null"]
run.log("E05 real arena (change vs real near-null):\n" + tab[[k for k in key if k in tab.columns]].round(4).to_string())
for m, r in tab.iterrows():
    for k in key:
        if k in r and pd.notna(r[k]):
            run.metric(f"{k}/{m}", float(r[k]))

# ---- dev / heldout split (decided before results)
srows = []
for sp in ["dev", "heldout"]:
    msk_sp = (man.split == sp).to_numpy()
    for m in methods:
        s_ = scores[m][msk_sp]; l_ = lab[msk_sp]
        wm = s_.reshape(len(s_), -1).mean(1)
        c_, n_ = l_ == "change", np.isin(l_, nulls)
        if c_.sum() == 0 or n_.sum() == 0:
            continue
        srows.append({"split": sp, "method": m, "n_change": int(c_.sum()), "n_null": int(n_.sum()),
                      "AUROC_patch": auroc(s_[c_].ravel(), s_[n_].ravel()), "AUROC_win_mean": auroc(wm[c_], wm[n_]),
                      "TPR@5%FCR_patch": tpr_at_fcr(s_[c_].ravel(), s_[n_].ravel()),
                      "NCR": float(np.median(s_[n_]) / np.median(s_[c_]))})
sp_tab = pd.DataFrame(srows)
sp_tab.to_csv(run.dir / "table_split.csv", index=False)
run.log("per split (conclusions on heldout):\n" + sp_tab.round(4).to_string())
for r in sp_tab.itertuples():
    run.metric(f"{r.split}/AUROC_patch/{r.method}", float(r.AUROC_patch))

# ---- paired scene-bootstrap (resample quadkeys) of AUROC_patch/all_null differences vs raw and vs classic
rng = np.random.default_rng(0)
qks = man.quadkey.astype(str).to_numpy()
uq = np.unique(qks)
groups = {q: np.nonzero(qks == q)[0] for q in uq}
cmp = [m for m in methods if m != "raw"]
boot = {m: [] for m in ["raw"] + cmp}
for _ in range(a.boot):
    idx = np.concatenate([groups[q] for q in rng.choice(uq, len(uq), replace=True)])
    l_ = lab[idx]
    if (l_ == "change").sum() == 0 or np.isin(l_, nulls).sum() == 0:
        continue
    for m in boot:
        if m not in scores:
            continue
        s = scores[m][idx].reshape(len(idx), -1).mean(1)       # window-level (patch AUROC too costly per draw)
        boot[m].append(auroc(s[l_ == "change"], s[np.isin(l_, nulls)]))
ci = []
for m in cmp:
    if m not in scores or not boot[m]:
        continue
    d = np.array(boot[m]) - np.array(boot["raw"])
    ci.append({"method": m, "AUROC_win_mean": tab.loc[m, "AUROC_win_mean/all_null"],
               "diff_vs_raw": tab.loc[m, "AUROC_win_mean/all_null"] - tab.loc["raw", "AUROC_win_mean/all_null"],
               "ci_lo": np.quantile(d, .025), "ci_hi": np.quantile(d, .975), "P(diff>0)": float((d > 0).mean())})
ci = pd.DataFrame(ci)
ci.to_csv(run.dir / "table_bootstrap_vs_raw.csv", index=False)
run.log("window-AUROC difference vs raw, 95% quadkey-bootstrap CI:\n" + ci.round(4).to_string())

# ---- null score vs off-nadir difference (does the operator remove parallax-type nuisance?)
nm = np.isin(lab, nulls)
bins = [0, 5, 10, 20, 40]
man["don_bin"] = pd.cut(man.d_offnadir, bins, include_lowest=True).astype(str)
strat = []
for m in methods:
    if m not in scores:
        continue
    med = scores[m].reshape(len(man), -1).mean(1)
    chg_med = np.median(scores[m][is_chg])
    for bname, g in man[nm].groupby("don_bin"):
        strat.append({"method": m, "d_offnadir_bin": bname, "n": len(g), "mean_null_score": float(med[g.index].mean()),
                      "NCR": float(np.median(scores[m][g.index]) / chg_med),
                      "cross_sensor_frac": float(g.cross_sensor.mean())})
st = pd.DataFrame(strat)
st.to_csv(run.dir / "table_null_vs_offnadir.csv", index=False)
run.log("null-window NCR by off-nadir difference:\n" + st.pivot_table(index="d_offnadir_bin", columns="method", values="NCR").round(3).to_string())

# ---- figures
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
for m in methods:
    if m not in scores:
        continue
    s = scores[m].reshape(len(man), -1).mean(1)
    pos, neg = s[is_chg], s[nm]
    th = np.unique(np.concatenate([pos, neg]))[::-1]
    tpr = [(pos >= t).mean() for t in th]; fpr = [(neg >= t).mean() for t in th]
    axs[0].plot(fpr, tpr, label=f"{m} ({tab.loc[m, 'AUROC_win_mean/all_null']:.3f})",
                lw=2.5 if "localmin+" in m else 1.2)
axs[0].plot([0, 1], [0, 1], "k:", lw=.8); axs[0].set_xlabel("false-change rate (real near-null windows)")
axs[0].set_ylabel("detection rate (burned windows)"); axs[0].legend(fontsize=7); axs[0].grid(alpha=.3)
axs[0].set_title("E06 window-level ROC, Maxar LA (real pairs)")
pv = st.pivot_table(index="d_offnadir_bin", columns="method", values="NCR")
pv.plot(ax=axs[1], marker="o"); axs[1].set_ylabel("NCR (median null / median change)"); axs[1].grid(alpha=.3)
axs[1].set_title("real-null nuisance vs off-nadir difference (lower = better)")
run.figure(fig, "fig_e06_roc_offnadir.png")
run.close()
