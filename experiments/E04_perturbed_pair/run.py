"""E04 — Perturbed real pairs (A, T(B)) (FAF-CD-style arm): does each method keep detecting REAL change when the
t2 acquisition carries a nuisance? Thresholds are re-used from an E03 run (clean-val tau*); nothing tuned on test.
Reports patch-F1 / pixel-F1 / AUROC per (family, dose, method) and the drop vs the same crops at dose 0.
Usage: PYTHONPATH=src nullshift-venv/bin/python experiments/E04_perturbed_pair/run.py --e03-run <E03 run dir> --eps <eps*>
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
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from nullshift import metrics as M  # noqa: E402
from nullshift.data import LevirScenes  # noqa: E402
from nullshift.encoder import Encoder  # noqa: E402
from nullshift.nuisance import FAMILIES  # noqa: E402
from nullshift.pipeline import ALL_METHODS, Engine  # noqa: E402
from nullshift.sweep import run_sweep, save_hists  # noqa: E402
from nullshift.tracking import Run  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--tag", default="dinov3L_main")
p.add_argument("--e03-run", required=True)
p.add_argument("--eps", type=float, required=True)
p.add_argument("--encoder", default="dinov3_vitl16_sat")
p.add_argument("--layer", type=int, default=-1)
p.add_argument("--subspaces", default="dinov3_vitl16_sat_L-1_kf8_ks16.pt")
p.add_argument("--tau-pr", type=float, default=0.5)
p.add_argument("--n", type=int, default=512)
p.add_argument("--doses", default="1,3,5")
p.add_argument("--families", default="all")
p.add_argument("--methods", default="all")
p.add_argument("--experiment", default="E04_perturbed_pair", help="experiment folder to write the run into")
p.add_argument("--batch", type=int, default=16)
p.add_argument("--run-dir", default=None, help="resume an interrupted run in place")
p.add_argument("--resume", action="store_true")
a = p.parse_args()
methods = ALL_METHODS if a.methods == "all" else a.methods.split(",")
doses = [int(x) for x in a.doses.split(",")]
e03 = Path(a.e03_run)
tau = json.loads((e03 / "thresholds.json").read_text())
run = Run(a.experiment, a.tag, dict(vars(a), methods=methods, doses=doses), resume_dir=a.run_dir)
torch.manual_seed(0)
enc = Encoder(a.encoder, a.layer)
S = {k: v.cuda() for k, v in torch.load(paths.NSB / "subspaces" / a.subspaces).items() if k.startswith("V_")}
eng = Engine(enc, S, eps=a.eps, tau_pr=a.tau_pr)

sub = pd.read_csv(paths.NSB / "subset_test512.csv").iloc[:a.n]
fams = list(FAMILIES) if a.families == "all" else a.families.split(",")
conds = [("clean", 0)] + [(f, d) for f in fams for d in doses]
H, ncrop = run_sweep(eng, LevirScenes("test"), sub, conds, methods, "perturbed", a.batch, run.log,
                    ckpt=run.dir / "ckpt_perturbed", resume=a.resume)
save_hists(H, run.dir / "hists_perturbed_test512.npz")
rows = []
for fam, d in conds:
    key = f"{fam}|{d}"
    for m in methods:
        if (key, m, "pos") not in H:
            continue
        P, N = H[(key, m, "pos")], H[(key, m, "neg")]
        nan = {"f1": float("nan"), "precision": float("nan"), "recall": float("nan")}
        rp = M.f1_at(P, N, tau[m]["patch_bin"]) if m in tau else nan
        rx = M.f1_at(H[(key, m, "pix_pos")], H[(key, m, "pix_neg")], tau[m]["pix_bin"]) if m in tau else nan
        rows.append({"family": fam, "dose_idx": d, "dose": 0 if fam == "clean" else FAMILIES[fam]["doses"][d],
                     "method": m, "patch_F1": rp["f1"], "patch_P": rp["precision"], "patch_R": rp["recall"],
                     "patch_AUROC": M.auroc(P, N), "pix_F1": rx["f1"], "FPR_unchanged": N.frac_above(tau[m]["patch_bin"]) if m in tau else float("nan")})
tab = pd.DataFrame(rows)
base = tab[tab.family == "clean"].set_index("method")
tab["dF1_patch"] = tab.patch_F1 - tab.method.map(base.patch_F1)
tab["dAUROC_patch"] = tab.patch_AUROC - tab.method.map(base.patch_AUROC)
tab.to_csv(run.dir / "table_perturbed.csv", index=False)
lb = tab[tab.family != "clean"].pivot_table(index="method", columns="family", values="dAUROC_patch", aggfunc="mean")
lb["mean_dAUROC"] = lb.mean(axis=1)   # threshold-free (new methods have no LEVIR tau)
lb = lb.join(base[["patch_F1", "patch_AUROC"]].rename(columns=lambda c: "clean_" + c))
lb.sort_values("mean_dAUROC", ascending=False).to_csv(run.dir / "table_leaderboard_dAUROC.csv")
run.log("mean patch-AUROC change vs clean (subset), per family:\n" + lb.round(3).to_string())
for m, r in lb.iterrows():
    run.metric(f"mean_dAUROC/{m}", float(r.mean_dAUROC)); run.metric(f"clean_F1_subset/{m}", float(r.clean_patch_F1))

SHOW = [m for m in ["raw", "geo", "rad", "geo+rad", "localmin", "phasecorr", "classic", "classic+ns"] if m in methods]
fig, axs = plt.subplots(3, 4, figsize=(20, 12))
for ax, fam in zip(axs.ravel(), fams):
    for m in SHOW:
        t = tab[(tab.method == m) & ((tab.family == fam) | (tab.family == "clean"))].sort_values("dose_idx")
        ax.plot(t.dose_idx, t.patch_AUROC, marker="o", ms=3, label=m, lw=2.2 if "geo" in m or m == "classic+ns" else 1.1)
    ax.set_title(fam, fontsize=9); ax.grid(alpha=.3); ax.set_xticks([0] + doses); ax.set_ylabel("patch AUROC")
axs.ravel()[-1].axis("off")
axs.ravel()[0].legend(fontsize=7)
fig.suptitle("E04 perturbed real pairs (A, T(B)): patch-F1 vs dose (higher is better)")
run.figure(fig, "fig_perturbed_f1.png")
run.close()
