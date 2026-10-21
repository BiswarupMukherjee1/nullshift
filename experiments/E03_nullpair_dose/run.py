"""E03 — Null-pair dose-response of false-change rate (FCR) for every method (H3, H5), with clean accuracy.

Stage 1  calibration  : clean LEVIR val (1024 crops) -> per-method thresholds tau* (patch-F1 and pixel-F1 optimal).
Stage 2  clean test   : clean LEVIR test (2048 crops, or --n-test) -> patch/pixel F1 at tau*, AUROC.
Stage 3  null pairs   : NSB-v1 test512 x 11 families x 5 doses -> FCR@tau*, AUROC(true change vs null).
Usage: PYTHONPATH=src python experiments/E03_nullpair_dose/run.py --eps <eps* from E01>
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

from nullshift import metrics as M  # noqa: E402
from nullshift.data import LevirScenes, crop_index  # noqa: E402
from nullshift.encoder import Encoder  # noqa: E402
from nullshift.nuisance import FAMILIES  # noqa: E402
from nullshift.pipeline import ALL_METHODS, Engine  # noqa: E402
from nullshift.sweep import conditions, run_sweep, save_hists  # noqa: E402
from nullshift.tracking import Run  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--tag", default="dinov3L_main")
p.add_argument("--encoder", default="dinov3_vitl16_sat")
p.add_argument("--layer", type=int, default=-1)
p.add_argument("--eps", type=float, required=True)
p.add_argument("--subspaces", default="dinov3_vitl16_sat_L-1_kf8_ks16.pt")
p.add_argument("--tau-pr", type=float, default=0.5)
p.add_argument("--n-val", type=int, default=1024)
p.add_argument("--n-test", type=int, default=2048)
p.add_argument("--n-null", type=int, default=512)
p.add_argument("--families", default="all")
p.add_argument("--methods", default="all")
p.add_argument("--batch", type=int, default=16)
p.add_argument("--reuse-calib", default=None, help="run dir whose stage-1/2 outputs (thresholds, clean hists) are reused")
p.add_argument("--resume", action="store_true", help="resume the null sweep from this run's checkpoint")
p.add_argument("--run-dir", default=None, help="existing run dir to resume into")
p.add_argument("--two-sided", action="store_true", help="null2 mode: both dates independently perturbed (E03c)")
p.add_argument("--doses", default="1,2,3,4,5")
a = p.parse_args()
methods = ALL_METHODS if a.methods == "all" else a.methods.split(",")
fams = list(FAMILIES) if a.families == "all" else a.families.split(",")
run = Run("E03_nullpair_dose", a.tag, resume_dir=a.run_dir, config=dict(vars(a), methods=methods, families=fams,
                                           threshold_protocol="per-method tau* = argmax patch-F1 on clean val",
                                           patch_label="pos: >=25% changed px, neg: 0%, mixed ignored"))
torch.manual_seed(0)
enc = Encoder(a.encoder, a.layer)
S = {k: v.cuda() for k, v in torch.load(paths.NSB / "subspaces" / a.subspaces).items() if k.startswith("V_")}
run.log("subspace ranks: " + str({k: tuple(v.shape) for k, v in S.items()}))
eng = Engine(enc, S, eps=a.eps, tau_pr=a.tau_pr)

import shutil  # noqa: E402
from nullshift.sweep import load_hists  # noqa: E402
dst = LevirScenes("test")
if a.reuse_calib:
    src = Path(a.reuse_calib)
    for f in ["thresholds.json", "hists_val_clean.npz", "hists_test_clean.npz", "table_clean_test.csv",
              "table_val_calibration.csv"]:
        if (src / f).resolve() != (run.dir / f).resolve():
            shutil.copy(src / f, run.dir / f)
    tau = json.loads((run.dir / "thresholds.json").read_text())
    Ht = load_hists(run.dir / "hists_test_clean.npz")
    clean_tab = pd.read_csv(run.dir / "table_clean_test.csv")
    run.log(f"reused calibration + clean test from {src}")
if not a.reuse_calib:
  # ---------------------------------------------------------------- stage 1: calibration on clean val
  val = crop_index("val").iloc[:a.n_val]
  Hv, _ = run_sweep(eng, LevirScenes("val"), val, [("clean", 0)], methods, "clean", a.batch, run.log)
  save_hists(Hv, run.dir / "hists_val_clean.npz")
  tau, rows = {}, []
  for m in methods:
      tp, f1p = M.best_f1(Hv[("clean|0", m, "pos")], Hv[("clean|0", m, "neg")])
      tx, f1x = M.best_f1(Hv[("clean|0", m, "pix_pos")], Hv[("clean|0", m, "pix_neg")])
      r90 = M.tau_at_recall(Hv[("clean|0", m, "pos")], 0.9)
      r99 = M.tau_at_recall(Hv[("clean|0", m, "pos")], 0.99)
      tau[m] = {"patch_bin": tp, "patch_tau": float(M.EDGES[tp]), "pix_bin": tx, "pix_tau": float(M.EDGES[tx]),
                "r90_bin": r90, "r90_tau": float(M.EDGES[r90]), "r99_bin": r99, "r99_tau": float(M.EDGES[r99])}
      rows.append({"method": m, "tau_patch": M.EDGES[tp], "val_patch_F1": f1p, "tau_R90": M.EDGES[r90],
                   "val_F1_at_R90": M.f1_at(Hv[("clean|0", m, "pos")], Hv[("clean|0", m, "neg")], r90)["f1"],
                   "tau_pix": M.EDGES[tx], "val_pix_F1": f1x,
                   "val_patch_AUROC": M.auroc(Hv[("clean|0", m, "pos")], Hv[("clean|0", m, "neg")])})
  run.save_json("thresholds.json", tau)
  pd.DataFrame(rows).to_csv(run.dir / "table_val_calibration.csv", index=False)
  run.log("calibration done:\n" + pd.DataFrame(rows).round(4).to_string())

  # ---------------------------------------------------------------- stage 2: clean test
  test = crop_index("test").iloc[:a.n_test]
  Ht, _ = run_sweep(eng, dst, test, [("clean", 0)], methods, "clean", a.batch, run.log)
  save_hists(Ht, run.dir / "hists_test_clean.npz")
  rows = []
  for m in methods:
      P, N = Ht[("clean|0", m, "pos")], Ht[("clean|0", m, "neg")]
      PX, NX = Ht[("clean|0", m, "pix_pos")], Ht[("clean|0", m, "pix_neg")]
      rp, rx = M.f1_at(P, N, tau[m]["patch_bin"]), M.f1_at(PX, NX, tau[m]["pix_bin"])
      rows.append({"method": m, "patch_F1": rp["f1"], "patch_P": rp["precision"], "patch_R": rp["recall"],
                   "patch_AUROC": M.auroc(P, N), "pix_F1": rx["f1"], "pix_IoU": rx["iou"], "pix_AUROC": M.auroc(PX, NX),
                   "FPR_real_unchanged_patches": N.frac_above(tau[m]["patch_bin"]),
                   "patch_F1_at_R90": M.f1_at(P, N, tau[m]["r90_bin"])["f1"],
                   "recall_at_R90": M.f1_at(P, N, tau[m]["r90_bin"])["recall"],
                   "FPR_real_unchanged_at_R90": N.frac_above(tau[m]["r90_bin"]),
                   "median_score_true_change": M.median(P), "median_score_real_unchanged": M.median(N)})
      run.metric(f"clean_patch_F1/{m}", rp["f1"]); run.metric(f"clean_pix_F1/{m}", rx["f1"])
  clean_tab = pd.DataFrame(rows)
  clean_tab.to_csv(run.dir / "table_clean_test.csv", index=False)
  run.log("clean test:\n" + clean_tab.round(4).to_string())

# ---------------------------------------------------------------- stage 3: null pairs
sub = pd.read_csv(paths.NSB / "subset_test512.csv").iloc[:a.n_null]
doses = [int(x) for x in a.doses.split(",")]
Hn, ncrop = run_sweep(eng, dst, sub, conditions(fams, doses=doses), methods, "null2" if a.two_sided else "null",
                      a.batch, run.log, ckpt=run.dir / "ckpt_null", resume=a.resume)
save_hists(Hn, run.dir / "hists_null_test512.npz")
run.save_json("null_crop_counts.json", ncrop)
rows = []
for (key, m, g), h in Hn.items():
    fam, d = key.split("|")
    d = int(d)
    dose = 0 if fam == "identity" else FAMILIES[fam]["doses"][d]
    rows.append({"family": fam, "dose_idx": d, "dose": dose, "method": m, "n_patches": h.n,
                 "FCR": h.frac_above(tau[m]["patch_bin"]), "FCR_R90": h.frac_above(tau[m]["r90_bin"]),
                 "FCR_R99": h.frac_above(tau[m]["r99_bin"]),
                 "AUROC_change_vs_null": M.auroc(Ht[("clean|0", m, "pos")], h),
                 "median_score": M.median(h), "NCR": M.median(h) / M.median(Ht[("clean|0", m, "pos")])})
null_tab = pd.DataFrame(rows).sort_values(["family", "dose_idx", "method"])
null_tab.to_csv(run.dir / "table_null_fcr.csv", index=False)

# leaderboards: mean over doses 1..5 per family, per method, for FCR@tau*, FCR@R90, NCR, AUROC
ct = clean_tab.set_index("method")[["patch_F1", "patch_F1_at_R90", "pix_F1", "patch_AUROC"]]
for val in ["FCR", "FCR_R90", "FCR_R99", "NCR", "AUROC_change_vs_null"]:
    lb = null_tab[null_tab.family != "identity"].pivot_table(index="method", columns="family", values=val, aggfunc="mean")
    lb["mean_geo"] = lb[[c for c in lb.columns if c.startswith("G")]].mean(1)
    lb["mean_rad"] = lb[[c for c in lb.columns if c.startswith("R")]].mean(1)
    lb = lb.copy().join(ct)
    lb.sort_values("mean_geo", ascending=(val != "AUROC_change_vs_null")).to_csv(run.dir / f"table_leaderboard_{val}.csv")
    run.log(f"leaderboard {val} (mean over doses 1-5):\n" + lb.round(4).to_string())
    for m, r in lb.iterrows():
        run.metric(f"{val}_geo/{m}", float(r.mean_geo)); run.metric(f"{val}_rad/{m}", float(r.mean_rad))

# ---------------------------------------------------------------- figures
SHOW = [m for m in ["raw", "geo", "geoP", "rad", "geoP+rad", "shiftV+gate", "localmin", "blur", "phasecorr",
                    "histmatch", "classic", "classic+ns"] if m in methods]
sty = {m: dict(lw=2.6 if m in ("geoP", "geoP+rad", "classic+ns") else 1.2,
               ls="-" if m in ("raw", "geo", "geoP", "geoP+rad", "rad", "shiftV+gate", "classic+ns") else "--")
       for m in SHOW}
nf = len(fams)
for val, ylab, fname, ylim in [("FCR_R99", "FCR @ tau_R99 (99% recall on clean val)", "fig_fcr_r99_dose.png", (-0.02, 1)),
                               ("FCR_R90", "FCR @ tau_R90 (90% recall on clean val)", "fig_fcr_r90_dose.png", (-0.02, 1)),
                               ("FCR", "FCR @ tau* (F1-optimal on clean val)", "fig_fcr_dose.png", (-0.02, 1)),
                               ("NCR", "median null score / median true-change score", "fig_ncr_dose.png", (0, 1.1)),
                               ("AUROC_change_vs_null", "AUROC true change vs null", "fig_auroc_dose.png", (0.5, 1.001))]:
    fig, axs = plt.subplots(3, 4, figsize=(20, 13))
    for ax, fam in zip(axs.ravel(), fams):
        s = null_tab[null_tab.family == fam]
        for m in SHOW:
            t = s[s.method == m].sort_values("dose_idx")
            ax.plot(t.dose_idx, t[val], marker="o", ms=3, label=m, **sty[m])
        ax.set_title(f"{fam}  ({FAMILIES[fam]['unit']})", fontsize=9)
        ax.set_xticks(range(1, 6)); ax.set_xticklabels([str(FAMILIES[fam]["doses"][i]) for i in range(1, 6)], fontsize=8)
        ax.set_ylim(*ylim); ax.grid(alpha=.3); ax.set_ylabel(ylab, fontsize=8)
    for ax in axs.ravel()[nf:]:
        ax.axis("off")
    axs.ravel()[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"E03 null-pair dose-response — {ylab}")
    run.figure(fig, fname)
    plt.close(fig)
run.close()
