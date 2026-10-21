"""E03b — Scene-level bootstrap confidence intervals for the headline E03 comparisons.

E03 pools histograms over tiles, which gives exact point estimates but no uncertainty. Here the key methods are re-run
on the same NSB-v1 test512 null pairs (same seeds, same thresholds from the E03 run) keeping PER-TILE statistics:
FCR_R99 (fraction of the tile's 256 patches above tau_R99) and the tile's median score. 95 % CIs by resampling the
128 test SCENES with replacement (tiles of a scene are correlated), 2000 bootstrap draws. Paired differences vs raw and
vs classic are bootstrapped jointly (same resample) so CIs of differences are valid.
Usage: PYTHONPATH=src python experiments/E03b_bootstrap_ci/run.py --e03-run <dir> --eps 1.0
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from nullshift import paths  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from nullshift import metrics as M  # noqa: E402
from nullshift.data import LevirScenes  # noqa: E402
from nullshift.encoder import Encoder  # noqa: E402
from nullshift.nuisance import apply_nuisance, center_crop, nuisance_seed  # noqa: E402
from nullshift.pipeline import Engine  # noqa: E402
from nullshift.tracking import Run  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--tag", default="dinov3L_main")
p.add_argument("--e03-run", required=True)
p.add_argument("--eps", type=float, default=1.0)
p.add_argument("--subspaces", default="dinov3_vitl16_sat_L-1_kf8_ks16.pt")
p.add_argument("--methods", default="raw,geo,geoP,rad,geoP+rad,phasecorr,classic,classic+ns")
p.add_argument("--conds", default="G1_shift:2,G1_shift:3,G2_field:3,G2_field:4,G3_parallax:3,G3_parallax:4,"
                                  "C1_compound:2,C1_compound:3,R3_haze:3,R4_colour:3,R5_jpeg:3,R6_noise:3,R7_blur:3")
p.add_argument("--boot", type=int, default=2000)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--run-dir", default=None, help="resume an interrupted run in place (per-tile rows are checkpointed)")
a = p.parse_args()
methods = a.methods.split(",")
conds = [(c.split(":")[0], int(c.split(":")[1])) for c in a.conds.split(",")]
tau = json.loads((Path(a.e03_run) / "thresholds.json").read_text())
run = Run("E03b_bootstrap_ci", a.tag, dict(vars(a), methods=methods, conds=conds), resume_dir=a.run_dir)
PART = run.dir / "per_tile_partial.csv"
enc = Encoder()
S = {k: v.cuda() for k, v in torch.load(paths.NSB / "subspaces" / a.subspaces).items() if k.startswith("V_")}
eng = Engine(enc, S, eps=a.eps)
sub = pd.read_csv(paths.NSB / "subset_test512.csv")
ds = LevirScenes("test", scenes=sorted(sub.scene.unique()))
rows = pd.read_csv(PART).to_dict("records") if (a.run_dir and PART.exists()) else []
done_ids = {r["crop_id"] for r in rows}
start = 0
while start < len(sub) and set(sub.iloc[start:start + a.batch].crop_id) <= done_ids and len(done_ids):
    start += a.batch
if start:
    run.log(f"resumed from tile {start}")
for i in range(start, len(sub), a.batch):
    rr = sub.iloc[i:i + a.batch]
    b = ds.batch(rr, "cuda")
    g3_ok = center_crop(b.label).float().mean(dim=(1, 2)) >= 0.01
    base = eng.reference(b.b, methods)
    for fam, d in conds:
        seeds = [nuisance_seed(c, fam, d) for c in b.ids]
        x1, x2 = apply_nuisance(b.b, fam, d, seeds, b.label)
        ref = eng.reference(x1, methods) if fam == "C1_compound" else base
        sc = eng.scores(ref, x2, methods)
        for m, s in sc.items():
            thr = M.EDGES[tau[m]["r99_bin"]]
            fcr = (s >= thr).float().flatten(1).mean(1).cpu().numpy()
            med = s.flatten(1).median(1).values.cpu().numpy()
            for j, r in enumerate(rr.itertuples()):
                if fam == "G3_parallax" and not bool(g3_ok[j]):
                    continue
                rows.append({"scene": r.scene, "crop_id": r.crop_id, "cond": f"{fam}:{d}", "method": m,
                             "fcr_r99": float(fcr[j]), "median_score": float(med[j])})
    run.log(f"{min(i + a.batch, len(sub))}/{len(sub)}")
    pd.DataFrame(rows).to_csv(PART, index=False)          # checkpoint every batch (reboot-safe)
df = pd.DataFrame(rows)
df.to_csv(run.dir / "per_tile.csv.gz", index=False)

rng = np.random.default_rng(0)
scenes = np.array(sorted(df.scene.unique()))
out = []
for cond, g in df.groupby("cond"):
    piv = g.pivot_table(index=["scene", "crop_id"], columns="method", values="fcr_r99")
    by_scene = piv.groupby(level=0).agg(["sum", "count"])
    sc_list = [s for s in scenes if s in by_scene.index]
    idx = rng.integers(0, len(sc_list), size=(a.boot, len(sc_list)))
    for m in methods:
        sums = by_scene[(m, "sum")].loc[sc_list].to_numpy()
        cnts = by_scene[(m, "count")].loc[sc_list].to_numpy()
        est = sums.sum() / cnts.sum()
        bs = sums[idx].sum(1) / cnts[idx].sum(1)
        rec = {"cond": cond, "method": m, "FCR_R99": est, "ci_lo": np.quantile(bs, .025), "ci_hi": np.quantile(bs, .975)}
        for refm in ["raw", "classic"]:
            if refm in methods and refm != m:
                rs = by_scene[(refm, "sum")].loc[sc_list].to_numpy()
                diff = bs - rs[idx].sum(1) / cnts[idx].sum(1)
                rec[f"diff_vs_{refm}"] = est - rs.sum() / cnts.sum()
                rec[f"diff_vs_{refm}_lo"] = np.quantile(diff, .025)
                rec[f"diff_vs_{refm}_hi"] = np.quantile(diff, .975)
        out.append(rec)
ci = pd.DataFrame(out)
ci.to_csv(run.dir / "table_bootstrap_fcr_r99.csv", index=False)
run.log("FCR_R99 with 95% scene-bootstrap CI (and paired differences):\n" + ci.round(4).to_string())
run.close()
