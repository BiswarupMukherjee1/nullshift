"""Paired quadkey-cluster bootstrap of patch-level AUROC and NCR differences for a real-pair run (E05/E06/E08/E09).
Resamples map tiles (quadkeys) with replacement; both methods of a pair are evaluated on the SAME resample.
Usage: python scripts/paired_bootstrap.py <run_dir> <dataset> "a:b,a:b,..." [B] [null_labels e.g. null_long]
Writes <run_dir>/posthoc_paired_bootstrap.csv and prints a table. Optional landscape-matched control if
Data_Created/<dataset>/manifest_fire_flags.csv exists."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
run, ds, pairs = Path(sys.argv[1]), sys.argv[2], [tuple(p.split(":")) for p in sys.argv[3].split(",")]
B = int(sys.argv[4]) if len(sys.argv) > 4 else 400
NULLS = sys.argv[5].split(",") if len(sys.argv) > 5 else None     # restrict negatives to these null labels
z = np.load(run / "patch_scores.npz")
man = pd.read_csv(ROOT / "Data_Created" / ds / "manifest.csv")
key = lambda m: m.replace("+", "_plus_")
chg = (man.label == "change").to_numpy()
if NULLS:                                    # keep change + selected null types only
    keep = (man.label == "change") | man.label.isin(NULLS)
    man = man[keep].reset_index(drop=True)
    z = {k: z[k][keep.to_numpy()] for k in z.files}
    chg = (man.label == "change").to_numpy()


def pauc(m, idx, nm=None):
    s = z[key(m)][idx]; l = chg[idx]; nm = (~l) if nm is None else nm[idx]
    pos, neg = s[l].ravel(), s[nm].ravel()
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


def ncr(m, idx):
    s = z[key(m)][idx]
    return np.median(s[~chg[idx]]) / np.median(s[chg[idx]])


qk = man.quadkey.astype(str).to_numpy(); uq = np.unique(qk); G = {q: np.nonzero(qk == q)[0] for q in uq}
rng = np.random.default_rng(1)
res = {p: ([], []) for p in pairs}
for _ in range(B):
    idx = np.concatenate([G[q] for q in rng.choice(uq, len(uq))])
    if chg[idx].sum() == 0 or (~chg[idx]).sum() == 0:
        continue
    for a_, b_ in pairs:
        res[(a_, b_)][0].append(pauc(a_, idx) - pauc(b_, idx))
        res[(a_, b_)][1].append(ncr(a_, idx) - ncr(b_, idx))
al = np.arange(len(man))
rows = []
for (a_, b_), (da, dn) in res.items():
    da, dn = np.array(da), np.array(dn)
    rows.append({"a": a_, "b": b_, "AUROC_a": pauc(a_, al), "AUROC_b": pauc(b_, al), "dAUROC": pauc(a_, al) - pauc(b_, al),
                 "dAUROC_lo": np.quantile(da, .025), "dAUROC_hi": np.quantile(da, .975),
                 "dNCR": ncr(a_, al) - ncr(b_, al), "dNCR_lo": np.quantile(dn, .025), "dNCR_hi": np.quantile(dn, .975),
                 "n_quadkeys": len(uq), "n_change_quadkeys": int(man[chg].quadkey.nunique())})
out = pd.DataFrame(rows)
out["negatives"] = ",".join(NULLS) if NULLS else "all_null"
out.to_csv(run / f"posthoc_paired_bootstrap_{'_'.join(NULLS) if NULLS else 'all'}.csv", index=False)
print(out.round(4).to_string())
