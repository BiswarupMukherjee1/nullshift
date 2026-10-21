"""E07 — Cost / design ablation of the coarse-to-fine operator on the real arena (MXR-LA-v1).
The 40 probe responses (8 directions x rings {0.5,1,2,4,8} px) of each reference are computed ONCE; every variant
uses a subset of them, so variants differ only in design (probe rings K, rank r, search radius), not in noise.
Selection rule (fixed before running): on the DEV split choose the cheapest variant within 0.5 pt patch-AUROC of the
best; report it on HELDOUT. Cost = extra encoder passes of the reference (K) + 1 target pass.
Usage: PYTHONPATH=src python experiments/E07_cost_ablation/run.py [--encoder ...]
"""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from nullshift import paths  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from nullshift import operators as op  # noqa: E402
from nullshift.encoder import Encoder  # noqa: E402
from nullshift.nuisance import center_crop  # noqa: E402
from nullshift.tracking import Run  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--tag", default=None)
p.add_argument("--encoder", default="dinov3_vitl16_sat")
p.add_argument("--batch", type=int, default=8)
a = p.parse_args()
RINGS = (0.5, 1.0, 2.0, 4.0, 8.0)
DISPS = op.probe_displacements(8, RINGS)                    # 40 probes, ring-major order
ring_idx = {m: list(range(8 * i, 8 * i + 8)) for i, m in enumerate(RINGS)}
VARIANTS = {   # name: (rings, rank, search radius in tokens)
    "raw": (None, 0, 0), "localmin1_r1": (None, 0, 1), "localmin1_r2": (None, 0, 2),
    "geoP_K24_r8": ((0.5, 1, 2), 8, 0), "geoPms_K40_r16": (RINGS, 16, 0),
    "c2f_K40_r16_s1": (RINGS, 16, 1), "c2f_K40_r8_s1": (RINGS, 8, 1), "c2f_K40_r24_s1": (RINGS, 24, 1),
    "c2f_K16_r8_s1": ((1, 4), 8, 1), "c2f_K24_r8_s1": ((1, 2, 4), 8, 1), "c2f_K24_r12_s1": ((0.5, 2, 8), 12, 1),
    "c2f_K8_r4_s1": ((2,), 4, 1), "c2f_K40_r16_s2": (RINGS, 16, 2), "c2f_K24_r8_s2": ((1, 2, 4), 8, 2),
}
D_ = paths.CREATED / "MXR-LA-v1"
man = pd.read_csv(D_ / "manifest.csv")
man["split"] = ["heldout" if int(hashlib.sha256(str(q).encode()).hexdigest(), 16) % 2 else "dev" for q in man.quadkey]
run = Run("E07_cost_ablation", a.tag or f"{a.encoder}_mxrla", dict(vars(a), variants=VARIANTS))
enc = Encoder(a.encoder)


def load(pid, t):
    return torch.from_numpy(np.asarray(Image.open(D_ / "images" / f"{pid}_{t}.png").convert("RGB"))).permute(2, 0, 1).float() / 255


def search(z1, z2, U, radius):
    best = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            z1s = torch.nn.functional.pad(z1, (radius,) * 4, mode="replicate")[..., radius + dy:radius + dy + z1.shape[-2], radius + dx:radius + dx + z1.shape[-1]]
            r = z2 - z1s
            if U is not None:
                B, C, k, h, w = U.shape
                Up = torch.nn.functional.pad(U.reshape(B, C * k, h, w), (radius,) * 4, mode="replicate")
                Us = Up[..., radius + dy:radius + dy + h, radius + dx:radius + dx + w].reshape(B, C, k, h, w)
                r = op.project_basis(r, Us)
            s = r.norm(dim=1)
            best = s if best is None else torch.minimum(best, s)
    return best


scores = {v: np.zeros((len(man), 16, 16), np.float32) for v in VARIANTS}
for i in range(0, len(man), a.batch):
    rr = man.iloc[i:i + a.batch]
    x1 = torch.stack([load(p_, "t1") for p_ in rr.pair_id]).cuda()
    x2 = torch.stack([load(p_, "t2") for p_ in rr.pair_id]).cuda()
    z1, z2 = enc(center_crop(x1)), enc(center_crop(x2))
    D = op.probe_responses(enc, x1, z1, DISPS)
    for v, (rings, rank, rad) in VARIANTS.items():
        U = None
        if rings is not None:
            idx = sum((ring_idx[float(m)] for m in rings), [])
            U = op.probe_basis(D[:, :, idx], rank)
        s = (z2 - z1).norm(dim=1) if (U is None and rad == 0) else (
            op.project_basis(z2 - z1, U).norm(dim=1) if rad == 0 else search(z1, z2, U, rad))
        scores[v][i:i + len(rr)] = s.cpu().numpy()
    run.log(f"{min(i + a.batch, len(man))}/{len(man)}")
np.savez_compressed(run.dir / "patch_scores.npz", **scores)

chg = (man.label == "change").to_numpy()
rows = []
for v, (rings, rank, rad) in VARIANTS.items():
    K = 0 if rings is None else 8 * len(rings)
    for sp in ["dev", "heldout", "all"]:
        m = np.ones(len(man), bool) if sp == "all" else (man.split == sp).to_numpy()
        s, c = scores[v][m], chg[m]
        pos, neg = s[c].ravel(), s[~c].ravel()
        rows.append({"variant": v, "split": sp, "K_probes": K, "rank": rank, "search_radius": rad,
                     "extra_passes_per_pair": K, "AUROC_patch": roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]),
                     "NCR": float(np.median(neg) / np.median(pos)),
                     "TPR@5%FCR_patch": float((pos > np.quantile(neg, 0.95)).mean())})
tab = pd.DataFrame(rows)
tab.to_csv(run.dir / "table_ablation.csv", index=False)
pv = tab.pivot_table(index="variant", columns="split", values=["AUROC_patch", "NCR"]).round(4)
run.log("ablation:\n" + pv.to_string())
dev = tab[(tab.split == "dev") & tab.variant.str.startswith("c2f")]
best = dev.AUROC_patch.max()
ok = dev[dev.AUROC_patch >= best - 0.005].sort_values(["extra_passes_per_pair", "rank", "search_radius"])
chosen = ok.iloc[0].variant
ho = tab[(tab.split == "heldout")].set_index("variant")
run.log(f"selected on dev (cheapest within 0.5 pt of best c2f): {chosen}; heldout AUROC {ho.loc[chosen, 'AUROC_patch']:.4f} "
        f"NCR {ho.loc[chosen, 'NCR']:.4f} vs localmin1_r1 {ho.loc['localmin1_r1', 'AUROC_patch']:.4f} / raw {ho.loc['raw', 'AUROC_patch']:.4f}")
run.metric("chosen_heldout_AUROC", float(ho.loc[chosen, "AUROC_patch"]))
run.save_json("selection.json", {"chosen": chosen, "rule": "cheapest c2f within 0.5 pt of best on dev"})
run.close()
