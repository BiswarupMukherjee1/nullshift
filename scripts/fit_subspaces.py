"""Fit the global nuisance subspaces on LEVIR **val** (never test) and store them in Data_Created/NSB-v1/subspaces/.

V_rad   : NFAD transfer. Per radiometric family R1..R7: uncentred second moment of dz = f(T(x)) - f(x) over the
          val256 fit manifest (doses 2,3,4, seeds with "fit/" prefix); top-k_f eigvecs per family; QR-stacked.
V_shift : same construction from global geometric interventions (G1 shifts, G2 fields at doses 2,3,4) -> the
          "one global subspace for misregistration" that H6 says cannot work.
*_g     : NFAD incidence-gate bases (columns with participation ratio > lower quartile on clean val features).
Usage: PYTHONPATH=src python scripts/fit_subspaces.py [--kf 8] [--layer -1]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nullshift import paths  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from nullshift import operators as op  # noqa: E402
from nullshift.data import LevirScenes  # noqa: E402
from nullshift.encoder import Encoder  # noqa: E402
from nullshift.nuisance import FAMILIES, apply_nuisance, center_crop, nuisance_seed  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--encoder", default="dinov3_vitl16_sat")
p.add_argument("--layer", type=int, default=-1)
p.add_argument("--kf", type=int, default=8, help="eigvecs per radiometric family")
p.add_argument("--kshift", type=int, default=16, help="eigvecs per geometric family (G1, G2) -> rank 32 as NFAD")
p.add_argument("--batch", type=int, default=32)
a = p.parse_args()

torch.manual_seed(0)
enc = Encoder(a.encoder, a.layer)
fit = pd.read_csv(paths.NSB / "subset_val256_fit.csv")
ds = LevirScenes("val", scenes=sorted(fit.scene.unique()))
C = enc.dim
RAD = [f for f in FAMILIES if FAMILIES[f]["kind"] == "rad"]
GEO = ["G1_shift", "G2_field"]
moments = {f: op.SecondMoment(C) for f in RAD + GEO}
clean = []
for i in range(0, len(fit), a.batch):
    rows = fit.iloc[i:i + a.batch]
    b = ds.batch(rows, "cuda")
    z = enc(center_crop(b.b))
    clean.append(z.half().cpu())
    for fam in RAD + GEO:
        for d in (2, 3, 4):
            seeds = [nuisance_seed("fit/" + c, fam, d) for c in b.ids]
            _, x2 = apply_nuisance(b.b, fam, d, seeds, b.label)
            moments[fam].add(enc(center_crop(x2)) - z)
    print(f"{i + len(rows)}/{len(fit)}", flush=True)

Z = torch.cat(clean).float().cuda()
report = {"encoder": a.encoder, "layer": a.layer, "kf": a.kf, "kshift": a.kshift, "n_fit_crops": len(fit),
          "families": {}}
bases_rad, bases_geo = [], []
for fam, M in moments.items():
    k = a.kf if fam in RAD else a.kshift
    V, ev = M.top(k)
    (bases_rad if fam in RAD else bases_geo).append(V)
    tot = float(ev.clamp_min(0).sum())
    report["families"][fam] = {"k": k, "energy_captured_topk": float(ev[:k].sum() / tot),
                               "energy_top1": float(ev[0] / tot), "mean_sq_disp": float(tot)}
V_rad = op.stack_subspaces(bases_rad)
V_shift = op.stack_subspaces(bases_geo)
rho_rad = op.participation_ratio_columns(V_rad, Z)
rho_shift = op.participation_ratio_columns(V_shift, Z)
V_rad_g = V_rad[:, rho_rad > torch.quantile(rho_rad, 0.25)]
V_shift_g = V_shift[:, rho_shift > torch.quantile(rho_shift, 0.25)]
report.update({"rank_V_rad": V_rad.shape[1], "rank_V_shift": V_shift.shape[1],
               "rank_V_rad_g": V_rad_g.shape[1], "rank_V_shift_g": V_shift_g.shape[1],
               "rho_rad_quartiles": np.quantile(rho_rad.cpu(), [0, .25, .5, .75, 1]).tolist(),
               "rho_shift_quartiles": np.quantile(rho_shift.cpu(), [0, .25, .5, .75, 1]).tolist(),
               "principal_cos_rad_vs_shift_max": float(torch.linalg.svdvals(V_rad.T @ V_shift).max())})
out = paths.NSB / "subspaces"
out.mkdir(parents=True, exist_ok=True)
tag = f"{a.encoder}_L{a.layer}_kf{a.kf}_ks{a.kshift}"
torch.save({"V_rad": V_rad.cpu(), "V_rad_g": V_rad_g.cpu(), "V_shift": V_shift.cpu(), "V_shift_g": V_shift_g.cpu(),
            "per_family": {f: b.cpu() for f, b in zip(RAD + GEO, bases_rad + bases_geo)},
            "S": {f: (m.S / m.n).float().cpu() for f, m in moments.items()}, "report": report},
           out / f"{tag}.pt")
(out / f"{tag}.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
