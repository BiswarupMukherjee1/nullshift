"""E01b — Fallback after the first-order model failed (E01): per-patch PROBE subspace instead of first-order tangent.

Probe subspace U_r(p) = top-r left singular vectors of D(p) = [f(shift(I, d_j)) - f(I)]_{j=1..K} (secant directions of
the shift manifold, no linearity assumed). Compared with: tangent (rank 2, eps=1) and multi-scale tangent (eps 0.5/1/2,
rank 6, the planned fallback). Evaluated on HELD-OUT displacements: random-direction global shifts at magnitudes
incl. between probe rings, and smooth displacement fields (G2). Selectivity against true-change / real-unchanged
residuals of real pairs. LEVIR val only (this selects the operator used on test).
Usage: PYTHONPATH=src python experiments/E01b_probe_subspace/run.py [--layer -1]
"""
import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from nullshift import paths  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from nullshift import operators as op  # noqa: E402
from nullshift.data import LevirScenes  # noqa: E402
from nullshift.encoder import Encoder  # noqa: E402
from nullshift.metrics import EnergyMeter, patch_fraction  # noqa: E402
from nullshift.nuisance import apply_nuisance, center_crop, fourier_shift, nuisance_seed  # noqa: E402
from nullshift.tracking import Run  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--tag", default=None)
p.add_argument("--encoder", default="dinov3_vitl16_sat")
p.add_argument("--layer", type=int, default=-1)
p.add_argument("--n", type=int, default=256)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--mags", default="0.5,1,2")
p.add_argument("--ndir", type=int, default=8)
a = p.parse_args()
mags = tuple(float(m) for m in a.mags.split(","))
RANKS = [2, 4, 8, 12, 16, 24]
TEST_MAGS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
tag = a.tag or f"dinov3L_L{a.layer}_probe{a.ndir}x{len(mags)}"
run = Run("E01b_probe_subspace", tag, dict(vars(a), ranks=RANKS, test_mags=TEST_MAGS, split="val"))
enc = Encoder(a.encoder, a.layer)
C = enc.dim
disps = op.probe_displacements(a.ndir, mags)
K = len(disps)
fit = pd.read_csv(paths.NSB / "subset_val256_fit.csv").iloc[:a.n]
ds = LevirScenes("val", scenes=sorted(fit.scene.unique()))
E = defaultdict(EnergyMeter)
gen = torch.Generator().manual_seed(7)

for i in range(0, len(fit), a.batch):
    b = ds.batch(fit.iloc[i:i + a.batch], "cuda")
    x, B = b.b, b.b.shape[0]
    z0 = enc(center_crop(x))
    D = op.probe_responses(enc, x, z0, disps)
    bases = {f"probe_r{r}": op.probe_basis(D, r) for r in RANKS if r <= K}
    J1 = op.shift_tangents(enc, x, 1.0)
    Jms = torch.cat([J1, op.shift_tangents(enc, x, 0.5), op.shift_tangents(enc, x, 2.0)], 2)

    def ops(r):
        out = {"tangent_r2": op.project_tangent(r, J1)[0], "tangent_ms_r6": op.project_tangent(r, Jms)[0]}
        out.update({k: op.project_basis(r, U) for k, U in bases.items()})
        return out

    # held-out global shifts (random direction per crop)
    for m in TEST_MAGS:
        ang = torch.rand(B, generator=gen) * 2 * math.pi
        d = torch.stack([m * torch.sin(ang), m * torch.cos(ang)], 1).to(x.device)
        dz = enc(center_crop(fourier_shift(x, d).clamp(0, 1))) - z0
        for k, rp in ops(dz).items():
            E[(f"shift_{m}", k)].add(dz, rp)
    # held-out smooth fields (G2, seeds disjoint from test via 'e01b/' prefix)
    for di in (2, 3, 4):
        seeds = [nuisance_seed("e01b/" + c, "G2_field", di) for c in b.ids]
        _, x2 = apply_nuisance(x, "G2_field", di, seeds)
        dz = enc(center_crop(x2)) - z0
        for k, rp in ops(dz).items():
            E[(f"field_{[0, .25, .5, 1, 2, 4][di]}", k)].add(dz, rp)
    # real pairs: bases from the t1 image A (as at deployment), residual B - A
    za = enc(center_crop(b.a))
    Da = op.probe_responses(enc, b.a, za, disps)
    Ja = op.shift_tangents(enc, b.a, 1.0)
    Jam = torch.cat([Ja, op.shift_tangents(enc, b.a, 0.5), op.shift_tangents(enc, b.a, 2.0)], 2)
    r = z0 - za
    fr = patch_fraction(center_crop(b.label))
    outs = {"tangent_r2": op.project_tangent(r, Ja)[0], "tangent_ms_r6": op.project_tangent(r, Jam)[0]}
    outs.update({f"probe_r{k}": op.project_basis(r, op.probe_basis(Da, k)) for k in RANKS if k <= K})
    for k, rp in outs.items():
        E[("true_change", k)].add(r, rp, fr >= 0.5)
        E[("real_unchanged", k)].add(r, rp, fr == 0)
    run.log(f"{min(i + a.batch, len(fit))}/{len(fit)}")

rank = {"tangent_r2": 2, "tangent_ms_r6": 6, **{f"probe_r{r}": r for r in RANKS}}
rows = [{"set": s, "operator": o, "rank": rank[o], "removed_frac": m.value, "chance": rank[o] / C} for (s, o), m in E.items()]
tab = pd.DataFrame(rows)
tc = tab[tab.set == "true_change"].set_index("operator").removed_frac
tab["selectivity"] = tab.removed_frac / tab.operator.map(tc)
tab.to_csv(run.dir / "table_probe.csv", index=False)
pv = tab.pivot_table(index="set", columns="operator", values="removed_frac")
order = ["tangent_r2", "tangent_ms_r6"] + [f"probe_r{r}" for r in RANKS if r <= K]
pv = pv[order]
run.log("removed energy fraction (rows: held-out nuisance / real residuals):\n" + pv.round(3).to_string())
sv = tab.pivot_table(index="set", columns="operator", values="selectivity")[order]
run.log("selectivity (removed on set / removed on true change):\n" + sv.round(2).to_string())
for o in order:
    run.metric(f"removed/shift_1.0/{o}", float(pv.loc["shift_1.0", o]))
    run.metric(f"removed/field_1/{o}", float(pv.loc["field_1", o]))
    run.metric(f"removed/true_change/{o}", float(pv.loc["true_change", o]))

fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
for o in order:
    s = [pv.loc[f"shift_{m}", o] for m in TEST_MAGS]
    axs[0].plot(TEST_MAGS, s, "o-", label=o, lw=2 if o.startswith("probe") else 1)
axs[0].set_xscale("log", base=2); axs[0].set_xlabel("held-out shift |d| (px)"); axs[0].set_ylabel("misregistration energy removed")
for m in mags:
    axs[0].axvline(m, color="grey", lw=0.5, ls=":")
axs[0].legend(fontsize=7); axs[0].grid(alpha=.3); axs[0].set_title("removal of held-out misregistration (dotted = probe rings)")
x_ = [rank[o] for o in order]
axs[1].plot(x_, [pv.loc["shift_1.0", o] for o in order], "o", label="shift 1 px")
axs[1].plot(x_, [pv.loc["field_1", o] for o in order], "s", label="field 1 px RMS")
axs[1].plot(x_, [pv.loc["true_change", o] for o in order], "^", label="true change")
axs[1].plot(x_, [pv.loc["real_unchanged", o] for o in order], "v", label="real unchanged")
axs[1].plot(sorted(set(x_)), [r / C for r in sorted(set(x_))], "k--", lw=1, label="chance r/C")
for o, xx in zip(order, x_):
    axs[1].annotate(o.replace("probe_", "p").replace("tangent_", "t"), (xx, pv.loc["shift_1.0", o]), fontsize=6)
axs[1].set_xlabel("rank"); axs[1].set_ylabel("energy removed"); axs[1].set_yscale("log"); axs[1].legend(fontsize=7)
axs[1].grid(alpha=.3); axs[1].set_title("nuisance vs true-change removal by rank")
run.figure(fig, "fig_probe_subspace.png")
run.close()
