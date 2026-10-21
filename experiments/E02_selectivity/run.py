"""E02 — Energy selectivity / entanglement (H2, H6).

For each operator O, the pooled fraction of residual energy it removes, rho = sum||r - O r||^2 / sum||r||^2, on
  (a) nuisance residuals: null pairs (B, T(B)) of NSB-v1 test512, every family x dose;
  (b) true-change residuals: real pairs (A, B), patches with >= 50 % changed pixels;
  (c) real unchanged residuals: real pairs, patches with 0 % change (seasonal/illumination/misregistration mix).
Selectivity = rho_nuisance / rho_change (1 = no separation; H2 kill if ~1). Chance = rank / C.
H6: the NFAD incidence gate (per-image gamma from the spatial participation ratio of residual energy) should keep
removing radiometric nuisance (spatially spread) but back off on misregistration (edge-local) -> reported via
mean gamma and phi per family. Usage: PYTHONPATH=src python experiments/E02_selectivity/run.py --eps <eps*>
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from nullshift import paths  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from nullshift import operators as op  # noqa: E402
from nullshift.data import LevirScenes  # noqa: E402
from nullshift.encoder import Encoder  # noqa: E402
from nullshift.metrics import EnergyMeter, patch_fraction  # noqa: E402
from nullshift.nuisance import FAMILIES, apply_nuisance, center_crop, nuisance_seed  # noqa: E402
from nullshift.tracking import Run  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--tag", default="dinov3L_test512")
p.add_argument("--eps", type=float, required=True)
p.add_argument("--encoder", default="dinov3_vitl16_sat")
p.add_argument("--layer", type=int, default=-1)
p.add_argument("--subspaces", default="dinov3_vitl16_sat_L-1_kf8_ks16.pt")
p.add_argument("--tau-pr", type=float, default=0.5)
p.add_argument("--n", type=int, default=512)
p.add_argument("--batch", type=int, default=16)
a = p.parse_args()
run = Run("E02_selectivity", a.tag, dict(vars(a), kill="selectivity(geo, misregistration vs true change) ~ 1"))
enc = Encoder(a.encoder, a.layer)
S = {k: v.cuda() for k, v in torch.load(paths.NSB / "subspaces" / a.subspaces).items() if k.startswith("V_")}
C = enc.dim
OPS = ["geo", "geo4", "geoP", "geo-smooth", "shiftV", "shiftV+gate", "rad", "rad+gate", "geo+rad", "geoP+rad",
       "radP", "geoP+radP"]
RR = 4   # radP rank (per-patch radiometric probe subspace)
DISPS = op.probe_displacements(8, (0.5, 1.0, 2.0))
PR = 8
RANK = {"geo": 2, "geo4": 4, "geoP": PR, "geoP+rad": PR + S["V_rad"].shape[1], "geo-smooth": 2, "radP": RR, "geoP+radP": PR + RR, "shiftV": S["V_shift"].shape[1], "shiftV+gate": S["V_shift_g"].shape[1],
        "rad": S["V_rad"].shape[1], "rad+gate": S["V_rad_g"].shape[1], "geo+rad": 2 + S["V_rad"].shape[1]}


def apply_ops(r, J1, J2, U1, UR=None, UG=None):
    out, gam = {}, {}
    if UR is not None:
        out["radP"] = op.project_basis(r, UR)
        out["geoP+radP"] = op.project_basis(r, UG)
    out["geo"] = op.project_tangent(r, J1)[0]
    out["geoP"] = op.project_basis(r, U1)
    out["geoP+rad"] = op.project_joint(r, U1, S["V_rad"])
    if J2 is not None:
        out["geo4"] = op.project_tangent(r, torch.cat([J1, J2], 2))[0]
    out["geo-smooth"] = op.project_global_translation(r, J1)[0]
    out["shiftV"] = op.project_global(r, S["V_shift"])
    out["shiftV+gate"], gam["shiftV+gate"] = op.incidence_gate(r, S["V_shift_g"], a.tau_pr)
    out["rad"] = op.project_global(r, S["V_rad"])
    out["rad+gate"], gam["rad+gate"] = op.incidence_gate(r, S["V_rad_g"], a.tau_pr)
    out["geo+rad"] = op.project_joint(r, J1, S["V_rad"])
    return out, gam


def phi(r):  # spatial participation ratio of residual energy, 1 = uniform, 1/P = one patch
    e = (r ** 2).sum(1).flatten(1)
    return e.sum(1) ** 2 / (e.shape[1] * (e ** 2).sum(1)).clamp_min(1e-12)


E = defaultdict(EnergyMeter)
G = defaultdict(list)
sub = pd.read_csv(paths.NSB / "subset_test512.csv").iloc[:a.n]
ds = LevirScenes("test", scenes=sorted(sub.scene.unique()))
for i in range(0, len(sub), a.batch):
    b = ds.batch(sub.iloc[i:i + a.batch], "cuda")
    lab = center_crop(b.label)
    fr = patch_fraction(lab)
    g3_ok = lab.float().mean(dim=(1, 2)) >= 0.01
    # ---- true change: real pair A -> B, tangents from A (and B for geo4)
    za, zb = enc(center_crop(b.a)), enc(center_crop(b.b))
    Ja, Jb = op.shift_tangents(enc, b.a, a.eps), op.shift_tangents(enc, b.b, a.eps)
    Dga, Dgb = op.probe_responses(enc, b.a, za, DISPS), op.probe_responses(enc, b.b, zb, DISPS)
    Dra, Drb = op.rad_probe_responses(enc, b.a, za), op.rad_probe_responses(enc, b.b, zb)
    Ua, Ub = op.probe_basis(Dga, PR), op.probe_basis(Dgb, PR)
    URa, URb = op.probe_basis(Dra, RR), op.probe_basis(Drb, RR)
    UGa, UGb = op.probe_basis(torch.cat([Dga, Dra], 2), PR + RR), op.probe_basis(torch.cat([Dgb, Drb], 2), PR + RR)
    del Dga, Dgb, Dra, Drb
    r = zb - za
    outs, gam = apply_ops(r, Ja, Jb, Ua, URa, UGa)
    for o, rp in outs.items():
        E[("true_change", 0, o)].add(r, rp, fr >= 0.5)
        E[("real_unchanged", 0, o)].add(r, rp, fr == 0)
    for o, g in gam.items():
        G[("real_pair", 0, o)] += g.tolist()
    G[("real_pair", 0, "phi")] += phi(r).tolist()
    # ---- nuisance: null pairs (B, T(B)), tangents from B (J1) and T(B) (J2, geometric families only)
    for fam in FAMILIES:
        for d in range(1, 6):
            seeds = [nuisance_seed(c, fam, d) for c in b.ids]
            x1, x2 = apply_nuisance(b.b, fam, d, seeds, b.label)
            keep = g3_ok if fam == "G3_parallax" else torch.ones_like(g3_ok)
            if keep.sum() == 0:
                continue
            z1 = zb if fam != "C1_compound" else enc(center_crop(x1))
            J1 = Jb if fam != "C1_compound" else op.shift_tangents(enc, x1, a.eps)
            if fam != "C1_compound":
                U1, UR1, UG1 = Ub, URb, UGb
            else:
                Dg1, Dr1 = op.probe_responses(enc, x1, z1, DISPS), op.rad_probe_responses(enc, x1, z1)
                U1, UR1, UG1 = op.probe_basis(Dg1, PR), op.probe_basis(Dr1, RR), op.probe_basis(torch.cat([Dg1, Dr1], 2), PR + RR)
            J2 = op.shift_tangents(enc, x2, a.eps) if FAMILIES[fam]["kind"] != "rad" else None
            rn = (enc(center_crop(x2)) - z1)[keep]
            outs, gam = apply_ops(rn, J1[keep], None if J2 is None else J2[keep], U1[keep], UR1[keep], UG1[keep])
            for o, rp in outs.items():
                E[(fam, d, o)].add(rn, rp)
            for o, g in gam.items():
                G[(fam, d, o)] += g.tolist()
            G[(fam, d, "phi")] += phi(rn).tolist()
    run.log(f"{min(i + a.batch, len(sub))}/{len(sub)} crops")

rows = [{"set": k[0], "dose_idx": k[1], "operator": k[2], "rank": RANK[k[2]], "chance": RANK[k[2]] / C,
         "removed_frac": m.value, "energy_total": m.tot} for k, m in E.items()]
tab = pd.DataFrame(rows)
tc = tab[tab.set == "true_change"].set_index("operator").removed_frac
tab["selectivity_vs_true_change"] = tab.removed_frac / tab.operator.map(tc)
tab.to_csv(run.dir / "table_energy.csv", index=False)
gtab = pd.DataFrame([{"set": k[0], "dose_idx": k[1], "what": k[2], "mean": float(np.mean(v)), "median": float(np.median(v))}
                     for k, v in G.items()])
gtab.to_csv(run.dir / "table_gate_phi.csv", index=False)

# headline numbers at 1 px (G1 dose 3), G2 1 px RMS (dose 3), G3 2 px (dose 3), radiometric dose 3
for fam in ["G1_shift", "G2_field", "G3_parallax", "R1_gain", "R3_haze", "R4_colour", "C1_compound"]:
    for o in OPS:
        s = tab[(tab.set == fam) & (tab.dose_idx == 3) & (tab.operator == o)]
        if len(s):
            run.metric(f"removed/{fam}_d3/{o}", float(s.removed_frac.iloc[0]))
            run.metric(f"selectivity/{fam}_d3/{o}", float(s.selectivity_vs_true_change.iloc[0]))
for o in OPS:
    run.metric(f"removed/true_change/{o}", float(tc[o]))
for o in ["geo", "geoP", "radP", "geoP+radP"]:
    sel = float(tab[(tab.set == "G1_shift") & (tab.dose_idx == 3) & (tab.operator == o)].selectivity_vs_true_change.iloc[0])
    rem = float(tab[(tab.set == "G1_shift") & (tab.dose_idx == 3) & (tab.operator == o)].removed_frac.iloc[0])
    run.metric(f"H2_selectivity_{o}_G1_1px", sel)
    run.log(f"H2 [{o}]: removes {rem:.1%} of 1-px misregistration energy, {sel:.1f}x more than of true change "
            f"({'PASS' if sel > 2 else 'FAIL (kill: ~1)'})")
pv = tab[tab.dose_idx.isin([0, 3])].pivot_table(index="set", columns="operator", values="removed_frac")
run.log("removed energy fraction (dose idx 3; true_change/real_unchanged at 0):\n" + pv[OPS].round(3).to_string())
gp = gtab[gtab.dose_idx.isin([0, 3])].pivot_table(index="set", columns="what", values="mean")
run.log("mean gate gamma / phi (dose idx 3):\n" + gp.round(3).to_string())

# ---------------- figures
fig, ax = plt.subplots(figsize=(13, 5))
sets = ["G1_shift", "G2_field", "G3_parallax", "R1_gain", "R2_gamma", "R3_haze", "R4_colour", "R5_jpeg", "R6_noise",
        "R7_blur", "C1_compound", "true_change", "real_unchanged"]
w = 0.8 / len(OPS)
for j, o in enumerate(OPS):
    vals = [pv.loc[s, o] if s in pv.index and o in pv.columns else np.nan for s in sets]
    ax.bar(np.arange(len(sets)) + j * w, vals, w, label=f"{o} (rank {RANK[o]})")
ax.set_xticks(np.arange(len(sets)) + 0.4); ax.set_xticklabels(sets, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("fraction of residual energy removed"); ax.legend(fontsize=7, ncol=4); ax.grid(axis="y", alpha=.3)
ax.set_title("E02: energy removed per operator (null pairs at dose idx 3 vs real change) — want high on nuisance, low on true_change")
run.figure(fig, "fig_energy_removed.png")
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for fam in ["G1_shift", "G2_field", "G3_parallax", "R3_haze", "R4_colour"]:
    for o, ax in zip(["geoP", "rad"], axs):
        s = tab[(tab.set == fam) & (tab.operator == o)].sort_values("dose_idx")
        ax.plot(s.dose_idx, s.selectivity_vs_true_change, "o-", label=fam)
for o, ax in zip(["geoP", "rad"], axs):
    ax.axhline(1, color="k", ls="--", lw=1); ax.set_yscale("log"); ax.set_title(f"selectivity of {o} (nuisance / true change)")
    ax.set_xlabel("dose idx"); ax.legend(fontsize=8); ax.grid(alpha=.3)
run.figure(fig, "fig_selectivity.png")
run.close()
