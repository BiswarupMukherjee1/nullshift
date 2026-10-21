"""Build the README figures from the CSV files in results/ (no GPU, no raw data needed).

Usage: python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1] / "results"
F = R / "figures"
F.mkdir(exist_ok=True)
PEAK_MIN = 0.15        # phase-correlation peak height above which a global shift estimate is considered reliable
NAMES = {"raw": "raw residual", "phasecorr": "phase-corr. registration", "blur": "Gaussian blur", "classic": "registration + radiometric norm.",
         "geoP": "NullShift geoP", "geoP+rad": "geoP + radiometric subspace", "classic+ns": "classic + NullShift",
         "localmin": "3×3 token search", "geoPms": "geoP, wide probes", "localmin+geoPms": "NullShift coarse-to-fine", "c2f": "NullShift c2f (16 probes)"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def fig_token_displacement():
    d = pd.read_csv(R / "synthetic/E01_token_displacement.csv", comment="#")
    x, y = d.shift_px.values, d.mean_token_displacement.values
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.loglog(x, y, "o-", color="k", label="measured (DINOv3 ViT-L/16 SAT)")
    ax.loglog(x, y[2] * x, "--", color="tab:red", label="linear (first-order) model")
    ax.loglog(x, y[2] * np.sqrt(x), ":", color="tab:blue", label="∝ √shift")
    ax.set_xlabel("sub-pixel translation (px)"); ax.set_ylabel("token displacement ‖Δz‖")
    ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in x]); ax.minorticks_off()
    ax.set_yticks([0.1, 0.2, 0.4]); ax.set_yticklabels(["0.1", "0.2", "0.4"])
    ax.set_title("Token response to sub-pixel shift"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(F / "fig1_token_displacement.png", dpi=150); plt.close(fig)


def fig_null_pairs():
    d = pd.read_csv(R / "synthetic/E03_NCR.csv").set_index("method")
    fams = {"global shift": "G1_shift", "smooth warp": "G2_field", "parallax": "G3_parallax", "haze": "R3_haze",
            "JPEG": "R5_jpeg", "sensor noise": "R6_noise", "compound": "C1_compound"}
    meths = ["raw", "phasecorr", "blur", "classic", "geoP", "geoP+rad", "classic+ns"]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    w = 0.8 / len(meths)
    for i, m in enumerate(meths):
        ax.bar(np.arange(len(fams)) + i * w, [d.loc[m, c] for c in fams.values()], w, label=NAMES[m])
    ax.set_xticks(np.arange(len(fams)) + 0.4 - w / 2); ax.set_xticklabels(fams.keys())
    ax.set_ylabel("nuisance-to-change ratio (lower = better)")
    ax.set_title("Synthetic null pairs (LEVIR-CD test): false-change score relative to real change, mean over 5 strengths")
    ax.legend(ncol=4, frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); fig.savefig(F / "fig2_null_pairs.png", dpi=150); plt.close(fig)


def fig_misregistration():
    lab = {"null_pre": "pre-event pairs\n(≤ 14 days)", "null_post": "post-event pairs\n(≤ 7 days)", "null_long": "1–5 week gap"}
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for k, (ev, c) in enumerate([("LA", "tab:orange"), ("TX", "tab:blue")]):
        d = pd.read_csv(R / f"real/misregistration_{ev}.csv")
        d = d[(d.peak > PEAK_MIN) & d.label.isin(lab)]
        for j, (l, name) in enumerate(lab.items()):
            v = d[d.label == l].shift_px.values
            if len(v):
                pos = j + (k - 0.5) * 0.35
                ax.boxplot(v, positions=[pos], widths=0.3, patch_artist=True, showfliers=True,
                           boxprops=dict(facecolor=c, alpha=0.6), medianprops=dict(color="k"))
        ax.plot([], [], "s", color=c, alpha=0.6, label={"LA": "Los Angeles 2025", "TX": "Texas 2024"}[ev])
    ax.set_xticks(range(3)); ax.set_xticklabels(lab.values())
    ax.set_ylabel("residual misregistration (px at 0.5 m)")
    ax.set_title("Measured misregistration between real Maxar acquisitions"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(F / "fig3_misregistration.png", dpi=150); plt.close(fig)


def fig_real_pairs():
    meths = ["raw", "geoP", "geoPms", "localmin", "localmin+geoPms"]
    sets = []
    for bb, f in [("LA, ViT-L sat", "E06_LA_vitl_dev_heldout.csv"), ("LA, ViT-B web", "E06_LA_vitb_dev_heldout.csv")]:
        d = pd.read_csv(R / "real" / f); d = d[d.split == "heldout"].set_index("method")
        sets.append((bb + "\n(held-out tiles)", d["AUROC_patch"], d["NCR"]))
    for bb, f in [("TX, ViT-L sat", "E09_TX_vitl.csv"), ("TX, ViT-B web", "E09_TX_vitb.csv")]:
        d = pd.read_csv(R / "real" / f).set_index("method")
        sets.append((bb + "\n(all near-null)", d["AUROC_patch/all_null"], d["NCR/all_null"]))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    w = 0.8 / len(meths)
    for ax, k, yl in [(axes[0], 1, "patch AUROC, change vs near-null (higher = better)"), (axes[1], 2, "nuisance-to-change ratio (lower = better)")]:
        for i, m in enumerate(meths):
            ax.bar(np.arange(len(sets)) + i * w, [s[k].get(m, np.nan) for s in sets], w, label=NAMES[m])
        ax.set_xticks(np.arange(len(sets)) + 0.4 - w / 2); ax.set_xticklabels([s[0] for s in sets], fontsize=7.5)
        ax.set_ylabel(yl, fontsize=8)
    axes[0].set_ylim(0.75, 1.0)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, frameon=False, fontsize=8, loc="lower center")
    fig.suptitle("Real acquisition pairs (Maxar Open Data): two events, two backbones", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(F / "fig4_real_pairs.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig_token_displacement(); fig_null_pairs(); fig_misregistration(); fig_real_pairs()
    print("figures ->", F)
