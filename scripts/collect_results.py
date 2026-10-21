"""Copy the final tables of each experiment run into results/ (the files the README and docs/RESULTS.md cite).

Usage: python scripts/collect_results.py [--root <project root containing experiments/*/runs>]
The latest run matching each pattern is used.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# (run glob, file in run dir, destination under results/)
FILES = [
    ("E01_linearity/runs/*_dinov3L_val256", "linearity_summary.csv", "synthetic/E01_linearity.csv"),
    ("E02_selectivity/runs/*_dinov3L_test512", "table_energy.csv", "synthetic/E02_energy_removed.csv"),
    ("E02_selectivity/runs/*_radP", "table_energy.csv", "synthetic/E02_energy_removed_radP.csv"),
    ("E03b_bootstrap_ci/runs/*_dinov3L_main", "table_bootstrap_fcr_r99.csv", "synthetic/E03b_bootstrap_FCR_R99.csv"),
    ("E04_perturbed_pair/runs/*_dinov3L_main", "table_perturbed.csv", "synthetic/E04_perturbed_pairs.csv"),
    ("E05_maxar_realpairs/runs/*_dinov3L_mxrla", "table_mxr.csv", "real/E05_LA_vitl.csv"),
    ("E06_coarse_to_fine/runs/*_dinov3L_mxrla_c2f", "table_split.csv", "real/E06_LA_vitl_dev_heldout.csv"),
    ("E06_coarse_to_fine/runs/*_dinov3L_mxrla_c2f", "posthoc_paired_bootstrap.txt", "real/E06_LA_vitl_bootstrap.txt"),
    ("E06_coarse_to_fine/runs/*_vitb16web", "table_split.csv", "real/E06_LA_vitb_dev_heldout.csv"),
    ("E06_coarse_to_fine/runs/*_vitb16web", "posthoc_paired_bootstrap.txt", "real/E06_LA_vitb_bootstrap.txt"),
    ("E06_coarse_to_fine/runs/*_dinov3L_mxrla_c2f", "posthoc_paired_bootstrap_all.csv", "real/E06_LA_vitl_bootstrap.csv"),
    ("E06_coarse_to_fine/runs/*_vitb16web", "posthoc_paired_bootstrap_all.csv", "real/E06_LA_vitb_bootstrap.csv"),
    ("E07_cost_ablation/runs/*_dinov3_vitl16_sat_mxrla", "table_ablation.csv", "real/E07_ablation_vitl.csv"),
    ("E07_cost_ablation/runs/*_dinov3_vitb16_web_mxrla", "table_ablation.csv", "real/E07_ablation_vitb.csv"),
    ("E08_c2f_radiometric/runs/*_e08_mxrla_L", "table_split.csv", "real/E08_LA_vitl_dev_heldout.csv"),
    ("E08_c2f_radiometric/runs/*_e08_mxrla_B", "table_split.csv", "real/E08_LA_vitb_dev_heldout.csv"),
    ("E08_c2f_radiometric/runs/*_e08_levir_perturbed", "table_perturbed.csv", "synthetic/E08_perturbed_pairs.csv"),
    ("E09_second_event/runs/*_e09_mxrtx_L", "table_mxr.csv", "real/E09_TX_vitl.csv"),
    ("E09_second_event/runs/*_e09_mxrtx_L", "posthoc_paired_bootstrap_all.csv", "real/E09_TX_vitl_bootstrap_all_null.csv"),
    ("E09_second_event/runs/*_e09_mxrtx_L", "posthoc_paired_bootstrap_null_long.csv", "real/E09_TX_vitl_bootstrap_null_long.csv"),
    ("E09_second_event/runs/*_e09_mxrtx_B", "table_mxr.csv", "real/E09_TX_vitb.csv"),
    ("E09_second_event/runs/*_e09_mxrtx_B", "posthoc_paired_bootstrap_all.csv", "real/E09_TX_vitb_bootstrap_all_null.csv"),
    ("E09_second_event/runs/*_e09_mxrtx_B", "posthoc_paired_bootstrap_null_long.csv", "real/E09_TX_vitb_bootstrap_null_long.csv"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results"))
    a = ap.parse_args()
    root, out = Path(a.root) / "experiments", Path(a.out)
    for pattern, name, dest in FILES:
        runs = sorted(root.glob(pattern))
        if not runs or not (runs[-1] / name).exists():
            print(f"[missing] {pattern}/{name}")
            continue
        d = out / dest
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(runs[-1] / name, d)
        print(f"[ok] {dest}  <- {runs[-1].name}")
    # token displacement vs shift (E01 metrics)
    runs = sorted(root.glob("E01_linearity/runs/*_dinov3L_val256"))
    if runs:
        m = json.loads((runs[-1] / "metrics.json").read_text())
        rows = ["shift_px,mean_token_displacement"] + [f"{d},{m[f'mean_dz_norm_d{d}']:.4f}" for d in ("0.25", "0.5", "1.0", "2.0", "4.0")]
        rows += [f"# reference: mean residual norm, real unchanged {m['mean_r_norm_unchanged_real']:.3f}, real change {m['mean_r_norm_true_change']:.3f}"]
        (out / "synthetic" / "E01_token_displacement.csv").write_text("\n".join(rows) + "\n")
        print("[ok] synthetic/E01_token_displacement.csv")


if __name__ == "__main__":
    main()
