"""Create the NullShift null-pair benchmark NSB-v1 (manifests, seeds, previews, dose statistics, datasheet).

No perturbed images are stored: every null pair is (crop, family, dose_idx, seed) and is regenerated bit-exactly by
`nullshift.nuisance.apply_nuisance`. Usage: PYTHONPATH=src nullshift-venv/bin/python scripts/create_nsb.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nullshift import paths  # noqa: E402  (redirect caches first)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from nullshift.data import LevirScenes, crop_index, list_scenes, sha256_file  # noqa: E402
from nullshift.nuisance import FAMILIES, MARGIN, N_DOSES, apply_nuisance, center_crop, nuisance_seed  # noqa: E402

SEED = 20260928
OUT = paths.NSB
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "previews").mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---------------------------------------------------------------- provenance
    prov = {"source": "LEVIR-CD (Chen & Shi 2020), HF mirror satellite-image-deep-learning/LEVIR-CD",
            "splits": {}, "margin_px": MARGIN, "tile_px": 256, "seed": SEED}
    for split in ["val", "test"]:
        sc = list_scenes(split)
        files = sorted((paths.LEVIR / split).rglob("*.png"))
        prov["splits"][split] = {"n_scenes": len(sc), "n_png": len(files),
                                 "sha256_first_A": sha256_file(paths.LEVIR / split / "A" / f"{sc[0]}.png")}
    zips = paths.DATA_RAW / "SHA256SUMS.txt"
    if zips.exists():
        prov["zip_sha256"] = zips.read_text().strip().splitlines()

    # ---------------------------------------------------------------- crop indices with label statistics
    subsets = {}
    for split in ["val", "test"]:
        idx = crop_index(split)
        ds = LevirScenes(split)
        fr = []
        for r in idx.itertuples():
            lab = ds.L[r.scene][r.y + MARGIN:r.y + MARGIN + 256, r.x + MARGIN:r.x + MARGIN + 256]
            fr.append(lab.mean())
        idx["change_frac"] = np.round(fr, 5)
        idx.to_csv(OUT / f"crops_{split}.csv", index=False)
        # 4 crops per scene, fixed seed
        # (explicit loop: groupby.apply drops the grouping column in pandas >= 3)
        parts = []
        for s in sorted(idx.scene.unique()):
            g = idx[idx.scene == s]
            parts.append(g.iloc[np.sort(rng.choice(len(g), 4, replace=False))])
        subsets[split] = pd.concat(parts).reset_index(drop=True)
        del ds

    test512 = subsets["test"]
    test512["g3_eligible"] = test512.change_frac >= 0.01
    test512.to_csv(OUT / "subset_test512.csv", index=False)
    fit256 = subsets["val"]
    fit256.to_csv(OUT / "subset_val256_fit.csv", index=False)

    # ---------------------------------------------------------------- null-pair manifest (test) and fit manifest (val)
    rows = []
    for r in test512.itertuples():
        for fam in FAMILIES:
            if fam == "G3_parallax" and not r.g3_eligible:
                continue
            for d in range(1, N_DOSES):
                rows.append({"crop_id": r.crop_id, "family": fam, "dose_idx": d,
                             "dose": FAMILIES[fam]["doses"][d], "seed": nuisance_seed(r.crop_id, fam, d)})
    man = pd.DataFrame(rows)
    man.to_csv(OUT / "nullpairs_test512.csv", index=False)
    fit_rows = []
    for r in fit256.itertuples():
        for fam in FAMILIES:
            if FAMILIES[fam]["kind"] != "rad":
                continue
            for d in (2, 3, 4):   # fit doses; distinct seeds from test via the "fit/" prefix
                fit_rows.append({"crop_id": r.crop_id, "family": fam, "dose_idx": d,
                                 "seed": nuisance_seed("fit/" + r.crop_id, fam, d)})
    pd.DataFrame(fit_rows).to_csv(OUT / "fit_val256.csv", index=False)

    with open(OUT / "nuisances.yaml", "w") as f:
        yaml.safe_dump({"margin_px": MARGIN, "families": FAMILIES}, f, sort_keys=False)

    # ---------------------------------------------------------------- dose statistics (pixel space) on 64 crops
    ds = LevirScenes("test", scenes=sorted(test512.scene.unique()))
    sub = test512[test512.g3_eligible].iloc[:64]
    b = ds.batch(sub, DEV)
    stats = []
    for fam in FAMILIES:
        for d in range(N_DOSES):
            seeds = [nuisance_seed(c, fam, d) for c in b.ids]
            x1, x2 = apply_nuisance(b.b, fam, d, seeds, b.label)
            c1, c2 = center_crop(x1), center_crop(x2)
            mse = ((c1 - c2) ** 2).mean(dim=(1, 2, 3))
            psnr = (10 * torch.log10(1 / mse.clamp_min(1e-12))).clamp(max=99)
            stats.append({"family": fam, "dose_idx": d, "dose": FAMILIES[fam]["doses"][d],
                          "mean_abs_diff": float((c1 - c2).abs().mean()),
                          "psnr_db_median": float(psnr.median()),
                          "frac_px_diff_gt_2dn": float(((c1 - c2).abs() > 2 / 255).float().mean())})
    pd.DataFrame(stats).to_csv(OUT / "dose_stats_pixel.csv", index=False)

    # ---------------------------------------------------------------- previews: one PNG per family (2 crops x 6 doses)
    show = sub.iloc[[0, 5]]
    bb = ds.batch(show, DEV)
    for fam in FAMILIES:
        fig, ax = plt.subplots(4, N_DOSES, figsize=(2.2 * N_DOSES, 9))
        for d in range(N_DOSES):
            seeds = [nuisance_seed(c, fam, d) for c in bb.ids]
            x1, x2 = apply_nuisance(bb.b, fam, d, seeds, bb.label)
            for i in range(2):
                im = center_crop(x2)[i].permute(1, 2, 0).cpu().numpy()
                df = (center_crop(x2)[i] - center_crop(x1)[i]).abs().mean(0).cpu().numpy()
                ax[2 * i, d].imshow(im)
                ax[2 * i + 1, d].imshow(df, cmap="magma", vmin=0, vmax=0.15)
                ax[2 * i, d].set_title(f"dose {d}: {FAMILIES[fam]['doses'][d]}", fontsize=8)
        for a in ax.ravel():
            a.axis("off")
        fig.suptitle(f"{fam} — rows: T(I) and |T(I) - I| (0-0.15) for 2 crops; unit: {FAMILIES[fam]['unit']}", fontsize=9)
        fig.savefig(OUT / "previews" / f"{fam}.png", dpi=70, bbox_inches="tight")
        plt.close(fig)

    summary = {"n_crops_val": 1024, "n_crops_test": 2048, "n_test512": len(test512),
               "n_g3_eligible": int(test512.g3_eligible.sum()), "n_nullpairs": len(man),
               "n_fit_interventions": len(fit_rows), "provenance": prov}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
