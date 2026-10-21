"""Shared sweep loop for E03/E04: iterate crops x nuisance conditions x methods and accumulate score histograms.

mode = "clean"     : real pair (A, B)                     groups: pos/neg patches, pix_pos/pix_neg pixels
mode = "null"      : null pair (B, T(B))                  group : all (every patch is a true negative)
mode = "perturbed" : real pair with nuisance (A, T(B))    groups: as clean
mode = "null2"     : TWO-SIDED null pair (T_a(B), T_b(B)) with independent draws a != b of the same family/dose —
                     the operator is built from T_a(B), which is NOT the image that was perturbed into t2
                     (tests whether the gain depends on building the operator from the very image that is perturbed).
G3 (parallax) only counts crops whose centre has >= 1 % labelled building pixels (the footprint source).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .metrics import Hist, patch_fraction, patch_masks, upsample
from .nuisance import FAMILIES, apply_nuisance, center_crop, nuisance_seed

GEO4_FAMILIES = {"G1_shift", "G2_field", "G3_parallax", "C1_compound", "clean"}


def conditions(families=None, doses=range(1, 6), include_identity=True):
    fams = families or list(FAMILIES)
    cond = [("identity", 0)] if include_identity else []
    return cond + [(f, d) for f in fams for d in doses]


def run_sweep(engine, ds, rows, conds, methods, mode: str, batch: int = 16, log=print, pix: bool = True,
              ckpt=None, ckpt_every: int = 4, resume: bool = False):
    """ckpt: path prefix; every `ckpt_every` batches the histograms + progress are saved; resume=True continues."""
    H = defaultdict(Hist)
    n_crops = defaultdict(int)
    start = 0
    if resume and ckpt is not None and Path(f"{ckpt}.npz").exists():
        H.update(load_hists(f"{ckpt}.npz"))
        meta = json.loads(Path(f"{ckpt}.json").read_text())
        start, n_crops = meta["next_row"], defaultdict(int, meta["n_crops"])
        log(f"  resumed {mode} sweep from row {start}")
    for bi, i in enumerate(range(start, len(rows), batch)):
        b = ds.batch(rows.iloc[i:i + batch], "cuda")
        lab = center_crop(b.label)
        frac = patch_fraction(lab)
        pos, neg = patch_masks(frac)
        g3_ok = lab.float().mean(dim=(1, 2)) >= 0.01
        base_ref = engine.reference(b.b if mode == "null" else b.a, methods) if mode != "null2" else None
        for fam, d in conds:
            if fam in ("clean", "identity"):
                x1, x2 = (b.b, b.b.clone()) if mode in ("null", "null2") else (b.a, b.b)
                ref = base_ref if base_ref is not None else engine.reference(x1, methods)
            elif mode == "null2":
                seeds_a = [nuisance_seed("ref/" + c, fam, d) for c in b.ids]
                seeds_b = [nuisance_seed(c, fam, d) for c in b.ids]
                _, x1 = apply_nuisance(b.b, fam, d, seeds_a, b.label)
                _, x2 = apply_nuisance(b.b, fam, d, seeds_b, b.label)
                ref = engine.reference(x1, methods)
            else:
                seeds = [nuisance_seed(c, fam, d) for c in b.ids]
                x1, x2 = apply_nuisance(b.b, fam, d, seeds, b.label, ref=None if mode == "null" else b.a)
                ref = engine.reference(x1, methods) if fam == "C1_compound" else base_ref
            ms = [m for m in methods if m != "geo4" or fam in GEO4_FAMILIES]
            keep = g3_ok if fam == "G3_parallax" else torch.ones_like(g3_ok)
            if keep.sum() == 0:
                continue
            sc = engine.scores(ref, x2, ms)
            key = f"{fam}|{d}"
            n_crops[key] += int(keep.sum())
            for m, s in sc.items():
                s = s[keep]
                if mode in ("null", "null2"):
                    H[(key, m, "all")].add(s)
                else:
                    H[(key, m, "pos")].add(s, pos[keep])
                    H[(key, m, "neg")].add(s, neg[keep])
                    if pix:
                        up = upsample(s)
                        H[(key, m, "pix_pos")].add(up, lab[keep])
                        H[(key, m, "pix_neg")].add(up, ~lab[keep])
        log(f"  {mode}: {min(i + batch, len(rows))}/{len(rows)} crops")
        if ckpt is not None and (bi + 1) % ckpt_every == 0:
            save_hists(H, f"{ckpt}.npz")
            Path(f"{ckpt}.json").write_text(json.dumps({"next_row": i + batch, "n_crops": dict(n_crops)}))
    return H, dict(n_crops)


def save_hists(H, path):
    keys = sorted(H)
    np.savez_compressed(path, keys=np.array(["||".join(k) for k in keys]),
                        counts=np.stack([H[k].c for k in keys]).astype(np.int32))


def load_hists(path):
    z = np.load(path)
    H = {}
    for k, c in zip(z["keys"], z["counts"]):
        h = Hist()
        h.c = c.astype(np.int64)
        H[tuple(str(k).split("||"))] = h
    return H
