"""LEVIR-CD loading and the deterministic NSB-v1 crop index.

Scenes are 1024x1024 RGB PNG (A = t1, B = t2, label = building change mask). Crops are the standard non-overlapping
4x4 grid of 256x256 tiles (16 per scene). Each crop is served with a MARGIN of context taken from the scene
(reflect-padded at scene borders) so that geometric nuisances never wrap content into the evaluated centre.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .nuisance import MARGIN
from .paths import LEVIR

TILE = 256
GRID = [0, 256, 512, 768]


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def list_scenes(split: str) -> list[str]:
    d = LEVIR / split / "A"
    return sorted(p.stem for p in d.glob("*.png"))


def crop_index(split: str) -> pd.DataFrame:
    rows = []
    for s in list_scenes(split):
        for y in GRID:
            for x in GRID:
                rows.append({"crop_id": f"{split}/{s}/{y:04d}_{x:04d}", "split": split, "scene": s, "y": y, "x": x})
    return pd.DataFrame(rows)


@dataclass
class Batch:
    ids: list
    a: torch.Tensor      # (B,3,H+2M,W+2M) float [0,1]  t1
    b: torch.Tensor      # (B,3,H+2M,W+2M) float [0,1]  t2
    label: torch.Tensor  # (B,H+2M,W+2M) bool           change mask (t2 buildings)


class LevirScenes:
    """Holds reflect-padded uint8 scenes of one split in RAM (~1 GB for test)."""

    def __init__(self, split: str, margin: int = MARGIN, scenes: list[str] | None = None):
        self.split, self.m = split, margin
        self.scenes = scenes or list_scenes(split)
        self.A, self.B, self.L = {}, {}, {}
        pad3 = ((margin, margin), (margin, margin), (0, 0))
        for s in self.scenes:
            a = np.asarray(Image.open(LEVIR / split / "A" / f"{s}.png").convert("RGB"))
            b = np.asarray(Image.open(LEVIR / split / "B" / f"{s}.png").convert("RGB"))
            l = np.asarray(Image.open(LEVIR / split / "label" / f"{s}.png").convert("L")) > 127
            self.A[s] = np.pad(a, pad3, mode="reflect")
            self.B[s] = np.pad(b, pad3, mode="reflect")
            self.L[s] = np.pad(l, margin, mode="reflect")

    def batch(self, rows: pd.DataFrame, device="cpu") -> Batch:
        m, T = self.m, TILE
        sl = lambda r: (slice(r.y, r.y + T + 2 * m), slice(r.x, r.x + T + 2 * m))
        a = np.stack([self.A[r.scene][sl(r)] for r in rows.itertuples()])
        b = np.stack([self.B[r.scene][sl(r)] for r in rows.itertuples()])
        l = np.stack([self.L[r.scene][sl(r)] for r in rows.itertuples()])
        to = lambda z: torch.from_numpy(z).permute(0, 3, 1, 2).float().div_(255).to(device)
        return Batch(list(rows.crop_id), to(a), to(b), torch.from_numpy(l).to(device))
