"""Histogram-based metrics so that thresholds, F1, FCR and AUROC are exact up to bin width without storing maps.
Scores are cosine-feature residual norms in [0, 2]."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

NBINS, SMAX = 8000, 2.0
EDGES = np.linspace(0, SMAX, NBINS + 1)


class Hist:
    def __init__(self):
        self.c = np.zeros(NBINS, np.int64)

    def add(self, s: torch.Tensor, mask: torch.Tensor | None = None):
        v = s[mask] if mask is not None else s.flatten()
        idx = (v.float().clamp(0, SMAX - 1e-9) / SMAX * NBINS).long()
        self.c += torch.bincount(idx.flatten(), minlength=NBINS).cpu().numpy()

    @property
    def n(self):
        return int(self.c.sum())

    def frac_above(self, tau_bin: int) -> float:
        """Fraction of samples with score >= EDGES[tau_bin]."""
        return float(self.c[tau_bin:].sum() / max(self.n, 1))

    def to_list(self):
        return self.c.tolist()


def best_f1(pos: Hist, neg: Hist):
    """Threshold (bin index) maximising F1 with predictions score >= EDGES[t]."""
    tp = np.cumsum(pos.c[::-1])[::-1].astype(float)
    fp = np.cumsum(neg.c[::-1])[::-1].astype(float)
    fn = pos.n - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    t = int(np.argmax(f1))
    return t, float(f1[t])


def tau_at_recall(pos: Hist, recall: float) -> int:
    """Largest threshold bin t whose recall (fraction of positives with score >= EDGES[t]) is still >= `recall`."""
    tp = np.cumsum(pos.c[::-1])[::-1]
    ok = np.nonzero(tp >= recall * pos.n)[0]
    return int(ok.max()) if len(ok) else 0


def median(h: Hist) -> float:
    return float(EDGES[np.searchsorted(np.cumsum(h.c), h.n / 2)]) if h.n else float("nan")


def f1_at(pos: Hist, neg: Hist, t: int):
    tp, fp = pos.c[t:].sum(), neg.c[t:].sum()
    fn = pos.n - tp
    prec = tp / max(tp + fp, 1)
    rec = tp / max(pos.n, 1)
    return {"f1": float(2 * tp / max(2 * tp + fp + fn, 1)), "precision": float(prec), "recall": float(rec),
            "iou": float(tp / max(tp + fp + fn, 1))}


def auroc(pos: Hist, neg: Hist) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie at bin resolution)."""
    if pos.n == 0 or neg.n == 0:
        return float("nan")
    neg_below = np.concatenate([[0], np.cumsum(neg.c)[:-1]])
    return float((pos.c * (neg_below + 0.5 * neg.c)).sum() / (pos.n * neg.n))


def patch_fraction(label_center: torch.Tensor, patch: int = 16) -> torch.Tensor:
    """(B,H,W) bool -> (B,H/p,W/p) changed-pixel fraction per patch."""
    return F.avg_pool2d(label_center.float()[:, None], patch)[:, 0]


def patch_masks(frac: torch.Tensor, pos_thr: float = 0.25):
    """Patch labels: positive if >= pos_thr changed pixels, negative if none; mixed patches ignored."""
    return frac >= pos_thr, frac == 0


def upsample(s: torch.Tensor, size: int = 256) -> torch.Tensor:
    return F.interpolate(s[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0]


class EnergyMeter:
    """Pooled fraction of residual energy removed by an operator, sum ||r - r'||^2 / sum ||r||^2."""

    def __init__(self):
        self.rem, self.tot = 0.0, 0.0

    def add(self, r, rp, mask=None):
        e_tot = (r ** 2).sum(1)
        e_rem = ((r - rp) ** 2).sum(1)
        if mask is not None:
            e_tot, e_rem = e_tot[mask], e_rem[mask]
        self.rem += float(e_rem.sum())
        self.tot += float(e_tot.sum())

    @property
    def value(self):
        return self.rem / self.tot if self.tot > 0 else float("nan")
