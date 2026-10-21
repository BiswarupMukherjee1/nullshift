"""Pixel-space baselines applied to (padded) images before encoding. All normalise/align x2 towards x1."""
from __future__ import annotations

import numpy as np
import torch
from skimage.registration import phase_cross_correlation

from .nuisance import MARGIN, center_crop, fourier_shift, gaussian_blur

GRAY = torch.tensor([0.299, 0.587, 0.114])


def phase_peak(c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
    """Height of the phase-correlation peak (Hann-windowed, mean-removed), in [0, 1]: ~1 for a clean translation,
    ~0.02 for unrelated content. Measured on LEVIR val: real multi-year A-B pairs <= 0.12 (median 0.022); null pairs
    G1 >= 0.64, G2 1 px RMS ~0.29, C1 dose 3 ~0.28."""
    c1, c2 = c1.double(), c2.double()
    w = torch.outer(torch.hann_window(c1.shape[-2], periodic=False, dtype=torch.float64, device=c1.device),
                    torch.hann_window(c1.shape[-1], periodic=False, dtype=torch.float64, device=c1.device))
    F1 = torch.fft.fft2((c1 - c1.mean((-2, -1), keepdim=True)) * w)
    F2 = torch.fft.fft2((c2 - c2.mean((-2, -1), keepdim=True)) * w)
    R = F1 * F2.conj()
    return torch.fft.ifft2(R / R.abs().clamp_min(1e-12)).real.flatten(1).max(1).values.float()


def phasecorr_align(x1: torch.Tensor, x2: torch.Tensor, upsample: int = 20, margin: int = MARGIN, max_shift: float = 4.0,
                    min_peak: float = 0.15):
    """Global sub-pixel translation by phase correlation (estimated on the central region), undone by Fourier shift.
    Operational refinement stage: estimates with |s| > max_shift are rejected (no correction). Without this bound the
    baseline "registers" already co-registered multi-year LEVIR pairs by a median 8 px (up to 62 px), because A-B
    correlation is low (median 0.34) and spurious peaks win. Estimates whose phase-correlation peak is < min_peak are
    also rejected (standard peak-confidence gate; see `phase_peak`). Returns (x2_aligned, shifts (B,2))."""
    g = GRAY.to(x1.device).view(1, 3, 1, 1)
    c1 = center_crop((x1 * g).sum(1), margin // 2).cpu().numpy()
    c2 = center_crop((x2 * g).sum(1), margin // 2).cpu().numpy()
    # Hann window: without it the crop-boundary discontinuity dominates the whitened cross-power spectrum and pins
    # the peak at 0 on smooth scenes (caught by tests/test_core.py::test_phasecorr_recovers_shift*).
    win = np.outer(np.hanning(c1.shape[-2]), np.hanning(c1.shape[-1]))
    sh = np.stack([phase_cross_correlation((a - a.mean()) * win, (b - b.mean()) * win, upsample_factor=upsample)[0]
                   for a, b in zip(c1, c2)])
    s = torch.from_numpy(sh).float().to(x1.device)       # shift that registers x2 onto x1
    ok = (s.norm(dim=1) <= max_shift) & (phase_peak(torch.from_numpy(c1), torch.from_numpy(c2)).to(s.device) >= min_peak)
    s = s * ok[:, None]
    return fourier_shift(x2, s).clamp(0, 1), s


def blur_both(x1, x2, sigma: float = 1.0):
    return gaussian_blur(x1, sigma), gaussian_blur(x2, sigma)


def histmatch(x1: torch.Tensor, x2: torch.Tensor, margin: int = MARGIN):
    """Per-band histogram matching of x2 to x1 (statistics from the central region, mapping applied everywhere)."""
    out = []
    for a, b in zip(x1, x2):
        ref = center_crop(a, margin).permute(1, 2, 0).cpu().numpy()
        mov = b.permute(1, 2, 0).cpu().numpy()
        res = np.empty_like(mov)
        for ch in range(3):
            # map via sorted quantiles of the central region, applied to the full padded band
            src = np.sort(center_crop(b, margin)[ch].cpu().numpy().ravel())
            tgt = np.sort(ref[..., ch].ravel())
            res[..., ch] = np.interp(mov[..., ch], src, tgt)
        out.append(torch.from_numpy(res).permute(2, 0, 1))
    return torch.stack(out).to(x2.device, x2.dtype)


def rrn(x1: torch.Tensor, x2: torch.Tensor, iters: int = 3, k: float = 2.0, margin: int = MARGIN):
    """Relative radiometric normalisation, PIF style: per band, x2' = a*x2 + c with a = sigma1/sigma2, c = m1 - a*m2
    (mean-std matching) computed on pseudo-invariant pixels, selected iteratively as |residual| < k * 1.4826 * MAD.
    Mean-std (not OLS) because OLS of x1 on x2 suffers regression dilution on multi-year pairs: with LEVIR's median
    A-B correlation of 0.34 the OLS slope was 0.48 and halved B's contrast (first implementation, fixed)."""
    a1 = center_crop(x1, margin).flatten(2)   # (B,3,P)
    a2 = center_crop(x2, margin).flatten(2)
    w = torch.ones_like(a1)
    for _ in range(iters):
        sw = w.sum(-1).clamp_min(1)
        m1, m2 = (w * a1).sum(-1) / sw, (w * a2).sum(-1) / sw
        s1 = ((w * (a1 - m1[..., None]) ** 2).sum(-1) / sw).sqrt()
        s2 = ((w * (a2 - m2[..., None]) ** 2).sum(-1) / sw).sqrt().clamp_min(1e-4)
        a = s1 / s2
        c = m1 - a * m2
        res = a1 - (a[..., None] * a2 + c[..., None])
        mad = (res - res.median(-1, keepdim=True).values).abs().median(-1, keepdim=True).values.clamp_min(1e-4)
        w = ((res - res.median(-1, keepdim=True).values).abs() < k * 1.4826 * mad).float()
    return (a[..., None, None] * x2 + c[..., None, None]).clamp(0, 1)
