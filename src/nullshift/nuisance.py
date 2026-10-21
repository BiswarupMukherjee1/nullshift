"""Physically parameterised acquisition nuisances (NSB-v1).

All functions take float tensors in [0, 1] of shape (B, 3, H, W) that INCLUDE a margin (default 64 px) around
the evaluated 256x256 centre, so geometric warps never wrap real content into the evaluated region.
Sign convention for geometry: a displacement d = (dy, dx) moves image CONTENT by +d, i.e. out(p) = in(p - d).

Every stochastic parameter is drawn from a per-sample torch.Generator seeded by `nuisance_seed(crop_id, family, dose)`,
so any null pair can be regenerated bit-exactly without storing images.
"""
from __future__ import annotations

import hashlib
import io
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

MARGIN = 64

# ---------------------------------------------------------------- dose grids (index 0 = identity, 1..5 = increasing)
FAMILIES = {
    # geometric
    "G1_shift":    {"kind": "geo", "unit": "px (|d|, random direction)", "doses": [0, 0.25, 0.5, 1.0, 2.0, 4.0]},
    "G2_field":    {"kind": "geo", "unit": "px RMS, zero-mean smooth field (L=32 px)", "doses": [0, 0.25, 0.5, 1.0, 2.0, 4.0]},
    "G3_parallax": {"kind": "geo", "unit": "px max roof displacement", "doses": [0, 0.5, 1.0, 2.0, 4.0, 8.0]},
    # radiometric
    "R1_gain":     {"kind": "rad", "unit": "|g-1|", "doses": [0, 0.02, 0.05, 0.10, 0.20, 0.30]},
    "R2_gamma":    {"kind": "rad", "unit": "|log gamma|", "doses": [0, 0.05, 0.10, 0.20, 0.30, 0.50]},
    "R3_haze":     {"kind": "rad", "unit": "optical depth (green), Rayleigh-weighted, spatial ramp", "doses": [0, 0.02, 0.05, 0.10, 0.20, 0.30]},
    "R4_colour":   {"kind": "rad", "unit": "delta of 3x3 mixing matrix I + delta*U(-1,1)", "doses": [0, 0.02, 0.05, 0.10, 0.20, 0.30]},
    "R5_jpeg":     {"kind": "rad", "unit": "JPEG quality (lower = stronger)", "doses": [None, 90, 75, 50, 30, 15]},
    "R6_noise":    {"kind": "rad", "unit": "Gaussian sigma (0-1 scale)", "doses": [0, 0.005, 0.01, 0.02, 0.04, 0.08]},
    "R7_blur":     {"kind": "rad", "unit": "Gaussian PSF sigma (px)", "doses": [0, 0.3, 0.5, 0.8, 1.2, 2.0]},
    # compound: G2 field + R4 colour + independent R6 noise on BOTH dates (dose index shared)
    "C1_compound": {"kind": "mix", "unit": "dose index of G2/R4/R6", "doses": [0, 1, 2, 3, 4, 5]},
}
N_DOSES = 6

WAVELENGTH_UM = torch.tensor([0.65, 0.55, 0.47])   # R, G, B band centres
AIRLIGHT = torch.tensor([0.78, 0.80, 0.86])        # slightly bluish path radiance


def nuisance_seed(crop_id: str, family: str, dose_idx: int) -> int:
    h = hashlib.sha256(f"{crop_id}|{family}|{dose_idx}".encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def center_crop(x: torch.Tensor, margin: int = MARGIN) -> torch.Tensor:
    return x[..., margin:x.shape[-2] - margin, margin:x.shape[-1] - margin] if margin > 0 else x


# ---------------------------------------------------------------- geometric primitives
def fourier_shift(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Sub-pixel translation by phase ramp. x: (B,C,H,W); d: (B,2) = (dy, dx) in px. out(p) = x(p - d) (circular)."""
    B, C, H, W = x.shape
    d = d.to(x.device, torch.float64)
    ky = torch.fft.fftfreq(H, device=x.device, dtype=torch.float64).view(1, H, 1)
    kx = torch.fft.fftfreq(W, device=x.device, dtype=torch.float64).view(1, 1, W)
    phase = torch.exp(-2j * math.pi * (ky * d[:, 0].view(B, 1, 1) + kx * d[:, 1].view(B, 1, 1)))
    X = torch.fft.fft2(x.to(torch.float64))
    return torch.fft.ifft2(X * phase[:, None]).real.to(x.dtype)


def warp_field(x: torch.Tensor, u: torch.Tensor, mode: str = "bicubic") -> torch.Tensor:
    """Backward warp: out(p) = x(p - u(p)). u: (B,2,H,W) = (uy, ux) in px."""
    B, C, H, W = x.shape
    yy, xx = torch.meshgrid(torch.arange(H, device=x.device, dtype=x.dtype),
                            torch.arange(W, device=x.device, dtype=x.dtype), indexing="ij")
    sy = yy[None] - u[:, 0]
    sx = xx[None] - u[:, 1]
    grid = torch.stack([2 * sx / (W - 1) - 1, 2 * sy / (H - 1) - 1], dim=-1)
    return F.grid_sample(x, grid, mode=mode, padding_mode="border", align_corners=True)


def _gauss_smooth_fft(n: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian filter of the last two dims via FFT (circular; used only on random noise)."""
    H, W = n.shape[-2:]
    ky = torch.fft.fftfreq(H, dtype=torch.float64).view(H, 1)
    kx = torch.fft.fftfreq(W, dtype=torch.float64).view(1, W)
    g = torch.exp(-2 * math.pi ** 2 * sigma ** 2 * (ky ** 2 + kx ** 2))
    return torch.fft.ifft2(torch.fft.fft2(n.to(torch.float64)) * g).real.to(torch.float32)


def random_field(H: int, W: int, rms: float, corr_len: float, gen: torch.Generator, margin: int = MARGIN) -> torch.Tensor:
    """Zero-mean (over the evaluated centre) smooth displacement field with vector RMS = rms. Returns (2,H,W)."""
    n = torch.randn(2, H, W, generator=gen)
    u = _gauss_smooth_fft(n, corr_len)
    c = center_crop(u, margin)
    u = u - c.mean(dim=(-2, -1), keepdim=True)
    c = center_crop(u, margin)
    cur = torch.sqrt((c ** 2).sum(0).mean()).clamp_min(1e-12)
    return u * (rms / cur)


def parallax_field(mask: np.ndarray, dmax: float, gen: torch.Generator, edge_sigma: float = 1.0) -> torch.Tensor:
    """Height-proportional roof displacement proxy. mask: (H,W) bool building footprints.
    Each connected footprint gets height factor h~U(0.5,1) (so max displacement = dmax); common view direction."""
    H, W = mask.shape
    lab, n = ndimage.label(mask)
    theta = float(torch.rand(1, generator=gen)) * 2 * math.pi
    hts = (0.5 + 0.5 * torch.rand(max(n, 1), generator=gen)).numpy()
    hmap = np.zeros((H, W), np.float32)
    if n > 0:
        hmap = np.concatenate([[0.0], hts]).astype(np.float32)[lab]
    if edge_sigma > 0:
        hmap = ndimage.gaussian_filter(hmap, edge_sigma)
    u = np.stack([hmap * dmax * math.sin(theta), hmap * dmax * math.cos(theta)])
    return torch.from_numpy(u.astype(np.float32))


# ---------------------------------------------------------------- radiometric primitives
def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    r = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    C = x.shape[1]
    xp = F.pad(x, (r, r, r, r), mode="reflect")
    xp = F.conv2d(xp, k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(xp, k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)


def jpeg(x: torch.Tensor, quality: int, offsets=None) -> torch.Tensor:
    """JPEG re-encode. offsets: per-image (oy, ox) in 1..7 -> 8x8 block grid NOT aligned with the source's grid.
    (LEVIR-CD tiles already carry a q~75 JPEG history on their original grid; aligned re-encoding at q75 is nearly
    idempotent, which made the dose non-monotonic — see Data_Created/NSB-v1/DATASHEET.md.)"""
    if offsets is not None:
        out = x.clone()
        for i, (oy, ox) in enumerate(offsets):
            out[i:i + 1, :, oy:, ox:] = jpeg(x[i:i + 1, :, oy:, ox:], quality)
        return out
    out = []
    for im in x:
        a = (im.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, format="JPEG", quality=int(quality))
        buf.seek(0)
        b = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
        out.append(torch.from_numpy(b).permute(2, 0, 1))
    return torch.stack(out).to(x.device, x.dtype)


def haze(x: torch.Tensor, tau_g: float, gen: torch.Generator) -> torch.Tensor:
    """I' = I*t + A*(1-t), t_b = exp(-tau_b), tau_b = tau_g * (lambda_b/0.55)^-4 * ramp(p); ramp in [0.5,1.5] along a random direction."""
    _, _, H, W = x.shape
    theta = float(torch.rand(1, generator=gen)) * 2 * math.pi
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    proj = math.cos(theta) * xx + math.sin(theta) * yy
    ramp = 1.0 + 0.5 * proj / proj.abs().max().clamp_min(1e-6)
    tau = tau_g * (WAVELENGTH_UM / 0.55) ** -4
    t = torch.exp(-tau.view(3, 1, 1) * ramp[None]).to(x.device, x.dtype)
    A = AIRLIGHT.view(3, 1, 1).to(x.device, x.dtype)
    return x * t + A * (1 - t)


# ---------------------------------------------------------------- family dispatcher
def apply_nuisance(x: torch.Tensor, family: str, dose_idx: int, seeds, masks=None, ref: torch.Tensor | None = None):
    """Return (x1, x2) with x2 = T(x) for padded images x (B,3,H,W) in [0,1].
    x1 is the reference: `ref` if given (perturbed-pair setting, ref = t1 image) else x (null-pair setting);
    it is unchanged except for the independent C1 sensor noise. masks: (B,H,W) bool footprints (G3 only)."""
    x1 = x if ref is None else ref
    if dose_idx == 0:
        return x1, x.clone()
    dose = FAMILIES[family]["doses"][dose_idx]
    B, _, H, W = x.shape
    gens = [_gen(s) for s in seeds]
    if family == "G1_shift":
        ang = torch.stack([torch.rand(1, generator=g) for g in gens]).view(B) * 2 * math.pi
        d = torch.stack([dose * torch.sin(ang), dose * torch.cos(ang)], dim=1)
        x2 = fourier_shift(x, d)
    elif family == "G2_field":
        u = torch.stack([random_field(H, W, dose, 32.0, g) for g in gens]).to(x.device)
        x2 = warp_field(x, u)
    elif family == "G3_parallax":
        assert masks is not None, "G3 needs building masks"
        u = torch.stack([parallax_field(m.cpu().numpy().astype(bool), dose, g) for m, g in zip(masks, gens)]).to(x.device)
        x2 = warp_field(x, u)
    elif family == "R1_gain":
        sgn = torch.stack([torch.randint(0, 2, (1,), generator=g) for g in gens]).view(B, 1, 1, 1).float() * 2 - 1
        x2 = x * (1 + sgn.to(x.device) * dose)
    elif family == "R2_gamma":
        sgn = torch.stack([torch.randint(0, 2, (1,), generator=g) for g in gens]).view(B, 1, 1, 1).float() * 2 - 1
        x2 = x.clamp_min(0) ** torch.exp(sgn.to(x.device) * dose)
    elif family == "R3_haze":
        x2 = torch.cat([haze(x[i:i + 1], dose, gens[i]) for i in range(B)])
    elif family == "R4_colour":
        M = torch.stack([torch.eye(3) + dose * (2 * torch.rand(3, 3, generator=g) - 1) for g in gens]).to(x.device, x.dtype)
        x2 = torch.einsum("bij,bjhw->bihw", M, x)
    elif family == "R5_jpeg":
        offs = [tuple(int(v) for v in torch.randint(1, 8, (2,), generator=g)) for g in gens]
        x2 = jpeg(x, dose, offs)
    elif family == "R6_noise":
        n = torch.stack([torch.randn(3, H, W, generator=g) for g in gens]).to(x.device, x.dtype)
        x2 = x + dose * n
    elif family == "R7_blur":
        x2 = gaussian_blur(x, dose)
    elif family == "C1_compound":
        g2 = FAMILIES["G2_field"]["doses"][dose]
        r4 = FAMILIES["R4_colour"]["doses"][dose]
        r6 = FAMILIES["R6_noise"]["doses"][dose]
        u = torch.stack([random_field(H, W, g2, 32.0, g) for g in gens]).to(x.device)
        x2 = warp_field(x, u)
        M = torch.stack([torch.eye(3) + r4 * (2 * torch.rand(3, 3, generator=g) - 1) for g in gens]).to(x.device, x.dtype)
        x2 = torch.einsum("bij,bjhw->bihw", M, x2)
        n2 = torch.stack([torch.randn(3, H, W, generator=g) for g in gens]).to(x.device, x.dtype)
        n1 = torch.stack([torch.randn(3, H, W, generator=g) for g in gens]).to(x.device, x.dtype)
        x2 = x2 + r6 * n2
        x1 = (x1 + r6 * n1).clamp(0, 1)
    else:
        raise KeyError(family)
    return x1, x2.clamp(0, 1)
