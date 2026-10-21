"""NullShift operators on feature residuals r = z2 - z1 (shape (B,C,h,w)).

geo        : per-patch projection off span J(p), J = d f(shift(I1,d))/d d estimated by central differences (rank 2).
geo4       : same with J from I1 and I2 stacked (rank 4).
geo-smooth : one global displacement d per image, least squares over all patches (feature-space Lucas-Kanade step).
rad        : NFAD-style global subspace V (uncentred second moment of intervention displacements), r - V V^T r.
gate       : NFAD incidence gate, r - gamma(x) Vg Vg^T r, gamma = min(1, PR(e)/tau), e_p = ||r(p)||^2.
"""
from __future__ import annotations

import math

import torch

from .nuisance import MARGIN, center_crop, fourier_shift


# ------------------------------------------------------------------ tangents
@torch.no_grad()
def shift_tangents(enc, x_pad: torch.Tensor, eps: float = 1.0, margin: int = MARGIN) -> torch.Tensor:
    """Central-difference Jacobian of patch features w.r.t. image translation.
    x_pad: (B,3,H+2m,W+2m). Returns J: (B,C,2,h,w), columns = (d/d dy, d/d dx), units 1/px."""
    B = x_pad.shape[0]
    dev = x_pad.device
    shifts = torch.tensor([[eps, 0.0], [-eps, 0.0], [0.0, eps], [0.0, -eps]], device=dev)
    xs = torch.cat([fourier_shift(x_pad, s.expand(B, 2)).clamp(0, 1) for s in shifts])
    z = enc(center_crop(xs, margin))
    zyp, zym, zxp, zxm = z.split(B)
    return torch.stack([(zyp - zym) / (2 * eps), (zxp - zxm) / (2 * eps)], dim=2)


def _flat(J: torch.Tensor):
    B, C, k, h, w = J.shape
    return J.permute(0, 3, 4, 1, 2).reshape(B * h * w, C, k)


def project_tangent(r: torch.Tensor, J: torch.Tensor, ridge_rel: float = 1e-3):
    """Remove the least-squares fit J(p) a(p) from r(p) independently per patch.
    Returns (r_proj, a) with a: (B,k,h,w) the per-patch coefficients (px displacement for shift tangents).
    A small ridge (relative to the mean tangent energy) keeps near-flat patches from projecting random directions."""
    B, C, k, h, w = J.shape
    Jf = _flat(J).double()                                  # (N,C,k)
    rf = r.permute(0, 2, 3, 1).reshape(-1, C, 1).double()    # (N,C,1)
    G = Jf.transpose(1, 2) @ Jf                              # (N,k,k)
    lam = ridge_rel * G.diagonal(dim1=1, dim2=2).mean()
    a = torch.linalg.solve(G + lam * torch.eye(k, device=G.device, dtype=G.dtype), Jf.transpose(1, 2) @ rf)
    rp = (rf - Jf @ a).squeeze(-1).float().reshape(B, h, w, C).permute(0, 3, 1, 2)
    return rp.contiguous(), a.squeeze(-1).float().reshape(B, h, w, k).permute(0, 3, 1, 2)


def project_global_translation(r: torch.Tensor, J: torch.Tensor):
    """One displacement per image: d = argmin sum_p ||r(p) - J(p) d||^2. Returns (r - J d, d (B,k))."""
    B, C, k, h, w = J.shape
    Jb = J.permute(0, 3, 4, 1, 2).reshape(B, h * w * C, k).double()
    rb = r.permute(0, 2, 3, 1).reshape(B, h * w * C, 1).double()
    G = Jb.transpose(1, 2) @ Jb
    d = torch.linalg.solve(G + 1e-9 * torch.eye(k, device=G.device, dtype=G.dtype), Jb.transpose(1, 2) @ rb)
    rp = r - torch.einsum("bckhw,bk->bchw", J, d.squeeze(-1).float())
    return rp, d.squeeze(-1).float()


# ------------------------------------------------------------------ global subspaces (NFAD transfer)
class SecondMoment:
    """Accumulates the uncentred second moment S = E[dz dz^T] of feature displacements (per family)."""

    def __init__(self, C: int, device="cuda"):
        self.S = torch.zeros(C, C, dtype=torch.float64, device=device)
        self.n = 0

    def add(self, dz: torch.Tensor):
        X = dz.permute(0, 2, 3, 1).reshape(-1, dz.shape[1]).double()
        self.S += X.T @ X
        self.n += X.shape[0]

    def top(self, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        evals, evecs = torch.linalg.eigh(self.S / max(self.n, 1))
        order = torch.argsort(evals, descending=True)
        return evecs[:, order[:k]].float(), evals[order].float()


def stack_subspaces(bases: list[torch.Tensor]) -> torch.Tensor:
    """QR-orthonormalise stacked bases (NFAD: per-family top-k, then QR)."""
    Q, R = torch.linalg.qr(torch.cat(bases, dim=1).double())
    keep = R.diagonal().abs() > 1e-6
    return Q[:, keep].float()


def project_global(r: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    c = torch.einsum("ck,bchw->bkhw", V, r)
    return r - torch.einsum("ck,bkhw->bchw", V, c)


def participation_ratio_columns(V: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    """NFAD rho_j = Var_i(mean_p c_ipj) / Var_{i,p}(c_ipj), c = v_j^T z(x_i, p). Z: (N,C,h,w) clean features."""
    c = torch.einsum("ck,nchw->nkhw", V, Z).flatten(2)       # (N,k,P)
    return c.mean(-1).var(0) / c.permute(1, 0, 2).reshape(V.shape[1], -1).var(-1).clamp_min(1e-12)


def select_global_columns(V: torch.Tensor, Z: torch.Tensor, q: float = 0.25) -> torch.Tensor:
    rho = participation_ratio_columns(V, Z)
    return V[:, rho > torch.quantile(rho, q)]


def incidence_gate(r: torch.Tensor, Vg: torch.Tensor, tau_pr: float = 0.5):
    """NFAD per-image gate on the spatial concentration of residual energy. Returns (r', gamma (B,))."""
    e = (r ** 2).sum(1).flatten(1)                           # (B,P)
    P = e.shape[1]
    phi = e.sum(1) ** 2 / (P * (e ** 2).sum(1)).clamp_min(1e-12)   # in [1/P, 1]; 1 = spatially uniform
    gamma = (phi / tau_pr).clamp(max=1.0)
    c = torch.einsum("ck,bchw->bkhw", Vg, r)
    return r - gamma.view(-1, 1, 1, 1) * torch.einsum("ck,bkhw->bchw", Vg, c), gamma


def score(r: torch.Tensor) -> torch.Tensor:
    """Change score per patch = ||r(p)||, (B,h,w)."""
    return r.norm(dim=1)


def localmin_score(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Symmetric 3x3-neighbourhood minimum feature distance (handles >= 1-patch shifts)."""
    def one(a, b):
        B, C, h, w = b.shape
        bp = torch.nn.functional.pad(b, (1, 1, 1, 1), mode="replicate")
        nb = torch.nn.functional.unfold(bp, 3).view(B, C, 9, h, w)
        return (a[:, :, None] - nb).norm(dim=1).min(dim=1).values
    return 0.5 * (one(z2, z1) + one(z1, z2))


def project_joint(r: torch.Tensor, J: torch.Tensor, V: torch.Tensor, ridge_rel: float = 1e-3):
    """Exact projection off span([J(p) | V]) per patch (NullShift full = geo + rad).
    span(V, J) = span(V, J_perp) with J_perp = J - V V^T J, and V is orthogonal to J_perp, so
    r' = P_{J_perp}^perp (r - V V^T r). (Sequential geo-then-rad would not be a projection.)"""
    rv = project_global(r, V)
    Jv = J - torch.einsum("ck,bkjhw->bcjhw", V, torch.einsum("ck,bcjhw->bkjhw", V, J))
    return project_tangent(rv, Jv, ridge_rel)[0]


# ------------------------------------------------------------------ probe (secant) subspace — E01 fallback
def probe_displacements(n_dir: int = 8, mags=(0.5, 1.0, 2.0)) -> torch.Tensor:
    """K = n_dir * len(mags) probe shifts (dy, dx) on rings; directions offset by half a step between rings."""
    out = []
    for j, m in enumerate(mags):
        for i in range(n_dir):
            a = 2 * math.pi * (i + 0.5 * (j % 2)) / n_dir
            out.append([m * math.sin(a), m * math.cos(a)])
    return torch.tensor(out)


@torch.no_grad()
def probe_responses(enc, x_pad: torch.Tensor, z0: torch.Tensor, disps: torch.Tensor, margin: int = MARGIN):
    """D(p) = [f(shift(x, d_j)) - f(x)]_j : (B,C,K,h,w). Matched intervention responses, per patch, no linearity assumed."""
    B = x_pad.shape[0]
    zs = []
    for d in disps.to(x_pad.device):
        zs.append(enc(center_crop(fourier_shift(x_pad, d.expand(B, 2)).clamp(0, 1), margin)) - z0)
    return torch.stack(zs, dim=2)


def probe_basis(D: torch.Tensor, rank: int, rel_floor: float = 1e-6, chunk: int = 1024) -> torch.Tensor:
    """Per-patch orthonormal basis of the top-`rank` left singular vectors of D(p) (via the K x K Gram).
    Processed in chunks of `chunk` patches so float64 memory stays bounded (E08 OOM with 40 probes x 16 tiles)."""
    B, C, K, h, w = D.shape
    Dflat = _flat(D)                                           # (N,C,K) float
    out = torch.empty(Dflat.shape[0], C, rank, device=D.device, dtype=torch.float32)
    for i in range(0, Dflat.shape[0], chunk):
        Df = Dflat[i:i + chunk].double()
        lam, W = torch.linalg.eigh(Df.transpose(1, 2) @ Df)    # ascending
        lam, W = lam.flip(-1)[:, :rank], W.flip(-1)[:, :, :rank]
        ok = lam > rel_floor * lam[:, :1].clamp_min(1e-30)
        U = (Df @ W) / lam.clamp_min(1e-30).sqrt()[:, None, :]
        out[i:i + chunk] = (U * ok[:, None, :]).float()        # drop numerically empty directions
        del Df, U
    return out.reshape(B, h, w, C, rank).permute(0, 3, 4, 1, 2).contiguous()


def project_basis(r: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """r - U U^T r with per-patch orthonormal U (B,C,k,h,w)."""
    c = torch.einsum("bckhw,bchw->bkhw", U, r)
    return r - torch.einsum("bckhw,bkhw->bchw", U, c)


# ------------------------------------------------------------------ radiometric probe subspace (radP) — Phase 1b
RAD_PROBES = [("gain", 0.9), ("gain", 1.1), ("gamma", 0.85), ("gamma", 1.18), ("haze", 0.08),
              ("colour", 0), ("colour", 1), ("blur", 0.8), ("jpeg", 50)]


@torch.no_grad()
def rad_probe_images(x_pad: torch.Tensor) -> list:
    """Deterministic, content-preserving radiometric interventions of the reference (same for every image), mirroring
    the NFAD intervention families but used per image & per patch instead of for one global subspace."""
    from .nuisance import gaussian_blur, haze, jpeg
    out = []
    for kind, v in RAD_PROBES:
        if kind == "gain":
            y = x_pad * v
        elif kind == "gamma":
            y = x_pad.clamp_min(0) ** v
        elif kind == "haze":
            y = haze(x_pad, v, torch.Generator().manual_seed(11))
        elif kind == "colour":
            g = torch.Generator().manual_seed(100 + v)
            Mx = (torch.eye(3) + 0.1 * (2 * torch.rand(3, 3, generator=g) - 1)).to(x_pad.device, x_pad.dtype)
            y = torch.einsum("ij,bjhw->bihw", Mx, x_pad)
        elif kind == "blur":
            y = gaussian_blur(x_pad, v)
        else:
            y = jpeg(x_pad, int(v))
        out.append(y.clamp(0, 1))
    return out


@torch.no_grad()
def rad_probe_responses(enc, x_pad: torch.Tensor, z0: torch.Tensor, margin: int = MARGIN) -> torch.Tensor:
    """D_rad(p) = [f(T_k(x)) - f(x)]_k : (B,C,K_rad,h,w)."""
    return torch.stack([enc(center_crop(y, margin)) - z0 for y in rad_probe_images(x_pad)], dim=2)


# ------------------------------------------------------------------ coarse-to-fine (E06): neighbourhood search + probe projection
def _shift_map(t: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Token map shifted by (dy, dx) tokens with replicate borders: out[..., i, j] = t[..., i+dy, j+dx]."""
    lead, (h, w) = t.shape[:-2], t.shape[-2:]
    f = t.reshape(-1, 1, h, w) if t.dim() != 4 else t
    f = f.reshape(1, -1, h, w)
    fp = torch.nn.functional.pad(f, (1, 1, 1, 1), mode="replicate")
    out = fp[..., 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return out.reshape(*lead, h, w)


def localmin_scores(z1: torch.Tensor, z2: torch.Tensor, U1: torch.Tensor | None = None,
                    V: torch.Tensor | None = None) -> torch.Tensor:
    """One-directional coarse-to-fine score: min over the 3x3 token neighbourhood q of p of
    || P_perp(U1(q)) (z2(p) - z1(q)) ||  (U1 = None -> plain distance, i.e. one-directional localmin).
    Coarse search absorbs multi-pixel (up to one token = 16 px) displacement, the probe projection the sub-pixel rest."""
    best = None
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            z1s = _shift_map(z1, dy, dx)
            r = z2 - z1s
            if U1 is not None and V is not None:          # joint: matched per-patch basis + global subspace V
                r = project_joint(r, _shift_map(U1, dy, dx), V)
            elif U1 is not None:
                r = project_basis(r, _shift_map(U1, dy, dx))
            s = r.norm(dim=1)
            best = s if best is None else torch.minimum(best, s)
    return best
