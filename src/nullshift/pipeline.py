"""Evaluation engine: computes the change-score map of every method for a batch of (reference, target) pairs.

Reference-side quantities (z1, shift tangents J1, blurred z1) are computed once per crop batch and reused across all
nuisance conditions (`Ref`), which is what makes dose sweeps affordable.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import baselines as bl
from . import operators as op
from .nuisance import MARGIN, center_crop, gaussian_blur

# geoP = per-patch probe subspace (E01b fallback after H1 failed); geo = the proposal's rank-2 tangent (kept for honesty).
# geo4 (tangents from both dates) is available but not in the default list (E02 ablation only).
FEATURE_METHODS = ["raw", "geo", "geoP", "geo-smooth", "rad", "geo+rad", "geoP+rad", "shiftV", "shiftV+gate", "rad+gate",
                   "localmin"]
PIXEL_METHODS = ["blur", "phasecorr", "histmatch", "rrn", "classic", "classic+ns"]
ALL_METHODS = FEATURE_METHODS + PIXEL_METHODS


@dataclass
class Ref:
    x1: torch.Tensor           # padded reference images
    z1: torch.Tensor           # (B,C,h,w)
    J1: torch.Tensor | None    # (B,C,2,h,w)   shift tangents
    U1: torch.Tensor | None    # (B,C,r,h,w)   probe-subspace basis (orthonormal per patch)
    z1_blur: torch.Tensor | None
    zbar1: torch.Tensor | None = None   # shift-averaged reference features (shiftavg baseline)
    UR1: torch.Tensor | None = None     # per-patch radiometric probe basis (radP)
    UG1: torch.Tensor | None = None     # joint per-patch geometric+radiometric probe basis (geoP+radP)
    UB1: torch.Tensor | None = None     # probe basis of the BLURRED reference (blur+geoP)
    UM1: torch.Tensor | None = None     # multi-scale probe basis, rings 0.5..8 px (geoPms, E06)
    UC1: torch.Tensor | None = None     # cheap c2f basis: 16 probes (rings 1, 4 px), rank 8 (E07 recommendation)
    UCR1: torch.Tensor | None = None    # c2f basis + 9 radiometric probes, rank 12 (E08)


class Engine:
    def __init__(self, enc, subspaces: dict, eps: float = 1.0, tau_pr: float = 0.5, ridge_rel: float = 1e-3,
                 blur_sigma: float = 1.0, margin: int = MARGIN, probe_ndir: int = 8, probe_mags=(0.5, 1.0, 2.0),
                 probe_rank: int = 8, rad_rank: int = 4):
        self.enc, self.S = enc, subspaces
        self.eps, self.tau_pr, self.ridge, self.blur_sigma, self.m = eps, tau_pr, ridge_rel, blur_sigma, margin
        self.disps = op.probe_displacements(probe_ndir, probe_mags)
        self.probe_rank = probe_rank
        self.rad_rank = rad_rank

    def f(self, x):
        return self.enc(center_crop(x, self.m))

    def reference(self, x1, methods) -> Ref:
        need_J = any(m in methods for m in ["geo", "geo4", "geo-smooth", "geo+rad"])
        need_U = any(m in methods for m in ["geoP", "geoP+rad", "classic+ns", "classic+geoP", "shiftavg+geoP",
                                            "localmin+geoP"])
        z1 = self.f(x1)
        J1 = op.shift_tangents(self.enc, x1, self.eps, self.m) if need_J else None
        U1 = UR1 = UG1 = None
        need_R = any(m in methods for m in ["radP", "geoP+radP"])
        Dg = op.probe_responses(self.enc, x1, z1, self.disps, self.m) if (need_U or "geoP+radP" in methods) else None
        if need_U:
            U1 = op.probe_basis(Dg, self.probe_rank)
        if need_R:
            Dr = op.rad_probe_responses(self.enc, x1, z1, self.m)
            if "radP" in methods:
                UR1 = op.probe_basis(Dr, self.rad_rank)
            if "geoP+radP" in methods:
                UG1 = op.probe_basis(torch.cat([Dg, Dr], 2), self.probe_rank + self.rad_rank)
            del Dr
        del Dg
        z1b = self.f(gaussian_blur(x1, self.blur_sigma)) if ("blur" in methods or "blur+geoP" in methods) else None
        UB1 = None
        if "blur+geoP" in methods:
            xb = gaussian_blur(x1, self.blur_sigma)
            UB1 = op.probe_basis(op.probe_responses(self.enc, xb, z1b, self.disps, self.m), self.probe_rank)
        zbar1 = self.shift_avg(x1, z1) if any(m.startswith("shiftavg") for m in methods) else None
        UM1 = None
        if any(m in methods for m in ["geoPms", "localmin+geoPms"]):
            UM1 = op.probe_basis(op.probe_responses(self.enc, x1, z1, op.probe_displacements(8, (0.5, 1.0, 2.0, 4.0, 8.0)),
                                                    self.m), 16)
        UC1 = UCR1 = None
        if any(m in methods for m in ["c2f", "c2f+radP", "c2f+rad"]):
            Dc = op.probe_responses(self.enc, x1, z1, op.probe_displacements(8, (1.0, 4.0)), self.m)
            UC1 = op.probe_basis(Dc, 8)
            if "c2f+radP" in methods:
                UCR1 = op.probe_basis(torch.cat([Dc, op.rad_probe_responses(self.enc, x1, z1, self.m)], 2), 12)
            del Dc
        return Ref(x1, z1, J1, U1, z1b, zbar1, UR1, UG1, UB1, UM1, UC1, UCR1)

    def shift_avg(self, x, z0=None):
        """Test-time shift averaging (PatchShift / DVT-style baseline): mean of f over x and its 8 one-pixel shifts."""
        z0 = self.f(x) if z0 is None else z0
        acc = z0.clone()
        for d in op.probe_displacements(8, (1.0,)).to(x.device):
            acc += self.f(torch.nan_to_num(op.fourier_shift(x, d.expand(x.shape[0], 2))).clamp(0, 1))
        return torch.nn.functional.normalize(acc / 9, dim=1)

    @torch.no_grad()
    def scores(self, ref: Ref, x2: torch.Tensor, methods, return_residual=False) -> dict:
        S, out = self.S, {}
        z2 = self.f(x2)
        r = z2 - ref.z1
        if "raw" in methods:
            out["raw"] = op.score(r)
        if "geo" in methods:
            out["geo"] = op.score(op.project_tangent(r, ref.J1, self.ridge)[0])
        if "geoP" in methods:
            out["geoP"] = op.score(op.project_basis(r, ref.U1))
        if "geoP+rad" in methods:
            out["geoP+rad"] = op.score(op.project_joint(r, ref.U1, S["V_rad"], self.ridge))
        if "shiftavg" in methods or "shiftavg+geoP" in methods:
            zbar2 = self.shift_avg(x2, z2)
            rs = zbar2 - ref.zbar1
            if "shiftavg" in methods:
                out["shiftavg"] = op.score(rs)
            if "shiftavg+geoP" in methods:
                out["shiftavg+geoP"] = op.score(op.project_basis(rs, ref.U1))
        if "radP" in methods:
            out["radP"] = op.score(op.project_basis(r, ref.UR1))
        if "geoP+radP" in methods:
            out["geoP+radP"] = op.score(op.project_basis(r, ref.UG1))
        if "geo4" in methods:
            J2 = op.shift_tangents(self.enc, x2, self.eps, self.m)
            out["geo4"] = op.score(op.project_tangent(r, torch.cat([ref.J1, J2], 2), self.ridge)[0])
        if "geo-smooth" in methods:
            out["geo-smooth"] = op.score(op.project_global_translation(r, ref.J1)[0])
        if "rad" in methods:
            out["rad"] = op.score(op.project_global(r, S["V_rad"]))
        if "geo+rad" in methods:
            out["geo+rad"] = op.score(op.project_joint(r, ref.J1, S["V_rad"], self.ridge))
        if "shiftV" in methods:
            out["shiftV"] = op.score(op.project_global(r, S["V_shift"]))
        if "shiftV+gate" in methods:
            out["shiftV+gate"] = op.score(op.incidence_gate(r, S["V_shift_g"], self.tau_pr)[0])
        if "rad+gate" in methods:
            out["rad+gate"] = op.score(op.incidence_gate(r, S["V_rad_g"], self.tau_pr)[0])
        if "localmin" in methods:
            out["localmin"] = op.localmin_score(ref.z1, z2)
        if "localmin1" in methods:
            out["localmin1"] = op.localmin_scores(ref.z1, z2)
        if "localmin+geoP" in methods:
            out["localmin+geoP"] = op.localmin_scores(ref.z1, z2, ref.U1)
        if "c2f" in methods:
            out["c2f"] = op.localmin_scores(ref.z1, z2, ref.UC1)
        if "c2f+radP" in methods:
            out["c2f+radP"] = op.localmin_scores(ref.z1, z2, ref.UCR1)
        if "c2f+rad" in methods:
            out["c2f+rad"] = op.localmin_scores(ref.z1, z2, ref.UC1, S["V_rad"])
        if "geoPms" in methods:
            out["geoPms"] = op.score(op.project_basis(r, ref.UM1))
        if "localmin+geoPms" in methods:
            out["localmin+geoPms"] = op.localmin_scores(ref.z1, z2, ref.UM1)
        # ---------------- pixel-space baselines (re-encode the corrected target)
        if "blur" in methods or "blur+geoP" in methods:
            rb = self.f(gaussian_blur(x2, self.blur_sigma)) - ref.z1_blur
            if "blur" in methods:
                out["blur"] = op.score(rb)
            if "blur+geoP" in methods:
                out["blur+geoP"] = op.score(op.project_basis(rb, ref.UB1))
        if "phasecorr" in methods or "classic" in methods or "classic+ns" in methods or "classic+geoP" in methods:
            x2p, _ = bl.phasecorr_align(ref.x1, x2, margin=self.m)
            if "phasecorr" in methods:
                out["phasecorr"] = op.score(self.f(x2p) - ref.z1)
            if "classic" in methods or "classic+ns" in methods or "classic+geoP" in methods:
                rc = self.f(bl.rrn(ref.x1, x2p, margin=self.m)) - ref.z1
                out["classic"] = op.score(rc)
                if "classic+geoP" in methods:
                    out["classic+geoP"] = op.score(op.project_basis(rc, ref.U1))
                if "classic+ns" in methods:
                    out["classic+ns"] = op.score(op.project_joint(rc, ref.U1, S["V_rad"], self.ridge))
        if "histmatch" in methods:
            out["histmatch"] = op.score(self.f(bl.histmatch(ref.x1, x2, margin=self.m)) - ref.z1)
        if "rrn" in methods:
            out["rrn"] = op.score(self.f(bl.rrn(ref.x1, x2, margin=self.m)) - ref.z1)
        if return_residual:
            out["_r"], out["_z2"] = r, z2
        return {k: v for k, v in out.items() if k in methods or k.startswith("_")}
