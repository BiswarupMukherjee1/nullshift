"""Unit tests for the parts where a silent bug would invalidate every experiment."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nullshift import baselines, metrics, nuisance, operators  # noqa: E402

torch.manual_seed(0)


def smooth_img(B=2, H=96, W=96):
    x = torch.rand(B, 3, H, W)
    return nuisance.gaussian_blur(x, 2.0).clamp(0, 1)


# ------------------------------------------------------------------ geometry
def test_fourier_shift_zero_is_identity():
    x = torch.rand(2, 3, 64, 64)
    y = nuisance.fourier_shift(x, torch.zeros(2, 2))
    assert torch.allclose(x, y, atol=1e-6)


def test_fourier_shift_integer_matches_roll_and_sign_convention():
    x = torch.rand(1, 3, 64, 64)
    y = nuisance.fourier_shift(x, torch.tensor([[2.0, -3.0]]))   # content moves +2 in y, -3 in x
    assert torch.allclose(y, torch.roll(x, shifts=(2, -3), dims=(-2, -1)), atol=1e-5)


def test_fourier_shift_composes_except_nyquist():
    # On even-sized images the Nyquist bin cannot carry a sub-pixel phase ramp (taking .real scales it by cos(pi d)),
    # so composition is exact only below Nyquist. Verified: 100 % of the error energy sits in the Nyquist row/col.
    x = smooth_img()
    a = nuisance.fourier_shift(nuisance.fourier_shift(x, torch.full((2, 2), 0.3)), torch.full((2, 2), 0.2))
    b = nuisance.fourier_shift(x, torch.full((2, 2), 0.5))
    lp = lambda t: nuisance.center_crop(nuisance.gaussian_blur(t, 1.0), 8)
    assert torch.allclose(lp(a), lp(b), atol=1e-4)


def test_warp_field_constant_matches_fourier_in_interior():
    x = smooth_img(1, 96, 96)
    u = torch.zeros(1, 2, 96, 96)
    u[:, 0], u[:, 1] = 1.0, -2.0
    w = nuisance.warp_field(x, u)
    f = nuisance.fourier_shift(x, torch.tensor([[1.0, -2.0]]))
    assert torch.allclose(nuisance.center_crop(w, 16), nuisance.center_crop(f, 16), atol=1e-3)


def test_random_field_rms_and_zero_mean():
    g = torch.Generator().manual_seed(1)
    u = nuisance.random_field(384, 384, 1.5, 32.0, g)
    c = nuisance.center_crop(u)
    assert abs(math.sqrt((c ** 2).sum(0).mean()) - 1.5) < 1e-4
    assert c.mean(dim=(-2, -1)).abs().max() < 1e-4


# ------------------------------------------------------------------ nuisances
@pytest.mark.parametrize("fam", list(nuisance.FAMILIES))
def test_dose0_identity_and_determinism(fam):
    x = smooth_img(2, 160, 160)
    masks = torch.zeros(2, 160, 160, dtype=torch.bool)
    masks[:, 60:90, 60:100] = True
    seeds = [nuisance.nuisance_seed(f"c{i}", fam, 3) for i in range(2)]
    x1, x2 = nuisance.apply_nuisance(x, fam, 0, seeds, masks)
    assert torch.equal(x1, x) and torch.equal(x2, x)
    a1, a2 = nuisance.apply_nuisance(x, fam, 3, seeds, masks)
    b1, b2 = nuisance.apply_nuisance(x, fam, 3, seeds, masks)
    assert torch.equal(a2, b2) and torch.equal(a1, b1)
    assert not torch.allclose(a2, x, atol=1e-4), f"{fam} dose 3 did nothing"
    assert a2.min() >= 0 and a2.max() <= 1


def test_seeds_differ_across_crops_and_doses():
    s = {nuisance.nuisance_seed(f"c{i}", f, d) for i in range(20) for f in nuisance.FAMILIES for d in range(6)}
    assert len(s) == 20 * len(nuisance.FAMILIES) * 6


# ------------------------------------------------------------------ operators
def test_project_tangent_is_orthogonal_projection():
    B, C, k, h, w = 2, 64, 2, 4, 4
    J = torch.randn(B, C, k, h, w)
    r = torch.randn(B, C, h, w)
    rp, a = operators.project_tangent(r, J, ridge_rel=0.0)
    # residual orthogonal to every tangent column
    dots = torch.einsum("bckhw,bchw->bkhw", J, rp)
    assert dots.abs().max() < 1e-4
    # idempotent
    rpp, _ = operators.project_tangent(rp, J, ridge_rel=0.0)
    assert torch.allclose(rp, rpp, atol=1e-5)
    # exact recovery of an in-span residual: r = J d -> r' = 0, a = d
    d = torch.randn(B, k, h, w)
    r2 = torch.einsum("bckhw,bkhw->bchw", J, d)
    rp2, a2 = operators.project_tangent(r2, J, ridge_rel=0.0)
    assert rp2.norm() / r2.norm() < 1e-4 and torch.allclose(a2, d, atol=1e-4)


def test_global_translation_recovers_common_d():
    B, C, h, w = 2, 32, 5, 5
    J = torch.randn(B, C, 2, h, w)
    d = torch.tensor([[0.3, -0.7], [1.2, 0.1]])
    r = torch.einsum("bckhw,bk->bchw", J, d)
    rp, dh = operators.project_global_translation(r, J)
    assert torch.allclose(dh, d, atol=1e-4) and rp.norm() < 1e-3


def test_project_global_and_stack():
    C = 50
    V = operators.stack_subspaces([torch.randn(C, 3), torch.randn(C, 3)])
    assert V.shape == (C, 6) and torch.allclose(V.T @ V, torch.eye(6), atol=1e-5)
    r = torch.randn(3, C, 2, 2)
    rp = operators.project_global(r, V)
    assert torch.einsum("ck,bchw->bkhw", V, rp).abs().max() < 1e-4


def test_incidence_gate_range():
    r = torch.zeros(2, 8, 4, 4)
    r[0] = 1.0                     # uniform energy -> phi = 1 -> gamma = 1
    r[1, :, 0, 0] = 1.0            # one patch -> phi = 1/16 -> gamma = 1/8 at tau=0.5
    V = torch.eye(8)[:, :2]
    _, g = operators.incidence_gate(r, V, 0.5)
    assert abs(g[0] - 1) < 1e-6 and abs(g[1] - 0.125) < 1e-6


def test_localmin_zero_for_integer_patch_shift():
    z = torch.randn(1, 8, 6, 6)
    z2 = torch.roll(z, 1, dims=-1)
    s = operators.localmin_score(z, z2)
    assert s[..., 1:-1, 1:-1].abs().max() < 1e-5


# ------------------------------------------------------------------ baselines
def test_rrn_inverts_gain_offset():
    x = smooth_img(1, 160, 160) * 0.8 + 0.1
    x2 = x * 1.2 - 0.05
    y = baselines.rrn(x, x2)
    assert (nuisance.center_crop(y) - nuisance.center_crop(x)).abs().max() < 1e-3


@pytest.mark.parametrize("blur", [0.7, 2.0])     # sharp and very smooth scenes
def test_phasecorr_recovers_shift(blur):
    x = nuisance.gaussian_blur(torch.rand(1, 3, 256, 256), blur)
    x2 = nuisance.fourier_shift(x, torch.tensor([[0.6, -1.3]]))
    y, s = baselines.phasecorr_align(x, x2)
    assert torch.allclose(s, torch.tensor([[-0.6, 1.3]]), atol=0.1)
    assert (nuisance.center_crop(y) - nuisance.center_crop(x)).abs().max() < 0.02


def test_histmatch_inverts_gamma():
    x = smooth_img(1, 160, 160)
    y = baselines.histmatch(x, x ** 1.5)
    assert (nuisance.center_crop(y) - nuisance.center_crop(x)).abs().mean() < 5e-3


# ------------------------------------------------------------------ metrics
def test_hist_metrics_against_bruteforce():
    rng = np.random.default_rng(0)
    pos = torch.from_numpy(rng.normal(0.8, 0.2, 5000).clip(0, 1.99)).float()
    neg = torch.from_numpy(rng.normal(0.4, 0.2, 20000).clip(0, 1.99)).float()
    P, N = metrics.Hist(), metrics.Hist()
    P.add(pos), N.add(neg)
    # AUROC brute force
    ps, ns = pos.numpy(), neg.numpy()
    bf = (ps[:, None] > ns[None, :2000]).mean() + 0.5 * (ps[:, None] == ns[None, :2000]).mean()
    assert abs(metrics.auroc(P, N) - bf) < 0.01
    t, f1 = metrics.best_f1(P, N)
    tau = metrics.EDGES[t]
    tp = (ps >= tau).sum(); fp = (ns >= tau).sum(); fn = (ps < tau).sum()
    assert abs(f1 - 2 * tp / (2 * tp + fp + fn)) < 2e-3
    assert abs(N.frac_above(t) - (ns >= tau).mean()) < 2e-3


def test_patch_fraction():
    lab = torch.zeros(1, 32, 32, dtype=torch.bool)
    lab[0, :16, :8] = True
    f = metrics.patch_fraction(lab)
    assert torch.allclose(f, torch.tensor([[[0.5, 0.0], [0.0, 0.0]]]))


def test_project_joint_matches_stacked_projection():
    B, C, h, w = 2, 40, 3, 3
    J = torch.randn(B, C, 2, h, w)
    V = operators.stack_subspaces([torch.randn(C, 4)])
    r = torch.randn(B, C, h, w)
    rp = operators.project_joint(r, J, V, ridge_rel=0.0)
    Jall = torch.cat([J, V[None, :, :, None, None].expand(B, C, 4, h, w)], dim=2)
    ref, _ = operators.project_tangent(r, Jall, ridge_rel=0.0)
    assert torch.allclose(rp, ref, atol=1e-4)


def test_probe_basis_orthonormal_and_spans_responses():
    B, C, K, h, w = 2, 64, 6, 3, 3
    D = torch.randn(B, C, K, h, w)
    U = operators.probe_basis(D, K)
    G = torch.einsum("bckhw,bcjhw->bkjhw", U, U)
    eye = torch.eye(K)[None, :, :, None, None]
    assert (G - eye).abs().max() < 1e-4
    r = D[:, :, 2]                       # a probe response lies in the span -> removed exactly
    assert operators.project_basis(r, U).norm() / r.norm() < 1e-4
    Ur = operators.probe_basis(D, 2)     # rank-2 truncation leaves orthogonal complement
    rp = operators.project_basis(torch.randn(B, C, h, w), Ur)
    assert torch.einsum("bckhw,bchw->bkhw", Ur, rp).abs().max() < 1e-4


def test_probe_displacements_rings():
    d = operators.probe_displacements(8, (0.5, 1.0))
    assert d.shape == (16, 2)
    assert torch.allclose(d.norm(dim=1), torch.tensor([0.5] * 8 + [1.0] * 8), atol=1e-6)


def test_tau_at_recall():
    P = metrics.Hist()
    P.add(torch.linspace(0.1, 1.0, 1000))
    t = metrics.tau_at_recall(P, 0.9)
    assert abs(P.frac_above(t) - 0.9) < 0.005 and P.frac_above(t) >= 0.9


def test_rrn_no_regression_dilution_on_uncorrelated_pairs():
    # two unrelated scenes with similar statistics: normalisation must not collapse contrast (OLS would give slope ~0)
    x1 = smooth_img(1, 200, 200) * 0.6 + 0.2
    x2 = nuisance.gaussian_blur(torch.rand(1, 3, 200, 200), 2.0) * 0.6 + 0.2
    y = baselines.rrn(x1, x2)
    s1, sy = nuisance.center_crop(x1).std(), nuisance.center_crop(y).std()
    assert 0.6 < float(sy / s1) < 1.6


def test_phasecorr_rejects_implausible_shift():
    x1 = nuisance.gaussian_blur(torch.rand(2, 3, 256, 256), 0.7)
    x2 = nuisance.gaussian_blur(torch.rand(2, 3, 256, 256), 0.7)     # unrelated -> any peak is spurious
    x2[1] = nuisance.fourier_shift(x1[1:2], torch.tensor([[10.0, -10.0]]))[0]   # true but > max_shift
    _, s = baselines.phasecorr_align(x1, x2)
    assert s.norm(dim=1).max() <= 4.0 + 1e-6
    assert s[1].abs().max() == 0          # > 4 px rejected -> no correction
    assert s[0].abs().max() == 0          # unrelated content -> low peak -> no correction


def test_rad_probe_images_deterministic_and_bounded():
    x = smooth_img(2, 160, 160)
    a, b = operators.rad_probe_images(x), operators.rad_probe_images(x)
    assert len(a) == len(operators.RAD_PROBES)
    for u, v in zip(a, b):
        assert torch.equal(u, v) and u.min() >= 0 and u.max() <= 1 and not torch.equal(u, x)


class _FakeEnc(torch.nn.Module):
    """Stride-16 conv 'encoder' (C=32) so every Engine code path can run on CPU in the unit tests."""
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(3, 32, 16, 16)
        self.dim = 32

    @torch.no_grad()
    def forward(self, x, batch=32):
        return torch.nn.functional.normalize(torch.tanh(self.conv(x)), dim=1)


def test_engine_runs_every_method_end_to_end():
    from nullshift import pipeline
    enc = _FakeEnc()
    C = enc.dim
    S = {"V_rad": operators.stack_subspaces([torch.randn(C, 4)]), "V_rad_g": operators.stack_subspaces([torch.randn(C, 3)]),
         "V_shift": operators.stack_subspaces([torch.randn(C, 4)]), "V_shift_g": operators.stack_subspaces([torch.randn(C, 3)])}
    eng = pipeline.Engine(enc, S, margin=32)
    x1 = nuisance.gaussian_blur(torch.rand(2, 3, 128, 128), 1.0).clamp(0, 1)
    x2 = nuisance.fourier_shift(x1, torch.tensor([[0.5, 0.5], [1.0, 0.0]])).clamp(0, 1)
    methods = pipeline.ALL_METHODS + ["geo4", "shiftavg", "shiftavg+geoP", "radP", "geoP+radP", "blur+geoP", "classic+geoP", "localmin1",
                                          "localmin+geoP", "geoPms", "localmin+geoPms",
                                          "c2f", "c2f+radP", "c2f+rad"]
    sc = eng.scores(eng.reference(x1, methods), x2, methods)
    assert set(methods) <= set(sc), set(methods) - set(sc)
    for m in methods:
        assert sc[m].shape == (2, 4, 4) and torch.isfinite(sc[m]).all(), m
    ident = eng.scores(eng.reference(x1, methods), x1.clone(), ["raw", "geoP", "geoP+radP"])
    assert all(ident[m].abs().max() < 1e-5 for m in ident)


def test_localmin_scores_absorbs_one_token_shift_and_matches_plain_distance():
    z = torch.nn.functional.normalize(torch.randn(1, 8, 6, 6), dim=1)
    z2 = torch.roll(z, 1, dims=-1)                         # content moved by one token
    s = operators.localmin_scores(z, z2)
    assert s[..., 1:-1, 1:-1].abs().max() < 1e-6
    U = torch.zeros(1, 8, 2, 6, 6)                         # zero basis -> projection is identity
    assert torch.allclose(operators.localmin_scores(z, z2, U), s)
    assert (operators.localmin_scores(z, z) == 0).all()


def test_probe_basis_chunking_matches_unchunked():
    D = torch.randn(2, 16, 6, 4, 4)
    assert torch.allclose(operators.probe_basis(D, 3, chunk=5).abs(), operators.probe_basis(D, 3, chunk=10_000).abs(), atol=1e-5)
