import numpy as np
import pytest

from apr_tma import pipeline as P
from apr_tma.augment import APRTMA, APRTMAConfig
from apr_tma.tma import EllipseTrace, blend, modulation_map


def synthetic(h=224, w=224, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([xx / w, yy / h, 0.5 + 0.3 * np.sin(8 * xx / w)], -1)
    return np.clip(base + 0.03 * rng.standard_normal((h, w, 3)).astype(np.float32), 0, 1)


def test_acquisition_depth():
    assert P.acquisition_depth(0) == 1 and P.acquisition_depth(1) == 2


@pytest.mark.parametrize("pattern", P.BAYER_PATTERNS)
@pytest.mark.parametrize("op", P.DEMOSAIC_OPS)
def test_demosaic_recovers_smooth_image(pattern, op):
    yy, xx = np.mgrid[0:64, 0:64].astype(np.float32) / 63.0
    ref = np.stack([xx, yy, 0.5 * (xx + yy)], -1)
    out = P.demosaic(P.mosaic(ref, pattern), pattern, op)
    assert np.abs(out[6:-6, 6:-6] - ref[6:-6, 6:-6]).mean() < 0.02


def test_pipeline_range_dtype_shape():
    rng = np.random.default_rng(1)
    for y in (0, 1):
        out = P.apply_pipeline(synthetic(), y, rng)
        assert out.shape == (224, 224, 3) and out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0 and np.isfinite(out).all()


def test_spoof_branch_runs_two_passes_and_live_one(monkeypatch):
    calls = []
    real = P.acquisition_pass
    monkeypatch.setattr(P, "acquisition_pass", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    rng = np.random.default_rng(2)
    P.apply_pipeline(synthetic(), 0, rng)
    assert len(calls) == 1
    calls.clear()
    P.apply_pipeline(synthetic(), 1, rng)
    assert len(calls) == 2


def test_coupling_is_convex_combination_between_passes():
    rng = np.random.default_rng(3)
    t1 = P.PipelineParams.sample(rng)
    t2 = P.couple_params(t1, rng, rho=1.0, q_c=0.0)  # rho = 1 copies the first pass
    for name in P.PipelineParams.CONTINUOUS:
        assert getattr(t2, name) == pytest.approx(getattr(t1, name))
    assert t2.bayer == t1.bayer and t2.demosaic == t1.demosaic


def test_sampled_parameters_inside_table2_ranges():
    rng = np.random.default_rng(4)
    for _ in range(300):
        p = P.PipelineParams.sample(rng)
        assert 1.8 <= p.inv_gamma <= 2.6 and 0.6 <= p.inv_tone_strength <= 1.0
        assert 1.6 <= p.red_gain <= 2.4 and 1.3 <= p.blue_gain <= 2.1
        assert 1e-4 <= p.shot_scale <= 1.2e-2 and 1e-4 <= p.read_std <= 4e-3
        assert 0.3 <= p.sharp_radius <= 1.5 and 0.0 <= p.sharp_amount <= 1.2
        assert p.bayer in P.BAYER_PATTERNS and p.demosaic in P.DEMOSAIC_OPS
        assert np.allclose(p.ccm().sum(1), 1.0, atol=1e-5)
        xi = P.ReproductionParams.sample(rng)
        assert 0.0 <= xi.blur_sigma <= 1.2 and 0.0 <= xi.moire_strength <= 0.35


def test_modulation_map_bounds_and_blend_energy_cap():
    rng = np.random.default_rng(5)
    prior = rng.random((64, 64)).astype(np.float32)
    m = modulation_map(prior)
    assert m.min() >= 0.15 - 1e-6 and m.max() <= 1.0 + 1e-6
    x, p = rng.random((64, 64, 3)).astype(np.float32), rng.random((64, 64, 3)).astype(np.float32)
    out = blend(x, p, m, alpha=0.35)
    assert np.all(np.abs(out - x) <= 0.35 * np.abs(p - x) + 1e-6)  # |x~ - x| <= alpha * max(M) * |P - x|


def test_gate_probability_and_clean_passthrough():
    aug = APRTMA(APRTMAConfig(p_apr=0.5), trace_prior=EllipseTrace())
    x = synthetic(64, 64)
    flags = []
    for i in range(400):
        out, applied = aug(x, i % 2, np.random.default_rng([0, i]))
        flags.append(applied)
        if not applied:
            assert out is x
    assert 0.42 < np.mean(flags) < 0.58


def test_magnitude_is_class_symmetric_by_construction():
    cfg = APRTMAConfig()
    assert cfg.alpha == 0.35 and cfg.alpha_spoof is None
    assert cfg.alpha_for(0) == cfg.alpha_for(1) == 0.35


def test_determinism():
    x = synthetic(64, 64)
    aug = APRTMA(APRTMAConfig(p_apr=1.0))
    a, _ = aug(x, 1, np.random.default_rng(7))
    b, _ = aug(x, 1, np.random.default_rng(7))
    assert np.array_equal(a, b)
