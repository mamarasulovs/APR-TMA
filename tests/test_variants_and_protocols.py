import numpy as np
import pytest

from apr_tma import pipeline as P
from apr_tma.data import Sample
from apr_tma.splits import make_split, targets_of
from apr_tma.variants import VARIANTS, build_augmentation


def img(seed=0):
    return np.random.default_rng(seed).random((64, 64, 3)).astype(np.float32)


def test_all_table6_variants_present_and_runnable():
    assert len(VARIANTS) == 13 and "full" in VARIANTS
    x = img()
    for name in VARIANTS:
        aug, _, _ = build_augmentation("apr_tma", name)
        aug.cfg = type(aug.cfg)(**{**aug.cfg.__dict__, "p_apr": 1.0})
        for y in (0, 1):
            out, applied = aug(x, y, np.random.default_rng(1))
            assert applied and out.shape == x.shape and np.isfinite(out).all()


def test_symmetric_composition_uses_one_pass_for_spoof(monkeypatch):
    calls = []
    real = P.acquisition_pass
    monkeypatch.setattr(P, "acquisition_pass", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    ctl = VARIANTS["symmetric_composition"].config.stage
    P.apply_pipeline(img(), 1, np.random.default_rng(0), ctl=ctl)
    assert len(calls) == 1


def test_fixed_pipeline_is_deterministic_in_parameters():
    ctl = VARIANTS["fixed_pipeline"].config.stage
    a = P.PipelineParams.sample(np.random.default_rng(1), ctl)
    b = P.PipelineParams.sample(np.random.default_rng(2), ctl)
    assert all(np.array_equal(getattr(a, f), getattr(b, f)) for f in a.__dataclass_fields__)


@pytest.mark.parametrize("variant,field,expected", [
    ("drop_noise", "noise_on", False), ("drop_sharpening", "sharp_amount", 0.0),
    ("drop_demosaic_randomization", "demosaic", "MHC"), ("drop_tone_wb_gamma", "tone_curve", "gamma")])
def test_stage_drops_applied_to_both_passes(variant, field, expected):
    ctl = VARIANTS[variant].config.stage
    rng = np.random.default_rng(3)
    t1 = P.PipelineParams.sample(rng, ctl)
    t2 = P.couple_params(t1, rng, ctl=ctl)
    assert getattr(t1, field) == expected and getattr(t2, field) == expected


def test_uniform_modulation_and_class_dependent_alpha():
    aug, _, _ = build_augmentation("apr_tma", "uniform_modulation")
    assert np.all(aug.modulation(img()) == 1.0)
    cfg = VARIANTS["class_dependent_magnitude"].config
    assert cfg.alpha_for(0) != cfg.alpha_for(1)
    assert not VARIANTS["no_sare"].use_sare and VARIANTS["full"].use_sare


def test_comparators_are_class_agnostic():
    for m in ("ti_aug", "fas_aug_proxy"):
        aug, _, _ = build_augmentation(m)
        aug.p = 1.0
        out0, _ = aug(img(), 0, np.random.default_rng(4))
        out1, _ = aug(img(), 1, np.random.default_rng(4))
        assert np.array_equal(out0, out1)  # same randomness -> same output regardless of the label


def S(domain, label, vid, **kw):
    return Sample("x.png", label, domain, f"{domain}{vid}", **kw)


def oulu_samples():
    rows = []
    for ph in "123456":
        for sub in ("train", "dev", "test"):
            for att in ("", "print1", "print2", "replay1", "replay2"):
                for ses in "123":
                    rows.append(S("O", int(att != ""), f"{ph}{sub}{att}{ses}", phone=ph, subset=sub, attack=att, session=ses))
    return rows


def test_ocim_split_has_no_target_leakage():
    rows = [S(d, v % 2, f"{d}{v}") for d in "OCIM" for v in range(10) for _ in range(3)]
    for tgt in "MCIO":
        sp = make_split("OCIM", tgt, rows, seed=1)
        assert {s.domain for s in sp.test} == {tgt}
        assert tgt not in {s.domain for s in sp.train + sp.val}
        assert {s.video_id for s in sp.train}.isdisjoint({s.video_id for s in sp.val})
        assert {s.label for s in sp.val} == {0, 1}


def test_limited_and_modern_sources():
    rows = [S(d, v % 2, f"{d}{v}") for d in ("M", "I", "C", "O", "CS", "SW") for v in range(10)]
    assert {s.domain for s in make_split("LIMITED", "C", rows).train} <= {"M", "I"}
    sp = make_split("MODERN", "SW", rows)
    assert {s.domain for s in sp.train} == {"CS"} and {s.domain for s in sp.test} == {"SW"}


def test_oulu_p3_and_p4_follow_protocol():
    rows = oulu_samples()
    p3 = make_split("OULU_P3", "2", rows)
    assert all(s.phone != "2" and s.subset == "train" for s in p3.train)
    assert all(s.phone == "2" and s.subset == "test" for s in p3.test)
    p4 = make_split("OULU_P4", "2", rows)
    assert all(s.session in "12" and s.attack in ("", "print1", "replay1") for s in p4.train)
    assert all(s.session == "3" and s.attack in ("", "print2", "replay2") and s.phone == "2" for s in p4.test)
    assert targets_of("OULU_P3") == [str(k) for k in range(1, 7)]


def test_wmca_unseen_holds_out_attack_type():
    rows = [S("WM", 0, f"l{i}") for i in range(10)] + \
           [S("WM", 1, f"{a}{i}", attack=a) for a in ("glasses", "replay", "mask") for i in range(5)]
    assert targets_of("WMCA_UNSEEN", rows) == ["glasses", "mask", "replay"]
    sp = make_split("WMCA_UNSEEN", "glasses", rows)
    assert "glasses" not in {s.attack for s in sp.train + sp.val}
    assert {s.attack for s in sp.test if s.label == 1} == {"glasses"}
    assert any(s.label == 0 for s in sp.test)
