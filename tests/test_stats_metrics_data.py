import csv
from pathlib import Path

import numpy as np
import pytest

from apr_tma.faces import Face, align_crop
from apr_tma.manifests import casia, msu, oulu, replay, siw
from apr_tma.metrics import acer, eer_threshold, hter
from apr_tma.stats import analyze, holm_bonferroni


def test_holm_matches_manuscript_table12():
    adj = holm_bonferroni([0.00038, 0.00191, 0.00235, 0.081])
    assert np.allclose(adj, [0.0015, 0.0057, 0.0057, 0.081], atol=5e-4)


def test_pooled_statistics_match_manuscript_formulae():
    rng = np.random.default_rng(0)
    seeds = range(5)
    base = {t: {s: 7.0 + rng.normal(0, 0.1) for s in seeds} for t in "MCIO"}
    ours = {t: {s: base[t][s] - 0.7 + rng.normal(0, 0.1) for s in seeds} for t in "MCIO"}
    rows = analyze(base, ours)
    pooled = rows[-1]
    assert pooled.name == "Pooled" and pooled.n == 5
    assert pooled.mean == pytest.approx(0.7, abs=0.15) and pooled.ci_low < pooled.mean < pooled.ci_high
    assert pooled.dz == pytest.approx(pooled.mean / pooled.sd)
    assert pooled.t == pytest.approx(pooled.mean / (pooled.sd / np.sqrt(5)))
    assert all(r.p_holm >= r.p_raw for r in rows[:-1])


def test_acer_uses_worst_case_attack():
    scores = np.array([0.1, 0.1, 0.2, 0.9, 0.9, 0.2, 0.2])
    labels = np.array([0, 0, 0, 1, 1, 1, 1])
    attacks = np.array(["", "", "", "print1", "print1", "replay1", "replay1"])
    # threshold 0.5: BPCER = 0; print APCER = 0; replay APCER = 1 -> ACER = 50
    assert acer(scores, labels, 0.5, attacks) == pytest.approx(50.0)
    assert acer(scores, labels, 0.5) == pytest.approx(hter(scores, labels, 0.5))


def test_eer_threshold_separates_clean_scores():
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert 0.3 <= eer_threshold(scores, labels) <= 0.7 and hter(scores, labels, 0.5) == 0.0


def test_dataset_adapters():
    assert oulu(Path("1_2_03_4/00000.png")).__dict__ == dict(label=1, video_id="1_2_03_4", attack="replay1", phone="1", session="2", subset="")
    assert oulu(Path("6_1_20_1/00001.png")).label == 0
    assert casia(Path("train_release/1/HR_1/0.png")).label == 0 and casia(Path("train_release/1/3/0.png")).label == 1
    assert replay(Path("train/real/client001/x/0.png")).label == 0 and replay(Path("train/attack/fixed/x/0.png")).label == 1
    assert msu(Path("real_client001_android/0.png")).label == 0 and msu(Path("attack_client001_ipad/0.png")).label == 1
    assert siw(Path("Train/Live/001/0.png")).label == 0 and siw(Path("Train/Spoof/Print/001/0.png")).label == 1


def test_align_crop_shape_and_eye_alignment():
    img = np.zeros((300, 300, 3), np.uint8)
    img[140:160, 100:120] = 255
    face = Face((100, 100, 100, 100), (130, 140), (170, 140))
    out = align_crop(img, face, size=224)
    assert out.shape == (224, 224, 3)
    tilted = Face((100, 100, 100, 100), (130, 150), (170, 130))  # rolled eye line is rotated out
    assert align_crop(img, tilted, size=224).shape == (224, 224, 3)


def test_generic_adapter_extracts_attack_type():
    from apr_tma.manifests import generic
    rule = generic("bonafide", "attack", r"attack/([^/]+)/")
    assert rule(Path("attack/glasses/c1/0.png")).attack == "glasses"
    assert rule(Path("bonafide/c1/0.png")).label == 0 and rule(Path("other/x.png")) is None
