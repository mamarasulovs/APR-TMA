"""Evaluation protocols of the manuscript.

``make_split(protocol, target, samples, seed, val_fraction)`` returns a :class:`Split` with
``train``, ``val`` (source-only, used for model selection and the decision threshold) and
``test`` (the unseen target). No target data ever enters ``train`` or ``val``.

Protocols and targets
---------------------
OCIM         target in {M, C, I, O}   leave-one-domain-out over OULU-NPU, CASIA-MFSD, Replay, MSU
MICY         target in {I, M, C, Y}   as OCIM with ROSE-YOUTU replacing OULU-NPU as a source
LIMITED      target in {C, O}         train on M & I only
MODERN       target in {SW, CS}       CelebA-Spoof (CS) <-> SiW-Mv2 (SW)
WMCA_UNSEEN  target = attack type     leave-one-attack-type-out on WMCA (domain WM); ACER
OULU_P3      target in 1..6           leave-one-camera-out (official train / dev / test subsets)
OULU_P4      target in 1..6           leave-one-camera-out with unseen session and unseen attacks
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .data import Sample, split_by_video

DOMAIN_PROTOCOLS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "OCIM": {"M": ("O", "C", "I"), "C": ("O", "M", "I"), "I": ("O", "C", "M"), "O": ("I", "C", "M")},
    "MICY": {"Y": ("M", "I", "C"), "C": ("M", "I", "Y"), "I": ("M", "C", "Y"), "M": ("I", "C", "Y")},
    "LIMITED": {"C": ("M", "I"), "O": ("M", "I")},
    "MODERN": {"SW": ("CS",), "CS": ("SW",)},
}
OULU_PROTOCOLS = ("OULU_P3", "OULU_P4")
ALL_PROTOCOLS = tuple(DOMAIN_PROTOCOLS) + ("WMCA_UNSEEN",) + OULU_PROTOCOLS
# Metric reported by each protocol (Tables 4-5, 7-10 of the manuscript).
PROTOCOL_METRIC = {**{p: "hter" for p in DOMAIN_PROTOCOLS}, "WMCA_UNSEEN": "acer",
                   "OULU_P3": "acer", "OULU_P4": "acer"}

_P4_TRAIN_ATTACKS = {"print1", "replay1"}
_P4_TEST_ATTACKS = {"print2", "replay2"}


@dataclass
class Split:
    train: List[Sample]
    val: List[Sample]
    test: List[Sample]


def targets_of(protocol: str, samples: Sequence[Sample] = ()) -> List[str]:
    """All targets (folds) of a protocol; WMCA folds are read from the manifest."""
    if protocol in DOMAIN_PROTOCOLS:
        return list(DOMAIN_PROTOCOLS[protocol])
    if protocol in OULU_PROTOCOLS:
        return [str(k) for k in range(1, 7)]
    if protocol == "WMCA_UNSEEN":
        return sorted({s.attack for s in samples if s.domain == "WM" and s.label == 1 and s.attack})
    raise KeyError(protocol)


def _live_holdout(samples: Sequence[Sample], fraction: float, seed: int):
    """Deterministic video-level split of live samples into (kept for training, held out)."""
    vids = sorted({s.video_id for s in samples})
    rng = np.random.default_rng(seed)
    rng.shuffle(vids)
    held = set(vids[:max(1, int(round(fraction * len(vids))))])
    return [s for s in samples if s.video_id not in held], [s for s in samples if s.video_id in held]


def make_split(protocol: str, target: str, samples: Sequence[Sample],
               seed: int = 0, val_fraction: float = 0.1) -> Split:
    if protocol in DOMAIN_PROTOCOLS:
        sources = set(DOMAIN_PROTOCOLS[protocol][target])
        src = [s for s in samples if s.domain in sources]
        tgt = [s for s in samples if s.domain == target]
        train, val = split_by_video(src, val_fraction, seed)
        return Split(train, val, tgt)

    if protocol == "WMCA_UNSEEN":
        wm = [s for s in samples if s.domain == "WM"]
        live = [s for s in wm if s.label == 0]
        spoof = [s for s in wm if s.label == 1]
        live_train, live_test = _live_holdout(live, 0.2, seed=0)  # fixed across folds and seeds
        held = [s for s in spoof if s.attack == target]
        train, val = split_by_video(live_train + [s for s in spoof if s.attack != target], val_fraction, seed)
        return Split(train, val, live_test + held)

    if protocol in OULU_PROTOCOLS:
        o = [s for s in samples if s.domain == "O"]
        phone = str(target)
        if protocol == "OULU_P3":
            keep = lambda s: s.phone != phone
            test = [s for s in o if s.subset == "test" and s.phone == phone]
        else:
            ok_attack = lambda s, allowed: s.label == 0 or s.attack in allowed
            keep = lambda s: s.phone != phone and s.session in ("1", "2") and ok_attack(s, _P4_TRAIN_ATTACKS)
            test = [s for s in o if s.subset == "test" and s.phone == phone and s.session == "3"
                    and ok_attack(s, _P4_TEST_ATTACKS)]
        train = [s for s in o if s.subset == "train" and keep(s)]
        val = [s for s in o if s.subset == "dev" and keep(s)]
        return Split(train, val, test)

    raise KeyError(f"unknown protocol {protocol!r}; choose from {ALL_PROTOCOLS}")
