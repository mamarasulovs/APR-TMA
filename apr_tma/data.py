"""Manifest-driven datasets and the leave-one-domain-out protocols used in the manuscript.

Face crops are expected to be detected, aligned and cropped beforehand (any detector). Every
dataset is described by one CSV manifest with the columns::

    path,label,domain,video_id
    /data/oulu/train/0001/000.png,0,O,oulu_0001
    ...

``label`` is 0 for live and 1 for spoof, ``domain`` is a dataset tag (O, C, I, M, Y, CS, SW, WM),
and ``video_id`` groups frames of one video so that train / validation splits never leak frames.
Optional columns ``attack``, ``phone``, ``session`` and ``subset`` are needed only by the OULU-NPU
and WMCA protocols (see ``docs/DATA.md``). Images are resized to 224 x 224.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import APRTMA

IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

@dataclass
class Sample:
    path: str
    label: int
    domain: str
    video_id: str
    attack: str = ""      # presentation-attack instrument, "" for live
    phone: str = ""       # capture device (OULU-NPU protocols 3 and 4)
    session: str = ""     # recording session (OULU-NPU)
    subset: str = ""      # official train / dev / test partition, when the dataset defines one


def read_manifest(path: str) -> List[Sample]:
    with open(path, newline="") as f:
        return [Sample(r["path"], int(r["label"]), r["domain"], r["video_id"], r.get("attack", "") or "",
                       r.get("phone", "") or "", r.get("session", "") or "", r.get("subset", "") or "")
                for r in csv.DictReader(f)]


def select_domains(samples: Sequence[Sample], domains: Sequence[str]) -> List[Sample]:
    keep = set(domains)
    return [s for s in samples if s.domain in keep]


def split_by_video(samples: Sequence[Sample], val_fraction: float = 0.1, seed: int = 0):
    """Source-only train / validation split at video level, stratified by class.

    No target data is involved. Each class keeps at least one validation video so that the
    source-fixed threshold is always defined.
    """
    rng = np.random.default_rng(seed)
    by_class = {0: {}, 1: {}}
    for s in samples:
        by_class[s.label].setdefault(s.video_id, s.label)
    val_ids = set()
    for cls, vids in by_class.items():
        ids = sorted(vids)
        if not ids:
            raise ValueError(f"no videos of class {cls} among the source samples")
        rng.shuffle(ids)
        val_ids.update(ids[:max(1, int(round(val_fraction * len(ids))))])
    train = [s for s in samples if s.video_id not in val_ids]
    val = [s for s in samples if s.video_id in val_ids]
    return train, val


def trace_key(path: str) -> str:
    """Stable cache file name for the precomputed trace prior of an image."""
    return hashlib.md5(path.encode("utf-8")).hexdigest() + ".npy"


def load_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.float32) / 255.0


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1).copy())


class FASDataset(Dataset):
    """Returns ``(image, label, augmented)``; APR-TMA is applied online when ``augment`` is set.

    Randomness is a pure function of ``(seed, epoch, index)``, so a run is reproducible regardless
    of the number of data-loader workers. Call :meth:`set_epoch` once per epoch.
    """

    def __init__(self, samples: Sequence[Sample], augment: Optional[APRTMA] = None,
                 seed: int = 0, trace_dir: Optional[str] = None):
        self.samples = list(samples)
        self.augment = augment
        self.seed = seed
        self.epoch = 0
        self.trace_dir = Path(trace_dir) if trace_dir else None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def _cached_trace(self, s: Sample) -> Optional[np.ndarray]:
        if self.trace_dir is None:
            return None
        f = self.trace_dir / trace_key(s.path)
        if not f.exists():
            return None
        m = np.load(f).astype(np.float32)
        return cv2.resize(m, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = load_rgb(s.path)
        applied = False
        if self.augment is not None:
            rng = np.random.default_rng([self.seed, self.epoch, i])
            img, applied = self.augment(img, s.label, rng, trace=self._cached_trace(s))
        return to_tensor(img), torch.tensor(s.label, dtype=torch.long), torch.tensor(applied)
