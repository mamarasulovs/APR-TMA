"""Trace-Modulated Attention: label-free face-parsing prior and modulation map M(x).

The prior A(x) is a soft map over facial regions (skin, brows, eyes, nose, mouth, lips)
produced by a frozen BiSeNet face parser pretrained on CelebAMask-HQ. The parser is only used to
generate masks during training and is never invoked at inference.

Two prior providers are available:

* ``BiSeNetTrace``  - the parser used in the manuscript (needs external weights, see README).
* ``EllipseTrace``  - a weight-free elliptical face prior, for smoke tests and quick demos only.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

# CelebAMask-HQ label ids of the BiSeNet parser (19 classes).
# 1 skin, 2 l_brow, 3 r_brow, 4 l_eye, 5 r_eye, 6 eye_g, 7 l_ear, 8 r_ear, 9 ear_r,
# 10 nose, 11 mouth, 12 u_lip, 13 l_lip, 14 neck, 15 neck_l, 16 cloth, 17 hair, 18 hat
FACIAL_CLASSES = (1, 2, 3, 4, 5, 10, 11, 12, 13)

LAMBDA = 0.70          # trace / uniform mix, Eq. (6)
MASK_CLIP = (0.15, 1.0)


class TracePrior(Protocol):
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Return A(x) in [0, 1] with shape (H, W) for an RGB float image (H, W, 3)."""


def modulation_map(prior: np.ndarray, lam: float = LAMBDA, clip=MASK_CLIP) -> np.ndarray:
    """Eq. (6): M(x) = clip(lam * A(x) + (1 - lam) * 1, 0.15, 1.0)."""
    m = lam * prior.astype(np.float32) + (1.0 - lam)
    return np.clip(m, clip[0], clip[1]).astype(np.float32)


def blend(x: np.ndarray, p_x: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Eq. (7): x_tilde = x + alpha * M ⊙ (P(x) - x). The magnitude alpha is class-symmetric."""
    return (x + alpha * mask[..., None] * (p_x - x)).astype(np.float32)


class EllipseTrace:
    """Weight-free soft elliptical prior centred on the crop (smoke tests only)."""

    def __init__(self, sigma_scale: float = 0.35):
        self.sigma_scale = sigma_scale

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cy, cx = 0.5 * h, 0.5 * w
        d = ((yy - cy) / (self.sigma_scale * h * 1.4)) ** 2 + ((xx - cx) / (self.sigma_scale * w)) ** 2
        return np.exp(-0.5 * d).astype(np.float32)


class BiSeNetTrace:
    """Frozen BiSeNet face parser (CelebAMask-HQ) producing the trace prior.

    Expects the reference implementation ``zllrunning/face-parsing.PyTorch`` on disk:

        git clone https://github.com/zllrunning/face-parsing.PyTorch  external/face-parsing
        # download 79999_iter.pth into external/face-parsing/res/cp/

    The parser is run once per image. For training we recommend caching the masks with
    ``scripts/precompute_traces.py`` so the parser never runs inside the data loader.
    """

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, repo_dir: str, weights: str | None = None, device: str = "cpu",
                 smooth_sigma: float = 3.0):
        import torch

        repo = Path(repo_dir)
        weights_path = Path(weights) if weights else repo / "res" / "cp" / "79999_iter.pth"
        if not weights_path.exists():
            raise FileNotFoundError(f"BiSeNet weights not found: {weights_path}")
        sys.path.insert(0, str(repo))
        from model import BiSeNet  # type: ignore  # provided by the external repository

        self._torch = torch
        self.device = torch.device(device)
        net = BiSeNet(n_classes=19)
        net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        net.eval().requires_grad_(False)
        self.net = net.to(self.device)
        self.smooth_sigma = smooth_sigma

    def __call__(self, img: np.ndarray) -> np.ndarray:
        torch = self._torch
        h, w = img.shape[:2]
        x = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        x = (x - self.MEAN) / self.STD
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.net(t)[0]
            parsing = logits.argmax(1).squeeze(0).cpu().numpy()
        mask = np.isin(parsing, FACIAL_CLASSES).astype(np.float32)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        if self.smooth_sigma > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), self.smooth_sigma)
        return np.clip(mask, 0.0, 1.0)
