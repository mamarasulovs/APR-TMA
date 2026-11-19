"""APR-TMA augmentation operator A_y (Algorithm 1 of the manuscript, augmentation part)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

from .pipeline import DEFAULT_CONTROL, StageControl, apply_pipeline
from .tma import EllipseTrace, blend, modulation_map


@dataclass(frozen=True)
class APRTMAConfig:
    """Frozen operating point of Table 2 (plus the switches used by the ablation)."""

    p_apr: float = 0.50                # per-sample application probability
    lam: float = 0.70                  # trace / uniform mix
    alpha: float = 0.35                # blend magnitude; identical for both classes by default
    alpha_spoof: Optional[float] = None  # set only for the class-dependent-magnitude ablation
    q_c: float = 0.50                  # categorical re-sampling probability in the second pass
    mask_clip: Tuple[float, float] = (0.15, 1.0)
    uniform_modulation: bool = False   # ablation: M == 1
    stage: StageControl = field(default_factory=lambda: DEFAULT_CONTROL)

    def alpha_for(self, y: int) -> float:
        """alpha_L = alpha_S = alpha unless the leakage-control ablation overrides the spoof value."""
        return self.alpha_spoof if (y == 1 and self.alpha_spoof is not None) else self.alpha


class APRTMA:
    """Class-asymmetric acquisition-pipeline randomization with trace modulation.

    ``__call__(img, y, rng)`` takes an RGB float32 image in [0, 1] and the label
    (0 = live, 1 = spoof) and returns ``(x_tilde, applied)``. With probability ``1 - p_apr`` the
    clean image is returned unchanged, so clean and transformed samples are both seen in training.
    """

    def __init__(self, config: APRTMAConfig = APRTMAConfig(),
                 trace_prior: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.cfg = config
        self.trace_prior = trace_prior if trace_prior is not None else EllipseTrace()

    def modulation(self, img: np.ndarray, trace: Optional[np.ndarray] = None) -> np.ndarray:
        if self.cfg.uniform_modulation:
            return np.ones(img.shape[:2], np.float32)
        prior = trace if trace is not None else self.trace_prior(img)
        return modulation_map(prior, self.cfg.lam, self.cfg.mask_clip)

    def __call__(self, img: np.ndarray, y: int, rng: np.random.Generator,
                 trace: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        if rng.random() >= self.cfg.p_apr:
            return img, False
        p_x = apply_pipeline(img, y, rng, q_c=self.cfg.q_c, ctl=self.cfg.stage)
        return blend(img, p_x, self.modulation(img, trace), self.cfg.alpha_for(y)), True
