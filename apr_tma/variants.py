"""Method registry: baselines, the proxy data-centric comparators, and the ablation variants.

``build_augmentation(method, variant)`` returns ``(augmentation, use_sare, trace_source)``:

* ``augmentation``  callable ``(img, y, rng, trace=None) -> (img, applied)`` or ``None``
* ``use_sare``      whether the SARE term is part of the objective (gamma = 0.1)
* ``trace_source``  which cached trace prior the variant needs (``bisenet``, ``gradcam``,
                    ``stdn``) or ``None``

Methods
-------
``baseline``        host with no augmentation.
``ti_aug``          traditional image augmentation (flip + colour / noise / blur / JPEG).
``fas_aug_proxy``   class-agnostic physics-inspired surface artifacts (colour, noise, blur, JPEG,
                    recapture blur and moire) applied identically to both classes, with SARE.
                    This is a *proxy* for the FAS-Aug family, not the official implementation;
                    use the official code for any head-to-head claim.
``apr_tma``         the proposed method; ``--variant`` selects the full model or one ablation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .augment import APRTMA, APRTMAConfig
from .pipeline import (ReproductionParams, StageControl, reproduce, surface_pass)


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    config: APRTMAConfig = APRTMAConfig()
    use_sare: bool = True
    trace_source: Optional[str] = "bisenet"


def _cfg(**stage_kw) -> APRTMAConfig:
    return replace(APRTMAConfig(), stage=StageControl(**stage_kw))


# Ordered as in Table 6 of the manuscript.
VARIANTS: Dict[str, Variant] = {v.name: v for v in [
    Variant("no_internal_isp", "Surface operations only (no internal ISP)", _cfg(isp="surface")),
    Variant("symmetric_composition", "Both classes one pass (no structural asymmetry)",
            _cfg(symmetric_depth=True)),
    Variant("fixed_pipeline", "Asymmetric structure with fixed (non-randomized) parameters",
            _cfg(randomize=False)),
    Variant("uniform_modulation", "Uniform modulation, M == 1",
            replace(APRTMAConfig(), uniform_modulation=True), trace_source=None),
    # alpha_L / alpha_S taken from the two ends of the source-only grid {0.20, 0.35, 0.50}
    Variant("class_dependent_magnitude", "Class-dependent blend magnitude (alpha_L=0.20, alpha_S=0.50)",
            replace(APRTMAConfig(), alpha=0.20, alpha_spoof=0.50)),
    Variant("no_sare", "Without the SARE term", use_sare=False),
    Variant("gradcam_modulation", "Grad-CAM attention instead of the face-parsing prior",
            trace_source="gradcam"),
    Variant("drop_tone_wb_gamma", "Fix tone / white balance / gamma", _cfg(drop_tone_wb=True)),
    Variant("drop_noise", "Remove shot and read noise", _cfg(drop_noise=True)),
    Variant("drop_demosaic_randomization", "Fix demosaicing operator and CFA pattern",
            _cfg(drop_demosaic_rand=True)),
    Variant("stdn_trace_modulation", "STDN spoof-trace maps instead of the face-parsing prior",
            trace_source="stdn"),
    Variant("drop_sharpening", "Remove sharpening", _cfg(drop_sharpen=True)),
    Variant("full", "Full APR-TMA + SARE"),
]}


class _Gated:
    """Shared gate and interface for the comparator augmentations."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def _apply(self, img, y, rng):  # pragma: no cover - interface
        raise NotImplementedError

    def __call__(self, img, y, rng, trace=None):
        if rng.random() >= self.p:
            return img, False
        return self._apply(img, y, rng), True


class TIAug(_Gated):
    """Traditional augmentation: horizontal flip and generic surface distortions."""

    def _apply(self, img, y, rng):
        out = surface_pass(img, rng)
        return out[:, ::-1].copy() if rng.random() < 0.5 else out


class FASAugProxy(_Gated):
    """Class-agnostic physics-inspired surface artifacts (proxy, see module docstring)."""

    def _apply(self, img, y, rng):
        out = surface_pass(img, rng)
        if rng.random() < 0.5:  # recapture-style artifacts, identical for live and spoof
            out = reproduce(out, ReproductionParams.sample(rng))
        return out


def build_augmentation(method: str, variant: str = "full",
                       trace_prior: Optional[Callable] = None
                       ) -> Tuple[Optional[Callable], bool, Optional[str]]:
    if method == "baseline":
        return None, False, None
    if method == "ti_aug":
        return TIAug(), False, None
    if method == "fas_aug_proxy":
        return FASAugProxy(), True, None
    if method == "apr_tma":
        if variant not in VARIANTS:
            raise KeyError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
        v = VARIANTS[variant]
        return APRTMA(v.config, trace_prior=trace_prior), v.use_sare, v.trace_source
    raise KeyError(f"unknown method {method!r}")


METHODS = ("baseline", "ti_aug", "fas_aug_proxy", "apr_tma")
