"""Training objective, Eqs. (8) and (9): L_total = L_cls + gamma * L_SARE with gamma = 0.1."""
from __future__ import annotations

import torch
import torch.nn.functional as F

GAMMA = 0.1


def sare_penalty(per_sample_loss: torch.Tensor, labels: torch.Tensor, augmented: torch.Tensor) -> torch.Tensor:
    """Spoofing-attack risk equalization (variance form).

    The batch is split into groups defined by (class, augmented-or-clean). The penalty is the
    variance of the group-wise mean risks, so no single artifact configuration (or the absence of
    one) can carry a disproportionate share of the loss. Groups with no samples are skipped and
    the penalty is zero when fewer than two groups are present.

    NOTE: this is a re-implementation of the idea described in the manuscript; the exact SARE
    formulation of the FAS-Aug reference code may differ in grouping and weighting.
    """
    risks = []
    for cls in (0, 1):
        for aug in (False, True):
            sel = (labels == cls) & (augmented == aug)
            if sel.any():
                risks.append(per_sample_loss[sel].mean())
    if len(risks) < 2:
        return per_sample_loss.new_zeros(())
    return torch.stack(risks).var(unbiased=False)


def total_loss(logits: torch.Tensor, labels: torch.Tensor, augmented: torch.Tensor,
               gamma: float = GAMMA, use_sare: bool = True):
    """Return (L_total, L_cls, L_sare). ``logits`` has shape (B,), ``labels`` in {0, 1}."""
    per_sample = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    l_cls = per_sample.mean()
    l_sare = sare_penalty(per_sample, labels, augmented) if use_sare else l_cls.new_zeros(())
    return l_cls + gamma * l_sare, l_cls.detach(), l_sare.detach()
