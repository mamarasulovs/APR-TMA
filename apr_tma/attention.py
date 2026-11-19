"""Grad-CAM attention source (ablation "Grad-CAM modulation").

Grad-CAM maps are computed once with a trained baseline host and cached by
``scripts/precompute_traces.py --source gradcam``; they then replace the face-parsing prior in
Eq. (6). The maps are label-free at use time: the class-score gradient is taken w.r.t. the
predicted-spoof logit, never the ground-truth label.
"""
from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn


def vit_reshape(tokens: torch.Tensor, grid: int = 14) -> torch.Tensor:
    """(B, 1 + N, C) token sequence -> (B, C, grid, grid) map for Grad-CAM on ViT blocks."""
    patches = tokens[:, 1:, :]
    return patches.reshape(patches.size(0), grid, grid, -1).permute(0, 3, 1, 2)


def default_target_layer(model: nn.Module):
    """Last convolutional stage of ResNet-18 or the last transformer block of a ViT host."""
    bb = model.backbone
    if hasattr(bb, "blocks"):
        return bb.blocks[-1].block.norm1 if hasattr(bb.blocks[-1], "block") else bb.blocks[-1].norm1, vit_reshape
    return bb[7], None  # layer4 of the torchvision ResNet children list


class GradCAMTrace:
    def __init__(self, model: nn.Module, device: str = "cpu", target_layer=None,
                 reshape: Optional[Callable] = None, mean=None, std=None):
        self.model = model.eval().to(device)
        self.device = device
        layer, rs = (target_layer, reshape) if target_layer is not None else default_target_layer(model)
        self.reshape = rs
        self.mean = np.array(mean if mean is not None else [0.485, 0.456, 0.406], np.float32)
        self.std = np.array(std if std is not None else [0.229, 0.224, 0.225], np.float32)
        self._act = self._grad = None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "_act", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "_grad", go[0]))

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        x = torch.from_numpy(((img - self.mean) / self.std).transpose(2, 0, 1).copy()).unsqueeze(0)
        x = x.to(self.device).requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x)
        logit.sum().backward()
        act, grad = self._act, self._grad
        if self.reshape is not None:
            act, grad = self.reshape(act), self.reshape(grad)
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(1)).squeeze(0).detach().cpu().numpy()
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        cam -= cam.min()
        return (cam / cam.max() if cam.max() > 0 else cam).astype(np.float32)
