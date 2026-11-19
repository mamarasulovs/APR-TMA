"""Host models. APR-TMA is backbone-agnostic: f = h o g with a single-logit liveness head.

* ``resnet18``      - ImageNet-initialised ResNet-18 (torchvision).
* ``clip_vit``      - CLIP ViT-B/16 image encoder (timm, OpenAI weights) fine-tuned with a linear head:
                      a FLIP-V-style vision-language host for the plug-in experiment.
* ``vit_convpass``  - ViT-B/16 (timm) with convolutional bypass adapters (ConvPass,
                      Jie & Deng, 2022). The ViT weights are frozen; adapters and head train.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResNet18FAS(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(*list(net.children())[:-1])  # g_phi
        self.head = nn.Linear(512, 1)                              # h_psi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x).flatten(1)).squeeze(1)


class ConvPass(nn.Module):
    """Convolutional bypass: 1x1 down-projection, 3x3 conv on the token grid, 1x1 up-projection."""

    def __init__(self, dim: int, hidden: int = 8, grid: int = 14):
        super().__init__()
        self.grid = grid
        self.down = nn.Linear(dim, hidden)
        self.conv = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.up = nn.Linear(hidden, dim)
        self.act = nn.GELU()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        cls, patches = tokens[:, :1], tokens[:, 1:]
        b, n, _ = patches.shape
        z = self.act(self.down(patches)).transpose(1, 2).reshape(b, -1, self.grid, self.grid)
        z = self.act(self.conv(z)).flatten(2).transpose(1, 2)
        up_patches = self.up(z)
        up_cls = self.up(self.act(self.down(cls)))
        return torch.cat([up_cls, up_patches], dim=1)


class _AdaptedBlock(nn.Module):
    def __init__(self, block: nn.Module, dim: int):
        super().__init__()
        self.block = block
        self.adapt_attn = ConvPass(dim)
        self.adapt_mlp = ConvPass(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.block
        h = b.norm1(x)
        x = x + b.drop_path1(b.ls1(b.attn(h))) + self.adapt_attn(h)
        h = b.norm2(x)
        x = x + b.drop_path2(b.ls2(b.mlp(h))) + self.adapt_mlp(h)
        return x


class ViTConvPassFAS(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        import timm

        vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)
        for p in vit.parameters():
            p.requires_grad_(False)
        dim = vit.embed_dim
        vit.blocks = nn.Sequential(*[_AdaptedBlock(blk, dim) for blk in vit.blocks])
        self.backbone = vit
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(1)


class CLIPViTFAS(nn.Module):
    """Image-only CLIP host. For the full FLIP-MCL host, use the official code and plug
    :class:`apr_tma.augment.APRTMA` into its data pipeline (see docs/PLUGIN.md)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import timm

        self.backbone = timm.create_model("vit_base_patch16_clip_224.openai", pretrained=pretrained,
                                          num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).squeeze(1)


BACKBONES = ("resnet18", "vit_convpass", "clip_vit")


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    if name == "resnet18":
        return ResNet18FAS(pretrained)
    if name == "vit_convpass":
        return ViTConvPassFAS(pretrained)
    if name == "clip_vit":
        return CLIPViTFAS(pretrained)
    raise ValueError(f"unknown backbone: {name}")
