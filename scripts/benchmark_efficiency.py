"""Computational-cost measurements of Table 11: throughput, peak memory, parameters, FLOPs.

    python scripts/benchmark_efficiency.py --backbone vit_convpass --steps 50 --out tables/efficiency.json

Reports, for the host without augmentation, the FAS-Aug proxy, and the host with APR-TMA:
  * end-to-end training throughput (images/s) including data loading and augmentation workers,
  * peak GPU memory (CUDA only), parameters and inference FLOPs of the deployed model,
  * per-image augmentation latency for the live (1 pass) and spoof (2 passes) branches,

The deployed model is the host itself: APR-TMA lives entirely in the data pipeline, so parameters,
FLOPs and latency at inference are those of the host by construction.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apr_tma.augment import APRTMA  # noqa: E402
from apr_tma.data import to_tensor  # noqa: E402
from apr_tma.losses import total_loss  # noqa: E402
from apr_tma.models import BACKBONES, build_model  # noqa: E402
from apr_tma.tma import EllipseTrace  # noqa: E402
from apr_tma.variants import build_augmentation  # noqa: E402


class SyntheticFaces(Dataset):
    def __init__(self, n, augment, seed=0):
        self.n, self.augment, self.seed = n, augment, seed
        self.base = np.random.default_rng(seed).random((224, 224, 3)).astype(np.float32)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        y, applied, img = i % 2, False, self.base
        if self.augment is not None:
            img, applied = self.augment(img, y, np.random.default_rng([self.seed, i]))
        return to_tensor(img), torch.tensor(y), torch.tensor(applied)


def train_throughput(method, variant, backbone, steps, batch, workers, device, pretrained=False):
    aug, use_sare, _ = build_augmentation(method, variant, trace_prior=EllipseTrace())
    dl = DataLoader(SyntheticFaces(steps * batch, aug), batch_size=batch, num_workers=workers, drop_last=True)
    model = build_model(backbone, pretrained=pretrained).to(device).train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    n, t0 = 0, None
    for i, (x, y, g) in enumerate(dl):
        if i == 2:                       # discard warm-up iterations
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0, n = time.perf_counter(), 0
        x, y, g = x.to(device), y.to(device), g.to(device)
        loss, _, _ = total_loss(model(x), y, g, use_sare=use_sare)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        n += x.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None
    return {"images_per_s": n / dt, "peak_vram_gb": peak}


def augmentation_latency(n=50):
    aug = APRTMA(trace_prior=EllipseTrace())
    aug.cfg = type(aug.cfg)(p_apr=1.0)
    img = np.random.default_rng(0).random((224, 224, 3)).astype(np.float32)
    out = {}
    for name, y in (("live_one_pass", 0), ("spoof_two_pass", 1)):
        t0 = time.perf_counter()
        for i in range(n):
            aug(img, y, np.random.default_rng(i))
        out[f"{name}_ms"] = 1000 * (time.perf_counter() - t0) / n
    return out


def model_cost(backbone):
    from torch.utils.flop_counter import FlopCounterMode

    m = build_model(backbone, pretrained=False).eval()
    params = sum(p.numel() for p in m.parameters())
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(torch.randn(1, 3, 224, 224))
    return {"params": params, "inference_gflops": fc.get_total_flops() / 1e9,
            "state": [(k, tuple(v.shape)) for k, v in m.state_dict().items()]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=BACKBONES, default="vit_convpass")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="tables/efficiency.json")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = {"device": str(device), "backbone": a.backbone, "batch_size": a.batch_size}
    cost = model_cost(a.backbone)
    result["deployed_model"] = {k: v for k, v in cost.items() if k != "state"}
    result["augmentation_latency"] = augmentation_latency()
    runs = {}
    for name, (m, v) in {"host": ("baseline", "full"), "fas_aug_proxy": ("fas_aug_proxy", "full"),
                         "apr_tma": ("apr_tma", "full")}.items():
        runs[name] = train_throughput(m, v, a.backbone, a.steps, a.batch_size, a.workers, device)
        print(name, runs[name], flush=True)
    host = runs["host"]["images_per_s"]
    for r in runs.values():
        r["throughput_vs_host"] = r["images_per_s"] / host
    result["training"] = runs
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["deployed_model"], indent=2), "\nsaved", a.out)


if __name__ == "__main__":
    main()
