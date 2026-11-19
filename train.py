"""Train and evaluate one run of a protocol (one target / fold, one seed).

Example (OCIM, target MSU-MFSD, ViT-ConvPass host, APR-TMA + SARE):

    python train.py --manifest data/manifest.csv --protocol OCIM --target M \\
        --backbone vit_convpass --method apr_tma --seed 3407 --trace-dir cache/traces/bisenet \\
        --out runs/ocim_M_vit_apr_tma_s3407

Use ``scripts/run_suite.py`` to launch whole tables (all targets, methods, variants and seeds).
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from apr_tma.data import FASDataset, read_manifest
from apr_tma.losses import GAMMA, total_loss
from apr_tma.metrics import evaluate_cross_domain
from apr_tma.models import BACKBONES, build_model
from apr_tma.splits import ALL_PROTOCOLS, PROTOCOL_METRIC, make_split
from apr_tma.tma import EllipseTrace
from apr_tma.variants import METHODS, VARIANTS, build_augmentation

SEEDS = (3407, 4519, 5633, 6749, 7861)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--protocol", choices=ALL_PROTOCOLS, default="OCIM")
    p.add_argument("--target", required=True, help="held-out domain / camera / attack type")
    p.add_argument("--backbone", choices=BACKBONES, default="resnet18")
    p.add_argument("--method", choices=METHODS, default="apr_tma")
    p.add_argument("--variant", choices=sorted(VARIANTS), default="full",
                   help="ablation variant (only for --method apr_tma)")
    p.add_argument("--seed", type=int, default=SEEDS[0])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--p-apr", type=float, default=None, help="override the gate (default 0.50)")
    p.add_argument("--trace-dir", default=None, help="cached trace prior (scripts/precompute_traces.py)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--amp", action="store_true", help="bfloat16 autocast on CUDA")
    p.add_argument("--no-pretrained", action="store_true", help="random init (offline smoke tests)")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    """Linear warm-up followed by cosine decay."""
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * base * (1.0 + math.cos(math.pi * t))


@torch.no_grad()
def predict(model, loader, device, amp=False):
    model.eval()
    scores = []
    for x, _, _ in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            logits = model(x.to(device))
        scores.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(scores)


def run(a: argparse.Namespace) -> dict:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = make_split(a.protocol, a.target, read_manifest(a.manifest), seed=a.seed,
                       val_fraction=a.val_fraction)
    for name, part in (("train", split.train), ("val", split.val), ("test", split.test)):
        if not part:
            raise SystemExit(f"empty {name} split for {a.protocol}/{a.target}: check the manifest columns")

    if len(split.train) < a.batch_size:
        raise SystemExit(f"training split ({len(split.train)} frames) is smaller than the batch size "
                         f"({a.batch_size}); check the manifest or lower --batch-size")

    augment, use_sare, trace_source = build_augmentation(
        a.method, a.variant if a.method == "apr_tma" else "full", trace_prior=EllipseTrace())
    if augment is not None and a.p_apr is not None:
        if hasattr(augment, "cfg"):
            from dataclasses import replace
            augment.cfg = replace(augment.cfg, p_apr=a.p_apr)
        else:
            augment.p = a.p_apr
    if trace_source and a.trace_dir is None:
        print(f"WARNING: variant needs the '{trace_source}' trace prior but --trace-dir is not set; "
              "falling back to a weight-free elliptical stand-in (smoke tests only).", flush=True)

    train_ds = FASDataset(split.train, augment=augment, seed=a.seed,
                          trace_dir=a.trace_dir if trace_source else None)
    val_ds, tgt_ds = FASDataset(split.val), FASDataset(split.test)
    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, drop_last=True,
                          num_workers=a.workers, pin_memory=pin,
                          generator=torch.Generator().manual_seed(a.seed))
    val_dl = DataLoader(val_ds, batch_size=a.batch_size, num_workers=a.workers, pin_memory=pin)
    tgt_dl = DataLoader(tgt_ds, batch_size=a.batch_size, num_workers=a.workers, pin_memory=pin)
    val_y = np.array([s.label for s in split.val])
    tgt_y = np.array([s.label for s in split.test])
    tgt_attacks = np.array([s.attack for s in split.test])

    model = build_model(a.backbone, pretrained=not a.no_pretrained).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=a.weight_decay)
    steps = len(train_dl)
    total, warmup = a.epochs * steps, a.warmup_epochs * steps

    log, best_val, step = [], float("inf"), 0
    for epoch in range(a.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        run_loss = 0.0
        for x, y, aug in train_dl:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, warmup, a.lr)
            x, y, aug = x.to(device), y.to(device), aug.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=a.amp and device.type == "cuda"):
                logits = model(x)
            loss, _, _ = total_loss(logits.float(), y, aug, GAMMA, use_sare)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.item()
            step += 1
        val_err = float(np.mean((predict(model, val_dl, device, a.amp) >= 0.5) != val_y))
        log.append({"epoch": epoch, "train_loss": run_loss / steps, "val_err": val_err})
        print(json.dumps(log[-1]), flush=True)
        if val_err <= best_val:                     # source-only model selection
            best_val = val_err
            torch.save(model.state_dict(), out / "best.pt")

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    vs, ts = predict(model, val_dl, device, a.amp), predict(model, tgt_dl, device, a.amp)
    result = evaluate_cross_domain(vs, val_y, ts, tgt_y, tgt_attacks if tgt_attacks.any() else None)
    result["metric"] = PROTOCOL_METRIC[a.protocol]
    result.update({k: v for k, v in vars(a).items() if k != "manifest"})
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "log.json").write_text(json.dumps(log, indent=2))
    np.savez_compressed(out / "scores.npz", val_scores=vs, val_labels=val_y,
                        tgt_scores=ts, tgt_labels=tgt_y, tgt_attacks=tgt_attacks)
    print(f"{a.protocol}/{a.target}  HTER={result['hter']:.2f}  AUC={result['auc']:.2f}  "
          f"ACER={result['acer']:.2f}")
    return result


if __name__ == "__main__":
    run(parse_args())
