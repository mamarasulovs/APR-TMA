"""Cache trace priors A(x) so that no parser or attention model runs inside the data loader.

    # face-parsing prior used by the full model
    python scripts/precompute_traces.py --source bisenet --manifest data/manifest.csv \\
        --out cache/traces/bisenet --parser-repo external/face-parsing --device cuda

    # Grad-CAM attention source (ablation): needs a trained baseline checkpoint
    python scripts/precompute_traces.py --source gradcam --manifest data/manifest.csv \\
        --out cache/traces/gradcam --checkpoint runs/<baseline_run>/best.pt --backbone vit_convpass

    # STDN spoof-trace maps (ablation) come from the STDN authors' code; save one float map per
    # image as <out>/<trace_key>.npy (see apr_tma.data.trace_key) and pass --trace-root accordingly.

Only source-domain frames matter for training, but the full manifest is processed by default; use
--domains to restrict it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apr_tma.data import load_rgb, read_manifest, trace_key  # noqa: E402


def build_extractor(a):
    if a.source == "bisenet":
        from apr_tma.tma import BiSeNetTrace
        return BiSeNetTrace(a.parser_repo, a.weights, a.device)
    if a.source == "gradcam":
        from apr_tma.attention import GradCAMTrace
        from apr_tma.models import build_model
        model = build_model(a.backbone, pretrained=False)
        model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))
        return GradCAMTrace(model, a.device)
    raise SystemExit("--source must be bisenet or gradcam")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=("bisenet", "gradcam"), required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--parser-repo", default=None)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--backbone", default="vit_convpass")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    extractor = build_extractor(a)
    samples = [s for s in read_manifest(a.manifest) if not a.domains or s.domain in a.domains]
    for i, s in enumerate(samples):
        f = out / trace_key(s.path)
        if f.exists():
            continue
        np.save(f, extractor(load_rgb(s.path)).astype(np.float16))
        if i % 1000 == 0:
            print(f"{i}/{len(samples)}", flush=True)


if __name__ == "__main__":
    main()
