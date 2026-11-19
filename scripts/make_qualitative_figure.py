"""Qualitative figure (Fig. 6): input, trace mask, modulation map, augmented image and residual.

    python scripts/make_qualitative_figure.py --live live.png --spoof spoof.png --out fig6.png \\
        [--trace-dir cache/traces/bisenet]

Live row: one acquisition pass. Spoof row: two coupled passes with a reproduction layer. The
residual |x_tilde - x| is amplified by 10. Without ``--trace-dir`` the weight-free elliptical prior
is used for the mask.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apr_tma.augment import APRTMA, APRTMAConfig  # noqa: E402
from apr_tma.data import load_rgb, trace_key  # noqa: E402
from apr_tma.tma import blend  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True)
    ap.add_argument("--spoof", required=True)
    ap.add_argument("--out", default="fig6.png")
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--seed", type=int, default=3407)
    a = ap.parse_args()

    aug = APRTMA(APRTMAConfig(p_apr=1.0))
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.6))
    titles = ["input $x$", "trace mask $M_{trace}$", "modulation $M_\\lambda$", "augmented $\\tilde{x}$",
              "residual $|\\tilde{x}-x|\\times10$"]
    for row, (path, y, name) in enumerate(((a.live, 0, "live (1 pass)"), (a.spoof, 1, "spoof (2 passes)"))):
        img = load_rgb(path)
        trace = None
        if a.trace_dir:
            f = Path(a.trace_dir) / trace_key(path)
            trace = np.load(f).astype(np.float32) if f.exists() else None
        prior = trace if trace is not None else aug.trace_prior(img)
        mask = aug.modulation(img, prior)
        rng = np.random.default_rng([a.seed, y])
        from apr_tma.pipeline import apply_pipeline
        x_tilde = blend(img, apply_pipeline(img, y, rng), mask, aug.cfg.alpha_for(y))
        panels = [img, prior, mask, x_tilde, np.clip(np.abs(x_tilde - img) * 10, 0, 1)]
        for col, p in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(p, cmap=None if p.ndim == 3 else "magma", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=10)
        axes[row, 0].set_ylabel(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print("saved", a.out)


if __name__ == "__main__":
    main()
