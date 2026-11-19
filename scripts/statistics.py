"""Paired tests, Holm-Bonferroni correction, pooled CI and forest plot (Table 12 / Fig. 7).

    python scripts/statistics.py --runs runs --protocol OCIM --backbone vit_convpass \\
        --baseline fas_aug_proxy --ours apr_tma --out tables/
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apr_tma.stats import analyze  # noqa: E402


def collect(runs: Path, protocol, backbone, method, variant, metric):
    out = defaultdict(dict)
    for f in runs.glob("*/result.json"):
        r = json.loads(f.read_text())
        if (r["protocol"], r["backbone"], r["method"]) != (protocol, backbone, method):
            continue
        if method == "apr_tma" and r["variant"] != variant:
            continue
        out[str(r["target"])][int(r["seed"])] = r[metric]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--protocol", default="OCIM")
    ap.add_argument("--backbone", default="vit_convpass")
    ap.add_argument("--baseline", default="fas_aug_proxy")
    ap.add_argument("--baseline-variant", default="full")
    ap.add_argument("--ours", default="apr_tma")
    ap.add_argument("--ours-variant", default="full")
    ap.add_argument("--metric", default="hter")
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()

    runs = Path(a.runs)
    base = collect(runs, a.protocol, a.backbone, a.baseline, a.baseline_variant, a.metric)
    ours = collect(runs, a.protocol, a.backbone, a.ours, a.ours_variant, a.metric)
    if not base or not ours:
        sys.exit("no matching runs found for one of the two methods")
    rows = analyze(base, ours)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["| Target | Mean Δ | SD(Δ) | t | Raw p | Holm-adj. p | Outcome |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.name == "Pooled":
            lines.append(f"| Pooled | {r.mean:.3f} | {r.sd:.3f} | {r.t:.2f} | {r.p_raw:.4f} | – | "
                         f"95% CI [{r.ci_low:.3f}, {r.ci_high:.3f}]; d_z={r.dz:.2f} |")
        else:
            lines.append(f"| {r.name} | {r.mean:.3f} | {r.sd:.3f} | {r.t:.2f} | {r.p_raw:.5f} | {r.p_holm:.4f} | "
                         f"{'Significant' if r.p_holm < 0.05 else 'Not significant'} |")
    text = "\n".join(lines)
    (out / "statistics.md").write_text(text + "\n")
    (out / "statistics.json").write_text(json.dumps([r.__dict__ for r in rows], indent=2))
    print(text)

    # forest plot: mean paired difference with 95% CI per target, pooled estimate last
    from scipy import stats as st
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(rows) + 1.2))
    for i, r in enumerate(rows):
        n = max(r.n, 2)
        lo, hi = ((r.ci_low, r.ci_high) if r.ci_low is not None
                  else (r.mean - st.t.ppf(0.975, n - 1) * r.sd / n ** 0.5,
                        r.mean + st.t.ppf(0.975, n - 1) * r.sd / n ** 0.5))
        y = len(rows) - i
        ax.plot([lo, hi], [y, y], color="#1f4e79", lw=2)
        ax.plot(r.mean, y, "D" if r.name == "Pooled" else "o", color="#1f4e79", ms=8 if r.name == "Pooled" else 6)
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.set_yticks(range(len(rows), 0, -1), [r.name for r in rows])
    ax.set_xlabel("Paired HTER difference (baseline − ours), %  (> 0 favours ours)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "forest_plot.png", dpi=200)
    print(f"saved {out / 'statistics.md'} and {out / 'forest_plot.png'}")


if __name__ == "__main__":
    main()
