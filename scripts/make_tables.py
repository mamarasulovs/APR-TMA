"""Collect runs/*/result.json into the result tables of the manuscript.

    python scripts/make_tables.py --runs runs --out tables

Writes, per protocol and backbone, ``<protocol>_<backbone>.{md,csv,tex}`` (rows: method / variant;
columns: targets and the average) and, for OCIM hosts that have the ablation runs,
``ablation_<backbone>.{md,csv,tex}`` with Delta-HTER against the full model (Table 6). Values are means over the available seeds; the number of
seeds per cell is recorded in the CSV.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apr_tma.variants import VARIANTS  # noqa: E402

ACER_PROTOCOLS = {"OULU_P3", "OULU_P4", "WMCA_UNSEEN"}
METHOD_LABEL = {"baseline": "Host (no augmentation)", "ti_aug": "TI-Aug",
                "fas_aug_proxy": "FAS-Aug proxy + SARE", "apr_tma": "APR-TMA + SARE (ours)"}


def label(method, variant):
    if method != "apr_tma" or variant == "full":
        return METHOD_LABEL[method]
    return f"APR-TMA: {VARIANTS[variant].description}"


def load(runs: Path):
    cells = defaultdict(lambda: defaultdict(list))  # (protocol, backbone, method, variant)[target] -> results
    for f in sorted(runs.glob("*/result.json")):
        r = json.loads(f.read_text())
        key = (r["protocol"], r["backbone"], r["method"], r["variant"] if r["method"] == "apr_tma" else "-")
        cells[key][str(r["target"])].append(r)
    return cells


def summarize(per_target, protocol):
    """Per-target seed means, then the summary across targets."""
    rows = {}
    for t, rs in per_target.items():
        rows[t] = {"hter": np.mean([r["hter"] for r in rs]), "auc": np.mean([r["auc"] for r in rs]),
                   "acer": np.mean([r["acer"] for r in rs]), "n": len(rs)}
    return rows


def fmt_cell(v, protocol):
    return f"{v['acer']:.2f}" if protocol in ACER_PROTOCOLS else f"{v['hter']:.2f} / {v['auc']:.2f}"


def sort_targets(ts):
    return sorted(ts, key=lambda s: (len(s), s))


def to_latex(header, body, caption):
    cols = "l" + "c" * (len(header) - 1)
    out = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\begin{{tabular}}{{{cols}}}",
           "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    out += [" & ".join(r) + " \\\\" for r in body]
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def write(out: Path, stem, header, body, caption):
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in body]
    (out / f"{stem}.md").write_text("\n".join(md) + "\n")
    with open(out / f"{stem}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(body)
    tex = lambda t: (t.replace("→", "$\\to$").replace("↓", "$\\downarrow$").replace("↑", "$\\uparrow$")
                     .replace("Δ", "$\\Delta$").replace("±", "$\\pm$").replace("%", "\\%").replace("&", "\\&")
                     .replace("_", "\\_"))
    (out / f"{stem}.tex").write_text(to_latex([tex(h) for h in header], [[tex(c) for c in r] for r in body],
                                              caption.replace("%", "\\%")) + "\n")
    print(f"wrote {stem}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = load(Path(a.runs))
    if not cells:
        sys.exit(f"no results found under {a.runs}/")

    groups = defaultdict(dict)
    for (proto, bb, method, variant), per_target in cells.items():
        groups[(proto, bb)][(method, variant)] = summarize(per_target, proto)

    order = {m: i for i, m in enumerate(METHOD_LABEL)}
    var_order = {v: i for i, v in enumerate(VARIANTS)}
    for (proto, bb), rows in groups.items():
        acer = proto in ACER_PROTOCOLS
        targets = sort_targets({t for r in rows.values() for t in r})
        header = ["Method"] + [f"→{t}" for t in targets] + (["Mean ± SD (folds)"] if acer else ["Avg. HTER"])
        body = []
        for (m, v), r in sorted(rows.items(), key=lambda kv: (order[kv[0][0]], var_order.get(kv[0][1], 0))):
            cells_ = [fmt_cell(r[t], proto) if t in r else "–" for t in targets]
            vals = [r[t]["acer" if acer else "hter"] for t in targets if t in r]
            summ = f"{np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f}" if acer and len(vals) > 1 else f"{np.mean(vals):.2f}"
            body.append([label(m, v)] + cells_ + [summ])
        metric = "ACER (%)" if acer else "HTER (%) / AUC (%)"
        write(out, f"{proto.lower()}_{bb}", header, body, f"{proto}, {bb} host. {metric}; mean over seeds.")

        if proto == "OCIM":
            full = rows.get(("apr_tma", "full"))
            if full and len(targets) == len(full) and len(rows) > 5:
                fh = np.mean([full[t]["hter"] for t in targets])
                ab = []
                for (m, v), r in sorted(rows.items(), key=lambda kv: var_order.get(kv[0][1], -1)):
                    if m != "apr_tma" or len(r) != len(targets):
                        continue
                    h = np.mean([r[t]["hter"] for t in targets])
                    u = np.mean([r[t]["auc"] for t in targets])
                    ab.append((h, [label(m, v), f"{h:.2f}", f"{u:.2f}", f"{h - fh:+.2f}"]))
                ab.sort(key=lambda x: -x[0])
                write(out, f"ablation_{bb}", ["Variant", "HTER ↓", "AUC ↑", "ΔHTER"], [r for _, r in ab],
                      "Component and leakage-control ablation on OCIM (average over four targets).")


if __name__ == "__main__":
    main()
