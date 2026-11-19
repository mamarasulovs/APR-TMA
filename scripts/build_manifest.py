"""Build (or extend) the CSV manifest from folders of face crops.

    python scripts/build_manifest.py --dataset oulu   --root /data/faces/oulu   --out data/manifest.csv
    python scripts/build_manifest.py --dataset casia  --root /data/faces/casia  --out data/manifest.csv --append
    python scripts/build_manifest.py --dataset generic --domain Y --root /data/faces/rose \\
        --live-regex '/genuine/' --spoof-regex '/(print|video|mask)' --out data/manifest.csv --append
    python scripts/build_manifest.py --dataset generic --domain WM --root /data/faces/wmca \\
        --live-regex 'bonafide' --spoof-regex 'attack' --attack-regex 'attack/([^/]+)/' \\
        --out data/manifest.csv --append

Adapters: oulu, casia, replay, msu, siw, generic. The domain tags used by the protocols are
O, C, I, M, Y (ROSE-YOUTU), CS (CelebA-Spoof), SW (SiW-Mv2), WM (WMCA).

OULU-NPU official partitions: pass ``--subset-dirs train dev test`` if the crops are stored in one
folder per partition (``<root>/train``, ``<root>/dev``, ``<root>/test``); the folder name becomes the
``subset`` column that protocols 3 and 4 require.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apr_tma.manifests import ADAPTERS, DOMAIN_TAG, generic  # noqa: E402

COLUMNS = ["path", "label", "domain", "video_id", "attack", "phone", "session", "subset"]


def scan(root: Path, rule, domain: str, subset: str):
    base = root / subset if subset else root
    rows = []
    for p in sorted(base.rglob("*.png")):
        row = rule(p.relative_to(base), subset)
        if row is None:
            continue
        rows.append({"path": str(p.resolve()), "label": row.label, "domain": domain,
                     "video_id": f"{domain}:{subset}:{row.video_id}" if subset else f"{domain}:{row.video_id}",
                     "attack": row.attack, "phone": row.phone, "session": row.session, "subset": row.subset})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=[*ADAPTERS, "generic"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--domain", default=None, help="domain tag (defaults to the adapter's tag)")
    ap.add_argument("--subset-dirs", nargs="*", default=None)
    ap.add_argument("--live-regex", default=None)
    ap.add_argument("--spoof-regex", default=None)
    ap.add_argument("--attack-regex", default=None,
                    help="regex whose first group names the attack type (needed for WMCA, domain WM)")
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()

    if a.dataset == "generic":
        if not (a.live_regex and a.spoof_regex and a.domain):
            sys.exit("generic adapter needs --domain, --live-regex and --spoof-regex")
        rule = generic(a.live_regex, a.spoof_regex, a.attack_regex)
    else:
        rule = ADAPTERS[a.dataset]
    domain = a.domain or DOMAIN_TAG[a.dataset]

    rows = []
    for subset in (a.subset_dirs or [""]):
        rows += scan(Path(a.root), rule, domain, subset)
    if not rows:
        sys.exit("no frames matched; check --root and the dataset naming")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not (a.append and out.exists())
    with open(out, "w" if new else "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerows(rows)
    live = sum(r["label"] == 0 for r in rows)
    print(f"{domain}: {len(rows)} frames ({live} live, {len(rows) - live} spoof) -> {out}")


if __name__ == "__main__":
    main()
