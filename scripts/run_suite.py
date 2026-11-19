"""Expand an experiment suite from configs/suites.yaml into train.py jobs and run them.

    python scripts/run_suite.py ocim_vit --manifest data/manifest.csv --trace-root cache/traces
    python scripts/run_suite.py ablation --manifest data/manifest.csv --trace-root cache/traces
    python scripts/run_suite.py all --manifest data/manifest.csv --dry-run

Finished runs (``result.json`` present) are skipped, so an interrupted sweep can be resumed by
repeating the command. Use ``--shard i/n`` to split one suite across several GPUs or machines.
"""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apr_tma.data import read_manifest  # noqa: E402
from apr_tma.splits import targets_of  # noqa: E402
from apr_tma.variants import VARIANTS  # noqa: E402


def expand(cfg: dict, suite: str, manifest: str, seeds, backbone=None, only_targets=None):
    s = cfg["suites"][suite]
    samples = read_manifest(manifest)
    targets = targets_of(s["protocol"], samples)
    if only_targets:
        targets = [t for t in targets if t in set(only_targets)]
    bb = backbone or s["backbone"]
    jobs = []
    for entry, target, seed in itertools.product(s["entries"], targets, seeds):
        method, variant = entry["method"], entry.get("variant", "full")
        trace = VARIANTS[variant].trace_source if method == "apr_tma" else None
        name = f"{s['protocol'].lower()}_{target}_{bb}_{method}" + (f"_{variant}" if method == "apr_tma" else "")
        jobs.append(dict(suite=suite, protocol=s["protocol"], target=target, backbone=bb, method=method,
                         variant=variant, seed=seed, trace=trace, run=f"{name}_s{seed}"))
    return jobs


def command(job, a, cfg):
    out = Path(a.out_root) / job["run"]   # shared across suites: identical runs are reused
    cmd = [sys.executable, str(ROOT / "train.py"), "--manifest", a.manifest, "--protocol", job["protocol"],
           "--target", str(job["target"]), "--backbone", job["backbone"], "--method", job["method"],
           "--variant", job["variant"], "--seed", str(job["seed"]), "--out", str(out),
           "--workers", str(a.workers)]
    if job["trace"] and a.trace_root:
        cmd += ["--trace-dir", str(Path(a.trace_root) / cfg["trace_sources"][job["trace"]])]
    if a.epochs is not None:
        cmd += ["--epochs", str(a.epochs)]
    if a.batch_size is not None:
        cmd += ["--batch-size", str(a.batch_size)]
    if a.amp:
        cmd.append("--amp")
    if a.no_pretrained:
        cmd.append("--no-pretrained")
    return cmd, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", help="suite name from the config, or 'all'")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "suites.yaml"))
    ap.add_argument("--trace-root", default=None, help="folder with bisenet/, gradcam/, stdn/ caches")
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--backbone", default=None, help="override the suite backbone")
    ap.add_argument("--targets", nargs="*", default=None, help="restrict to these targets / folds")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job starting at i")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(Path(a.config).read_text())
    seeds = a.seeds or cfg["seeds"]
    names = list(cfg["suites"]) if a.suite == "all" else [a.suite]
    jobs = [j for n in names for j in expand(cfg, n, a.manifest, seeds, a.backbone, a.targets)]
    i, n = map(int, a.shard.split("/"))
    jobs = jobs[i::n]
    print(f"{len(jobs)} jobs", flush=True)
    failed = []
    for k, job in enumerate(jobs, 1):
        cmd, out = command(job, a, cfg)
        if (out / "result.json").exists():
            print(f"[{k}/{len(jobs)}] skip   {job['run']}")
            continue
        print(f"[{k}/{len(jobs)}] run    {job['run']}", flush=True)
        if a.dry_run:
            print("   ", " ".join(cmd))
            continue
        if subprocess.run(cmd).returncode != 0:
            failed.append(job["run"])
    if failed:
        print("FAILED:", *failed, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
