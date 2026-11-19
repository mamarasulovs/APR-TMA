"""Paired statistical analysis of Section "Statistical and Error Analysis" (Table 12, Fig. 7).

Per-target paired differences  Delta = HTER_baseline - HTER_ours  over the shared seeds, a paired
t-test (df = n_seeds - 1) per target with Holm-Bonferroni correction across targets, and a pooled
test on the per-seed average over targets (mean, SD, t, p, 95% CI and d_z = mean / SD).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats


def holm_bonferroni(p: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1), same order as the input."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


@dataclass
class PairedRow:
    name: str
    mean: float
    sd: float
    t: float
    p_raw: float
    p_holm: float | None
    ci_low: float | None = None
    ci_high: float | None = None
    dz: float | None = None
    n: int = 0


def paired_test(delta: np.ndarray):
    delta = np.asarray(delta, float)
    n = len(delta)
    sd = delta.std(ddof=1)
    t, p = stats.ttest_1samp(delta, 0.0) if sd > 0 else (np.inf if delta.mean() > 0 else 0.0, 0.0)
    return float(delta.mean()), float(sd), float(t), float(p), n


def analyze(baseline: Dict[str, Dict[int, float]], ours: Dict[str, Dict[int, float]],
            alpha: float = 0.05) -> List[PairedRow]:
    """``baseline[target][seed]`` and ``ours[target][seed]`` are HTER values (lower is better)."""
    targets = sorted(set(baseline) & set(ours))
    rows, deltas = [], {}
    for t in targets:
        seeds = sorted(set(baseline[t]) & set(ours[t]))
        d = np.array([baseline[t][s] - ours[t][s] for s in seeds])
        deltas[t] = (seeds, d)
        mean, sd, tv, p, n = paired_test(d)
        rows.append(PairedRow(t, mean, sd, tv, p, None, n=n))
    for r, adj in zip(rows, holm_bonferroni([r.p_raw for r in rows])):
        r.p_holm = float(adj)
    common = sorted(set.intersection(*[set(deltas[t][0]) for t in targets]))
    per_seed = np.array([np.mean([deltas[t][1][deltas[t][0].index(s)] for t in targets]) for s in common])
    mean, sd, tv, p, n = paired_test(per_seed)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    rows.append(PairedRow("Pooled", mean, sd, tv, p, None, mean - half, mean + half,
                          mean / sd if sd > 0 else float("inf"), n))
    return rows
