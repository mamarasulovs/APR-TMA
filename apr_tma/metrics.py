"""Cross-domain face anti-spoofing metrics: HTER at a source-fixed threshold, AUC, ACER."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def eer_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold at the equal-error point. ``scores`` are P(spoof); labels 1 = spoof."""
    if len(np.unique(labels)) < 2:
        raise ValueError("threshold selection needs both live and spoof samples")
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    return float(thr[np.nanargmin(np.abs(fnr - fpr))])


def far_frr(scores: np.ndarray, labels: np.ndarray, thr: float):
    """FAR: spoofs accepted as live. FRR: live rejected as spoof."""
    pred_spoof = scores >= thr
    spoof, live = labels == 1, labels == 0
    far = float(np.mean(~pred_spoof[spoof])) if spoof.any() else float("nan")
    frr = float(np.mean(pred_spoof[live])) if live.any() else float("nan")
    return far, frr


def hter(scores: np.ndarray, labels: np.ndarray, thr: float) -> float:
    far, frr = far_frr(scores, labels, thr)
    return 100.0 * 0.5 * (far + frr)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    return 100.0 * float(roc_auc_score(labels, scores))


def acer(scores: np.ndarray, labels: np.ndarray, thr: float, attacks=None) -> float:
    """ACER = (APCER + BPCER) / 2 in percent (ISO/IEC 30107-3, OULU-NPU convention).

    APCER is the worst-case rate over presentation-attack instruments when ``attacks`` is given
    (one entry per sample, "" for live), otherwise the pooled spoof acceptance rate.
    """
    scores, labels = np.asarray(scores), np.asarray(labels)
    pred_spoof = scores >= thr
    live = labels == 0
    bpcer = float(np.mean(pred_spoof[live])) if live.any() else float("nan")
    if attacks is None:
        apcer = float(np.mean(~pred_spoof[labels == 1]))
    else:
        attacks = np.asarray(attacks)
        rates = [float(np.mean(~pred_spoof[(labels == 1) & (attacks == a)]))
                 for a in np.unique(attacks[labels == 1])]
        apcer = max(rates) if rates else float("nan")
    return 100.0 * 0.5 * (apcer + bpcer)


def evaluate_cross_domain(val_scores, val_labels, tgt_scores, tgt_labels, tgt_attacks=None) -> dict:
    """Fix the threshold on source validation data, then report target HTER, AUC and ACER."""
    tgt_scores, tgt_labels = np.asarray(tgt_scores), np.asarray(tgt_labels)
    thr = eer_threshold(np.asarray(val_scores), np.asarray(val_labels))
    return {"threshold": thr,
            "hter": hter(tgt_scores, tgt_labels, thr),
            "auc": auc(tgt_scores, tgt_labels) if len(np.unique(tgt_labels)) > 1 else float("nan"),
            "acer": acer(tgt_scores, tgt_labels, thr, tgt_attacks)}
