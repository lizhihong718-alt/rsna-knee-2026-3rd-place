"""Section 20: competition evaluation — per-pathology ROC-AUC + macro mean.

The competition metric is mean of the 12 per-target ROC-AUCs. AUC is
threshold-free (rank based, tie-corrected). Pathologies with no positive or
no negative sample in the split yield NaN and are excluded from the macro
mean (with a note). Gold-only AUC is reported separately.
"""
from __future__ import annotations

import numpy as np

from kneemri.cache import LABEL_COLS


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tie-corrected rank AUC (Mann-Whitney). NaN when only one class present."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    labels = np.nan_to_num(labels, nan=0.0)
    n_pos = float((labels > 0.5).sum())
    n_neg = float((labels <= 0.5).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # average ranks handle ties; scipy-free implementation
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0   # 1-based average rank
        i = j + 1
    sum_pos = ranks[labels > 0.5].sum()
    # Mann-Whitney U with tie-corrected average ranks
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def per_target_auc(logits: np.ndarray, targets: np.ndarray,
                   label_mask: np.ndarray | None = None) -> dict[str, float]:
    """AUC per pathology; NaN where the split has <2 classes for the target."""
    targets = np.asarray(targets, dtype=np.float64)
    aucs = {}
    for k, name in enumerate(LABEL_COLS):
        s, y = logits[:, k], targets[:, k]
        keep = np.isfinite(y)
        if label_mask is not None:
            keep &= np.asarray(label_mask)[:, k] > 0
        if not np.isfinite(s[keep]).all():
            raise ValueError(f"non-finite predictions for {name}")
        s, y = s[keep], y[keep]
        aucs[name] = roc_auc(s, y)
    return aucs


def macro_auc(aucs: dict[str, float]) -> float:
    vals = [v for v in aucs.values() if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def evaluate_predictions(logits: np.ndarray, targets: np.ndarray,
                         label_mask: np.ndarray | None = None,
                         is_gold: np.ndarray | None = None) -> dict:
    """Full validation report: per-target AUC, macro mean, and (when gold
    flags are given) the gold-only macro AUC. Report-derived CV thresholds
    soft labels at 0.5 and is a proxy, not the hidden expert-label benchmark."""
    logits,targets = np.asarray(logits),np.asarray(targets)
    valid = np.isfinite(targets)
    if label_mask is not None: valid &= np.asarray(label_mask)>0
    report = {"per_target": per_target_auc(logits, targets, label_mask)}
    report["macro_auc"] = macro_auc(report["per_target"])
    report['n_studies'] = len(targets)
    report['target_counts'] = {name:{'valid':int(valid[:,i].sum()),
        'positive':int((valid[:,i] & (targets[:,i]>.5)).sum()),
        'negative':int((valid[:,i] & (targets[:,i]<=.5)).sum())} for i,name in enumerate(LABEL_COLS)}
    confident = valid & ((targets<=.2) | (targets>=.8))
    report['per_target_confident'] = per_target_auc(logits,targets,confident)
    report['macro_auc_confident'] = macro_auc(report['per_target_confident'])
    safe = np.nan_to_num(targets,nan=0.0)
    bce = np.logaddexp(0,logits)-safe*logits
    report['soft_bce'] = float(bce[valid].mean()) if valid.any() else float('nan')
    if is_gold is not None and np.asarray(is_gold).any():
        keep = np.asarray(is_gold, dtype=bool)
        report['n_gold'] = int(keep.sum())
        report["per_target_gold"] = per_target_auc(
            logits[keep], np.asarray(targets)[keep],
            np.asarray(label_mask)[keep] if label_mask is not None else None)
        report["macro_auc_gold"] = macro_auc(report["per_target_gold"])
    return report


def format_report(report: dict) -> str:
    lines = []
    for name, auc in report["per_target"].items():
        auc_str = f"{auc:.4f}" if np.isfinite(auc) else "  n/a"
        extra = ""
        if "per_target_gold" in report:
            g = report["per_target_gold"].get(name)
            extra = f"  (gold {g:.4f})" if g is not None and np.isfinite(g) else ""
        lines.append(f"  {name:18s} AUC {auc_str}{extra}")
    lines.append(f"  {'MACRO MEAN':18s} AUC {report['macro_auc']:.4f}")
    if "macro_auc_gold" in report:
        lines.append(f"  {'MACRO (gold only)':18s} AUC {report['macro_auc_gold']:.4f}")
    return "\n".join(lines)
