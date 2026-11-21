"""Decision metrics (with deferrals), ranking/calibration metrics, and the objective."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def counts(y, d) -> dict:
    y, d = np.asarray(y, int), np.asarray(d, int)
    cov = d >= 0
    return {"n": len(y), "n_pos": int(y.sum()), "n_defer": int((~cov).sum()),
            "tp": int(((d == 1) & (y == 1)).sum()), "fp": int(((d == 1) & (y == 0)).sum()),
            "fn": int(((d == 0) & (y == 1)).sum()), "tn": int(((d == 0) & (y == 0)).sum()),
            "defer_pos": int(((~cov) & (y == 1)).sum())}


def from_counts(c: dict, costs: dict | None = None) -> dict:
    costs = costs or {}
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    n = max(c["n"], 1)
    out = dict(c)
    out["coverage"] = 1 - c["n_defer"] / n
    out["precision"] = tp / (tp + fp) if tp + fp else 0.0
    out["recall"] = tp / (tp + fn) if tp + fn else 0.0               # on covered rows
    out["f1"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0  # on covered rows
    # recall over all positives: a deferred positive is not caught automatically
    out["recall_all"] = tp / c["n_pos"] if c["n_pos"] else 0.0
    out["cost"] = (costs.get("fp", 1.0) * fp + costs.get("fn", 1.0) * fn
                   + costs.get("defer", 0.0) * c["n_defer"]) / n
    return out


def ece(y, p, bins: int = 15) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def score_metrics(y, p) -> dict:
    y, p = np.asarray(y, int), np.asarray(p, float)
    out = {"brier": float(np.mean((p - y) ** 2)), "ece": ece(y, p)}
    if 0 < y.sum() < len(y):
        out["auprc"] = float(average_precision_score(y, p))
        out["auroc"] = float(roc_auc_score(y, p))
    return out


def all_metrics(y, d, p_cal, costs=None) -> dict:
    return {**from_counts(counts(y, d), costs), **score_metrics(y, p_cal)}


def objective(m: dict, obj: dict) -> float:
    """Higher is better. obj: {metric: f1|recall_all|precision|cost|..., min_coverage: 0.9}.
    ``cost`` is negated. A run below min_coverage scores -inf."""
    if m["coverage"] < float(obj.get("min_coverage", 0.0)) - 1e-12:
        return float("-inf")
    key = obj.get("metric", "f1")
    v = float(m[key])
    return -v if key == "cost" else v
