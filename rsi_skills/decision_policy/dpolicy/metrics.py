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


def resolve(d, d_first):
    """End-to-end decisions: a row still deferred falls back to the first-stage threshold decision."""
    d = np.asarray(d, int).copy()
    if d_first is not None:
        m = d < 0
        d[m] = np.asarray(d_first, int)[m]
    return d


def defer_quality(y, d_first, deferred, d_final=None) -> dict:
    """Are the deferred rows the hard ones? First-stage error rate inside vs outside the deferred set,
    F1 on the accepted rows, and on the deferred rows: first stage vs what the policy finally decided
    (second stage, or still deferred)."""
    y, d_first, deferred = np.asarray(y, int), np.asarray(d_first, int), np.asarray(deferred, bool)
    f1 = lambda yy, dd: from_counts(counts(yy, dd))["f1"]
    err = d_first != y
    out = {"n_deferred": int(deferred.sum()), "deferred_share": float(deferred.mean()) if len(y) else 0.0}
    if deferred.any():
        out["first_err_rate_deferred"] = float(err[deferred].mean())
        out["deferred_pos_rate"] = float(y[deferred].mean())
    if (~deferred).any():
        out["first_err_rate_accepted"] = float(err[~deferred].mean())
    out["share_of_first_errors_deferred"] = float((err & deferred).sum() / err.sum()) if err.any() else 0.0
    out["f1_accepted"] = f1(y[~deferred], d_first[~deferred])
    if deferred.any():
        out["f1_deferred_first"] = f1(y[deferred], d_first[deferred])
        if d_final is not None:
            dd = resolve(np.asarray(d_final, int)[deferred], d_first[deferred])
            out["f1_deferred_final"] = f1(y[deferred], dd)
    return out


def all_metrics(y, d, p_cal, costs=None, d_first=None, deferred=None) -> dict:
    """Decision metrics on the policy's own decisions (deferred rows excluded: f1, precision, recall are on
    accepted rows), end-to-end metrics (``*_e2e``: every row decided, deferred rows fall back to the first
    stage), and score metrics. ``f1_e2e`` is the team's F1 on all rows."""
    out = {**from_counts(counts(y, d), costs), **score_metrics(y, p_cal)}
    if d_first is not None:
        e = from_counts(counts(y, resolve(d, d_first)), costs)
        out.update({f"{k}_e2e": e[k] for k in ("precision", "recall", "f1", "cost")})
        f = from_counts(counts(y, d_first), costs)
        out.update({f"{k}_first": f[k] for k in ("precision", "recall", "f1")})
        if deferred is not None:
            out.update(defer_quality(y, d_first, deferred, d))
    return out


def dec_metrics(dec, costs=None) -> dict:
    """all_metrics on a decisions DataFrame from policy.run_policy."""
    return all_metrics(dec.y, dec.decision, dec.p_cal, costs,
                       dec["decision_first"] if "decision_first" in dec else None,
                       dec["deferred"] if "deferred" in dec else None)


def objective(m: dict, obj: dict) -> float:
    """Higher is better. obj: {metric: f1_e2e|f1|recall_all|precision|cost|cost_e2e|..., min_coverage: 0.8}.
    Cost metrics are negated. Below min_coverage (share of rows the policy decides itself) scores -inf."""
    if m["coverage"] < float(obj.get("min_coverage", 0.0)) - 1e-12:
        return float("-inf")
    key = obj.get("metric", "f1_e2e")
    v = float(m[key])
    return -v if key.startswith("cost") else v
