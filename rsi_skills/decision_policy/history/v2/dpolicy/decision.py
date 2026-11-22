"""Threshold rules and the abstain band. A row is predicted positive when p_cal >= threshold."""
from __future__ import annotations

import numpy as np

from dpolicy.calibration import logit, sigmoid

RULES = ("max_f1", "cost", "precision_at", "recall_at", "top_frac", "fixed")


def sweep(p, y):
    """Every distinct cut: thresholds (descending) with TP and FP counts for ``p >= thr``."""
    p, y = np.asarray(p, float), np.asarray(y, int)
    o = np.argsort(-p, kind="mergesort")
    ps, ys = p[o], y[o]
    idx = np.r_[np.where(np.diff(ps))[0], len(ps) - 1]
    tp = np.cumsum(ys)[idx]
    fp = (idx + 1) - tp
    return ps[idx], tp, fp


def choose_threshold(p, y, rule: str = "max_f1", **kw) -> dict:
    """Return {"threshold", "rule", "fit_value", "note"?}. ``threshold = inf`` means "never positive"."""
    if rule not in RULES:
        raise ValueError(f"decision.rule '{rule}' not in {RULES}")
    p, y = np.asarray(p, float), np.asarray(y, int)
    if rule == "fixed":
        return {"threshold": float(kw["value"]), "rule": rule}
    if rule == "top_frac":
        frac = float(kw["frac"])
        if not 0 < frac < 1:
            raise ValueError("decision.frac must be in (0, 1)")
        return {"threshold": float(np.quantile(p, 1 - frac)), "rule": rule, "frac": frac}
    thr, tp, fp = sweep(p, y)
    pos = int(y.sum())
    fn = pos - tp
    if rule == "max_f1":
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        i = int(np.argmax(f1))
        return {"threshold": float(thr[i]), "rule": rule, "fit_value": float(f1[i])}
    if rule == "cost":
        c_fp, c_fn = float(kw.get("cost_fp", 1.0)), float(kw.get("cost_fn", 1.0))
        cost = np.r_[c_fn * pos, c_fp * fp + c_fn * fn]          # first entry: never positive
        i = int(np.argmin(cost))
        t = float("inf") if i == 0 else float(thr[i - 1])
        return {"threshold": t, "rule": rule, "fit_value": float(cost[i] / len(y)),
                "cost_fp": c_fp, "cost_fn": c_fn}
    target = float(kw["target"])
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(pos, 1)
    if rule == "precision_at":
        ok = np.where(prec >= target)[0]
        if not len(ok):
            return {"threshold": float("inf"), "rule": rule, "target": target,
                    "note": f"no cut reaches precision {target} on the fit rows; predicting none"}
        i = int(ok[np.argmax(rec[ok])])
        return {"threshold": float(thr[i]), "rule": rule, "target": target, "fit_value": float(rec[i])}
    ok = np.where(rec >= target)[0]                                 # recall_at
    i = int(ok[np.argmax(prec[ok])])
    return {"threshold": float(thr[i]), "rule": rule, "target": target, "fit_value": float(prec[i])}


def abstain_band(p_cal, threshold: float, coverage: float) -> tuple[float, float]:
    """Band (lo, hi) around the threshold, symmetric in logit space, holding ~(1 - coverage) of the
    fit rows: the rows the policy is least sure about. Rows with lo < p_cal < hi are deferred."""
    if not 0 < coverage <= 1:
        raise ValueError("abstain.coverage must be in (0, 1]")
    if coverage == 1 or not np.isfinite(threshold):
        return threshold, threshold
    t = float(np.clip(threshold, 1e-6, 1 - 1e-6))
    dist = np.abs(logit(p_cal) - logit(t))
    w = float(np.quantile(dist, 1 - coverage))
    return float(sigmoid(logit(t) - w)), float(sigmoid(logit(t) + w))


def decide(p_cal, threshold: float, band: tuple[float, float] | None = None):
    """1 = positive, 0 = negative, -1 = deferred (abstain)."""
    p_cal = np.asarray(p_cal, float)
    d = (p_cal >= threshold).astype(int)
    if band is not None and band[0] < band[1]:
        d[(p_cal > band[0]) & (p_cal < band[1])] = -1
    return d
