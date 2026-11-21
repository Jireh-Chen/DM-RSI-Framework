"""Paired bootstrap of the objective difference between two policies on the same rows,
stratified by step when a step column is given (as in crypto-fraud-detection eval_decisions.py)."""
from __future__ import annotations

import numpy as np

from dpolicy.metrics import counts, from_counts, objective


def _index(strata, rng):
    return np.concatenate([rng.choice(ix, len(ix)) for ix in strata])


def paired_diff(y, d_new, d_old, obj: dict, costs: dict | None = None, *, steps=None,
                reps: int = 1000, seed: int = 20261121, level: float = 0.95) -> dict:
    y, d_new, d_old = (np.asarray(a) for a in (y, d_new, d_old))
    point = objective(from_counts(counts(y, d_new), costs), obj) - objective(from_counts(counts(y, d_old), costs), obj)
    if reps <= 0 or not np.isfinite(point):
        return {"diff": point, "lo": None, "hi": None, "p_better": None, "reps": 0}
    if steps is None:
        strata = [np.arange(len(y))]
    else:
        steps = np.asarray(steps)
        strata = [np.where(steps == s)[0] for s in np.unique(steps)]
    # min_coverage is checked on the point estimate above; inside resamples compare the raw metric so that
    # coverage noise around the limit cannot produce infinite differences
    obj = {k: v for k, v in obj.items() if k != "min_coverage"}
    rng = np.random.default_rng(seed)
    diffs = np.empty(reps)
    for r in range(reps):
        ix = _index(strata, rng)
        a = objective(from_counts(counts(y[ix], d_new[ix]), costs), obj)
        b = objective(from_counts(counts(y[ix], d_old[ix]), costs), obj)
        diffs[r] = a - b
    alpha = (1 - level) / 2
    return {"diff": float(point), "lo": float(np.quantile(diffs, alpha)), "hi": float(np.quantile(diffs, 1 - alpha)),
            "p_better": float(np.mean(diffs > 0)), "reps": reps}
