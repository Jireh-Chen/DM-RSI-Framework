"""Lightweight second-stage model: re-decides the rows the first stage is unsure about (the abstain
band) or all rows, from [logit(p_cal), extra features]. Its own threshold is chosen on out-of-fold
predictions over the fit rows, so it is never chosen on in-sample scores."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dpolicy.calibration import logit
from dpolicy.decision import choose_threshold

MODELS = ("logreg", "hgb")
MIN_ROWS, MIN_POS = 50, 10


def _model(name: str, seed: int):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                              l2_regularization=1.0, random_state=seed)
    raise ValueError(f"second_stage.model '{name}' not in {MODELS}")


def matrix(ids, p_cal, features: pd.DataFrame | None) -> np.ndarray:
    X = logit(p_cal)[:, None]
    if features is not None:
        F = features.reindex(pd.Index(ids)).to_numpy(float)
        X = np.hstack([X, F])
    return X


class SecondStage:
    """``bags`` > 1 (1.3): average ``bags`` models and ``bags`` out-of-fold passes, each with its own seed and fold
    split. One model's decisions moved F1 by ~0.006 between seeds on Elliptic select rows (1.2 replicates), as much as
    the gain being measured; averaging makes the second stage a property of the data, not of the seed."""

    def __init__(self, model: str = "logreg", rule: str = "max_f1", seed: int = 0, bags: int = 1, **rule_kw):
        self.model_name, self.rule, self.rule_kw, self.seed, self.bags = model, rule, rule_kw, seed, max(1, int(bags))
        self.model = None
        self.threshold = None
        self.skipped: str | None = None

    def fit(self, X, y) -> "SecondStage":
        y = np.asarray(y, int)
        if len(y) < MIN_ROWS or y.sum() < MIN_POS or (len(y) - y.sum()) < MIN_POS:
            self.skipped = f"{len(y)} rows / {int(y.sum())} positives in its fit set: too few, first stage kept"
            return self
        if self.model_name != "hgb":  # HGB handles NaN natively
            X = np.nan_to_num(X)
        seeds = [self.seed + 1000 * b for b in range(self.bags)]
        oof = np.mean([cross_val_predict(_model(self.model_name, s), X, y, method="predict_proba",
                                         cv=StratifiedKFold(5, shuffle=True, random_state=s))[:, 1] for s in seeds], axis=0)
        self.threshold = choose_threshold(oof, y, self.rule, **self.rule_kw)["threshold"]
        self.model = [_model(self.model_name, s).fit(X, y) for s in seeds]
        return self

    def predict(self, X):
        if self.model is None:
            return None
        if self.model_name != "hgb":
            X = np.nan_to_num(X)
        q = np.mean([m.predict_proba(X)[:, 1] for m in self.model], axis=0)
        return q, (q >= self.threshold).astype(int)

    def describe(self) -> dict:
        return {"model": self.model_name, "bags": self.bags, "threshold": self.threshold, "skipped": self.skipped}
