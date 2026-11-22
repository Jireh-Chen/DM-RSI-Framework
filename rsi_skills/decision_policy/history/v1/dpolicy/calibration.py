"""Score calibration: maps p -> p_cal. All methods except isotonic are strictly monotone, so
they never change the ranking inside one fit; isotonic can create ties."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6
METHODS = ("none", "platt", "temperature", "isotonic")


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


class Calibrator:
    def __init__(self, method: str = "none"):
        if method not in METHODS:
            raise ValueError(f"calibration.method '{method}' not in {METHODS}")
        self.method = method
        self.params: dict = {}
        self._iso = None

    def fit(self, p, y) -> "Calibrator":
        z, y = logit(p), np.asarray(y, int)
        if self.method == "platt":
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], y)
            self.params = {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
        elif self.method == "temperature":
            def nll(log_t):
                q = np.clip(sigmoid(z / np.exp(log_t)), EPS, 1 - EPS)
                return -np.mean(y * np.log(q) + (1 - y) * np.log1p(-q))
            r = minimize_scalar(nll, bounds=(-5, 5), method="bounded")
            self.params = {"T": float(np.exp(r.x))}
        elif self.method == "isotonic":
            self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(np.asarray(p, float), y)
        return self

    def transform(self, p):
        p = np.asarray(p, float)
        if self.method == "none":
            return p
        if self.method == "platt":
            return sigmoid(self.params["a"] * logit(p) + self.params["b"])
        if self.method == "temperature":
            return sigmoid(logit(p) / self.params["T"])
        return self._iso.predict(p)

    def describe(self) -> dict:
        return {"method": self.method, **self.params}
