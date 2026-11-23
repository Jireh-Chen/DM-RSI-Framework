"""A decision policy = calibration -> threshold rule -> abstain band -> second stage, fitted on
labelled rows and applied to other rows, either once (static) or refitted per step on the labels
that have arrived so far (rolling)."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pandas as pd

from dpolicy.calibration import Calibrator
from dpolicy.decision import abstain_band, choose_threshold, decide
from dpolicy.second_stage import SecondStage, matrix

DEFAULT = {
    "name": "policy",
    "calibration": {"method": "none"},
    "decision": {"rule": "max_f1"},
    "abstain": {"coverage": 1.0},
    "second_stage": {"enabled": False, "model": "logreg", "apply_to": "band", "features": [], "rule": "max_f1"},
    "refit": {"mode": "static"},
}
TOP_KEYS = set(DEFAULT) | {"parent", "note"}


class PolicyError(ValueError):
    pass


def normalize(spec: dict) -> dict:
    """Fill defaults and validate. Unknown keys are an error so a typo never silently does nothing."""
    unknown = set(spec) - TOP_KEYS
    if unknown:
        raise PolicyError(f"unknown policy keys {sorted(unknown)}")
    out = copy.deepcopy(DEFAULT)
    for k, v in spec.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    ss, ab, rf = out["second_stage"], out["abstain"], out["refit"]
    if ss["enabled"] and ss["apply_to"] == "band" and float(ab["coverage"]) >= 1:
        raise PolicyError("second_stage.apply_to: band needs abstain.coverage < 1")
    if ss["apply_to"] not in ("band", "all"):
        raise PolicyError("second_stage.apply_to must be band | all")
    if not (ss["features"] == "all" or isinstance(ss["features"], list)):
        raise PolicyError("second_stage.features must be a list of column names or 'all'")
    if rf["mode"] not in ("static", "rolling"):
        raise PolicyError("refit.mode must be static | rolling")
    return out


def spec_hash(spec: dict) -> str:
    core = {k: v for k, v in normalize(spec).items() if k not in ("name", "parent", "note")}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:12]


class Fitted:
    def __init__(self, spec: dict, fit_df: pd.DataFrame, features: pd.DataFrame | None, seed: int = 0):
        self.spec = spec
        y, p = fit_df["y"].to_numpy(int), fit_df["p"].to_numpy(float)
        self.cal = Calibrator(spec["calibration"]["method"]).fit(p, y)
        pc = self.cal.transform(p)
        dec = dict(spec["decision"])
        rule = dec.pop("rule")
        self.thr = choose_threshold(pc, y, rule, **dec)
        cov = float(spec["abstain"]["coverage"])
        self.band = abstain_band(pc, self.thr["threshold"], cov) if cov < 1 else None
        self.ss = None
        ss = spec["second_stage"]
        if ss["enabled"]:
            cols = list(features.columns) if (features is not None and ss["features"] == "all") else ss["features"]
            self.feat = features[cols] if (features is not None and cols) else None
            if ss["features"] and features is None:
                raise PolicyError("second_stage.features given but no data.features_path in the run config")
            rows = np.ones(len(y), bool)
            if ss["apply_to"] == "band":
                rows = decide(pc, self.thr["threshold"], self.band) == -1
            X = matrix(fit_df["id"].to_numpy(), pc, self.feat)
            self.ss = SecondStage(ss["model"], ss.get("rule", "max_f1"), seed=seed).fit(X[rows], y[rows])
        self.n_fit, self.pos_fit = len(y), int(y.sum())

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        pc = self.cal.transform(df["p"].to_numpy(float))
        t = self.thr["threshold"]
        d = decide(pc, t, self.band)
        stage = np.where(d == -1, "defer", "first").astype(object)
        if self.ss is not None:
            rows = (d == -1) if self.spec["second_stage"]["apply_to"] == "band" else np.ones(len(d), bool)
            got = self.ss.predict(matrix(df["id"].to_numpy(), pc, self.feat)[rows]) if rows.any() else None
            if got is not None:
                d[rows] = got[1]
                stage[rows] = "second"
            elif self.ss.skipped:  # second stage could not be fitted: the band falls back to the threshold
                d[rows] = (pc[rows] >= t).astype(int)
                stage[rows] = "first"
        out = df[[c for c in ("id", "y", "p", "step", "post43") if c in df.columns]].copy()
        out["p_cal"], out["decision"], out["stage"], out["threshold"] = pc, d, stage, t
        # what the first stage alone (threshold, no abstain band) would decide: the fallback for deferred rows
        # and the reference for "are the deferred rows the bad ones?"
        out["decision_first"] = (pc >= t).astype(int)
        out["deferred"] = (stage == "defer") | (stage == "second")
        return out

    def describe(self) -> dict:
        return {"n_fit": self.n_fit, "pos_fit": self.pos_fit, "calibration": self.cal.describe(),
                "threshold": self.thr, "band": self.band,
                "second_stage": self.ss.describe() if self.ss else None}


def run_policy(spec: dict, fit_df: pd.DataFrame, eval_df: pd.DataFrame, features=None, seed: int = 0):
    """Fit on ``fit_df`` and decide ``eval_df``. Returns (decisions DataFrame, list of fit descriptions).

    rolling: before step t the policy is refitted on labelled rows with step in
    (t - delay - window, t - delay] drawn from fit_df and from eval_df itself (labels that would have
    arrived by then in deployment). window: null = all history."""
    spec = normalize(spec)
    rf = spec["refit"]
    if rf["mode"] == "static":
        f = Fitted(spec, fit_df, features, seed)
        return f.apply(eval_df), [{"steps": "all", **f.describe()}]
    if "step" not in eval_df.columns or "step" not in fit_df.columns:
        raise PolicyError("refit.mode rolling needs a step column")
    delay, window = int(rf.get("delay", 1)), rf.get("window")
    hist = pd.concat([fit_df, eval_df], ignore_index=True)
    parts, fits = [], []
    for t in sorted(eval_df["step"].unique()):
        hi = t - delay
        pool = hist[hist["step"] <= hi]
        if window is not None:
            win = pool[pool["step"] > hi - int(window)]
            pool = win if win["y"].nunique() == 2 else pool
        if pool.empty or pool["y"].nunique() < 2:
            raise PolicyError(f"step {t}: no labelled history with both classes")
        f = Fitted(spec, pool, features, seed)
        parts.append(f.apply(eval_df[eval_df["step"] == t]))
        fits.append({"step": int(t), **f.describe()})
    return pd.concat(parts, ignore_index=True), fits
