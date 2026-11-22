"""Score files and the validation protocol.

Input contract (same as crypto-fraud-detection `elephant/scripts/onejev_score.py` and the
non-LLM baselines): ``<scores_dir>/<name>_<split>.parquet`` with columns

    id  y (0/1)  p (P(positive) in [0, 1])   and optionally  step, post43, seen_in_train, logit_pos, logit_neg

The test split is only read through ``load_split(..., allow_test=True)``; nothing in the evolve
loop passes it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ("id", "y", "p")


class DataError(ValueError):
    pass


class SplitLocked(PermissionError):
    pass


def _check(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataError(f"{path}: missing columns {missing}")
    if df["id"].duplicated().any():
        raise DataError(f"{path}: {int(df['id'].duplicated().sum())} duplicated ids")
    y = df["y"]
    if y.isna().any() or not set(np.unique(y)).issubset({0, 1}):
        raise DataError(f"{path}: y must be 0/1 with no missing values")
    p = df["p"]
    if p.isna().any() or (p < 0).any() or (p > 1).any():
        raise DataError(f"{path}: p must be in [0, 1] with no missing values")
    return df


def add_step(df: pd.DataFrame, step_from_id: bool) -> pd.DataFrame:
    """Elliptic ids look like ``<address>_<step>``; parse the step when the column is absent."""
    if "step" in df.columns or not step_from_id:
        return df
    df = df.copy()
    df["step"] = df["id"].astype(str).str.rsplit("_", n=1).str[1].astype(int)
    return df


def load_split(scores_dir, name: str, split: str, *, allow_test: bool = False,
               step_from_id: bool = False) -> pd.DataFrame:
    if split.startswith("test") and not allow_test:
        raise SplitLocked(f"split '{split}' is locked: only `dpolicy final-test --allow-test` reads it")
    path = Path(scores_dir) / f"{name}_{split}.parquet"
    if not path.exists():
        raise DataError(f"{path} not found")
    df = _check(pd.read_parquet(path), path)
    df["y"] = df["y"].astype(int)
    df["p"] = df["p"].astype(float)
    return add_step(df, step_from_id).reset_index(drop=True)


def split_protocol(val: pd.DataFrame, protocol: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the labelled validation rows into a *fit* part (policies are fitted here) and a
    *select* part (candidates are compared here).

    protocol:
      by: step          -> select = rows whose step is in ``select_steps`` (default: the last
                           ``select_frac`` of the distinct steps, i.e. the later period; this mimics
                           deployment, where a policy is fitted on the past and used on the future)
      by: group         -> deterministic split by a group column (e.g. address), ``select_frac`` of
                           the groups go to select
    """
    by = protocol.get("by", "step")
    frac = float(protocol.get("select_frac", 0.4))
    if by == "step":
        if "step" not in val.columns:
            raise DataError("protocol by: step needs a step column (or step_from_id: true)")
        steps = sorted(val["step"].unique())
        sel = protocol.get("select_steps")
        if sel is None:
            k = max(1, int(round(len(steps) * frac)))
            if k >= len(steps):
                raise DataError(f"only {len(steps)} distinct steps: cannot keep a fit part")
            sel = steps[-k:]
        mask = val["step"].isin(set(sel)).to_numpy()
    elif by == "group":
        col = protocol["group"]
        seed = int(protocol.get("seed", 20261121))
        groups = np.array(sorted(val[col].astype(str).unique()))
        rng = np.random.default_rng(seed)
        chosen = set(rng.choice(groups, size=max(1, int(round(len(groups) * frac))), replace=False))
        mask = val[col].astype(str).isin(chosen).to_numpy()
    else:
        raise DataError(f"unknown protocol.by '{by}' (step | group)")
    fit, select = val[~mask].reset_index(drop=True), val[mask].reset_index(drop=True)
    for part, df in (("fit", fit), ("select", select)):
        if df.empty or df["y"].nunique() < 2:
            raise DataError(f"protocol leaves the {part} part with {len(df)} rows / one class")
    return fit, select


def load_features(path, columns: list[str] | None) -> pd.DataFrame:
    """Extra per-row features for a second-stage model, keyed by id (parquet or csv)."""
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "id" not in df.columns:
        raise DataError(f"{path}: features need an id column")
    cols = columns or [c for c in df.columns if c != "id" and np.issubdtype(df[c].dtype, np.number)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataError(f"{path}: missing feature columns {missing}")
    return df[["id", *cols]].drop_duplicates("id").set_index("id")
