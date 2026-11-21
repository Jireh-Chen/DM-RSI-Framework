"""Synthetic Elliptic-shaped score files for tests and the demo: steps 30-34 val, 35-49 test,
miscalibrated scores, and a prevalence/score drift from step 43. Not real data."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(seed: int = 0, rows_per_step: int = 800) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out = {}
    for split, steps in (("val", range(30, 35)), ("test", range(35, 50))):
        parts = []
        for s in steps:
            post = s >= 43
            prev = 0.03 if post else 0.07
            y = (rng.random(rows_per_step) < prev).astype(int)
            f1 = rng.normal(0, 1, rows_per_step) + 1.2 * y          # a feature the model partly ignores
            shift = -1.0 if post else 0.0
            z = -3.0 + 2.8 * y + 0.6 * f1 + rng.normal(0, 1.0, rows_per_step) + shift
            p = 1 / (1 + np.exp(-1.6 * z))                            # overconfident scores
            ids = [f"a{split}{s}x{i}_{s}" for i in range(rows_per_step)]
            parts.append(pd.DataFrame({"id": ids, "y": y, "p": p, "post43": post, "f1": f1}))
        out[split] = pd.concat(parts, ignore_index=True)
    return out


def write(out_dir, name: str = "SYN", seed: int = 0) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = make(seed)
    for split, df in data.items():
        df.drop(columns=["f1"]).to_parquet(out_dir / f"{name}_{split}.parquet", index=False)
    pd.concat(data.values())[["id", "f1"]].to_parquet(out_dir / f"{name}_features.parquet", index=False)
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="SYN")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(write(a.out, a.name, a.seed))
