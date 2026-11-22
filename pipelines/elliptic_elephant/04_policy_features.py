"""Numeric per-address features for the Decision Policy second stage, from the same OneJev rows the model saw.

Only the model's input (state.summary and state.transactions) is used: no label, no address string, no absolute
time step, so the second stage sees nothing the first stage could not. Output: one parquet keyed by id, covering
the val and test rows (dpolicy reads test rows only in final-test).

Usage: $JEV_PYTHON pipelines/elliptic_elephant/04_policy_features.py <rows dir> <out.parquet>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUMMARY = ("txs_sent", "txs_received", "first_seen_steps_ago", "active_steps", "txs_listed")
TX_NUM = ("tx_total_btc", "tx_fee_btc", "tx_num_input_addresses", "tx_num_output_addresses",
          "tx_smallest_output_btc", "tx_largest_output_btc")


def row_features(r: dict) -> dict:
    st = r["state"]
    out = {"id": r["id"]}
    s = st.get("summary", {})
    for k in SUMMARY:
        out[f"s_{k}"] = float(s.get(k, np.nan)) if s.get(k) is not None else np.nan
    txs = st.get("transactions", []) or []
    out["tx_n"] = len(txs)
    roles = [t.get("role") for t in txs]
    for role in ("sent", "received", "both"):
        out[f"tx_frac_{role}"] = roles.count(role) / len(txs) if txs else np.nan
    for k in TX_NUM:
        v = np.array([t[k] for t in txs if t.get(k) is not None], dtype=float)
        out[f"tx_mean_{k}"] = float(v.mean()) if v.size else np.nan
        out[f"tx_max_{k}"] = float(v.max()) if v.size else np.nan
    ago = np.array([t["steps_ago"] for t in txs if t.get("steps_ago") is not None], dtype=float)
    out["tx_min_steps_ago"] = float(ago.min()) if ago.size else np.nan
    out["tx_span_steps"] = float(ago.max() - ago.min()) if ago.size else np.nan
    v = np.array([t["tx_total_btc"] for t in txs if t.get("tx_total_btc") is not None], dtype=float)
    out["tx_log_sum_btc"] = float(np.log1p(v.sum())) if v.size else np.nan
    return out


def main(rows_dir: str, out_path: str) -> None:
    feats = []
    for split in ("val", "test"):
        with open(Path(rows_dir) / f"{split}.jsonl") as f:
            feats += [row_features(json.loads(line)) for line in f if line.strip()]
    df = pd.DataFrame(feats)
    if df["id"].duplicated().any():
        sys.exit(f"{int(df['id'].duplicated().sum())} duplicated ids across val/test")
    num = [c for c in df.columns if c != "id"]
    df[num] = df[num].astype(float).fillna(-1.0)   # hgb handles NaN, logreg does not: keep one convention
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"{len(df):,} rows, {len(num)} features -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
