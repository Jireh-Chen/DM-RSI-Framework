"""Check that the merged LoRA model, scored by onejev_score.py (text-only LM, batched: the path 03 uses), gives the
same probabilities as the harness's own scorer (base + adapter, multimodal, one row at a time, as in finalize) on the
first N prepared select rows. Also reports both timings.

Usage (dmrsi-env.sh sourced, one GPU visible as cuda:0):
  PYTHONPATH=$DMRSI_PYDEPS TORCH_COMPILE_DISABLE=1 $OJH_REPO/.venv/bin/python \
      pipelines/elliptic_elephant/05_check_merged.py <RUN dir> <merged dir> [N=200]
Writes <merged dir>/merge_check.json; exits 1 when the mean |p diff| exceeds MEAN_TOL (the max is reported against
the harness's finalize.TOL, which bf16 merging plus batching does not meet row by row).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

run, merged = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
sys.path.insert(0, os.environ["OJH_REPO"])
from harness.finalize import TOL, score_kwargs  # noqa: E402

MEAN_TOL = 0.005
from harness.score import score_checkpoint  # noqa: E402
from harness.select import checkpoint_dir, eval_events  # noqa: E402

lock = yaml.safe_load((run / "run.lock.yaml").read_text())
(seed,) = lock["seeds"]
seed_dir = run / "runs" / f"{lock['experiment']}-s{seed}"
events = [json.loads(l) for l in (seed_dir / "log.jsonl").read_text().splitlines() if l.strip()]
step = json.loads((merged / "merge_source.json").read_text())["step"]
ckpt = checkpoint_dir(seed_dir, step, "lora", max(e["step"] for e in eval_events(events)))
select = Path(lock["data"]["select"])
rows = [json.loads(l) for l in select.read_text().splitlines() if l.strip()][:n]

t0 = time.time()
ref = score_checkpoint(ckpt, rows, base_dir=Path(lock["machine"]["model_dir"]) / lock["base"]["name"],
                       **score_kwargs(lock))
t_ref = time.time() - t0


def p_pos(rec):
    pos = "Illicit" if "Illicit" in rec["labels"] else "yes"
    return rec["probs"][rec["labels"].index(pos)]


with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
    t0 = time.time()
    subprocess.run([sys.executable, f"{os.environ['CFD']}/elephant/scripts/onejev_score.py",
                    "--onejev", os.environ["ONEJEV_ROOT"], "--model", str(merged), "--rows", str(select),
                    "--name", "check", "--split", "select", "--out", tmp, "--device", "cuda:0", "--limit", str(n)],
                   check=True)
    t_m = time.time() - t0
    import pandas as pd
    df = pd.read_parquet(Path(tmp) / "check_select.parquet")

diffs = [abs(float(p) - p_pos(ref[i])) for i, p in zip(df["id"], df["p"])]
diff, mean = max(diffs), sum(diffs) / len(diffs)
# bf16 merge + batched scoring move single rows by up to ~0.1 (measured 2026-11-22 on step 2640: mean 0.0008,
# p99 0.014, same NLL); a skipped adapter moves the mean by ~0.1+. Gate on the mean, report the max.
out = {"rows": len(df), "step": step, "max_abs_p_diff": diff, "mean_abs_p_diff": mean, "harness_tolerance_max": TOL,
       "ok": mean <= MEAN_TOL, "mean_tolerance": MEAN_TOL,
       "seconds_harness_adapter": round(t_ref, 1), "seconds_onejev_score_merged": round(t_m, 1)}
(merged / "merge_check.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out), flush=True)
sys.exit(0 if out["ok"] else 1)
