"""Merge a LoRA adapter from a harness run into its base, on CPU, with the harness's own merge_lora (unchanged).

Why: onejev_score.py loads the text-only LM, while elephant-1 LoRA runs train the multimodal model, so the adapter's
keys do not match a text-only load and peft would silently skip them ("missing adapter keys"). A merged full model
loads correctly either way. CPU keeps a GPU from sitting idle (the server's idle-GPU watchdog kills such processes).

Usage (harness venv, dmrsi-env.sh sourced):
  $OJH_REPO/.venv/bin/python pipelines/elliptic_elephant/02b_merge_lora.py <RUN dir> [--step N] [--out DIR]
Default step: the step with the lowest select NLL in the run's log.jsonl (the harness's selection rule).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--step", type=int, default=None)
ap.add_argument("--out", default=None, help="default: <seed dir>/merged-step<N>-cpu")
a = ap.parse_args()

run = Path(a.run_dir).resolve()
sys.path.insert(0, os.environ["OJH_REPO"])
from harness.finalize import score_kwargs  # noqa: E402
from harness.select import checkpoint_dir, eval_events, select_step  # noqa: E402
from harness.merge import merge_lora  # noqa: E402

lock = yaml.safe_load((run / "run.lock.yaml").read_text())
if lock["mode"] != "lora":
    sys.exit(f"{run}: mode {lock['mode']}, nothing to merge")
(seed,) = lock["seeds"]
seed_dir = run / "runs" / f"{lock['experiment']}-s{seed}"
events = [json.loads(l) for l in (seed_dir / "log.jsonl").read_text().splitlines() if l.strip()]
evals = eval_events(events)
step = a.step if a.step is not None else select_step(events)
ckpt = checkpoint_dir(seed_dir, step, "lora", max(e["step"] for e in evals))
if ckpt is None:
    sys.exit(f"no adapter dir for step {step} under {seed_dir}")
nll = {e["step"]: e["nll"] for e in evals}
print(f"select NLL by step: {nll}\nselected step {step} -> {ckpt}", flush=True)

kw = score_kwargs(lock)
base = Path(lock["machine"]["model_dir"]) / lock["base"]["name"]
out = Path(a.out) if a.out else seed_dir / f"merged-step{step}-cpu"
t0 = time.time()
merged = merge_lora(base, ckpt, out, multimodal=kw["multimodal"], device="cpu", dtype=kw["dtype"],
                    chat_template=kw["chat_template"], disable_fla_kernels=kw["disable_fla_kernels"])
(out / "merge_source.json").write_text(json.dumps(
    {"run_dir": str(run), "step": step, "adapter": str(ckpt), "base": str(base), "select_nll": nll.get(step),
     "seconds": round(time.time() - t0, 1)}, indent=2))
print(f"merged -> {merged} in {time.time() - t0:.0f}s", flush=True)
