#!/bin/bash
# Run on the server in tmux right after `ojh run` starts, so the post-training steps go on without anyone attached:
#   wait for the run to end (done, or finalize killed) -> merge the selected LoRA step on CPU -> wait for one free GPU
#   -> check merged == harness scorer on 200 select rows -> score full val and test.
# Usage: tmux new -d -s dmrsi-after "bash -l pipelines/elliptic_elephant/after_training.sh <RUN dir> <score name>"
# Uses at most one GPU, and only a card with no process on it while this user holds no other card (2-GPU rule).
set -uo pipefail
source /shared/ssd/yile/dmrsi/dmrsi-env.sh
umask 077
RUN=${1:?RUN dir} NAME=${2:?score name}
LOG=$DMRSI_HOME/logs/after_training-$NAME.log
STATE=$DMRSI_HOME/logs/after_training-$NAME.state
PY=$OJH_REPO/.venv/bin/python
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; echo "$*" > "$STATE"; }
cd "$DMRSI_REPO"

say "waiting for $RUN to leave 'running'"
while grep -q running "$RUN/status" 2>/dev/null; do sleep 60; done
say "run status: $(cat "$RUN/status")"

MERGED=$(ls -d "$RUN"/runs/*/merged-step*-cpu 2>/dev/null | head -1)
if [ -z "$MERGED" ]; then
  say "merging on CPU"
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 nice -n 10 "$PY" pipelines/elliptic_elephant/02b_merge_lora.py "$RUN" >> "$LOG" 2>&1 \
    || { say "FAILED: merge"; exit 1; }
  MERGED=$(ls -d "$RUN"/runs/*/merged-step*-cpu | head -1)
fi
say "merged: $MERGED"

free_gpu() {  # a card with nothing on it, and only while this user holds no card
  local mine; mine=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read -r p; do
    [ "$(ps -o user= -p "$p" 2>/dev/null)" = "$USER" ] && echo "$p"; done | wc -l)
  [ "$mine" -gt 0 ] && return 1
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F', ' '$2 < 1000 && $3 < 5 {print $1; exit}'
}
say "waiting for a free GPU"
GPU=""; while [ -z "$GPU" ]; do GPU=$(free_gpu) || GPU=""; [ -z "$GPU" ] && sleep 120; done
say "using GPU $GPU"
export CUDA_VISIBLE_DEVICES=$GPU TORCH_COMPILE_DISABLE=1

say "checking merged vs harness scorer (200 select rows)"
PYTHONPATH=$DMRSI_PYDEPS "$PY" pipelines/elliptic_elephant/05_check_merged.py "$RUN" "$MERGED" 200 >> "$LOG" 2>&1
CHECK=$("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['mean_abs_p_diff'])" "$MERGED/merge_check.json" 2>/dev/null || echo nan)
say "merge check mean |p diff| = $CHECK"
"$PY" -c "import sys; d=float(sys.argv[1]); sys.exit(0 if d == d and d < 0.005 else 1)" "$CHECK" \
  || { say "FAILED: merged model disagrees with the harness scorer (mean diff $CHECK); not scoring"; exit 1; }

say "scoring val and test"
T0=$(date +%s)
bash pipelines/elliptic_elephant/03_score_checkpoint.sh "$MERGED" "$NAME" cuda:0 >> "$LOG" 2>&1 || { say "FAILED: scoring"; exit 1; }
say "scored in $(( $(date +%s) - T0 ))s"
unset CUDA_VISIBLE_DEVICES

say "DONE"
