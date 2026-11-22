#!/bin/bash
# A small GPU job queue for a shared server, run in tmux:  tmux new -d -s dmrsi-queue "bash -l pipelines/gpu_queue.sh"
#
# Drop a job into $QUEUE/pending/<order>_<name>.sh (a bash script; first line may be "# gpus: N", default 1). Every
# $POLL seconds the queue starts the oldest pending job when, twice in a row $CONFIRM seconds apart, enough cards are
# truly free: no process of anyone on them, < $FREE_MIB MiB used, < $FREE_UTIL % utilization. It never lets this user
# hold more than $MAX_MINE cards in total (the lab's 2-GPU rule; killgpu force-kills users above it), never shares a
# card with another user, and does nothing to keep cards busy (the idle watchdog stays in charge).
# A job runs in its own tmux session with CUDA_VISIBLE_DEVICES set and moves to done/ or failed/ when it exits.
set -uo pipefail
source /shared/ssd/yile/dmrsi/dmrsi-env.sh
umask 077
QUEUE=${QUEUE:-$DMRSI_HOME/queue} POLL=${POLL:-120} CONFIRM=${CONFIRM:-30}
FREE_MIB=${FREE_MIB:-1000} FREE_UTIL=${FREE_UTIL:-5} MAX_MINE=${MAX_MINE:-2}
mkdir -p "$QUEUE"/{pending,running,done,failed,logs}
LOG=$QUEUE/queue.log
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

busy_uuids() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u; }
mine() {  # cards with a process of this user
  nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | while IFS=', ' read -r p g; do
    [ "$(ps -o user= -p "$p" 2>/dev/null)" = "$USER" ] && echo "$g"; done | sort -u | wc -l
}
free_cards() {  # indices of cards nobody is on
  local busy; busy=$(busy_uuids)
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits |
    while IFS=', ' read -r i u m g; do
      grep -qx "$u" <<<"$busy" && continue
      [ "$m" -lt "$FREE_MIB" ] && [ "$g" -lt "$FREE_UTIL" ] && echo "$i"
    done
}

say "queue started (poll ${POLL}s, max ${MAX_MINE} cards for $USER)"
while true; do
  # reap finished jobs
  for f in "$QUEUE"/running/*.sh; do
    [ -e "$f" ] || continue
    n=$(basename "$f" .sh)
    tmux has-session -t "q-$n" 2>/dev/null && continue
    rc=$(cat "$QUEUE/logs/$n.rc" 2>/dev/null || echo 1)
    if [ "$rc" = 0 ]; then mv "$f" "$QUEUE/done/"; else mv "$f" "$QUEUE/failed/"; fi
    say "finished $n rc=$rc"
  done
  job=$(ls "$QUEUE"/pending/*.sh 2>/dev/null | sort | head -1)
  if [ -n "$job" ]; then
    n=$(basename "$job" .sh)
    need=$(sed -n '1s/^# gpus: *\([0-9]*\).*/\1/p' "$job"); need=${need:-1}
    # cards held = processes seen now, or what queued jobs asked for (a job may still be loading): the larger
    have=$(mine) held=0
    for r in "$QUEUE"/running/*.sh; do [ -e "$r" ] || continue
      k=$(sed -n '1s/^# gpus: *\([0-9]*\).*/\1/p' "$r"); held=$((held + ${k:-1})); done
    [ "$held" -gt "$have" ] && have=$held
    if [ $((have + need)) -le "$MAX_MINE" ]; then
      a=$(free_cards); sleep "$CONFIRM"; b=$(free_cards)
      cards=$(comm -12 <(sort <<<"$a") <(sort <<<"$b") | head -n "$need" | paste -sd, -)
      if [ -n "$cards" ] && [ "$(tr ',' '\n' <<<"$cards" | wc -l)" -eq "$need" ]; then
        mv "$job" "$QUEUE/running/"
        tmux new-session -d -s "q-$n" "CUDA_VISIBLE_DEVICES=$cards TORCH_COMPILE_DISABLE=\${TORCH_COMPILE_DISABLE:-1} \
          bash -l '$QUEUE/running/$n.sh' > '$QUEUE/logs/$n.log' 2>&1; echo \$? > '$QUEUE/logs/$n.rc'"
        say "started $n on GPU $cards (this user now on $((have + need)) cards)"
      fi
    fi
  fi
  sleep "$POLL"
done
