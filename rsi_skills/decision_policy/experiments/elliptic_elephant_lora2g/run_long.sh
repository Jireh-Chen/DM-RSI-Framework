#!/bin/bash
# Long validation: every version x 3 seeds x {10, 20} rounds, each with its own version's code (CPU only).
#   tmux new -d -s dmrsi-long "bash run_long.sh"          # configs/long/v<N>_it<K>_r<R>.yaml
# At most $JOBS runs at a time (4 threads each), so a shared box is not swamped. Same proposals as the 3-round runs
# (proposals_v<N>/, used in round 1); later rounds are the version's automatic search. Writes <out>/evolve_seconds.txt.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SKILL=$(cd "$HERE/../.." && pwd)
PY=${DPOLICY_PY:-/shared/ssd/yile/dmrsi/dpolicy-venv/bin/python}
JOBS=${JOBS:-6}
cd "$HERE"   # not the skill dir: python -m puts the cwd first on sys.path
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
one() {
  local cfg=$1 v out code
  v=$(basename "$cfg" | cut -d_ -f1)
  if [ -d "$SKILL/history/$v/dpolicy" ]; then code=$SKILL/history/$v; else code=$SKILL; fi
  out=$(PYTHONPATH=$code "$PY" -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['out_dir'])" "$cfg")
  [ -f "$out/evolve_seconds.txt" ] && return 0   # finished before: resumable
  rm -rf "$out"; mkdir -p "$out/proposals"
  PYTHONPATH=$code "$PY" -m dpolicy init "$cfg" > /dev/null || return 1
  cp "$HERE"/proposals_$v/*.yaml "$out/proposals/"
  PYTHONPATH=$code /usr/bin/time -f "%e" -o "$out/evolve_seconds.tmp" nice "$PY" -m dpolicy evolve "$cfg" > "$out/evolve.log" 2>&1 \
    && mv "$out/evolve_seconds.tmp" "$out/evolve_seconds.txt"
  echo "$(date '+%T') $(basename "$cfg" .yaml): $(cat "$out/evolve_seconds.txt" 2>/dev/null || echo FAILED)s, accepted $(grep -c 'new incumbent' "$out/evolve.log")"
}
export -f one; export HERE SKILL PY
ls "$HERE"/configs/long/*.yaml | xargs -P "$JOBS" -I{} bash -c 'one {}'
echo "$(date '+%T') ALL DONE"
