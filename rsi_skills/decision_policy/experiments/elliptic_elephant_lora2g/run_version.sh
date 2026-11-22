#!/bin/bash
# Run one Skill version's replicates in parallel (CPU only), each with that version's own code:
#   bash run_version.sh v3            # replicates 0 1 2
#   bash run_version.sh v4 1          # replicate 1 only
# v1-v3 run from ../../history/v<N>/dpolicy, v4 (current) from ../../dpolicy: PYTHONPATH picks the package, the
# installed dpolicy is not used. For each replicate: dpolicy init (if new) -> copy proposals_<v>/ -> dpolicy evolve,
# timed into <out>/evolve_seconds.txt.
# 4 BLAS/OpenMP threads each: on a loaded shared box, sklearn's HistGradientBoosting with all cores oversubscribes
# and one evolve went from ~20 s to >15 min (2026-11-22).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SKILL=$(cd "$HERE/../.." && pwd)
V=${1:?version, e.g. v4}; shift
REPS=${*:-0 1 2}
PY=${DPOLICY_PY:-/shared/ssd/yile/dmrsi/dpolicy-venv/bin/python}
if [ -d "$SKILL/history/$V/dpolicy" ]; then CODE=$SKILL/history/$V; else CODE=$SKILL; fi
cd "$HERE"   # not the skill dir: python -m puts the cwd first on sys.path, and ../../dpolicy would shadow history/
export PYTHONPATH=$CODE OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
echo "$V: dpolicy $("$PY" -c 'import dpolicy; print(dpolicy.__version__.split()[0], dpolicy.__file__)')"
for r in $REPS; do
  (
    CFG=$HERE/configs/${V}_r$r.yaml
    OUT=$("$PY" -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['out_dir'])" "$CFG")
    [ -f "$OUT/incumbent.yaml" ] || "$PY" -m dpolicy init "$CFG" > /dev/null
    mkdir -p "$OUT/proposals"
    [ -d "$HERE/proposals_$V" ] && cp "$HERE"/proposals_$V/*.yaml "$OUT/proposals/" 2>/dev/null
    /usr/bin/time -f "%e" -o "$OUT/evolve_seconds.txt" nice "$PY" -m dpolicy evolve "$CFG" > "$OUT/evolve.log" 2>&1
    echo "$V r$r: $(cat "$OUT/evolve_seconds.txt")s, accepted: $(grep -c 'new incumbent' "$OUT/evolve.log")"
  ) &
done
wait
