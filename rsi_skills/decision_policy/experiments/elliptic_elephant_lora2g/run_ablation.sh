#!/bin/bash
# Val-only ablation of the 1.5 lead's parts (dpolicy eval, current code, no test read). 2 protocols x 3 seeds.
#   bash run_ablation.sh   ->  prints one line per (protocol, seed, policy): F1 and gain vs baseline [95% CI]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
PY=${DPOLICY_PY:-/shared/ssd/yile/dmrsi/dpolicy-venv/bin/python}
export PYTHONPATH=$(cd ../.. && pwd) OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
run() {
  local cfg=$1 out
  out=$("$PY" -c "import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))['out_dir'])" "$cfg")
  rm -rf "$out"; "$PY" -m dpolicy init "$cfg" > /dev/null
  for p in ablation_1.5_lead/*.yaml; do
    "$PY" -m dpolicy eval "$cfg" "$p" | "$PY" -c "
import json,sys; d=json.loads(sys.stdin.read().split('\ndiagnosis:')[0]); v=d.get('vs_incumbent') or {}
print('$(basename "$cfg" .yaml)', '$(basename "$p" .yaml)', 'F1 %.4f gain %+.4f [%+.4f, %+.4f]' % (float(d['objective']), float(v['diff']), float(v['lo']), float(v['hi'])))"
  done
}
export -f run; export PY
ls configs/ablation_*.yaml | xargs -P 6 -I{} bash -c 'run {}' | sort
