#!/bin/bash
# Score the checkpoint the harness selected on full val and test with Bingqiao's onejev_score.py (raw slot logits,
# same parquet format as the baselines) -> $DMRSI_SCORES/<name>_{val,test}.parquet, the input of dpolicy.
# Uses the harness venv + its vendored OneJev; pandas comes from $DMRSI_PYDEPS (installed once, harness venv untouched).
# Usage: bash pipelines/elliptic_elephant/03_score_checkpoint.sh <RUN dir> <name> [cuda:0]
set -euo pipefail
: "${OJH_REPO:?source dmrsi-env.sh first}" "${CFD:?}" "${DMRSI_DATA:?}" "${DMRSI_SCORES:?}" "${DMRSI_PYDEPS:?}"
RUN=${1:?RUN dir (contains comparison.json)} NAME=${2:?name, e.g. EL_full_txhist_all_s0} DEV=${3:-cuda:0}
PY=$OJH_REPO/.venv/bin/python
ROWS=$DMRSI_DATA/data/onejev/addr_txhist_noul

if ! PYTHONPATH=$DMRSI_PYDEPS "$PY" -c "import pandas" 2>/dev/null; then
  echo "installing pandas into $DMRSI_PYDEPS (not into the harness venv)"
  "$PY" -m pip --version >/dev/null 2>&1 && PIP=("$PY" -m pip install) || PIP=(uv pip install --python "$PY")
  "${PIP[@]}" --target "$DMRSI_PYDEPS" --no-deps pandas==3.1.1 python-dateutil six tzdata
fi

CKPT=$("$PY" - "$RUN" <<'P'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
comp = json.loads((run / "comparison.json").read_text())
ck = [v.get("checkpoint") for v in comp.values() if isinstance(v, dict) and v.get("checkpoint")]
if len(ck) != 1:
    sys.exit(f"expected one selected checkpoint in comparison.json, got {ck}")
print((run / ck[0]).resolve())
P
)
echo "selected checkpoint: $CKPT"
mkdir -p "$DMRSI_SCORES"
for SPLIT in val test; do
  OUTF=$DMRSI_SCORES/${NAME}_${SPLIT}.parquet
  [ -e "$OUTF" ] && { echo "exists, not overwriting: $OUTF"; continue; }
  PYTHONPATH=$DMRSI_PYDEPS "$PY" "$CFD/elephant/scripts/onejev_score.py" --onejev "$ONEJEV_ROOT" --model "$CKPT" \
      --rows "$ROWS/$SPLIT.jsonl" --name "$NAME" --split "$SPLIT" --out "$DMRSI_SCORES" --device "$DEV"
done
echo "$CKPT" > "$DMRSI_SCORES/${NAME}.checkpoint"
ls -la "$DMRSI_SCORES"/${NAME}_*.parquet
