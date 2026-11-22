#!/bin/bash
# Score the checkpoint the harness selected on full val and test with Bingqiao's onejev_score.py (raw slot logits,
# same parquet format as the baselines) -> $DMRSI_SCORES/<name>_{val,test}.parquet, the input of dpolicy.
# Uses the harness venv + its vendored OneJev; pandas comes from $DMRSI_PYDEPS (installed once, harness venv untouched).
# Usage: bash pipelines/elliptic_elephant/03_score_checkpoint.sh <RUN dir | model dir> <name> [cuda:0]
#   RUN dir: the checkpoint comparison.json selected (full fine-tune). Model dir: a full model, e.g. the merged
#   LoRA from 02b_merge_lora.py. A LoRA adapter dir is refused: onejev_score.py loads the text-only LM and
#   would silently skip the adapter's multimodal keys, so merge it first.
# Env knobs: SCORE_SPLITS="val test" (default), SCORE_LIMIT=N (first N rows, smoke tests only; use a
# separate DMRSI_SCORES then). torch.compile is off (TORCH_COMPILE_DISABLE=1): its CPU compile phases leave the
# GPU idle long enough for the server's idle-GPU watchdog to kill the process.
set -euo pipefail
: "${OJH_REPO:?source dmrsi-env.sh first}" "${CFD:?}" "${DMRSI_DATA:?}" "${DMRSI_SCORES:?}" "${DMRSI_PYDEPS:?}"
RUN=${1:?RUN dir (contains comparison.json)} NAME=${2:?name, e.g. EL_full_txhist_all_s0} DEV=${3:-cuda:0}
PY=$OJH_REPO/.venv/bin/python
ROWS=$DMRSI_DATA/data/onejev/addr_txhist_noul

if ! PYTHONPATH=$DMRSI_PYDEPS "$PY" -c "import pandas, pyarrow" 2>/dev/null; then
  echo "installing pandas + pyarrow into $DMRSI_PYDEPS (not into the harness venv)"
  "$PY" -m pip --version >/dev/null 2>&1 && PIP=("$PY" -m pip install) || PIP=(uv pip install --python "$PY")
  "${PIP[@]}" --target "$DMRSI_PYDEPS" --no-deps pandas==3.1.1 pyarrow==26.0.0 python-dateutil six tzdata
fi

if [ -f "$RUN/config.json" ]; then CKPT=$(cd "$RUN" && pwd); else
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
fi
echo "model: $CKPT"
if [ -f "$CKPT/adapter_config.json" ]; then
  echo "$CKPT is a LoRA adapter: merge it first (02b_merge_lora.py), then pass the merged dir" >&2; exit 2
fi
export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
mkdir -p "$DMRSI_SCORES"
for SPLIT in ${SCORE_SPLITS:-val test}; do
  OUTF=$DMRSI_SCORES/${NAME}_${SPLIT}.parquet
  [ -e "$OUTF" ] && { echo "exists, not overwriting: $OUTF"; continue; }
  PYTHONPATH=$DMRSI_PYDEPS "$PY" "$CFD/elephant/scripts/onejev_score.py" --onejev "$ONEJEV_ROOT" --model "$CKPT" \
      --rows "$ROWS/$SPLIT.jsonl" --name "$NAME" --split "$SPLIT" --out "$DMRSI_SCORES" --device "$DEV" \
      ${SCORE_LIMIT:+--limit "$SCORE_LIMIT"}
done
echo "$CKPT" > "$DMRSI_SCORES/${NAME}.checkpoint"
ls -la "$DMRSI_SCORES"/${NAME}_*.parquet
