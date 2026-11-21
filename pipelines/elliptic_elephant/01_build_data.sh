#!/bin/bash
# Elliptic++ address level -> processed SQO jsonl -> OneJev txhist rows (noul), with Bingqiao's scripts, run read-only
# from the submodule; every output goes under $DMRSI_DATA. CPU only. Resumable: finished stages are skipped.
# Usage: source <dmrsi-env.sh> && bash pipelines/elliptic_elephant/01_build_data.sh
set -euo pipefail
: "${DMRSI_DATA:?source dmrsi-env.sh first}" "${CFD:?}" "${JEV_PYTHON:?}"
mkdir -p "$DMRSI_DATA" && cd "$DMRSI_DATA"

if [ ! -x "$JEV_PYTHON" ]; then
  echo "creating the data env at $(dirname "$(dirname "$JEV_PYTHON")")"
  python3 -m venv "$(dirname "$(dirname "$JEV_PYTHON")")"
  "$JEV_PYTHON" -m pip install -q -r "$CFD/requirements_jev.txt"
fi

if [ ! -f data/raw/elliptic_pp/wallets_features.csv ]; then
  "$(dirname "$JEV_PYTHON")/gdown" --folder https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l -O data/raw/elliptic_pp
fi
md5sum -c "$CFD/data/splits_ref/raw_elliptic_pp_md5.txt" --ignore-missing   # paths in the file are data/raw/elliptic_pp/...

[ -f data/processed/elliptic_pp_addr_sqo/test.jsonl ] || \
  "$JEV_PYTHON" "$CFD/scripts/build_addr_sqo.py" data/raw/elliptic_pp data/processed/elliptic_pp_addr_sqo
[ -f data/onejev/addr_txhist_noul/test.jsonl ] || \
  "$JEV_PYTHON" "$CFD/scripts/build_addr_txhist.py" --raw data/raw/elliptic_pp \
      --sqo data/processed/elliptic_pp_addr_sqo --out data/onejev/addr_txhist_noul

# val_sub4000 (uniform, seed 0) as export_onejev_data.py writes it; used for checkpoint selection during training
[ -f data/onejev/addr_txhist_noul/val_sub4000.jsonl ] || "$JEV_PYTHON" - <<'P'
import random
from pathlib import Path
d = Path("data/onejev/addr_txhist_noul")
lines = (d / "val.jsonl").read_text().splitlines()
pick = sorted(random.Random(0).sample(range(len(lines)), min(4000, len(lines))))
(d / "val_sub4000.jsonl").write_text("\n".join(lines[i] for i in pick) + "\n")
print(f"val_sub4000: {len(pick)} rows")
P
echo "next: verify the split ids against $CFD/data/splits_ref (DATA_PIPELINE.md section 5), then 02_train_baseline.sh"
wc -l data/onejev/addr_txhist_noul/*.jsonl
