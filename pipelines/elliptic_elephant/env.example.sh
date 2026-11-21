# Copy to a big disk (e.g. $DMRSI_HOME/dmrsi-env.sh), edit the CHANGE lines, then `source <full path>` in every shell.
# Nothing here is written into this repo or into external/ (the submodules stay clean).

export DMRSI_HOME=${DMRSI_HOME:-/CHANGE/ME/big/disk/dmrsi}          # CHANGE: a directory on a big disk, not /home
export DMRSI_REPO=${DMRSI_REPO:-/CHANGE/ME/DM-RSI-Framework}        # CHANGE: where this repo is cloned

# onejev-harness environment (its own env file; see external/onejev-harness/README.md "环境配置").
export OJH_HOME=${OJH_HOME:-$DMRSI_HOME/ojh}
[ -f "$OJH_HOME/ojh-env.sh" ] && source "$OJH_HOME/ojh-env.sh"
export OJH_REPO=$DMRSI_REPO/external/onejev-harness
export OJH=$OJH_REPO/.venv/bin/ojh

# Elliptic++ data (raw csv, processed jsonl, OneJev rows): outside the repo.
export DMRSI_DATA=${DMRSI_DATA:-$DMRSI_HOME/elliptic}
export CFD=$DMRSI_REPO/external/crypto-fraud-detection/crypto-fraud-detection   # Bingqiao's scripts (read-only)
export JEV_PYTHON=${JEV_PYTHON:-$DMRSI_HOME/jev-venv/bin/python}             # CPU env for data building (requirements_jev.txt)

# Scoring: harness venv (torch + OneJev) plus pandas installed into a side directory (the harness venv is not modified).
export DMRSI_PYDEPS=$DMRSI_HOME/pydeps
export ONEJEV_ROOT=$OJH_REPO/vendor/OneJev
export DMRSI_SCORES=${DMRSI_SCORES:-$DMRSI_HOME/scores}                     # <name>_{val,test}.parquet for dpolicy
