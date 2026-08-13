#!/usr/bin/env bash
# Executable launched by HTCondor for the reproduce_full pretraining run.
set -euo pipefail

source /data/snoplus/weiiiiiii/aiproj/driftmtplm/miniforge/etc/profile.d/conda.sh
conda activate /data/snoplus/weiiiiiii/aiproj/driftmtplm/.venv

exec python /home/huangp/aiproj/driftmtplm/third_party/mtp-lm/litgpt/pretrain.py \
    --config=/home/huangp/aiproj/driftmtplm/third_party/mtp-lm-patch/config_hub/pretrain/reproduce_full.yaml
