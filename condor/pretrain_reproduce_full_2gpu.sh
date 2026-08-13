#!/usr/bin/env bash
# Executable launched by HTCondor for the reproduce_full pretraining run (2 GPUs).
set -euo pipefail

source /data/snoplus/weiiiiiii/aiproj/driftmtplm/miniforge/etc/profile.d/conda.sh
conda activate /data/snoplus/weiiiiiii/aiproj/driftmtplm/.venv

# cd so that config_hub/... relative paths inside reproduce_full.yaml resolve
# regardless of HTCondor's initialdir (which defaults to the submit-time cwd).
cd /home/huangp/aiproj/driftmtplm/third_party/mtp-lm

exec python litgpt/pretrain.py \
    --config=/home/huangp/aiproj/driftmtplm/third_party/mtp-lm-patch/config_hub/pretrain/reproduce_full_2gpu.yaml
