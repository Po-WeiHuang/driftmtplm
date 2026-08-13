#!/usr/bin/env bash
# Quick GPU smoke test for the reproduce_full pretraining config:
# runs 2 optimizer steps, no compile, no wandb, into a throwaway out_dir.
set -euo pipefail

source /data/snoplus/weiiiiiii/aiproj/driftmtplm/miniforge/etc/profile.d/conda.sh
conda activate /data/snoplus/weiiiiiii/aiproj/driftmtplm/.venv

export WANDB_MODE=offline

exec python /home/huangp/aiproj/driftmtplm/third_party/mtp-lm/litgpt/pretrain.py \
    --config=/home/huangp/aiproj/driftmtplm/third_party/mtp-lm-patch/config_hub/pretrain/reproduce_full.yaml \
    --out_dir=/data/snoplus/weiiiiiii/aiproj/driftmtplm/outputs/smoke_test \
    --resume=false \
    --train.max_tokens=50000 \
    --train.initial_save=false \
    --train.save_latest_ckpt=false \
    --train.do_compile=false \
    --eval.initial_validation=false \
    --eval.final_validation=false
