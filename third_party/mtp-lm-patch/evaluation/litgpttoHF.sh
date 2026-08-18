#!/bin/bash
set -euo pipefail

export RUN_NAME=l3_magpie_metamath
export RUN_OUTPUT_DIR=/data/snoplus/weiiiiiii/aiproj/driftmtplm/outputs/reproduce
export CKPT_SUBDIR=latest

# convert_lit_checkpoint.py resolves model_class_path/config_class_path as a
# *relative filesystem path* (not via PYTHONPATH), so we must run from the
# directory that directly contains the `litgpt/` package folder.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}/mtp-lm"

litgpt convert_from_litgpt \
    --checkpoint_dir=$RUN_OUTPUT_DIR/$CKPT_SUBDIR \
    --output_dir=$RUN_OUTPUT_DIR/$CKPT_SUBDIR \
    --output_name=pytorch_model.bin \
    --skip_if_exists=True \
    --config_class_path=litgpt.transformers_local.llama.configuration_llama.LlamaConfig \
    --model_class_path=litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM