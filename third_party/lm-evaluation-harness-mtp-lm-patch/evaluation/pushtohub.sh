#!/bin/bash
set -euo pipefail

#export RUN_NAME=l3_magpie_metamath
export RUN_OUTPUT_DIR=/data/snoplus/weiiiiiii/aiproj/driftmtplm/outputs/Llama-3.2-1B-Instruct-drift
export CKPT_SUBDIR=latest

# convert_lit_checkpoint.py resolves model_class_path/config_class_path as a
# *relative filesystem path* (not via PYTHONPATH), so we must run from the
# directory that directly contains the `litgpt/` package folder.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}/mtp-lm"


litgpt push_to_hub \
--model_path=$RUN_OUTPUT_DIR/$CKPT_SUBDIR \
--model_class_path=litgpt.transformers_local.llama.modeling_llama.LlamaForCausalLM \
--org=weiiiiiiiiiiiiiiiii \
--private=False \
--model_name=Llama-3.2-1B-Instruct-drift \
--precision=bfloat16 \
--dry_run=False \
--update_existing=True \
--readme_path=litgpt/transformers_local/llama/README.md \
--readme_only=False