#!/bin/bash
export RUN_NAME=l3_magpie_metamath
export RUN_OUTPUT_DIR=/data/phys-snoplus-snews/exet5937/aiproj/driftmtplm/dataset/train/outputs/$RUN_NAME
export MODEL_NAME=Llama-3.1-8B-Magpie-Align-SFT-v0.1-MTPV128384
export INITIAL_CHECKPOINT=/data/phys-snoplus-snews/exet5937/aiproj/driftmtplm/checkpoints/Magpie-Align/$MODEL_NAME

# download and tokenize the training split
torchrun --standalone --nproc_per_node=1 litgpt/pull_raw_datasets.py \
--srcs_config_file=config_hub/data/metamath_strat_split_chat_train.yaml \
--script_configs=config_hub/data/p2p_generic_metamath.yaml \
--script_configs.num_raw_shards=32 \
--script_configs.target_shard_num=32 \
--script_configs.hf_tokenizer=$INITIAL_CHECKPOINT \
--script_configs.raw_dir=$RUN_OUTPUT_DIR/train/raw \
--script_configs.output_dir=$RUN_OUTPUT_DIR/train \
--script_configs.rm_cache=false || exit 1


torchrun --standalone --nproc_per_node=1 litgpt/p2p_tokenizer.py \
--srcs_config_file=config_hub/data/metamath_strat_split_chat_train.yaml \
--script_configs=config_hub/data/p2p_generic_metamath.yaml \
--script_configs.num_raw_shards=32 \
--script_configs.target_shard_num=32 \
--script_configs.hf_tokenizer=$INITIAL_CHECKPOINT \
--script_configs.raw_dir=$RUN_OUTPUT_DIR/train/raw \
--script_configs.output_dir=$RUN_OUTPUT_DIR/train \
--script_configs.train_split_pct=1.0 \
--script_configs.rm_cache=false || exit 1

# download and tokenize the validation split
torchrun --standalone --nproc_per_node=1 litgpt/pull_raw_datasets.py \
--srcs_config_file=config_hub/data/metamath_strat_split_chat_val.yaml \
--script_configs=config_hub/data/p2p_generic_metamath.yaml \
--script_configs.num_raw_shards=32 \
--script_configs.target_shard_num=32 \
--script_configs.hf_tokenizer=$INITIAL_CHECKPOINT \
--script_configs.raw_dir=$RUN_OUTPUT_DIR/val/raw \
--script_configs.output_dir=$RUN_OUTPUT_DIR/val \
--script_configs.rm_cache=false || exit 1

torchrun --standalone --nproc_per_node=1 litgpt/p2p_tokenizer.py \
--srcs_config_file=config_hub/data/metamath_strat_split_chat_val.yaml \
--script_configs=config_hub/data/p2p_generic_metamath.yaml \
--script_configs.num_raw_shards=32 \
--script_configs.target_shard_num=32 \
--script_configs.hf_tokenizer=$INITIAL_CHECKPOINT \
--script_configs.raw_dir=$RUN_OUTPUT_DIR/val/raw \
--script_configs.output_dir=$RUN_OUTPUT_DIR/val \
--script_configs.train_split_pct=1.0 \
--script_configs.rm_cache=false || exit 1


# consolidate the pair of datasets under one folder
mkdir -p $RUN_OUTPUT_DIR/processed
mv $RUN_OUTPUT_DIR/train/processed/train $RUN_OUTPUT_DIR/processed/train || exit 1
mv $RUN_OUTPUT_DIR/val/processed/train $RUN_OUTPUT_DIR/processed/val || exit 1