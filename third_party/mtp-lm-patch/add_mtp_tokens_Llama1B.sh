export MODEL_NAME=Llama-3.2-1B-Instruct-MTPV128384
export OUT=/data/phys-snoplus-snews/exet5937/aiproj/driftmtplm/checkpoints/extended/meta-llama/$MODEL_NAME
cd ~/aiproj/driftmtplm/third_party/mtp-lm

python -u litgpt/scripts/add_mtp_tokens.py \
--model_path=/data/phys-snoplus-snews/exet5937/aiproj/driftmtplm/checkpoints/meta-llama/Llama-3.2-1B \
--output_dir=$OUT \
--model_name=$MODEL_NAME