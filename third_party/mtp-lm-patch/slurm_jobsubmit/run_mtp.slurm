#!/bin/bash
#SBATCH --job-name=mtp_llama8b
#SBATCH --partition=medium
#SBATCH --nodes=1

# ---- GPU configuration ----
# Must pin a GPU type: the installed torch==2.8.0+cu128 wheel's compiled kernel
# arch list is [sm_70,75,80,86,90,100,120] - P100 (sm_60) has NO kernel image in
# this build at all (verified: even torch.nn.init.ones_ fails with "no kernel
# image is available for execution on the device"). An untyped "gpu:4" request
# can schedule onto a p100 node and crash on the very first op.
# Verified working on this build: v100, rtx8000, a100, h100, l40s (a6000 also
# kernel-compatible but htc-g[050-052] is reserved for account engs-a2i, and
# the rtx nodes htc-g[041-043] are under the same reservation - avoid both
# regardless of compatibility). Note p100/v100 nodes aren't even in the
# "medium" partition (only "short"/"everything"), so l40s/a100/h100/rtx8000
# are the relevant safe choices here; l40s has the most free capacity cluster-
# wide as of 2026-08-12, so it's used below.
#SBATCH --gres=gpu:l40s:4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1

# ---- CPU / Memory ----
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

# ---- Time ----
#SBATCH --time=24:00:00

# ---- Logs ----
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "==============================="
echo " SLURM INFORMATION"
echo "==============================="
echo "JOB_ID           = $SLURM_JOB_ID"
echo "NODELIST         = $SLURM_JOB_NODELIST"
echo "NTASKS           = $SLURM_NTASKS"
echo "GPUS_ON_NODE     = $SLURM_GPUS_ON_NODE"
echo

####################################################
# Environment
####################################################

source /data/phys-snoplus-snews/exet5937/aiproj/driftmtplm/miniforge3/etc/profile.d/conda.sh

conda activate /data/phys-snoplus-snews/exet5937/aiproj/driftmtplm/.venv

####################################################
# Distributed variables
####################################################

export WORLD_SIZE=${SLURM_NTASKS}
export RANK=${SLURM_PROCID}
export LOCAL_RANK=${SLURM_LOCALID}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo "==============================="
echo " DISTRIBUTED"
echo "==============================="
echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$RANK"
echo "LOCAL_RANK=$LOCAL_RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo

echo "==============================="
echo " CUDA"
echo "==============================="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

nvidia-smi -L

####################################################
# Move to project
####################################################

cd /home/exet5937/aiproj/driftmtplm/third_party/mtp-lm

####################################################
# Launch training
####################################################

srun python -u litgpt/pretrain.py \
    --config=../mtp-lm-patch/config_hub/pretrain/reproduce_full.yaml \
    --model_name=$MODEL_NAME \
    --tokenizer_dir=$INITIAL_CHECKPOINT \
    --initial_checkpoint_dir=$INITIAL_CHECKPOINT