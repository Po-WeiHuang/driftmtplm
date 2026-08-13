#!/bin/bash

# -------------------------
# Distributed Environment
# -------------------------

export WORLD_SIZE=${SLURM_NTASKS}
export RANK=${SLURM_PROCID}
export LOCAL_RANK=${SLURM_LOCALID}

# Master node
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Choose any free port
export MASTER_PORT=29500

echo "==========================="
echo "Distributed Environment"
echo "==========================="
echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$RANK"
echo "LOCAL_RANK=$LOCAL_RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo