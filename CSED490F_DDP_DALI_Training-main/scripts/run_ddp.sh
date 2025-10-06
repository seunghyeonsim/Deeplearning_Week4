#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

export DATA_DIR="/DATA/cifar10" 
export MODE="ddp"
CKPT_ROOT="runs"
NSIGHT_LOG_ROOT="nsight_logs"
export NSIGHT_LOG_DIR="$NSIGHT_LOG_ROOT/ddp_${TIMESTAMP}"
mkdir -p "$NSIGHT_LOG_DIR"

# run with one gpu
export LOCAL_GPU_IDS="0"
export NUM_GPUS=1
export CKPT_DIR="$CKPT_ROOT/ddp_${TIMESTAMP}/gpu_1"
export NSIGHT_FILE_NAME="gpu_1"

mkdir -p $CKPT_DIR

bash scripts/launch.sh

# run with two gpu
export LOCAL_GPU_IDS="0,1"
export NUM_GPUS=2
export CKPT_DIR="$CKPT_ROOT/ddp_${TIMESTAMP}/gpu_2"
export NSIGHT_FILE_NAME="gpu_2"

mkdir -p $CKPT_DIR

bash scripts/launch.sh