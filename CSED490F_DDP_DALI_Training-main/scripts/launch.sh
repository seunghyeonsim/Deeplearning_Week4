#!/bin/bash

export DATA_DIR
export CKPT_DIR
export NSIGHT_LOG_DIR
export NSIGHT_FILE_NAME
export MODE
export LOCAL_GPU_IDS
export NUM_GPUS

# ----- Problem 0: Nsight 로그 설정 -----
# 기본값: week4_nsys/ 디렉토리와 타임스탬프 파일명
NSIGHT_LOG_DIR=${NSIGHT_LOG_DIR:-week4_nsys}
NSIGHT_FILE_NAME=${NSIGHT_FILE_NAME:-$(date +%Y%m%d_%H%M%S)}

mkdir -p "${NSIGHT_LOG_DIR}"

# Nsight Systems로 프로파일링 (자식 프로세스 및 NVTX 포함)
NSYS_CMD="nsys profile \
  --output=${NSIGHT_LOG_DIR}/${NSIGHT_FILE_NAME} \
  --trace=cuda,nvtx,osrt \
  --capture-child-processes=true \
  --stats=true \
  --force-overwrite=true"

# ----- 학습 실행 -----
CUDA_VISIBLE_DEVICES=$LOCAL_GPU_IDS $NSYS_CMD python train_cifar.py \
  --num_gpu="$NUM_GPUS" \
  --data="$DATA_DIR" \
  --ckpt="$CKPT_DIR" \
  --mode="$MODE" \
  --save_ckpt
