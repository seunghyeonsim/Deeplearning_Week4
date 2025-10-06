#!/bin/bash

mkdir -m 777 -p runs
mkdir -m 777 -p slurm_logs
mkdir -m 777 -p nsight_logs

pip install -r requirements.txt

########################################################
# We skip downloading dataset, because our slurm cluster has limited network speed.
# Instead, we downolad dataset to /home/dataset direcotory and mount it to /DATA direcotory.
# If you want to download dataset in your local device, you can uncomment the following lines.
########################################################

# DATASET_DIR="/DATA" # /DATA is mounted in Slurm's /home/dataset directory

# python my_lib/init_dataset.py --seed 42 --dataset_dir $DATASET_DIR

# find $DATASET_DIR/cifar10 -exec chmod 777 {} \;
# find $DATASET_DIR/cifar10_images -exec chmod 777 {} \;