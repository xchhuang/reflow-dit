#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 24:00:00
#SBATCH -o slurm_outputs/slurm.out
#SBATCH --gres gpu:4
#SBATCH -a 1-5%1

mkdir -p slurm_outputs


# TRITON_CACHE_DIR='../../triton_cache'

# extract
# torchrun --nnodes=1 --nproc_per_node=1 extract_features.py --model DiT-XL/2 --data-path ../data/imagenet256 --features-path ../data/imagenet256_features


# dit
# accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train.py --model DiT-XL/2 --feature-path ../data/imagenet256_features --epochs=20


# rf
accelerate launch --multi_gpu --num_processes 4 --mixed_precision fp16 train_rf.py --model DiT-XL/2 --epochs=20


# srun -p gpu20 --gres gpu:1 accelerate launch --num_processes 1 --mixed_precision fp16 train_rf.py --model DiT-XL/2 --epochs=1


