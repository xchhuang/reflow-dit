#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 01:00:00
#SBATCH -o slurm_outputs/slurm.out
#SBATCH --gres gpu:1

mkdir -p slurm_outputs


# dit
srun -p gpu20 --gres gpu:1 python sample.py --model DiT-XL/2 --image-size 256 --ckpt ./results/001-DiT-XL-2/checkpoints/0040000.pt --num-sampling-steps=5


# rf
# srun -p gpu20 --gres gpu:1 python sample_rf.py --model DiT-XL/2 --image-size 256 --ckpt ./results/002-DiT-XL-2/checkpoints/model_v1.pt --num-sampling-steps=5 --cfg-scale=4.0
# srun -p gpu20 --gres gpu:1 python sample_rf.py --model DiT-XL/2 --image-size 256 --ckpt ./results/002-DiT-XL-2/checkpoints/model_v1.pt --num-sampling-steps=10 --cfg-scale=4.0
# srun -p gpu20 --gres gpu:1 python sample_rf.py --model DiT-XL/2 --image-size 256 --ckpt ./results/002-DiT-XL-2/checkpoints/model_v1.pt --num-sampling-steps=50 --cfg-scale=4.0

# srun -p gpu20 --gres gpu:1 python sample_rf.py --model DiT-XL/2 --image-size 256 --ckpt ./results/002-DiT-XL-2/checkpoints/model_v2.pt --num-sampling-steps=5 --cfg-scale=4.0
# srun -p gpu20 --gres gpu:1 python sample_rf.py --model DiT-XL/2 --image-size 256 --ckpt ./results/002-DiT-XL-2/checkpoints/model_v2.pt --num-sampling-steps=10 --cfg-scale=4.0
# srun -p gpu20 --gres gpu:1 python sample_rf.py --model DiT-XL/2 --image-size 256 --ckpt ./results/002-DiT-XL-2/checkpoints/model_v2.pt --num-sampling-steps=50 --cfg-scale=4.0
