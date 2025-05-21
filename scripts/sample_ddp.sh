#!/bin/bash
#SBATCH -p gpu20
#SBATCH -t 00:15:00
#SBATCH -o slurm_outputs/sample.out
#SBATCH --gres gpu:4
#SBATCH -a 1-999


mkdir -p slurm_outputs



# cluster

# srun -p gpu20 --gres gpu:4

# torchrun --nnodes=1 --nproc_per_node=4 sample_ddp.py --model DiT-XL/2 --image-size 256 --ckpt ./results/001-DiT-XL-2/checkpoints/0040000.pt --num-sampling-steps=50 --per-proc-batch-size 65 --num-fid-samples 520 --ycls=${SLURM_ARRAY_TASK_ID}



# srun -p gpu20 --gres gpu:1 torchrun --nnodes=1 --nproc_per_node=1 sample_ddp_rf.py --model DiT-XL/2 --image-size 256 --ckpt ./results/002-DiT-XL-2/checkpoints/model.pt --num-sampling-steps=50 --per-proc-batch-size 4 --num-fid-samples 4 --ycls=207


torchrun --nnodes=1 --nproc_per_node=4 sample_ddp_rf.py --model DiT-XL/2 --image-size 256 --ckpt ./results/002-DiT-XL-2/checkpoints/model.pt --num-sampling-steps=50 --per-proc-batch-size 65 --num-fid-samples 520 --ycls=${SLURM_ARRAY_TASK_ID}


# sbatch scripts/sample_ddp.sh
