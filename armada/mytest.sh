#!/bin/bash
{
    source "/data/yuwenye/miniconda3/etc/profile.d/conda.sh"; conda activate human_dp
    export WANDB_API_KEY=956bd553be1193c6fdfce758ad1eded5190eac56
    export HYDRA_FULL_ERROR=1
    export CUDA_VISIBLE_DEVICES=0

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py train_diffusion_transformer_real_hybrid_dino_multi_gpu_workspace
    # OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py train_diffusion_transformer_real_hybrid_dino_weighted_multi_gpu_workspace
    exit
}