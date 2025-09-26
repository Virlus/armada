#!/bin/bash
{
    source "/path/to/miniconda3/etc/profile.d/conda.sh"; conda activate armada
    export WANDB_API_KEY=${your_wandb_api_key}
    export HYDRA_FULL_ERROR=1
    export CUDA_VISIBLE_DEVICES=0

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py train_diffusion_transformer_real_hybrid_dino_multi_gpu_workspace
    exit
}