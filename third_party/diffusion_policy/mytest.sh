#!/bin/bash
source "/home/jinyang/miniconda3/etc/profile.d/conda.sh"; conda activate dp
export WANDB_API_KEY=956bd553be1193c6fdfce758ad1eded5190eac56
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3

# python train.py --config-name=train_diffusion_transformer_hybrid_multi_gpu_workspace task.dataset_path=/home/jinyang/yuwenye/human-in-the-loop/data/0219_PnP_fixed_init
# python train.py train_diffusion_transformer_real_hybrid_multi_gpu_workspace
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train.py train_diffusion_transformer_real_hybrid_weighted_multi_gpu_workspace
