#!/bin/bash
source "/home/jinyang/miniconda3/etc/profile.d/conda.sh"; conda activate dp
export WANDB_API_KEY=956bd553be1193c6fdfce758ad1eded5190eac56
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=3

python train.py --config-name=train_diffusion_transformer_real_hybrid_workspace task.dataset_path=/home/jinyang/yuwenye/human-in-the-loop/data/0219_PnP_fixed_init
