#!/bin/bash
source "/home/yuwenye/miniconda3/etc/profile.d/conda.sh"; conda activate human_dp

export WANDB_API_KEY=956bd553be1193c6fdfce758ad1eded5190eac56
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=3

python train.py --config-name=train_diffusion_transformer_real_hybrid_workspace task.dataset_path=/mnt/workspace/DP/0217_PnP
