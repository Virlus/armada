#!/bin/bash
source "/root/miniforge3/etc/profile.d/conda.sh"; conda activate human_dp

export CUDA_VISIBLE_DEVICES=0
python ot_reweigh.py -temp 1 -p /root/human-in-the-loop/data/0305_pour_sirius_round1_4/replay_buffer.zarr \
    -ckpt /root/human-in-the-loop/third_party/diffusion_policy/outputs/2025.03.05/14.10.45_train_diffusion_transformer_hybrid_multi_gpu_pouring/checkpoints/latest.ckpt