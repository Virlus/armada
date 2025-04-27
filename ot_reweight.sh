#!/bin/bash
source "/data/yuwenye/miniconda3/etc/profile.d/conda.sh"; conda activate human_dp

export CUDA_VISIBLE_DEVICES=0

# python ot_reweight.py -temp 1 -p /home/yuwenye/project/human-in-the-loop/third_party/diffusion_policy/data/0305_pour_sirius_round1_4/replay_buffer.zarr -skip 5 \
#     -ckpt /home/yuwenye/project/human-in-the-loop/third_party/diffusion_policy/outputs/0305_14.10.45_train_diffusion_transformer_hybrid_multi_gpu_pouring/checkpoints/latest.ckpt

python ot_match_episodes_and_reweight.py -temp 1 -p /home/yuwenye/project/human-in-the-loop/third_party/diffusion_policy/data/0305_pour_sirius_round1_4/replay_buffer.zarr -skip 5 \
    -ckpt /home/yuwenye/project/human-in-the-loop/third_party/diffusion_policy/outputs/0305_14.10.45_train_diffusion_transformer_hybrid_multi_gpu_pouring/checkpoints/latest.ckpt