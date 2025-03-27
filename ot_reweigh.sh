#!/bin/bash
source "/home/lvjun-local/miniconda3/etc/profile.d/conda.sh"; conda activate human_dp

export CUDA_VISIBLE_DEVICES=0
python ot_reweigh.py -temp 1 -p -ckpt 