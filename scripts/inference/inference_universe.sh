#!/bin/bash

num_gpus=2
/home/ae86/anaconda3/envs/universe/bin/torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 23333 \
    fastvideo/sample/batch_sample_universe.py \
    --model_path ./huggingfaces/Wan2.1-T2V-1.3B-Diffusers/ \
    --ace_path ./huggingfaces/ACE-Step-v1-3.5B/ \
    --num_frames 10 \
    --height 256 \
    --width 256 \
    --num_inference_steps 35 \
    --guidance_scale_vid 1.2 \
    --guidance_scale_aud 5.0 \
    --seed 10240 \
    --cpu_offload \
    --output_path ./output/ \
    --transformer_path checkpoints/UniVerse-1-Base/ \
    --csv_path ./meta.csv