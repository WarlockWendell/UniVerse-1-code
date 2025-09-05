#!/bin/bash

num_gpus=4
/data/workspace/env/miniconda/envs/universe_test/bin/torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 23333 \
    fastvideo/sample/sample_universe_test.py \
    --model_path /cpfs/wangduomin/workspace/models/huggingface/Wan2.1-T2V-1.3B-Diffusers/ \
    --ace_path /cpfs/wangduomin/workspace/models/huggingface/union/ACE-Step/ACE-Step-v1-3.5B/ \
    --num_frames 125 \
    --height 256 \
    --width 256 \
    --num_inference_steps 35 \
    --guidance_scale_vid 1.2 \
    --guidance_scale_aud 5.0 \
    --seed 10240 \
    --cpu_offload \
    --output_path ./output/ \
    --transformer_path /cpfs/wangduomin/workspace/models/preserved_models/fastvideo/veo3/wan/wan2.1_1.3b_veo_method2_multinodes_v4_10_ambient_exp1/checkpoint-10800/ \
    --refimg_path assets/images/321.jpg \
    --prompt_path assets/prompts/321.json \