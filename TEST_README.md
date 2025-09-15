1. 配好环境
    ```
    bash env_setup.sh
    ```
 
2. 下载权重
    ```
    hf download dorni/Universe-1-Base --local-dir ./checkpoints/Universe-1-Base
    hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers ./huggingfaces/Wan2.1-T2V-1.3B-Diffusers
    hf download ACE-Step/ACE-Step-v1-3.5B ./huggingfaces/ACE-Step-v1-3.5B
    ```

3. 推理脚本 `scripts/inference/inference_universe.sh` 

    ```
    num_gpus=2
    torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 23333 \
    fastvideo/sample/batch_sample_universe.py \
    --model_path ./huggingfaces/Wan2.1-T2V-1.3B-Diffusers/ \
    --ace_path ./huggingfaces/ACE-Step-v1-3.5B/ \
    --num_frames 100 \
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
    ```

    num_frames: 25FPS，100帧为4s

    height, width: 模型中预设的只有如下几种分辨率，详细见 `fastvideo/sample/batch_sample_universe.py` 的 211 行.

    ```
    ref_size = [(512, 512), (480, 768), (768, 480)] # (height, width)
    ref_size_ratio = np.array([1, 480 / 768, 768 / 480])
    ```

    我没法推理测试，所以没改，可以改下跑一下看看结果怎么样：
    ```
    ref_size = [(240, 436), (512, 512), (480, 768), (768, 480)]
    ref_size_ratio = np.array([240 / 436, 1, 480 / 768, 768 / 480])
    ```

    存储是分开存储音频和视频的，如果要存在一起，把 `fastvideo/sample/batch_sample_universe.py` 342 和 343 行注释取消掉。

    