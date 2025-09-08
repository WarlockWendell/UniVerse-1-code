<div align="center">
<img src=assets/logo.png width="30%"/>
</div>

# UniVerse-1: Unified Audio-Video Generation via Stitching of Experts.

<div align="center">
  <a href="https://huggingface.co/dorni/UniVerse-1-Base"><img src="https://img.shields.io/static/v1?label=UniVerse-1&message=HuggingFace&color=yellow"></a> &ensp;
  <a href="https://huggingface.co/datasets/dorni/Verse-Bench"><img src="https://img.shields.io/static/v1?label=Verse-Bench&message=HuggingFace&color=yellow"></a>
</div>
<div align="center">
  <a href="https://dorniwang.github.io/UniVerse-1"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=green"></a> &ensp;
  <a href=""><img src="assets/arxiv.svg"></a> &ensp;
  <a href="https://github.com/stepfun-ai/Step-Audio2/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue?&color=blue"/></a>
</div>

This is official inference code of UniVerse-1

## 🔥🔥🔥 News!!
<!-- * Sep 03, 2025: 👋 We release the  # TODO -->
<!-- * Sep 08, 2025: 👋 We release Verse-Bench metric tools, [Verse-Bench tools](https://github.com/Dorniwang/Verse-Bench). -->
<!-- * Sep 08, 2025: 👋 We release the technical report of [UniVerse-1](https://arxiv.org/pdf/2507.16632). -->
* Sep 08, 2025: 👋 We release Verse-Bench datasets, [Verse-Bench Dataset](https://huggingface.co/datasets/dorni/Verse-Bench).
* Sep 08, 2025: 👋 We release model weights of [UniVerse-1](https://huggingface.co/dorni/UniVerse-1-Base).
* Sep 03, 2025: 👋 We release the project page of [UniVerse-1](https://dorniwang.github.io/UniVerse-1).


## Introduction

UniVerse-1 is a unified, Veo-3-like model that simultaneously generates synchronized audio and video from a reference image and a text prompt.

- **Unified Audio-Video synthesis**: Features the fascinating ability to generate audio and video in tandem. It interprets the input prompt to produce a perfectly synchronized audio-visual output.

- **Speech audio generation**: The model can generate fluent speech directly from a text prompt, demonstrating a built-in text-to-speech (TTS) ability. Crucially, it tailors the voice timbre to match the specific character being generated.

- **Musical instrument playing sound generation**: The model is also highly proficient at creating sounds of musical instruments. Additionally, it offers some capability for "singing while playing," generating both vocal and instrumental tracks concurrently.

- **Ambient sound generation**: The model can generate ambient sounds, producing background audio that matches the visual environment of the video.

- **The first open-sourced Dit-based Audio-Video joint method**: We are the first to open-source a DiT-based, Veo-3-like model for joint audio-visual generation. 

## Model Download
| Models   | 🤗 Hugging Face |
|-------|-------|
| UniVerse-1 Base | [UniVerse-1](https://huggingface.co/dorni/UniVerse-1-Base) |

download our pretrained model into ./checkpoints/UniVerse-1-base/

## Model Usage
### 🔧 Dependencies and Installation
- Python >= 3.10
- [PyTorch >= 2.5.0-cu121](https://pytorch.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- Dependent models:
  - [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers), download into ./huggingfaces/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/
  - [ACE-Step/ACE-Step-v1-3.5B](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B), download into ./huggingfaces/ACE-Step/ACE-Step-v1-3.5B/

```bash
conda create -n universe python=3.10
conda activate universe
pip install torch==2.5.0 torchaudio==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install packaging ninja && pip install flash-attn==2.7.0.post2 --no-build-isolation 
pip install -r requirements-lint.txt
pip install -e .

git clone https://github.com/Dorniwang/UniVerse-1-code/
cd UniVerse-1-code
```

### 🚀 Inference Scripts

```bash
bash scipts/inference/inference_universe.sh
```

## Acknowledgements

Part of the code for this project comes from:
* [FastVideo](https://github.com/hao-ai-lab/FastVideo)
* [diffusers](https://github.com/huggingface/diffusers/tree/v0.33.1)
* [Wan2.1](https://github.com/huggingface/diffusers/blob/v0.33.1/src/diffusers/models/transformers/transformer_wan.py)
* [Ace-step](https://github.com/ace-step/ACE-Step)

Thank you to all the open-source projects for their contributions to this project!

## License

The code in the repository is licensed under [Apache 2.0](LICENSE) License.

## Citation

```
@misc{wang2025universe-1,
    title={UniVerse-1:A Unified Audio-Video Generation Framework via Stitching of Expertise},
    author={Wang, Duomin and Zuo, wei and Li, Aojie and Chen, Ling-Hao and Liao, Xinyao and Zhou, Deyu and Yin, Zixin and Dai, Xili and Jinag Daxin, Yu, Gang},
    journal={arxiv},
    year={2025}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Dorniwang/UniVerse-1-code&type=Date)](https://star-history.com/#Dorniwang/UniVerse-1-code&Date)