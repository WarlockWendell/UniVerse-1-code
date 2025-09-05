import argparse
import json
import os
import time

import re
import torch
import torch.nn as nn
import torch.distributed as dist
from diffusers.utils import export_to_video, load_image
from transformers import AutoTokenizer, UMT5EncoderModel
from einops import rearrange
from safetensors.torch import load_file

import sys
import PIL.Image as Image
import soundfile as sf
import librosa
import numpy as np

from fastvideo.models.wan.modeling_universe import UniVerseTransformer3DModel
from fastvideo.models.wan.pipeline_universe import UniVersePipeline
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state, nccl_info)
from fastvideo.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from fastvideo.models.music_dcae.music_dcae_pipeline import MusicDCAE
from fastvideo.models.language_segmentation import LangSegment, language_filters
from fastvideo.models.lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer

def initialize_distributed():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl",
                            init_method="env://",
                            world_size=world_size,
                            rank=local_rank)
    initialize_sequence_parallel_state(world_size)

SUPPORT_LANGUAGES = {
    "en": 259,
    "de": 260,
    "fr": 262,
    "es": 284,
    "it": 285,
    "pt": 286,
    "pl": 294,
    "tr": 295,
    "ru": 267,
    "cs": 293,
    "nl": 297,
    "ar": 5022,
    "zh": 5023,
    "ja": 5412,
    "hu": 5753,
    "ko": 6152,
    "hi": 6680,
}

structure_pattern = re.compile(r"\[.*?\]")

def get_lang(lang_segment, text):
    language = "en"
    try:
        _ = lang_segment.getTexts(text)
        langCounts = lang_segment.getCounts()
        language = langCounts[0][0]
        if len(langCounts) > 1 and language == "en":
            language = langCounts[1][0]
    except Exception as err:
        language = "en"
    return language

def tokenize_lyrics(lyric_tokenizer, lang_segment, text_speech):
    lyric_token_idx = [261]
    text_speech = text_speech.strip()
    if not text_speech:
        lyric_token_idx += [2]
        return lyric_token_idx

    lang = get_lang(lang_segment, text_speech)

    if lang not in SUPPORT_LANGUAGES:
        lang = "en"
    if "zh" in lang:
        lang = "zh"
    if "spa" in lang:
        lang = "es"

    try:
        if structure_pattern.match(text_speech):
            token_idx = lyric_tokenizer.encode(text_speech, "en")
        else:
            token_idx = lyric_tokenizer.encode(text_speech, lang)
        lyric_token_idx = lyric_token_idx + token_idx + [2]
    except Exception as e:
        print("tokenize error", e, "for text", text_speech, "major_language", lang)
    return lyric_token_idx

def inference(args):
    initialize_distributed()
    print(nccl_info.sp_size)
    device = torch.cuda.current_device()
    # Peiyuan: GPU seed will cause A100 and H100 to produce different results .....
    weight_dtype = torch.bfloat16

    if args.transformer_path is not None:
        transformer = UniVerseTransformer3DModel.from_pretrained(
            args.transformer_path, torch_dtype=weight_dtype)
    else:
        print("no model path provided, exit")
        exit(0)

    wan_text_encoder = UMT5EncoderModel.from_pretrained(
        args.model_path,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
    ).to(device)
    wan_tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        subfolder="tokenizer",
    )
    wan_text_encoder.requires_grad_(False)
    wan_text_encoder.eval()
    wan_max_sequence_length = 512
    audio_model_path = args.ace_path
 
    dcae_path = os.path.join(audio_model_path, "music_dcae_f8c8")
    vocoder_path = os.path.join(audio_model_path, "music_vocoder")

    dcae_pipeline = MusicDCAE(16000, 25600, dcae_path, vocoder_path).to(device)
    lyric_tokenizer = VoiceBpeTokenizer()
    lang_segment = LangSegment()
    lang_segment.setfilters(language_filters.default)

    text_model_path = os.path.join(audio_model_path, "umt5-base")
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_model_path,
        torch_dtype=torch.bfloat16,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        text_model_path)
    max_sequence_length = 256
    device = device
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    scheduler_vid = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=15, use_dynamic_shifting=False)
    scheduler_mel = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=15, use_dynamic_shifting=False)
    pipe = UniVersePipeline.from_pretrained(args.model_path,
                                                # vae=vae,
                                                transformer=transformer,
                                                scheduler=scheduler_vid,
                                                torch_dtype=weight_dtype)
    pipe.set_mel_scheduler(scheduler_mel)

    if args.cpu_offload:
        pipe.enable_model_cpu_offload(device)
    else:
        pipe.to(device)

    # Generate videos from the input prompt

    ref_size = [(512, 512), (480, 768), (768, 480)]
    ref_size_ratio = np.array([1, 480 / 768, 768 / 480])

    prompt_embeds = None
    encoder_attention_mask = None

    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref_img_path = args.refimg_path
        prompt_path = args.prompt_path
        with open(prompt_path, 'r') as f:
            prompt_dict = json.load(f)
        
        ref_img = Image.open(ref_img_path)
        ref_img = np.array(ref_img)[:, :, :3]
        h, w, c = ref_img.shape
        ref_img = Image.fromarray(ref_img)
        
        min_idx = np.argmin(np.abs(ref_size_ratio - h / w))
        height, width = ref_size[min_idx][0], ref_size[min_idx][1]

        num_frame = args.num_frames
        
        generator = torch.Generator("cpu").manual_seed(args.seed)

        text_speech = prompt_dict['speech_prompt']['text']
        text_ambient = prompt_dict['audio_prompt']
        text_conclusion = prompt_dict["video_prompt"]

        text_ambient = ", ".join(text_ambient)
        if "no ambient" in text_ambient.lower() or "none" in text_ambient.lower():
            text_ambient = ""

        print("text speech: {}".format(text_speech))
        print("text ambient: {}".format(text_ambient))
        print("text conclusion: {}".format(text_conclusion))

        text_speech = text_speech.strip()
        if not text_speech:
            speech_token = torch.from_numpy(np.array([0] * 32)).unsqueeze(0).to(device)
        else:
            speech_token = tokenize_lyrics(lyric_tokenizer, lang_segment, text_speech)
            speech_token = torch.from_numpy(np.array(speech_token)).to(device).unsqueeze(0)

        ambient_inputs = tokenizer(
            text_ambient,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_sequence_length,
        )
        ambient_inputs = {key: value.to(device) for key, value in ambient_inputs.items()}
        text_encoder.to(device)
        with torch.no_grad():
            ambient_outputs = text_encoder(**ambient_inputs)
            ambient_feats = ambient_outputs.last_hidden_state
        ambient_mask = ambient_inputs["attention_mask"]

        ##### positive prompt embedding
        text_inputs = tokenizer(
            text_conclusion,
            padding="max_length",
            max_length=wan_max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, prompt_attention_mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
        with torch.no_grad():
            prompt_embeds = wan_text_encoder(text_input_ids.to(device), prompt_attention_mask.to(device))[0]
        
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(wan_max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(1, seq_len, -1)

        ##### negative prompt embedding
        text_inputs_neg = tokenizer(
            "",
            padding="max_length",
            max_length=wan_max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids_neg, prompt_attention_mask_neg = text_inputs_neg.input_ids, text_inputs_neg.attention_mask
        seq_lens_neg = prompt_attention_mask_neg.gt(0).sum(dim=1).long()
        with torch.no_grad():
            prompt_embeds_neg = wan_text_encoder(text_input_ids_neg.to(device), prompt_attention_mask_neg.to(device))[0]
        
        prompt_embeds_neg = [u[:v] for u, v in zip(prompt_embeds_neg, seq_lens_neg)]
        prompt_embeds_neg = torch.stack(
            [torch.cat([u, u.new_zeros(wan_max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds_neg], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len_neg, _ = prompt_embeds_neg.shape
        prompt_embeds_neg = prompt_embeds_neg.view(1, seq_len_neg, -1)

        suffix = "{}_{}".format(ref_img_path.split("/")[-1].split(".")[0], prompt_path.split("/")[-1].split(".")[0])
        save_dir = args.output_path
        os.makedirs(save_dir, exist_ok=True)
        output = pipe(
            ref_img=ref_img,
            text_ut5=prompt_embeds,
            text_ut5_neg=prompt_embeds_neg,
            ambient=ambient_feats,
            ambient_mask=ambient_mask,
            speech_token=speech_token,
            height=height,
            width=width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale_vid=args.guidance_scale_vid,
            guidance_scale_aud=args.guidance_scale_aud,
            generator=generator,
            save=False,
            save_dir=save_dir,
        )
        video = output.frames
        mels = output.mels
        if nccl_info.global_rank <= 0:
            input_size = "h{}_w{}".format(height, width)

            with torch.no_grad():
                sr, reconstructed_wav, mels = dcae_pipeline.decode(latents=mels.float(), audio_lengths=torch.tensor([int(video.shape[1] / 25 * 16000)]), sr=16000) # bigvgan.decode_mel(mels)
            
            sf.write(os.path.join(args.output_path, f"{suffix}_{input_size}.wav"), reconstructed_wav[0].float().detach().cpu().numpy()[0], 16000)
            export_to_video(
                video[0],
                os.path.join(args.output_path, f"{suffix}_{input_size}.mp4"),
                fps=25,
            )

            ffmpeg_cmd = "ffmpeg -y -v quiet -i {} -i {} {}".format(os.path.join(args.output_path, f"{suffix}_{input_size}.mp4"), os.path.join(args.output_path, f"{suffix}_{input_size}.wav"), os.path.join(args.output_path, f"{suffix}_{input_size}_with_audio.mp4"))
            os.system(ffmpeg_cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--prompt_path", type=str, default=None)
    parser.add_argument("--refimg_path", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--model_path", type=str, default="data/hunyuan")
    parser.add_argument("--transformer_path", type=str, default=None)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--ace_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="./outputs/video")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=10)
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument(
        "--lora_checkpoint_dir",
        type=str,
        default=None,
        help="Path to the directory containing LoRA checkpoints",
    )
    # Additional parameters
    parser.add_argument(
        "--denoise-type",
        type=str,
        default="flow",
        help="Denoise type for noised inputs.",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="Seed for evaluation.")
    parser.add_argument("--neg_prompt",
                        type=str,
                        default=None,
                        help="Negative prompt for sampling.")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Classifier free guidance scale.",
    )
    parser.add_argument(
        "--guidance_scale_vid",
        type=float,
        default=1.0,
        help="Classifier free guidance scale.",
    )
    parser.add_argument(
        "--guidance_scale_aud",
        type=float,
        default=1.0,
        help="Classifier free guidance scale.",
    )
    parser.add_argument(
        "--embedded_cfg_scale",
        type=float,
        default=6.0,
        help="Embedded classifier free guidance scale.",
    )
    parser.add_argument("--flow_shift",
                        type=int,
                        default=7,
                        help="Flow shift parameter.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="Batch size for inference.")
    parser.add_argument(
        "--num_videos",
        type=int,
        default=1,
        help="Number of videos to generate per prompt.",
    )
    parser.add_argument(
        "--load-key",
        type=str,
        default="module",
        help=
        "Key to load the model states. 'module' for the main model, 'ema' for the EMA model.",
    )
    parser.add_argument(
        "--dit-weight",
        type=str,
        default=
        "data/hunyuan/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
    )
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help=
        "Enable reproducibility by setting random seeds and deterministic algorithms.",
    )
    parser.add_argument(
        "--disable-autocast",
        action="store_true",
        help=
        "Disable autocast for denoising loop and vae decoding in pipeline sampling.",
    )

    # Flow Matching
    parser.add_argument(
        "--flow-reverse",
        action="store_true",
        help="If reverse, learning/sampling from t=1 -> t=0.",
    )
    parser.add_argument("--flow-solver",
                        type=str,
                        default="euler",
                        help="Solver for flow matching.")
    parser.add_argument(
        "--use-linear-quadratic-schedule",
        action="store_true",
        help=
        "Use linear quadratic schedule for flow matching. Following MovieGen (https://ai.meta.com/static-resource/movie-gen-research-paper)",
    )
    parser.add_argument(
        "--linear-schedule-end",
        type=int,
        default=25,
        help="End step for linear quadratic schedule for flow matching.",
    )

    # Model parameters
    parser.add_argument("--model", type=str, default="HYVideo-T/2-cfgdistill")
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--precision",
                        type=str,
                        default="bf16",
                        choices=["fp32", "fp16", "bf16", "fp8"])
    parser.add_argument("--rope-theta",
                        type=int,
                        default=256,
                        help="Theta used in RoPE.")

    parser.add_argument("--vae", type=str, default="884-16c-hy")
    parser.add_argument("--vae-precision",
                        type=str,
                        default="fp16",
                        choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--vae-tiling", action="store_true", default=True)

    parser.add_argument("--text-encoder", type=str, default="llm")
    parser.add_argument(
        "--text-encoder-precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--text-states-dim", type=int, default=4096)
    parser.add_argument("--text-len", type=int, default=256)
    parser.add_argument("--tokenizer", type=str, default="llm")
    parser.add_argument("--prompt-template",
                        type=str,
                        default="dit-llm-encode")
    parser.add_argument("--prompt-template-video",
                        type=str,
                        default="dit-llm-encode-video")
    parser.add_argument("--hidden-state-skip-layer", type=int, default=2)
    parser.add_argument("--apply-final-norm", action="store_true")

    parser.add_argument("--text-encoder-2", type=str, default="clipL")
    parser.add_argument(
        "--text-encoder-precision-2",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--text-states-dim-2", type=int, default=768)
    parser.add_argument("--tokenizer-2", type=str, default="clipL")
    parser.add_argument("--text-len-2", type=int, default=77)

    args = parser.parse_args()
    if args.quantization:
        inference_quantization(args)
    else:
        inference(args)
