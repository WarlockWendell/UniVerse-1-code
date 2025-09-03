# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
import html
from typing import Any, Callable, Dict, List, Optional, Union
import os
import PIL
import ftfy
import regex as re
import torch
import numpy as np
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers.image_processor import PipelineImageInput

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from fastvideo.models.wan.modeling_universe import UniVerseTransformer3DModel
from fastvideo.models.wan.autoencoder_kl_wan import AutoencoderKLWan
from fastvideo.models.wan.pipeline_output import WanPipelineOutput
from fastvideo.models.wan.lora_pipeline import WanLoraLoaderMixin

from einops import rearrange
from fastvideo.utils.communications import all_gather
from fastvideo.utils.parallel_states import (get_sequence_parallel_state,
                                             nccl_info)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers.utils import export_to_video
        >>> from diffusers import AutoencoderKLWan, WanPipeline
        >>> from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        >>> # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
        >>> model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        >>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        >>> pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        >>> flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
        >>> pipe.to("cuda")

        >>> prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
        >>> negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        >>> output = pipe(
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     height=720,
        ...     width=1280,
        ...     num_frames=81,
        ...     guidance_scale=5.0,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=16)
        ```
"""


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


class UniVersePipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5Tokenizer),
            specifically the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/umt5-xxl](https://huggingface.co/google/umt5-xxl) variant.
        transformer ([`UniVerseTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`UniPCMultistepScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLWan`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: UniVerseTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.scheduler_mel = None
    
    def set_mel_scheduler(self, scheduler_mel):
        self.scheduler_mel = scheduler_mel

    def prepare_latents(
        self,
        ref_img: torch.Tensor,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        ref_img = ref_img.unsqueeze(2)  # [B, C, F, H, W]

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        ref_img_latents = retrieve_latents(self.vae.encode(ref_img), sample_mode="argmax")
        ref_img_latents = ref_img_latents.repeat(batch_size, 1, 1, 1, 1)
        ref_img_latents = (ref_img_latents - latents_mean) * latents_std

        mel_shape = (
            batch_size,
            8,
            16,
            num_latent_frames,
        )

        mel_latents = randn_tensor(mel_shape, generator=generator, device=device, dtype=dtype)

        return latents, mel_latents, ref_img_latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        ref_img,
        text_ut5: Optional[torch.Tensor] = None,
        text_ut5_neg: Optional[torch.Tensor] = None,
        ambient: Optional[torch.Tensor] = None,
        ambient_mask: Optional[torch.Tensor] = None,
        speech_token: Optional[torch.Tensor] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_vid: float = 5.0,
        guidance_scale_aud: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        save: bool = False,
        save_dir: str = ".",
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            autocast_dtype (`torch.dtype`, *optional*, defaults to `torch.bfloat16`):
                The dtype to use for the torch.amp.autocast.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct

        self._guidance_scale = guidance_scale_vid
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters
        batch_size = 1

        transformer_dtype = self.transformer.dtype

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.scheduler_mel.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        ref_img = self.video_processor.preprocess(ref_img, height=height, width=width).to(
            device, dtype=ambient.dtype
        )

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels #// 2
        latents, mel_latents, ref_img_latents = self.prepare_latents(
            ref_img,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            transformer_dtype,
            device,
            generator,
            latents,
        )

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        ref = ref_img_latents

        # check sequence_parallel
        world_size, rank = nccl_info.sp_size, nccl_info.rank_within_group
        if get_sequence_parallel_state():
            latents = rearrange(latents,
                                "b s (n t) h w -> b s n t h w",
                                n=world_size).contiguous()
            latents = latents[:, :, rank, :, :, :]

            mel_latents = rearrange(mel_latents,
                                "b s d (n t) -> b s d n t",
                                n=world_size).contiguous()
            mel_latents = mel_latents[:, :, :, rank, :]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                # print(latents.shape, mel_latents.shape)

                latent_model_input_vid = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input_mel = torch.cat([mel_latents] * 2) if self.do_classifier_free_guidance else mel_latents
                ref_input = torch.cat([torch.zeros_like(ref).to(ref), ref]) if self.do_classifier_free_guidance else ref
                
                ambient_input = torch.cat([torch.zeros_like(ambient).to(ambient), ambient]) if self.do_classifier_free_guidance else ambient
                ut5_input = torch.cat([text_ut5_neg, text_ut5]) if self.do_classifier_free_guidance else text_ut5
                ambient_mask_input = torch.cat([ambient_mask, ambient_mask]) if self.do_classifier_free_guidance else ambient_mask
                speech_token_input = torch.cat([torch.zeros_like(speech_token).to(speech_token), speech_token]) if self.do_classifier_free_guidance else speech_token

                timestep = t.expand(latent_model_input_vid.shape[0])

                with torch.no_grad():
                    # print(ambient_mask.shape)
                    noise_pred_vid, noise_pred_mel = self.transformer(
                        hidden_states_ref=ref_input,
                        hidden_states_vid=latent_model_input_vid,
                        hidden_states_mel=latent_model_input_mel,
                        timestep=timestep,
                        encoder_hidden_states_ambient=ambient_input,
                        encoder_hidden_states_ut5=ut5_input,
                        audio_text_attention_mask=ambient_mask_input,
                        speech_token=speech_token_input,
                        attention_mask=None,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                        is_cfg=self.do_classifier_free_guidance,
                    )[0]

                if self.do_classifier_free_guidance:
                    noise_uncond_vid, noise_pred_vid = noise_pred_vid.chunk(2)
                    noise_pred_vid = noise_uncond_vid + guidance_scale_vid * (noise_pred_vid - noise_uncond_vid)

                    noise_uncond_mel, noise_pred_mel = noise_pred_mel.chunk(2)
                    noise_pred_mel = noise_uncond_mel + guidance_scale_aud * (noise_pred_mel - noise_uncond_mel)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred_vid, t, latents, return_dict=False)[0]
                mel_latents = self.scheduler_mel.step(noise_pred_mel, t, mel_latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if get_sequence_parallel_state():
            latents = all_gather(latents, dim=2)
            mel_latents = all_gather(mel_latents, dim=3)
            if rank <= 0:
                print(latents.min(), latents.max(), latents.mean(), latents.std())
                print(mel_latents.min(), mel_latents.max(), mel_latents.mean(), mel_latents.std())
        latents = torch.cat([ref_img_latents, latents[:, :, 1:]], 2)

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video, mels=mel_latents)