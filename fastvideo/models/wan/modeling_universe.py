# This code is derive from Wan2.1 and Ace-step.
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

import math
from typing import Any, Dict, Optional, Tuple, List, Union
from pathlib import Path
from os import PathLike
import json

import torch
from torch import nn
from safetensors.torch import load_file

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils import BaseOutput, is_torch_version
from diffusers.models.normalization import RMSNorm

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List, Union


from fastvideo.models.flash_attn_no_pad import flash_attn_no_pad, flash_attn_varlen_kvpacked_func, flash_attn_kvpacked_func
from fastvideo.utils.communications import all_gather, all_to_all_4D
from fastvideo.utils.parallel_states import (get_sequence_parallel_state,
                                             nccl_info)
from fastvideo.models.lyrics_utils.lyric_encoder import ConformerEncoder as LyricEncoder
# from fastvideo.models.ace.attention import LinearTransformerBlock, t2i_modulate
from einops import rearrange


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def shrink_head(encoder_state, dim):
    local_heads = encoder_state.shape[dim] // nccl_info.sp_size
    return encoder_state.narrow(dim, nccl_info.rank_within_group * local_heads,
                                local_heads)


def val2list(x: list or tuple or any, repeat_time=1) -> list:  # type: ignore
    """Repeat `val` for `repeat_time` times and return the list or val if list/tuple."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:  # type: ignore
    """Return tuple with min_len by repeating element at idx_repeat."""
    # convert to list first
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


def get_same_padding(
    kernel_size: Union[int, Tuple[int, ...]],
) -> Union[int, Tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, f"kernel size {kernel_size} should be odd number"
        return kernel_size // 2


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        padding: Union[int, None] = None,
        use_bias=False,
        norm=None,
        act=None,
    ):
        super().__init__()
        if padding is None:
            padding = get_same_padding(kernel_size)
            padding *= dilation

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.use_bias = use_bias

        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
        )
        if norm is not None:
            self.norm = RMSNorm(out_dim, elementwise_affine=False)
        else:
            self.norm = None
        if act is not None:
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_feature=None,
        kernel_size=3,
        stride=1,
        padding: Union[int, None] = None,
        use_bias=False,
        norm=(None, None, None),
        act=("silu", "silu", None),
        dilation=1,
    ):
        out_feature = out_feature or in_features
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act = val2tuple(act, 3)

        self.glu_act = nn.SiLU(inplace=False)
        self.inverted_conv = ConvLayer(
            in_features,
            hidden_features * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act=act[0],
        )
        self.depth_conv = ConvLayer(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size,
            stride=stride,
            groups=hidden_features * 2,
            padding=padding,
            use_bias=use_bias[1],
            norm=norm[1],
            act=None,
            dilation=dilation,
        )
        self.point_conv = ConvLayer(
            hidden_features,
            out_feature,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act=act[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        x = x.transpose(1, 2)

        return x

class WanAttnProcessor2_0_self:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        hidden_states_mel: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb_vid: Optional[torch.Tensor] = None,
        num_frames: int = 1,
    ) -> torch.Tensor:
        hidden_states_vid = hidden_states
        batch_size, sequence_length_vid, _ = hidden_states_vid.shape

        if not hidden_states_mel is None:
            batch_size, sequence_length_mel, _ = hidden_states_mel.shape
        
        query_dtype = hidden_states_vid.dtype

        hidden_states = hidden_states_vid.to(query_dtype)
        encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2) # b h s d
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)


        if not hidden_states_mel is None:
            key_audio = attn.add_k_proj(hidden_states_mel)
            key_audio = attn.norm_added_k(key_audio)
            value_audio = attn.add_v_proj(hidden_states_mel)

            key_audio = key_audio.unflatten(2, (attn.heads, -1)).transpose(1, 2) # b h s d
            value_audio = value_audio.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
            x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
            x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
            return x_out.type_as(hidden_states)

        if rotary_emb_vid is not None: # only work when perform self attention

            if get_sequence_parallel_state(): ### TODO:
                rotary_emb_vid = shrink_head(rotary_emb_vid, dim=2)

            query = apply_rotary_emb(query, rotary_emb_vid).to(query_dtype)
            key = apply_rotary_emb(key, rotary_emb_vid).to(query_dtype)

        # shard the head dimension
        if get_sequence_parallel_state():
            # b, h, s, d

            query = all_to_all_4D(query, scatter_dim=1, gather_dim=2)  #scatter_dim=2, gather_dim=1
            key = all_to_all_4D(key, scatter_dim=1, gather_dim=2)
            value = all_to_all_4D(value, scatter_dim=1, gather_dim=2)
        
        qkv = torch.stack([query, key, value], dim=2)
        qkv = qkv.transpose(1, 3).to(query_dtype)
        seq_len = qkv.shape[1]
        
        attention_mask = torch.zeros(qkv.shape[0], seq_len).to(qkv).bool()
        attention_mask.fill_(True).to(query_dtype)

        hidden_states = flash_attn_no_pad(qkv,
                                        attention_mask,
                                        causal=False,
                                        dropout_p=0.0,
                                        softmax_scale=None)

        if get_sequence_parallel_state():
            hidden_states = all_to_all_4D(hidden_states,
                                          scatter_dim=1,
                                          gather_dim=2)
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.to(query_dtype)
        else:

            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.to(query_dtype)

        
        if not hidden_states_mel is None:
            ##### audio
            if get_sequence_parallel_state():
                # b, h, s, d
                key_audio = all_to_all_4D(key_audio, scatter_dim=1, gather_dim=2)
                value_audio = all_to_all_4D(value_audio, scatter_dim=1, gather_dim=2)

            q = query.transpose(1, 2).to(query_dtype) # b s h d
            kv = torch.stack([key_audio, value_audio], dim=2).to(query_dtype) 
            kv = kv.transpose(1, 3).to(query_dtype)  # b s 2 h d

            _, _, qh, qd = q.shape

            q = q.reshape(batch_size, num_frames * nccl_info.sp_size, sequence_length_vid // num_frames, qh, qd).reshape(batch_size * num_frames * nccl_info.sp_size, sequence_length_vid // num_frames, qh, qd)
            kv = kv.reshape(batch_size, num_frames * nccl_info.sp_size, sequence_length_mel // num_frames, 2, qh, qd).reshape(batch_size * num_frames * nccl_info.sp_size, sequence_length_mel // num_frames, 2, qh, qd)

            hidden_states_audio = flash_attn_kvpacked_func(
                                q,
                                kv,
                                softmax_scale=None,
                                causal=False
            )

            hidden_states_audio = hidden_states_audio.reshape(batch_size, num_frames * nccl_info.sp_size, sequence_length_vid // num_frames, qh, qd).reshape(batch_size, nccl_info.sp_size * sequence_length_vid, qh, qd)

            if get_sequence_parallel_state():
                hidden_states_audio = all_to_all_4D(hidden_states_audio,
                                            scatter_dim=1,
                                            gather_dim=2)
                hidden_states_audio = hidden_states_audio.flatten(2, 3)
                hidden_states_audio = hidden_states_audio.to(query_dtype)
            else:

                hidden_states_audio = hidden_states_audio.flatten(2, 3)
                hidden_states_audio = hidden_states_audio.to(query_dtype)

            hidden_states = hidden_states + hidden_states_audio

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

class WanAttnProcessor2_0_cross1:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        hidden_states_mel: List[torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        rotary_emb_vid: Optional[torch.Tensor] = None, 
        rotary_emb_mel: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states_vid = hidden_states
        batch_size, sequence_length_vid, _ = hidden_states_vid.shape
        batch_size, sequence_length_mel, _ = hidden_states_mel.shape
        batch_size, sequence_length_text, _ = encoder_hidden_states.shape
        
        cross_attn = True

        query_dtype = hidden_states_vid.dtype

        query = attn.to_q(hidden_states_vid)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        key_mel = attn.add_k_proj(hidden_states_mel)
        value_mel = attn.add_v_proj(hidden_states_mel)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        if attn.norm_added_k is not None:
            key_mel = attn.norm_added_k(key_mel)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2) # b h s d
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        
        key_mel = key_mel.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value_mel = value_mel.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
            x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
            x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
            return x_out.type_as(hidden_states)

        if rotary_emb_vid is not None: # only work when perform self attention
            if get_sequence_parallel_state(): ### TODO:
                rotary_emb_vid = shrink_head(rotary_emb_vid, dim=2)

            query = apply_rotary_emb(query, rotary_emb_vid).to(query_dtype)

        if rotary_emb_mel is not None:
            if get_sequence_parallel_state(): ### TODO:
                rotary_emb_mel = shrink_head(rotary_emb_mel, dim=2)

            key_mel = apply_rotary_emb(key_mel, rotary_emb_mel).to(query_dtype)
            

        # shard the head dimension
        if get_sequence_parallel_state():
            # b, h, s, d

            query = all_to_all_4D(query, scatter_dim=1, gather_dim=2)  #scatter_dim=2, gather_dim=1

            key = shrink_head(key, dim=1)
            value = shrink_head(value, dim=1)

            key_mel = all_to_all_4D(key_mel, scatter_dim=1, gather_dim=2)
            value_mel = all_to_all_4D(value_mel, scatter_dim=1, gather_dim=2)

        key = torch.cat([key, key_mel], 2)
        value = torch.cat([value, value_mel], 2)

        q = query.transpose(1, 2).to(query_dtype) # b s h d
        kv = torch.stack([key, value], dim=2).to(query_dtype) 
        kv = kv.transpose(1, 3).to(query_dtype)  # b s 3 h d

        hidden_states = flash_attn_kvpacked_func(
                            q,
                            kv,
                            softmax_scale=None,
                            causal=False
        )

        if get_sequence_parallel_state():
            hidden_states = all_to_all_4D(hidden_states,
                                          scatter_dim=1,
                                          gather_dim=2)
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.to(query_dtype)
        else:

            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.to(query_dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class CustomLiteLAProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections. add rms norm for query and key and apply RoPE"""

    def __init__(self):
        self.kernel_func = nn.ReLU(inplace=False)
        self.eps = 1e-15
        self.pad_val = 1.0

    def apply_rotary_emb(
        self,
        x: torch.Tensor,
        freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
        to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
        reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
        tensors contain rotary embeddings and are returned as real tensors.

        Args:
            x (`torch.Tensor`):
                Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
            freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
        """
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        rotary_freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        rotary_freqs_cis_cross: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        hidden_states_len = hidden_states.shape[1]

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)
        if encoder_hidden_states is not None:
            context_input_ndim = encoder_hidden_states.ndim
            if context_input_ndim == 4:
                batch_size, channel, height, width = encoder_hidden_states.shape
                encoder_hidden_states = encoder_hidden_states.view(
                    batch_size, channel, height * width
                ).transpose(1, 2)

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        dtype = hidden_states.dtype
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        has_encoder_hidden_state_proj = (
            hasattr(attn, "add_q_proj")
            and hasattr(attn, "add_k_proj")
            and hasattr(attn, "add_v_proj")
        )
        if encoder_hidden_states is not None and has_encoder_hidden_state_proj:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            # attention
            if not attn.is_cross_attention:
                query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
                key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
                value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)
            else:
                query = hidden_states
                key = encoder_hidden_states
                value = encoder_hidden_states

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.transpose(-1, -2).reshape(batch_size, attn.heads, head_dim, -1) # b, h, d, s
        key = (
            key.transpose(-1, -2)
            .reshape(batch_size, attn.heads, head_dim, -1)
            .transpose(-1, -2)
        ) # b, h, s, d
        value = value.transpose(-1, -2).reshape(batch_size, attn.heads, head_dim, -1) # b, h, d, s

        # RoPE需要 [B, H, S, D] 输入
        # 此时 query是 [B, H, D, S], 需要转成 [B, H, S, D] 才能应用RoPE
        query = query.permute(0, 1, 3, 2)  # [B, H, S, D]  (从 [B, H, D, S])

        # Apply query and key normalization if needed
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if rotary_freqs_cis is not None:
            
            if get_sequence_parallel_state(): ### TODO:
                rotary_freqs_cis = (shrink_head(rotary_freqs_cis[0], dim=0), shrink_head(rotary_freqs_cis[1], dim=0))

            query = self.apply_rotary_emb(query, rotary_freqs_cis)
            if not attn.is_cross_attention:
                key = self.apply_rotary_emb(key, rotary_freqs_cis)
            elif rotary_freqs_cis_cross is not None and has_encoder_hidden_state_proj:
                key = self.apply_rotary_emb(key, rotary_freqs_cis_cross)

        # 此时 query是 [B, H, S, D]，需要还原成 [B, H, D, S]
        query = query.permute(0, 1, 3, 2)  # [B, H, D, S]

        if attention_mask is not None:
            # attention_mask: [B, S] -> [B, 1, S, 1]
            attention_mask = attention_mask[:, None, :, None].to(
                key.dtype
            )  # [B, 1, S, 1]
            query = query * attention_mask.permute(
                0, 1, 3, 2
            )  # [B, H, S, D] * [B, 1, S, 1]
            if not attn.is_cross_attention:
                key = (
                    key * attention_mask
                )  # key: [B, h, S, D] 与 mask [B, 1, S, 1] 相乘
                value = value * attention_mask.permute(
                    0, 1, 3, 2
                )  # 如果 value 是 [B, h, D, S]，那么需调整mask以匹配S维度

        if (
            attn.is_cross_attention
            and encoder_attention_mask is not None
            and has_encoder_hidden_state_proj
        ):
            encoder_attention_mask = encoder_attention_mask[:, None, :, None].to(
                key.dtype
            )  # [B, 1, S_enc, 1]
            # key: [B, h, S_enc, D], value: [B, h, D, S_enc]
            key = key * encoder_attention_mask  # [B, h, S_enc, D] * [B, 1, S_enc, 1]
            value = value * encoder_attention_mask.permute(
                0, 1, 3, 2
            )  # [B, h, D, S_enc] * [B, 1, 1, S_enc]

        query = self.kernel_func(query)
        key = self.kernel_func(key)

        query, key, value = query.float(), key.float(), value.float()

        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=self.pad_val)

        if get_sequence_parallel_state():
            # b, h, d, s
            query = query.permute(0, 1, 3, 2) # b, h, s, d
            value = value.permute(0, 1, 3, 2) # b, h, s, d
            query = all_to_all_4D(query, scatter_dim=1, gather_dim=2)  #scatter_dim=2, gather_dim=1
            key = all_to_all_4D(key, scatter_dim=1, gather_dim=2)
            value = all_to_all_4D(value, scatter_dim=1, gather_dim=2)
            query = query.permute(0, 1, 3, 2) # b, h, d, s
            value = value.permute(0, 1, 3, 2) # b, h, d, s

        vk = torch.matmul(value, key)

        hidden_states = torch.matmul(vk, query)

        if hidden_states.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.float()

        hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + self.eps)


        if get_sequence_parallel_state():
            hidden_states = hidden_states.permute(0, 1, 3, 2) # b, h, s, d
            hidden_states = all_to_all_4D(hidden_states,
                                          scatter_dim=2,
                                          gather_dim=1)
            hidden_states = hidden_states.permute(0, 1, 3, 2) # b, h, d, s
            hidden_states = hidden_states.reshape(
                batch_size, attn.heads * head_dim, -1
            ).permute(0, 2, 1)
        else:
            hidden_states = hidden_states.view(
                batch_size, attn.heads * head_dim, -1
            ).permute(0, 2, 1)

        hidden_states = hidden_states.to(dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(dtype)

        # Split the attention outputs.
        if (
            encoder_hidden_states is not None
            and not attn.is_cross_attention
            and has_encoder_hidden_state_proj
        ):
            hidden_states, encoder_hidden_states = (
                hidden_states[:, :hidden_states_len],
                hidden_states[:, hidden_states_len:],
            )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if (
            encoder_hidden_states is not None
            and not attn.context_pre_only
            and not attn.is_cross_attention
            and hasattr(attn, "to_add_out")
        ):
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        if encoder_hidden_states is not None and context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if torch.get_autocast_gpu_dtype() == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return hidden_states, encoder_hidden_states


class CustomerAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def apply_rotary_emb(
        self,
        x: torch.Tensor,
        freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
        to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
        reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
        tensors contain rotary embeddings and are returned as real tensors.

        Args:
            x (`torch.Tensor`):
                Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
            freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
        """
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        hidden_states_vid: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        rotary_freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        rotary_freqs_cis_cross: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        rotary_freqs_cis_vid: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        key_vid = attn.add_k_proj(hidden_states_vid)
        value_vid = attn.add_v_proj(hidden_states_vid)
        

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key_vid = key_vid.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_vid = value_vid.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        if attn.norm_added_q is not None:
            query_vid = attn.norm_added_q(query_vid)
        if attn.norm_added_k is not None:
            key_vid = attn.norm_added_k(key_vid)

        # Apply RoPE if needed
        if rotary_freqs_cis is not None:

            if get_sequence_parallel_state(): ### TODO:
                rotary_freqs_cis = (shrink_head(rotary_freqs_cis[0], dim=0), shrink_head(rotary_freqs_cis[1], dim=0))
                rotary_freqs_cis_vid = (shrink_head(rotary_freqs_cis_vid[0], dim=0), shrink_head(rotary_freqs_cis_vid[1], dim=0))

            query = self.apply_rotary_emb(query, rotary_freqs_cis)
            key = self.apply_rotary_emb(key, rotary_freqs_cis_cross)
            key_vid = self.apply_rotary_emb(key_vid, rotary_freqs_cis_vid)

        attention_mask_vid = torch.ones([key_vid.shape[0], key_vid.shape[2] * nccl_info.sp_size]).to(key_vid)
        attention_mask_cond = torch.cat([encoder_attention_mask, attention_mask_vid], 1)
        # attention_mask: N x S1
        # encoder_attention_mask: N x S2
        # cross attention 整合attention_mask和encoder_attention_mask
        combined_mask = (
            attention_mask[:, :, None] * attention_mask_cond[:, None, :]
        )
        attention_mask = torch.where(combined_mask == 1, 0.0, -torch.inf)
        attention_mask = (
            attention_mask[:, None, :, :]
            .expand(-1, attn.heads, -1, -1)
            .to(query.dtype)
        )
        
        if get_sequence_parallel_state():
            # b, h, s, d

            query = all_to_all_4D(query, scatter_dim=1, gather_dim=2)  #scatter_dim=2, gather_dim=1
            attention_mask = all_to_all_4D(attention_mask, scatter_dim=1, gather_dim=2)
            key = shrink_head(key, dim=1)
            value = shrink_head(value, dim=1)
            key_vid = all_to_all_4D(key_vid, scatter_dim=1, gather_dim=2)  #scatter_dim=2, gather_dim=1
            value_vid = all_to_all_4D(value_vid, scatter_dim=1, gather_dim=2)  #scatter_dim=2, gather_dim=1

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        key = torch.cat([key, key_vid], 2)
        value = torch.cat([value, value_vid], 2)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        if get_sequence_parallel_state():
            hidden_states = all_to_all_4D(hidden_states,
                                          scatter_dim=2,
                                          gather_dim=1)
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            hidden_states = hidden_states.to(query.dtype)
        else:
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=torch.int64
        ).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

class Qwen2RotaryEmbeddingVid(nn.Module):
    def __init__(self, dim, patch_size: Tuple[int, int, int], max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.patch_size = patch_size
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=torch.int64
        ).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        
        # # x: [bs, num_channels, num_frames, height, width]
        batch_size, num_channels, num_frames, height, width = x.shape
        num_frames = num_frames * nccl_info.sp_size
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        cos_freqs = self.cos_cached[:ppf].view(ppf, -1)
        sin_freqs = self.sin_cached[:ppf].view(ppf, -1)
        return (cos_freqs, sin_freqs)


class T2IFinalLayer(nn.Module):
    """
    The final layer of Sana.
    """

    def __init__(self, hidden_size, patch_size=[16, 1], out_channels=256):
        super().__init__()
        self.norm_final = nn.RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size[0] * patch_size[1] * out_channels, bias=True
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size**0.5
        )
        self.out_channels = out_channels
        self.patch_size = patch_size

    def unpatchfy(
        self,
        hidden_states: torch.Tensor,
        width: int,
    ):
        # 4 unpatchify
        new_height, new_width = 1, hidden_states.size(1)
        hidden_states = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                new_height,
                new_width,
                self.patch_size[0],
                self.patch_size[1],
                self.out_channels,
            )
        ).contiguous()
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                self.out_channels,
                new_height * self.patch_size[0],
                new_width * self.patch_size[1],
            )
        ).contiguous()
        if width > new_width:
            output = torch.nn.functional.pad(
                output, (0, width - new_width, 0, 0), "constant", 0
            )
        elif width < new_width:
            output = output[:, :, :, :width]
        return output

    def forward(self, x, t, output_length):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        # unpatchify
        output = self.unpatchfy(x, output_length)
        return output


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        height=16,
        width=4096,
        patch_size=(16, 1),
        in_channels=8,
        embed_dim=1152,
        bias=True,
    ):
        super().__init__()
        patch_size_h, patch_size_w = patch_size
        self.early_conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels * 256,
                kernel_size=patch_size,
                stride=patch_size,
                padding=0,
                bias=bias,
            ),
            torch.nn.GroupNorm(
                num_groups=32, num_channels=in_channels * 256, eps=1e-6, affine=True
            ),
            nn.Conv2d(
                in_channels * 256,
                embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
        )
        self.patch_size = patch_size
        self.height, self.width = height // patch_size_h, width // patch_size_w
        self.base_size = self.width

    def forward(self, latent):
        # early convolutions, N x C x H x W -> N x 256 * sqrt(patch_size) x H/patch_size x W/patch_size
        latent = self.early_conv_layers(latent)
        latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return latent


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        self.in_features = in_features

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states

class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim)


    def apply_rotary_emb(self, hidden_states: torch.Tensor, freqs: torch.Tensor):
        x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
        x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
        return x_out.type_as(hidden_states)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = self.time_proj.weight.dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(timestep)
        timestep_proj = self.time_proj(self.act_fn(temb))

        if encoder_hidden_states is not None:
            encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image

class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
            )
            freqs.append(freq)
        self.freqs = torch.cat(freqs, dim=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        num_frames = num_frames * nccl_info.sp_size
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        self.freqs = self.freqs.to(hidden_states.device)
        freqs = self.freqs.split_with_sizes(
            [
                self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
                self.attention_head_dim // 6,
                self.attention_head_dim // 6,
            ],
            dim=1,
        )

        freqs_f = freqs[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_h = freqs[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_w = freqs[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        freqs = torch.cat([freqs_f, freqs_h, freqs_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs

class WanRotaryPosEmbed_mel_1D(nn.Module):
    def __init__(
        self, attention_head_dim: int, max_seq_len: int, theta: float = 1000000.0
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.max_seq_len = max_seq_len

        t_dim = attention_head_dim

        self.freqs = get_1d_rotary_pos_embed(
            t_dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, dim = hidden_states.shape
        num_frames = num_frames * nccl_info.sp_size

        self.freqs = self.freqs.to(hidden_states.device)

        freqs = self.freqs[:num_frames].view(1, 1, num_frames, -1)
        return freqs


class WanRotaryPosEmbed_vid_1D(nn.Module):
    def __init__(
        self, attention_head_dim: int, patch_size: Tuple[int, int, int], max_seq_len: int, theta: float = 10000.0
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.max_seq_len = max_seq_len
        self.patch_size = patch_size

        t_dim = attention_head_dim

        self.freqs = get_1d_rotary_pos_embed(
            t_dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float64
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        num_frames = num_frames * nccl_info.sp_size
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        self.freqs = self.freqs.to(hidden_states.device)

        freqs = self.freqs[:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1).reshape(1, 1, ppf * pph * ppw, -1)
        return freqs

class UniVerseTransformerBlock(nn.Module):
    def __init__(
        self,
        # wan params
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        context_pre_only: Optional[bool] = None,
        # ace params
        mel_dim: int = 2560,
        mel_num_heads: int = 20,
        use_adaln_single=True,
        cross_attention_dim=None,
        mlp_ratio=4.0,
        add_cross_attention=False,
        add_cross_attention_dim=None,
    ):
        super().__init__()

        ################################ wan block content start ################################
        # 1. Self-attention (with reference)
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=WanAttnProcessor2_0_self(),
        )

        # 2.1 Cross-attention (text)
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            context_pre_only=context_pre_only,
            processor=WanAttnProcessor2_0_cross1(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        ################################ wan block content end ##################################

        ################################ ace block content start ################################
        self.mel_norm1 = RMSNorm(mel_dim, elementwise_affine=False, eps=1e-6)
        self.mel_attn = Attention(
            query_dim=mel_dim,
            cross_attention_dim=cross_attention_dim,
            added_kv_proj_dim=mel_dim,
            dim_head=mel_dim // mel_num_heads,
            heads=mel_num_heads,
            out_dim=mel_dim,
            bias=True,
            qk_norm=None, #qk_norm, ##### TODO: add qk_norm in mel attention
            processor=CustomLiteLAProcessor2_0(),
        )

        self.add_cross_attention = add_cross_attention
        self.context_pre_only = context_pre_only

        if add_cross_attention and add_cross_attention_dim is not None:
            self.mel_cross_attn = Attention(
                query_dim=mel_dim,
                cross_attention_dim=add_cross_attention_dim,
                added_kv_proj_dim=add_cross_attention_dim,
                dim_head=mel_dim // mel_num_heads,
                heads=mel_num_heads,
                out_dim=mel_dim,
                added_proj_bias=True,
                context_pre_only=context_pre_only,
                bias=True,
                qk_norm=None, #qk_norm, ##### TODO: add qk_norm in mel attention
                processor=CustomerAttnProcessor2_0(),
            )

        self.mel_norm2 = RMSNorm(mel_dim, 1e-06, elementwise_affine=False)

        self.mel_ff = GLUMBConv(
            in_features=mel_dim,
            hidden_features=int(mel_dim * mlp_ratio),
            use_bias=(True, True, False),
            norm=(None, None, None),
            act=("silu", "silu", None),
        )
        self.use_adaln_single = use_adaln_single
        if use_adaln_single:
            self.mel_scale_shift_table = nn.Parameter(torch.randn(6, mel_dim) / dim**0.5)
        ################################ ace block content end ##################################

        projector_dim = mel_dim + dim

        self.mel_to_vid = nn.Sequential(
            nn.Linear(mel_dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, dim),
        )
        self.vid_to_mel = nn.Sequential(
            nn.Linear(dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, mel_dim),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(
        self,
        ## wan params
        hidden_states_vid: torch.Tensor,
        hidden_states_mel: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb_vid: torch.Tensor,
        rotary_emb_vid_audio: torch.Tensor,
        rotary_emb_mel: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,

        ## ace params
        encoder_hidden_states_mel: torch.FloatTensor = None,
        attention_mask_mel: torch.FloatTensor = None,
        encoder_attention_mask_mel: torch.FloatTensor = None,
        rotary_freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        rotary_freqs_cis_cross: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        rotary_freqs_cis_vid: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        temb_mel: torch.FloatTensor = None,
        num_frames: int = None,
        is_cfg: bool = False,
    ) -> torch.Tensor:

        ##### wan model scale_shift norm
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)
        batch_size, sequence_length_vid, inner_dim = hidden_states_vid.shape
        ############################################################################
        
        ##### ace model hidden states norm
        N = hidden_states_mel.shape[0]
        if self.use_adaln_single:
            shift_msa_mel, scale_msa_mel, gate_msa_mel, shift_mlp_mel, scale_mlp_mel, gate_mlp_mel = (
                self.mel_scale_shift_table[None] + temb_mel.reshape(N, 6, -1)
            ).chunk(6, dim=1)

        norm_hidden_states_mel = self.mel_norm1(hidden_states_mel)
        if self.use_adaln_single:
            norm_hidden_states_mel = norm_hidden_states_mel * (1 + scale_msa_mel) + shift_msa_mel
        ############################################################################

        # 1. Self-attention (with reference)

        ##### ace model self attention
        attn_output_mel, _ = self.mel_attn(
            hidden_states=norm_hidden_states_mel,
            attention_mask=attention_mask_mel,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            rotary_freqs_cis=rotary_freqs_cis,
            rotary_freqs_cis_cross=None,
        )
        if self.use_adaln_single:
            attn_output_mel = gate_msa_mel * attn_output_mel
        hidden_states_mel = attn_output_mel + hidden_states_mel
        ############################################################################
        hidden_states_mel_cond = self.mel_to_vid(hidden_states_mel.detach()) # NOTE

        ##### wan model self attention
        norm_hidden_states = (self.norm1(torch.cat([hidden_states_vid, hidden_states_mel_cond], 1).float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states_vid)
        norm_hidden_states_vid = norm_hidden_states[:, :sequence_length_vid]
        norm_hidden_states_mel_cond = norm_hidden_states[:, sequence_length_vid:]
        attn_output_vid = self.attn1(
            hidden_states=norm_hidden_states_vid,
            hidden_states_mel=norm_hidden_states_mel_cond, 
            rotary_emb_vid=rotary_emb_vid, 
            attention_mask=attention_mask,
            num_frames=num_frames,
        )
        hidden_states_vid = (hidden_states_vid.float() + attn_output_vid * gate_msa).type_as(hidden_states_vid)
        ############################################################################

        # 2.1. Cross-attention (text)

        ##### wan model cross attention
        norm_hidden_states = self.norm2(torch.cat([hidden_states_vid, hidden_states_mel_cond], 1).float()).type_as(hidden_states_vid)
        norm_hidden_states_vid = norm_hidden_states[:, :sequence_length_vid]
        norm_hidden_states_mel_cond = norm_hidden_states[:, sequence_length_vid:]
        if is_cfg:
            norm_hidden_states_mel_cond = torch.cat([torch.zeros_like(norm_hidden_states_mel_cond[:1]).to(norm_hidden_states_mel_cond), norm_hidden_states_mel_cond[1:]], 0)
        attn_output_vid = self.attn2(
            hidden_states=norm_hidden_states_vid, 
            hidden_states_mel=norm_hidden_states_mel_cond, 
            encoder_hidden_states=encoder_hidden_states, 
            rotary_emb_vid=rotary_emb_vid_audio, 
            rotary_emb_mel=rotary_emb_mel
        )
        hidden_states_vid = hidden_states_vid + attn_output_vid
        ############################################################################

        ##### ace model cross attention
        vid_to_mel_feat = norm_hidden_states_vid.detach().reshape(batch_size, num_frames, sequence_length_vid // num_frames, inner_dim)
        vid_to_mel_feat = self.pool(vid_to_mel_feat.reshape(batch_size*num_frames, sequence_length_vid // num_frames, inner_dim).transpose(1, 2))
        vid_to_mel_feat = vid_to_mel_feat.reshape(batch_size, num_frames, inner_dim, 1).squeeze(-1)
        norm_hidden_states_vid_cond = self.vid_to_mel(vid_to_mel_feat) # NOTE
        if is_cfg:
            norm_hidden_states_vid_cond = torch.cat([torch.zeros_like(norm_hidden_states_vid_cond[:1]).to(norm_hidden_states_vid_cond), norm_hidden_states_vid_cond[1:]], 0)
        attn_output_mel = self.mel_cross_attn(
            hidden_states=hidden_states_mel,
            hidden_states_vid=norm_hidden_states_vid_cond,
            attention_mask=attention_mask_mel,
            encoder_hidden_states=encoder_hidden_states_mel,
            encoder_attention_mask=encoder_attention_mask_mel,
            rotary_freqs_cis=rotary_freqs_cis,
            rotary_freqs_cis_cross=rotary_freqs_cis_cross,
            rotary_freqs_cis_vid=rotary_freqs_cis_vid,
        )
        hidden_states_mel = attn_output_mel + hidden_states_mel
        ############################################################################

        # 3. Feed-forward

        ##### wan model add norm and feed forward
        norm_hidden_states_vid = (self.norm3(hidden_states_vid.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states_vid)
        ff_output = self.ffn(norm_hidden_states_vid)
        hidden_states_vid = (hidden_states_vid.float() + ff_output.float() * c_gate_msa).type_as(hidden_states_vid)
        ############################################################################

        ##### ace model add norm and feed forward
        norm_hidden_states_mel = self.mel_norm2(hidden_states_mel)
        if self.use_adaln_single:
            norm_hidden_states_mel = norm_hidden_states_mel * (1 + scale_mlp_mel) + shift_mlp_mel

        # step 4: feed forward
        ff_output_mel = self.mel_ff(norm_hidden_states_mel)
        if self.use_adaln_single:
            ff_output_mel = gate_mlp_mel * ff_output_mel

        hidden_states_mel = hidden_states_mel + ff_output_mel
        ############################################################################

        return hidden_states_vid, hidden_states_mel

### TODO(DONE): 1. position embedding of mel should differ from that of vid; 2. norm of mel should apart from that of vid
class UniVerseTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "lyric_embs", "norm", "proj_in", "speaker_embedder", "genre_embedder"]
    _no_split_modules = ["UniVerseTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "mel_scale_shift_table", "norm1", "norm2", "norm3", "mel_norm1", "mel_norm2"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        mel_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        context_pre_only: Optional[bool] = None,
        rope_max_seq_len: int = 1024,
        rope_max_seq_len_text: int = 2048,

        mel_num_attention_heads: int = 20,
        mel_in_channels: Optional[int] = 8,
        mlp_ratio: float = 4.0,
        mel_out_channels: int = 8,
        max_position: int = 32768,
        rope_theta: float = 1000000.0,
        speaker_embedding_dim: int = 512,
        text_embedding_dim: int = 768,
        ssl_encoder_depths: List[int] = [8, 8],
        ssl_names: List[str] = ["mert", "m-hubert"],
        ssl_latent_dims: List[int] = [1024, 768],
        lyric_encoder_vocab_size: int = 6681,
        lyric_hidden_size: int = 1024,
        mel_patch_size: List[int] = [16, 1],
        max_height: int = 16,
        max_width: int = 4096,
        max_speech_token_num: int = 128
        
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.rope_mel = WanRotaryPosEmbed_mel_1D(attention_head_dim, rope_max_seq_len, theta=10000)
        self.rope_vid_audio = WanRotaryPosEmbed_vid_1D(attention_head_dim, patch_size, rope_max_seq_len, theta=10000)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        
        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
        )

        ###### ace blocks
        self.mel_out_channels = mel_out_channels
        self.max_position = max_position
        self.mel_patch_size = mel_patch_size
        self.max_speech_token_num = max_speech_token_num
        mel_inner_dim = mel_num_attention_heads * attention_head_dim

        # 3.1 Transformer blocks
        self.blocks = nn.ModuleList(
            [
                UniVerseTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim, context_pre_only, 
                    mel_dim=mel_inner_dim,
                    mel_num_heads=mel_num_attention_heads,
                    mlp_ratio=mlp_ratio,
                    add_cross_attention=True,
                    add_cross_attention_dim=mel_inner_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.rope_theta = rope_theta

        self.rotary_emb = Qwen2RotaryEmbedding(
            dim=attention_head_dim,
            max_position_embeddings=self.max_position,
            base=self.rope_theta,
        )
        self.rotary_emb_vid = Qwen2RotaryEmbeddingVid(
            dim=attention_head_dim,
            patch_size=patch_size,
            max_position_embeddings=self.max_position,
            base=self.rope_theta,
        )

        # 2. Define input layers
        self.mel_in_channels = mel_in_channels

        # 3. Define transformers blocks

        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=mel_inner_dim
        )
        self.t_block = nn.Sequential(
            nn.SiLU(), nn.Linear(mel_inner_dim, 6 * mel_inner_dim, bias=True)
        )

        # speaker
        self.speaker_embedder = nn.Linear(speaker_embedding_dim, mel_inner_dim)

        # genre
        self.genre_embedder = nn.Linear(text_embedding_dim, mel_inner_dim)

        # lyric
        self.lyric_embs = nn.Embedding(lyric_encoder_vocab_size, lyric_hidden_size)
        self.lyric_encoder = LyricEncoder(
            input_size=lyric_hidden_size, static_chunk_size=0
        )
        self.lyric_proj = nn.Linear(lyric_hidden_size, mel_inner_dim)
        self.lyric_embs.eval()
        self.lyric_encoder.eval()
        self.lyric_proj.eval() # NOTE

        projector_dim = 2 * mel_inner_dim

        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(mel_inner_dim, projector_dim),
                    nn.SiLU(),
                    nn.Linear(projector_dim, projector_dim),
                    nn.SiLU(),
                    nn.Linear(projector_dim, ssl_dim),
                )
                for ssl_dim in ssl_latent_dims
            ]
        )
                
        self.ssl_latent_dims = ssl_latent_dims
        self.ssl_encoder_depths = ssl_encoder_depths

        self.cosine_loss = torch.nn.CosineEmbeddingLoss(margin=0.0, reduction="mean")
        self.ssl_names = ssl_names

        self.proj_in = PatchEmbed(
            height=max_height,
            width=max_width,
            patch_size=mel_patch_size,
            in_channels=self.mel_in_channels,
            embed_dim=mel_inner_dim,
            bias=True,
        )

        self.final_layer = T2IFinalLayer(
            mel_inner_dim, patch_size=mel_patch_size, out_channels=mel_out_channels
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states_ref: torch.Tensor,
        hidden_states_vid: torch.Tensor,
        hidden_states_mel: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states_ambient: torch.Tensor,
        speech_token: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_ut5: Optional[torch.Tensor] = None,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_text_attention_mask: Optional[torch.Tensor] = None,
        ssl_hidden_states: Optional[List[torch.Tensor]] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        is_cfg: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )
                
        batch_size, num_channels, num_frames, height, width = hidden_states_vid.shape

        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb_vid = self.rope(hidden_states_vid)
        rotary_emb_vid_audio = self.rope_vid_audio(hidden_states_vid)
        rotary_freqs_cis_vid = self.rotary_emb_vid(hidden_states_vid, seq_len=num_frames * nccl_info.sp_size)

        if get_sequence_parallel_state():
            if nccl_info.rank_within_group == 0:
                hidden_states_vid = torch.cat([hidden_states_ref, hidden_states_vid[:, :, 1:]], 2).to(hidden_states_vid)
        else:
            hidden_states_vid = torch.cat([hidden_states_ref, hidden_states_vid[:, :, 1:]], 2).to(hidden_states_vid)
        hidden_states_vid = self.patch_embedding(hidden_states_vid)
        hidden_states_vid = hidden_states_vid.flatten(2).transpose(1, 2) # b, s, d

        ########## ace input condition encode
        encoder_hidden_states_ambient = self.genre_embedder(encoder_hidden_states_ambient)
        
        speech_token = speech_token[:, :self.max_speech_token_num]
        encoder_hidden_states_speech = self.lyric_embs(speech_token)
        encoder_hidden_states_speech, _mask = self.lyric_encoder(
            encoder_hidden_states_speech, torch.ones_like(speech_token).to(speech_token), decoding_chunk_size=1, num_decoding_left_chunks=-1
        )
        encoder_hidden_states_speech = self.lyric_proj(encoder_hidden_states_speech)

        encoder_hidden_states_audio = torch.cat([encoder_hidden_states_ambient, encoder_hidden_states_speech], dim=1)
        encoder_attention_mask_audio = torch.cat([audio_text_attention_mask.unsqueeze(1), _mask], dim=2).squeeze(1)

        ##### ace inference
        embedded_timestep = self.timestep_embedder(
            self.time_proj(timestep).to(dtype=hidden_states_mel.dtype)
        )
        temb = self.t_block(embedded_timestep)

        output_length = hidden_states_mel.shape[-1]
        hidden_states_mel = self.proj_in(hidden_states_mel)
        attention_mask_audio = torch.ones(hidden_states_mel.shape[0], hidden_states_mel.shape[1]).to(hidden_states_mel)

        inner_hidden_states = []

        rotary_freqs_cis = self.rotary_emb(
            hidden_states_mel, seq_len=hidden_states_mel.shape[1] * nccl_info.sp_size
        )
        encoder_rotary_freqs_cis = self.rotary_emb(
            encoder_hidden_states_audio, seq_len=encoder_hidden_states_audio.shape[1]
        )

        ########## wan input

        temb_vid, timestep_proj, encoder_hidden_states_ut5, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states_ut5, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))
        rotary_emb_mel = self.rope_mel(hidden_states_mel)
        
        encoder_hidden_states = encoder_hidden_states_ut5
        
        # 4. Transformer blocks
        for index_block, blocks in enumerate(self.blocks):
            
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states_vid, hidden_states_mel = self._gradient_checkpointing_func(
                    blocks, hidden_states_vid, hidden_states_mel, encoder_hidden_states, timestep_proj, rotary_emb_vid, rotary_emb_vid_audio, rotary_emb_mel, attention_mask,
                    encoder_hidden_states_audio, attention_mask_audio, encoder_attention_mask_audio, rotary_freqs_cis, encoder_rotary_freqs_cis, rotary_freqs_cis_vid, temb, post_patch_num_frames, is_cfg
                )
            else:
                hidden_states_vid, hidden_states_mel = blocks(
                    hidden_states_vid, hidden_states_mel, encoder_hidden_states, timestep_proj, rotary_emb_vid, rotary_emb_vid_audio, rotary_emb_mel, attention_mask,
                    encoder_hidden_states_audio, attention_mask_audio, encoder_attention_mask_audio, rotary_freqs_cis, encoder_rotary_freqs_cis, rotary_freqs_cis_vid, temb, post_patch_num_frames, is_cfg
                )

        mel_output = self.final_layer(hidden_states_mel, embedded_timestep, output_length)

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb_vid.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states_vid.device)
        scale = scale.to(hidden_states_vid.device)

        hidden_states_vid = (self.norm_out(hidden_states_vid.float()) * (1 + scale) + shift).type_as(hidden_states_vid)
        hidden_states_vid = self.proj_out(hidden_states_vid)

        hidden_states_vid = hidden_states_vid.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states_vid = hidden_states_vid.permute(0, 7, 1, 4, 2, 5, 3, 6)

        output = (hidden_states_vid.flatten(6, 7).flatten(4, 5).flatten(2, 3), mel_output)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, )

        return Transformer2DModelOutput(sample=output)
