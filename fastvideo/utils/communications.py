# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from fastvideo.utils.parallel_states import nccl_info


def broadcast(input_: torch.Tensor):
    src = nccl_info.group_id * nccl_info.sp_size
    dist.broadcast(input_, src=src, group=nccl_info.group)


def _all_to_all_4D(input: torch.tensor,
                   scatter_idx: int = 2,
                   gather_idx: int = 1,
                   group=None) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = (input.reshape(bs, shard_seqlen, seq_world_size, shard_hc,
                                 hs).transpose(0, 2).contiguous())

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            torch.cuda.synchronize()
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(
            bs, seqlen, shard_hc, hs)

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (input.reshape(
            bs, seq_world_size, shard_seqlen, shard_hc,
            hs).transpose(0, 3).transpose(0, 1).contiguous().reshape(
                seq_world_size, shard_hc, shard_seqlen, bs, hs))

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            torch.cuda.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(
            bs, shard_seqlen, hc, hs)

        return output
    else:
        raise RuntimeError(
            "scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int,
        gather_idx: int,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return _all_to_all_4D(input, scatter_idx, gather_idx, group=group)

    @staticmethod
    def backward(ctx: Any,
                 *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (
            None,
            SeqAllToAll4D.apply(ctx.group, *grad_output, ctx.gather_idx,
                                ctx.scatter_idx),
            None,
            None,
        )


def all_to_all_4D(
    input_: torch.Tensor,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return SeqAllToAll4D.apply(nccl_info.group, input_, scatter_dim,
                               gather_dim)


def _all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    input_list = [
        t.contiguous()
        for t in torch.tensor_split(input_, world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()

def _all_to_all_list(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    device, dtype = input_.device, input_.dtype
    input_shape_list = [
        torch.tensor(t.shape, device=device)
        for t in torch.tensor_split(input_, world_size, scatter_dim)
    ]
    output_shape_list = [torch.empty_like(input_shape_list[idx]) for idx in range(world_size)]
    dist.all_to_all(output_shape_list, input_shape_list, group=group)

    input_list = [
        t.contiguous()
        for t in torch.tensor_split(input_, world_size, scatter_dim)
    ]
    output_list = [torch.empty(output_shape_list[idx].tolist(), device=device, dtype=dtype) for idx in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)

    return output_list

class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        output = _all_to_all(input_, ctx.world_size, process_group,
                             scatter_dim, gather_dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )


def all_to_all(
    input_: torch.Tensor,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return _AllToAll.apply(input_, nccl_info.group, scatter_dim, gather_dim)


class _AllGather(torch.autograd.Function):
    """All-gather communication with autograd support.

    Args:
        input_: input tensor
        dim: dimension along which to concatenate
    """

    @staticmethod
    def forward(ctx, input_, dim):
        ctx.dim = dim
        world_size = nccl_info.sp_size
        group = nccl_info.group
        input_size = list(input_.size())

        ctx.input_size = input_size[dim]

        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        input_ = input_.contiguous()
        dist.all_gather(tensor_list, input_, group=group)

        output = torch.cat(tensor_list, dim=dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        world_size = nccl_info.sp_size
        rank = nccl_info.rank_within_group
        dim = ctx.dim
        input_size = ctx.input_size

        sizes = [input_size] * world_size

        grad_input_list = torch.split(grad_output, sizes, dim=dim)
        grad_input = grad_input_list[rank]

        return grad_input, None


def all_gather(input_: torch.Tensor, dim: int = 1):
    """Performs an all-gather operation on the input tensor along the specified dimension.

    Args:
        input_ (torch.Tensor): Input tensor of shape [B, H, S, D].
        dim (int, optional): Dimension along which to concatenate. Defaults to 1.

    Returns:
        torch.Tensor: Output tensor after all-gather operation, concatenated along 'dim'.
    """
    return _AllGather.apply(input_, dim)


def prepare_sequence_parallel_data(hidden_states, encoder_hidden_states,
                                   attention_mask, encoder_attention_mask):
    if nccl_info.sp_size == 1:
        return (
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
        )

    def prepare(hidden_states, encoder_hidden_states, attention_mask,
                encoder_attention_mask):
        hidden_states = all_to_all(hidden_states, scatter_dim=2, gather_dim=0)
        encoder_hidden_states = all_to_all(encoder_hidden_states,
                                           scatter_dim=1,
                                           gather_dim=0)
        attention_mask = all_to_all(attention_mask,
                                    scatter_dim=1,
                                    gather_dim=0)
        encoder_attention_mask = all_to_all(encoder_attention_mask,
                                            scatter_dim=1,
                                            gather_dim=0)
        return (
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
        )

    sp_size = nccl_info.sp_size
    frame = hidden_states.shape[2]
    assert frame % sp_size == 0, "frame should be a multiple of sp_size"

    (
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        encoder_attention_mask,
    ) = prepare(
        hidden_states,
        encoder_hidden_states.repeat(1, sp_size, 1),
        attention_mask.repeat(1, sp_size, 1, 1),
        encoder_attention_mask.repeat(1, sp_size),
    )

    return hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask


def sp_parallel_dataloader_wrapper(dataloader, device, train_batch_size,
                                   sp_size, train_sp_batch_size):
    while True:
        for data_item in dataloader:
            latents, cond, attn_mask, cond_mask = data_item
            latents = latents.to(device)
            cond = cond.to(device)
            attn_mask = attn_mask.to(device)
            cond_mask = cond_mask.to(device)
            frame = latents.shape[2]
            if frame == 1:
                yield latents, cond, attn_mask, cond_mask
            else:
                latents, cond, attn_mask, cond_mask = prepare_sequence_parallel_data(
                    latents, cond, attn_mask, cond_mask)
                assert (
                    train_batch_size * sp_size >= train_sp_batch_size
                ), "train_batch_size * sp_size should be greater than train_sp_batch_size"
                for iter in range(train_batch_size * sp_size //
                                  train_sp_batch_size):
                    st_idx = iter * train_sp_batch_size
                    ed_idx = (iter + 1) * train_sp_batch_size
                    encoder_hidden_states = cond[st_idx:ed_idx]
                    attention_mask = attn_mask[st_idx:ed_idx]
                    encoder_attention_mask = cond_mask[st_idx:ed_idx]
                    yield (
                        latents[st_idx:ed_idx],
                        encoder_hidden_states,
                        attention_mask,
                        encoder_attention_mask,
                    )

def prepare_sequence_parallel_data_hcvm(vid, ref, encoder_hidden_states, encoder_attention_mask, 
    audio, audio_ori, camera, pose, ref_pose, seg, image_rotary_emb_list):
    if nccl_info.sp_size == 1:
        input_dict = {}
        input_dict['vid'] = vid
        input_dict['ref'] = ref
        input_dict['encoder_hidden_states'] = encoder_hidden_states
        input_dict['encoder_attention_mask'] = encoder_attention_mask
        if not audio is None:
            input_dict['audio'] = audio
        if not audio_ori is None:
            input_dict['audio_ori'] = audio_ori
        
        for ire in range(len(image_rotary_emb_list)):
            input_dict['image_rotary_emb_{}'.format(ire)] = image_rotary_emb_list[ire]
        if not camera is None:
            input_dict['camera'] = camera
        if not pose is None:
            input_dict['pose'] = pose
            input_dict['ref_pose'] = ref_pose
        if not seg is None:
            input_dict['seg'] = seg
        return input_dict

    def prepare(vid, ref, encoder_hidden_states, encoder_attention_mask, 
        audio, audio_ori, camera, pose, ref_pose, seg, image_rotary_emb_list):
        input_dict = {}
        vid = all_to_all(vid, scatter_dim=2, gather_dim=0)
        ref = all_to_all(ref, scatter_dim=2, gather_dim=0)
        for ire in range(len(image_rotary_emb_list)):
            input_dict['image_rotary_emb_{}'.format(ire)] = all_to_all(image_rotary_emb_list[ire], scatter_dim=1, gather_dim=0)
        input_dict['vid'] = vid
        input_dict['ref'] = ref
        if not pose is None:
            pose = all_to_all(pose, scatter_dim=2, gather_dim=0)
            ref_pose = all_to_all(ref_pose, scatter_dim=2, gather_dim=0)
            input_dict['pose'] = pose
            input_dict['ref_pose'] = ref_pose
        if not seg is None:
            seg = all_to_all(seg, scatter_dim=2, gather_dim=0)
            input_dict['seg'] = seg
        if not camera is None:
            camera = all_to_all(camera, scatter_dim=1, gather_dim=0)
            input_dict['camera'] = camera

        if not audio is None:
            audio = all_to_all(audio, scatter_dim=1, gather_dim=0)
            input_dict['audio'] = audio

        if not audio_ori is None:
            # audio_ori = all_to_all(audio_ori, scatter_dim=1, gather_dim=0)
            input_dict['audio_ori'] = audio_ori

        encoder_hidden_states = all_to_all(encoder_hidden_states,
                                           scatter_dim=1,
                                           gather_dim=0)
        encoder_attention_mask = all_to_all(encoder_attention_mask,
                                            scatter_dim=1,
                                            gather_dim=0)
        input_dict['encoder_hidden_states'] = encoder_hidden_states
        input_dict['encoder_attention_mask'] = encoder_attention_mask
        return input_dict

    sp_size = nccl_info.sp_size
    frame = vid.shape[2]
    assert frame % sp_size == 0, "frame should be a multiple of sp_size"

    image_rotary_emb_list_to_prepare = []
    for ire in range(len(image_rotary_emb_list)):
        image_rotary_emb_list_to_prepare.append(image_rotary_emb_list[ire].repeat(1, sp_size, 1))

    input_dict = prepare(
        vid, 
        ref.repeat(1, 1, sp_size, 1, 1), 
        encoder_hidden_states.repeat(1, sp_size, 1), 
        encoder_attention_mask.repeat(1, sp_size), 
        audio.repeat(1, sp_size, 1, 1), 
        audio_ori, 
        camera, pose, 
        ref_pose.repeat(1, 1, sp_size, 1, 1) if not ref_pose is None else ref_pose, 
        seg,
        image_rotary_emb_list_to_prepare,
    )

    return input_dict

def sp_parallel_dataloader_wrapper_hcvm(dataloader, device, train_batch_size,
                                   sp_size, train_sp_batch_size):
    while True:
        for data_item in dataloader:
            # latents, cond, cond_mask = data_item

            vid = data_item['vid'] # b,c,f,h,w
            ref = data_item['reference'] # b,c,1,h,w
            encoder_hidden_states = data_item['prompt_embeds'] # b,1,d
            encoder_attention_mask = data_item['prompt_attention_masks'] # b,1

            image_rotary_emb_list = []
            ire_count = 0
            for key in data_item:
                if "image_rotary_emb" in key:
                    ire_count += 1
            for ire in range(ire_count):
                image_rotary_emb_list.append(data_item['image_rotary_emb_{}'.format(ire)].unsqueeze(0).to(device))
            
            vid = vid.to(device)
            ref = ref.to(device)
            encoder_hidden_states = encoder_hidden_states.to(device)
            encoder_attention_mask = encoder_attention_mask.to(device)

            if "audio" in data_item:
                audio = data_item['audio'] # b,t,c,d
                audio = audio.to(device)
            else:
                audio = None

            if "audio_ori" in data_item:
                audio_ori = data_item['audio_ori'] # b,t,c,d
                audio_ori = audio_ori.to(device)
            else:
                audio_ori = None

            if "camera" in data_item:
                camera = data_item['camera'] # b,t,d
                camera = camera.to(device)
            else:
                camera = None
            if "flow" in data_item:
                flow = data_item['flow'] # # b,t-1,d
                flow = flow.to(device)
            else:
                flow = None
            if "pose" in data_item:
                pose = data_item['pose'] # b,c,f,h,w
                ref_pose = data_item['ref_pose'] # b,c,1,h,w
                pose = pose.to(device)
                ref_pose = ref_pose.to(device)
            else:
                pose = None; ref_pose = None
            if "seg" in data_item:
                seg = data_item['seg'] # b,c,f,h,w
                seg = seg.to(device)
            else:
                seg = None

            frame = vid.shape[2]
            if frame == 1:
                input_dict = {}
                input_dict['vid'] = vid
                input_dict['ref'] = ref
                input_dict['encoder_hidden_states'] = encoder_hidden_states
                input_dict['encoder_attention_mask'] = encoder_attention_mask
                for ire in range(ire_count):
                    input_dict['image_rotary_emb_{}'.format(ire)] = image_rotary_emb_list[ire]

                if not camera is None:
                    input_dict['camera'] = camera
                if not audio is None:
                    input_dict['audio'] = audio
                if not audio_ori is None:
                    input_dict['audio_ori'] = audio_ori
                if not pose is None:
                    input_dict['pose'] = pose
                    input_dict['ref_pose'] = ref_pose
                if not seg is None:
                    input_dict['seg'] = seg
                if not flow is None:
                    input_dict['flow'] = flow

                yield input_dict
            else:
                input_dict = prepare_sequence_parallel_data_hcvm(
                    vid, ref, encoder_hidden_states, encoder_attention_mask, audio, audio_ori, camera, pose, ref_pose, seg, image_rotary_emb_list)
                assert (
                    train_batch_size * sp_size >= train_sp_batch_size
                ), "train_batch_size * sp_size should be greater than train_sp_batch_size"
                for iter in range(train_batch_size * sp_size //
                                  train_sp_batch_size):
                    st_idx = iter * train_sp_batch_size
                    ed_idx = (iter + 1) * train_sp_batch_size
                    
                    input_dict_sp = {}
                    for key in input_dict:
                        input_dict_sp[key] = input_dict[key][st_idx:ed_idx]
                    yield input_dict_sp