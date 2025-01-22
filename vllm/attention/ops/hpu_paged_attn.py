###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from vllm_hpu_extension import cache_ops, ops

# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512

def pa(attn, value, block_groups, block_mapping, block_scales, batch_size,
       matmul_av_op, batch2block_matmul_op, block2batch_matmul_op):
    #normalization
    attn_max = attn.amax(-1)
    missing_dims = attn_max.dim() - block_scales.dim()
    block_sum_attn = attn_max.mul(block_scales.reshape(-1, *[1 for _ in range(missing_dims)]))
    block_sum_attn = ops.block2batch(block_sum_attn, block_mapping, block2batch_matmul_op)
    block_sum_attn = ops.batch2block(block_sum_attn, block_mapping, batch2block_matmul_op)
    attn.sub_(block_sum_attn.unsqueeze(-1))
    attn_max.sub_(block_sum_attn)
    attn_max = attn_max.amax(0, keepdim=True)
    print("attn_max(mean, max, min) is ", torch.mean(attn_max).item(), torch.max(attn_max).item(), torch.min(attn_max).item())
    attn.sub_(attn_max.unsqueeze(-1))

    # end of norm
    attn = attn.exp()
    sums = attn.sum(dim=-1).unsqueeze(-1)
    block_sum = sums
    # Sum block's sums that belongs to the same sequeneces
    group_sums = ops.group_sum(sums, block_mapping, block_groups, 'mme')
    group_sums.add_(torch.finfo(group_sums.dtype).tiny)
    group_sums = torch.maximum(block_sum, group_sums)
    attn.div_(group_sums)
    attn = matmul_av_op(attn, value)
    return attn

def flat_pa(query, key_cache, value_cache, block_list, block_mapping,
            block_bias, block_scales, block_groups, scale, matmul_qk_op,
            matmul_av_op, batch2block_matmul_op, block2batch_matmul_op,
            keys_fetch_func, values_fetch_func):
    batch_size = query.size(0)
    q_heads = query.size(1)
    kv_heads = key_cache.size(2)

    query = ops.batch2block(scale * query, block_mapping, batch2block_matmul_op).unsqueeze(-2)
    key = keys_fetch_func(key_cache, block_list).transpose(1, 2)
    value = values_fetch_func(value_cache, block_list).transpose(1, 2)
    block_bias = block_bias.view(key.size(0), 1, 1, -1)
    if kv_heads != q_heads:
        block_bias = block_bias.unsqueeze(1)
        query = query.unflatten(1, (kv_heads, -1))
        key = key.unflatten(1, (kv_heads, 1))
        value = value.unflatten(1, (kv_heads, 1))
        key = key.transpose(3, 4)
    else:
        key = key.transpose(2, 3)

    attn = matmul_qk_op(query, key) + block_bias
    attn = pa(attn, value, block_groups, block_mapping, block_scales=block_scales,
                   batch_size=batch_size, matmul_av_op=matmul_av_op,
                   batch2block_matmul_op=batch2block_matmul_op, block2batch_matmul_op=block2batch_matmul_op)
    attn = ops.block2batch(attn, block_mapping, block2batch_matmul_op)
    attn = attn.squeeze(-2)
    if kv_heads != q_heads:
        attn = attn.flatten(1, 2)
    return attn

@dataclass
class HPUPagedAttentionMetadata:
    """Metadata for PagedAttention."""
    block_list: Optional[torch.Tensor]
    block_mapping: Optional[torch.Tensor]
    block_usage: Optional[torch.Tensor]
    block_indices: Optional[torch.Tensor]
    block_offsets: Optional[torch.Tensor]
    block_scales: Optional[torch.Tensor]
    block_groups: Optional[torch.Tensor]


class HPUPagedAttention:

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [64, 80, 96, 112, 128, 256]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]
        return key_cache, value_cache

    @staticmethod
    def write_to_paged_cache(key: torch.Tensor, value: torch.Tensor,
                             key_cache: torch.Tensor,
                             value_cache: torch.Tensor,
                             slot_mapping: torch.Tensor, kv_cache_dtype: str,
                             is_prompt: bool) -> None:
        cache_ops.reshape_and_cache(key, value, key_cache, value_cache,
                                    slot_mapping, kv_cache_dtype, is_prompt)

    @staticmethod
    def forward_decode(**kwargs) -> torch.Tensor:
        return flat_pa(**kwargs)

    @staticmethod
    def forward_prefix(**kwargs) -> torch.Tensor:
        return ops.prompt_attention_with_context(**kwargs)

    @staticmethod
    def swap_blocks(
        src_kv_cache: Tuple[torch.Tensor, torch.Tensor],
        dst_kv_cache: Tuple[torch.Tensor, torch.Tensor],
        src_to_dsts: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        cache_ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dsts)

        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        cache_ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dsts)

    @staticmethod
    def copy_blocks(
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        src_to_dsts: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        cache_ops.copy_blocks(key_caches, value_caches, src_to_dsts)
