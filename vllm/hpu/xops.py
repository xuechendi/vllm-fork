###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import torch
from typing import Optional

import vllm.hpu.utils

from habana_frameworks.torch.hpex.kernels import FusedSDPA
def repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = kv.shape
    if n_rep == 1:
        return kv
    kv = kv[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return kv.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def gaudi_flash_attn_v1(q, k, v, mask,  softmax_mode, scale, q_block_size):
        """
        Gaudi version of Flash Attention V1 to support long sequence at prompt phase
        Causal mask is not supported in this optimization
        """
        q_len = q.size(-2)
        q_tiles = (q_len // q_block_size) if (q_len % q_block_size == 0) else math.ceil(q_len / q_block_size)
        q_padding = q_tiles * q_block_size - q_len
        q = F.pad(q, (0, 0, 0, q_padding), "constant", 0)
        if mask is not None:
            mask = F.pad(mask, (0, 0, 0, q_padding), "constant", -10000.0)
        attn_output = torch.zeros_like(q)

        for i in range(q_tiles):
            s, e = i * q_block_size, (i + 1) * q_block_size
            row_q = q[:, :, s:e, :]
            row_mask = mask[:, :, s:e, :]
            row_o = attn_output[:, :, s:e, :]
            row_o.fill_(FusedSDPA.apply(row_q, k, v, row_mask, 0.0, False, scale, softmax_mode))

        if q_padding != 0:
            attn_output = attn_output[:, :, :-q_padding, :]

        return attn_output

def prompt_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        p: float = 0.0,
        scale: Optional[float] = None,
        qk_matmul_op=torch.matmul,
        softmax_op=torch.softmax,
        kv_matmul_op=torch.matmul,
        valid_sequence_lengths: Optional[torch.Tensor] = None
) -> torch.Tensor:
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    #TODO: remove the handle for query_heads != kv_head for fusedsdpa after SW-195415 fix
    query_heads = query.size(1)
    kv_heads = key.size(1)
    
    if attn_bias is not None:
        if query_heads != kv_heads:
            query = query.unflatten(1, (kv_heads, -1))
            key = key.unflatten(1, (kv_heads, 1))
            value = value.unflatten(1, (kv_heads, 1))
            attn_bias = attn_bias.unsqueeze(2)
        attn_weights = qk_matmul_op(query * scale, key.transpose(-1, -2))
        if attn_bias is not None:
            attn_weights.add_(attn_bias)
        attn_weights = softmax_op(attn_weights, dim=-1)
        attn_weights = kv_matmul_op(attn_weights, value)
        if query_heads != kv_heads:
            attn_weights = attn_weights.flatten(1, 2)
    else:
        #TODO: remove the handle for query_heads != kv_head for fusedsdpa after SW-195415 fix
        if query_heads != kv_heads:
            key = repeat_kv(key, int(query_heads//kv_heads))
            value = repeat_kv(value, int(query_heads//kv_heads))
        softmax_mode = 'fast'
        recompute_mode = True
        #attn_weights = gaudi_flash_attn_v1(query, key, value, attn_bias, softmax_mode, scale, 8192)
        attn_weights = FusedSDPA.apply(query, key, value, None, 0.0, True, scale, softmax_mode, recompute_mode, valid_sequence_lengths, 'left')
    attn_weights = attn_weights.transpose(1, 2)
    return attn_weights
