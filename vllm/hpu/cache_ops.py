###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import habana_frameworks.torch as htorch
import torch


def reshape_and_cache(key,
                      value,
                      key_cache,
                      value_cache,
                      slot_mapping,
                      dtype,
                      is_prompt=False):
    block_size = key_cache.size(1)
    slot_mapping = slot_mapping.flatten()
    indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    offsets = torch.fmod(slot_mapping, block_size)
    key_cache.index_put_((indices, offsets), key)
    value_cache.index_put_((indices, offsets), value)


def swap_blocks(src, dst, block_mapping):
    index_src = torch.zeros((1, ), dtype=torch.int32, device=src.device)
    index_dst = torch.zeros((1, ), dtype=torch.int32, device=dst.device)
    for src_idx, dst_idx in block_mapping.items():
        index_src[0] = src_idx
        index_dst[0] = dst_idx
        dst.index_put_([index_dst], src.index_select(0, index_src))
    if dst.device.type == 'hpu':
        htorch.core.mark_step()
        torch.hpu.synchronize()


def copy_blocks(key_caches, value_caches, block_mapping):
    index_src = torch.zeros((1, ),
                            dtype=torch.int32,
                            device=key_caches[0].device)
    index_dst = torch.zeros((1, ),
                            dtype=torch.int32,
                            device=key_caches[0].device)
    for src, dsts in block_mapping.items():
        index_src[0] = src
        for dst in dsts:
            index_dst[0] = dst
            for key_cache in key_caches:
                key_cache.index_copy_(0, index_dst,
                                      key_cache.index_select(0, index_src))
            for value_cache in value_caches:
                value_cache.index_copy_(0, index_dst,
                                        value_cache.index_select(0, index_src))
        if key_caches[0].device.type == 'hpu':
            htorch.core.mark_step()
