# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Triton implementation of dense varlen (cu_seqlens) scaled dot-product attention
# for MLA prefill, following the numerics and masking patterns of
# `triton_prefill_attention.context_attention_fwd` and the online softmax
# structure used in `triton_unified_attention`.

from __future__ import annotations

import math
from typing import Optional, Union

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import RCP_LN2


def _assert_varlen_tensors_same_accelerator(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> None:
    """Require Q/K/V and cu_seqlens on the same CUDA or XPU device."""
    dev = q.device
    if dev.type not in ("cuda", "xpu"):
        raise NotImplementedError(
            "triton_flash_attn_varlen expects tensors on CUDA or XPU; "
            f"got device type {dev.type!r}"
        )
    for name, t in (
        ("k", k),
        ("v", v),
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_k", cu_seqlens_k),
    ):
        if t.device != dev:
            raise ValueError(
                f"{name} must be on the same device as q ({dev}), got {t.device}"
            )


@triton.jit
def _fwd_kernel_varlen(
    Q,
    K,
    V,
    sm_scale,
    cu_seqlens_q_ptr,
    cu_seqlens_k_ptr,
    Out,
    lse_ptr,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_lse_h,
    stride_lse_t,
    kv_group_num: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DQK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_LSE: tl.constexpr,
    D_QK: tl.constexpr,
    D_V: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    q_start = tl.load(cu_seqlens_q_ptr + cur_batch)
    q_end = tl.load(cu_seqlens_q_ptr + cur_batch + 1)
    sq = q_end - q_start

    k_start = tl.load(cu_seqlens_k_ptr + cur_batch)
    k_end = tl.load(cu_seqlens_k_ptr + cur_batch + 1)
    sk = k_end - k_start

    block_start_loc = BLOCK_M * start_m

    offs_n = tl.arange(0, BLOCK_N)
    offs_d_qk = tl.arange(0, BLOCK_DQK)
    offs_d_v = tl.arange(0, BLOCK_DV)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    mask_qk = offs_d_qk < D_QK
    mask_v = offs_d_v < D_V

    off_q = (
        (q_start + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d_qk[None, :]
    )
    q = tl.load(
        Q + off_q,
        mask=(offs_m[:, None] < sq) & (mask_qk[None, :]),
        other=0.0,
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < sq, 1, 0)
    end_n = sk
    if IS_CAUSAL:
        end_n = tl.minimum(end_n, (start_m + 1) * BLOCK_M)
    end_n_limit = block_mask * end_n

    for start_n in range(0, end_n_limit, BLOCK_N):
        pos_q = offs_m[:, None]
        pos_k = start_n + offs_n[None, :]

        mask = pos_k < sk
        if IS_CAUSAL:
            mask &= pos_q >= pos_k

        start_n = tl.multiple_of(start_n, BLOCK_N)

        k_offs = (
            (k_start + start_n + offs_n[None, :]) * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d_qk[:, None]
        )
        k = tl.load(
            K + k_offs,
            mask=(pos_k < sk) & (mask_qk[:, None]),
            other=0.0,
        )

        qk = tl.dot(q, k)
        qk = tl.where(mask, qk * sm_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v_offs = (
            (k_start + start_n + offs_n[:, None]) * stride_vbs
            + cur_kv_head * stride_vh
            + offs_d_v[None, :]
        )
        v = tl.load(
            V + v_offs,
            mask=((start_n + offs_n[:, None]) < sk) & (mask_v[None, :]),
            other=0.0,
        )
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]

    off_o = (
        (q_start + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d_v[None, :]
    )
    tl.store(
        Out + off_o,
        acc,
        mask=(offs_m[:, None] < sq) & (mask_v[None, :]),
    )

    if HAS_LSE:
        # Natural-log LSE: max(L) + log(sum exp(L - max)); m_i is max of L / ln2
        lse_row = m_i * 0.6931471805599453 + tl.log(l_i)
        lse_off = cur_head * stride_lse_h + (q_start + offs_m) * stride_lse_t
        tl.store(lse_ptr + lse_off, lse_row, mask=offs_m < sq)


def _get_block_m(dtype: torch.dtype) -> int:
    if dtype == torch.float32:
        return 32
    try:
        from vllm.platforms import current_platform

        if current_platform.is_cuda_alike() and current_platform.has_device_capability(80):
            return 128
    except Exception:
        pass
    return 64


def flash_attn_varlen_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    softmax_scale: Optional[float] = None,
    return_softmax_lse: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Varlen multi-head attention on packed Q, K, V (FlashAttention-style layout).

    Args:
        q: [total_q, num_q_heads, D_qk]
        k: [total_k, num_kv_heads, D_qk]
        v: [total_k, num_kv_heads, D_v]
        cu_seqlens_q: int32, shape [num_seqs + 1], cumulative lengths for Q
        cu_seqlens_k: int32, shape [num_seqs + 1], cumulative lengths for K
        max_seqlen_q: upper bound on per-sequence query length (grid sizing)
        max_seqlen_k: unused in kernel (API parity with FlashAttention)
        causal: if True, mask keys j with j <= query index i per sequence; requires
            equal sequence lengths for Q and K in each batch row.
        softmax_scale: typically 1/sqrt(D_qk); defaults to 1/sqrt(D_qk)
        return_softmax_lse: if True, also return LSE with shape [num_q_heads, total_q]

    Returns:
        out: [total_q, num_q_heads, D_v]
        softmax_lse (optional): float32 [num_q_heads, total_q]
    """
    del max_seqlen_k  # API compatibility with flash_attn_varlen_func

    _assert_varlen_tensors_same_accelerator(q, k, v, cu_seqlens_q, cu_seqlens_k)
    assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
    num_seqs = cu_seqlens_q.numel() - 1
    assert cu_seqlens_k.numel() == num_seqs + 1
    assert q.shape[1] % k.shape[1] == 0

    if causal:
        # FlashAttention causal varlen assumes aligned Q/K lengths per sequence.
        lens_ok = torch.all(
            (cu_seqlens_q[1:] - cu_seqlens_q[:-1])
            == (cu_seqlens_k[1:] - cu_seqlens_k[:-1])
        )
        assert bool(lens_ok.item()), (
            "Triton varlen causal attention requires equal Q and K lengths "
            "per sequence; use causal=False for context vs. query length mismatch."
        )

    total_q = int(cu_seqlens_q[-1].item())
    total_k = int(cu_seqlens_k[-1].item())
    assert q.shape[0] == total_q
    assert k.shape[0] == total_k
    assert v.shape[0] == total_k

    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    kv_group_num = num_q_heads // num_kv_heads

    d_qk = q.shape[2]
    d_v = v.shape[2]
    assert k.shape[2] == d_qk

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(float(d_qk))
    sm_scale = float(softmax_scale) * RCP_LN2

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = torch.empty(
        total_q,
        num_q_heads,
        d_v,
        device=q.device,
        dtype=q.dtype,
    )
    lse = None
    if return_softmax_lse:
        lse = torch.empty(
            num_q_heads,
            total_q,
            device=q.device,
            dtype=torch.float32,
        )

    BLOCK_M = _get_block_m(q.dtype)
    BLOCK_DQK = triton.next_power_of_2(d_qk)
    BLOCK_DV = triton.next_power_of_2(d_v)
    BLOCK_N = BLOCK_M

    grid = (num_seqs, num_q_heads, triton.cdiv(max_seqlen_q, BLOCK_M))
    num_warps = 4 if d_qk <= 64 else 8

    _fwd_kernel_varlen[grid](
        q,
        k,
        v,
        sm_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        out,
        lse if return_softmax_lse else out,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        lse.stride(0) if return_softmax_lse else 0,
        lse.stride(1) if return_softmax_lse else 0,
        kv_group_num=kv_group_num,
        BLOCK_M=BLOCK_M,
        BLOCK_DQK=BLOCK_DQK,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=causal,
        HAS_LSE=return_softmax_lse,
        D_QK=d_qk,
        D_V=d_v,
        num_warps=num_warps,
        num_stages=1,
    )

    if return_softmax_lse:
        return out, lse
    return out
