# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import torch

from vllm.model_executor.kernels.linear import (  # noqa: E501
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .BlockScaledMMLinearKernel import Fp8BlockScaledMMLinearKernel

_TORCH_HAS_BLOCK_SCALED_MM = torch.__version__ >= "2.12" or "dev" in torch.__version__


def _convert_block_to_channel_scale(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert block-scaled FP8 weight to per-channel FP8 weight.

    Takes the max scale across the K dimension for each output channel,
    then rescales the weight so it remains valid under the per-channel scale.

    Args:
        weight: FP8 weight tensor of shape (N, K)
        weight_scale: block scale tensor of shape (N/block_n, K/block_k)

    Returns:
        (new_weight, channel_scale) where new_weight is (N, K)
        and channel_scale is (1, N) for row-wise _scaled_mm.
    """
    N, K = weight.shape
    block_n = N // weight_scale.shape[0]
    block_k = K // weight_scale.shape[1]

    # Per-channel scale: max across K blocks for each N block, then expand
    channel_scale_blocked = weight_scale.max(dim=1).values  # (N/block_n,)
    channel_scale = channel_scale_blocked.repeat_interleave(block_n)  # (N,)

    # Rescale weight: dequant with block scale, requant with channel scale
    weight_scale_expanded = weight_scale.repeat_interleave(
        block_n, dim=0
    ).repeat_interleave(block_k, dim=1)
    weight_dequant = weight.to(torch.float32) * weight_scale_expanded
    weight_requant = (weight_dequant / channel_scale.unsqueeze(1)).to(weight.dtype)

    new_weight = weight_requant.contiguous()
    new_channel_scale = channel_scale.reshape(1, N).contiguous()

    return new_weight, new_channel_scale


class XPUFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_xpu():
            return False, "XPUFP8ScaledMM only support on XPU"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if c.weight_quant_key not in {kFp8StaticChannelSym, kFp8StaticTensorSym}:
            return (
                False,
                "XPUFP8ScaledMM only support per-channel and per-tensor quantization",
            )
        if c.weight_quant_key.dtype not in {torch.float8_e5m2, torch.float8_e4m3fn}:
            return False, "XPUFP8ScaledMM only support FP8 weight dtype"
        return True, None

    def __init__(
        self, c: FP8ScaledMMLinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        assert self.can_implement(c)[0]
        assert self.is_supported()[0]
        self.config = c
        self.layer_param_names = layer_param_names

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        replace_parameter(layer, "weight", layer.weight.data.t())

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        weight_scale = layer.weight_scale
        return torch.ops._xpu_C.fp8_gemm_w8a16(x, weight, weight_scale, bias)

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        m = A.shape[0]
        A_2d = A.reshape(m, A.shape[-1])
        out = torch._scaled_mm(
            A_2d,
            B,
            scale_a=As,
            scale_b=Bs,
            bias=bias,
            out_dtype=out_dtype,
        )
        if type(out) is tuple and len(out) == 2:
            out = out[0]
        return out.reshape(*output_shape, out.shape[-1])


class _XPUFp8NativeBlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    """Native block FP8 kernel using torch._scaled_mm with block scales.
    Requires PyTorch >= 2.12.
    """

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_xpu():
            return False, "XPUFp8BlockScaledMM only support on XPU"
        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        out_dtype = self.config.out_dtype
        out = torch._scaled_mm(
            A,
            B.t().contiguous(),
            scale_a=As,
            scale_b=Bs.t().contiguous(),
            out_dtype=out_dtype,
        )
        if type(out) is tuple and len(out) == 2:
            out = out[0]
        return out


class _XPUFp8ChannelFallbackBlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    """Fallback kernel that converts block-scaled weights to per-channel
    at load time, then uses row-wise torch._scaled_mm.
    For PyTorch < 2.12 where block scales are not supported.
    """

    def __init__(self, config: FP8ScaledMMLinearLayerConfig) -> None:
        super().__init__(config)
        self.quant_fp8 = QuantFP8(
            static=False,
            group_shape=GroupShape.PER_TOKEN,
            num_token_padding=self.get_output_padding(),
            use_ue8m0=False,
        )

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_xpu():
            return False, "XPUFp8BlockScaledMM only support on XPU"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module):
        super().process_weights_after_loading(layer)

        params = self._get_layer_params(layer)
        weight = params.weight
        weight_scale = (
            params.weight_scale
            if params.weight_scale_inv is None
            else params.weight_scale_inv
        )
        scale_attr_name = (
            params.WEIGHT_SCALE
            if params.weight_scale_inv is None
            else params.WEIGHT_SCALE_INV
        )

        new_weight, new_scale = _convert_block_to_channel_scale(weight, weight_scale)
        # Pre-transpose weight to (K, N) to avoid per-call transpose
        replace_parameter(layer, params.WEIGHT, new_weight.data.t().contiguous())
        replace_parameter(layer, scale_attr_name, new_scale.data)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        params = self._get_layer_params(layer)
        weight = params.weight  # (K, N) after pre-transpose
        weight_scale = (
            params.weight_scale
            if params.weight_scale_inv is None
            else params.weight_scale_inv
        )
        input_scale = params.input_scale
        scale_up = params.input_scale_ub

        input_2d = x.view(-1, x.shape[-1])
        # weight is (K, N), so output dim is weight.shape[1]
        output_shape = [*x.shape[:-1], weight.shape[1]]

        q_input, input_scale = self.quant_fp8(
            input_2d, input_scale, scale_up, use_triton=False
        )

        output = self.apply_block_scaled_mm(
            A=q_input, B=weight, As=input_scale, Bs=weight_scale
        )

        if bias is not None:
            output = output + bias
        return output.to(dtype=self.config.out_dtype).view(*output_shape)

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        # B is already (K, N) from process_weights_after_loading
        out_dtype = self.config.out_dtype
        out = torch._scaled_mm(
            A,
            B,
            scale_a=As,
            scale_b=Bs,
            out_dtype=out_dtype,
        )
        if type(out) is tuple and len(out) == 2:
            out = out[0]
        return out


# Select the appropriate block kernel based on PyTorch version
XPUFp8BlockScaledMMKernel: type[Fp8BlockScaledMMLinearKernel]
if _TORCH_HAS_BLOCK_SCALED_MM:
    XPUFp8BlockScaledMMKernel = _XPUFp8NativeBlockScaledMMKernel
else:
    XPUFp8BlockScaledMMKernel = _XPUFp8ChannelFallbackBlockScaledMMKernel
