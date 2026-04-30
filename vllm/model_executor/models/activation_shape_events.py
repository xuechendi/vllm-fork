# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.autograd.profiler import record_function

import vllm.envs as envs

_ENABLED: bool | None = None

_seen_shapes: set[str] = set()
_print_enabled: bool = True


def _is_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = envs.VLLM_ACTIVATION_SHAPE_EVENTS
    return _ENABLED


def record_activation_shape(name: str, tensor: torch.Tensor) -> None:
    if not _is_enabled():
        return
    shape_str = "x".join(str(d) for d in tensor.shape)
    tag = f"activation_shape:{name}:{shape_str}"

    if _print_enabled and tag not in _seen_shapes:
        _seen_shapes.add(tag)
        print(tag, flush=True)

    with record_function(tag):
        pass
