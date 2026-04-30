# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extract activation_shape events from a torch profiler JSON trace.

Usage:
    python -m vllm.model_executor.models.dump_activation_shapes <trace.json>

Reads the Chrome-format trace exported by torch.profiler and prints only the
``activation_shape:*`` entries, one per line:

    activation_shape:attn.qkv_proj.input:4096x8192
    activation_shape:attn.qkv_proj.output:4096x12288
    ...
"""

import json
import sys


def dump_activation_shapes(trace_path: str) -> list[str]:
    with open(trace_path) as f:
        data = json.load(f)

    events = data if isinstance(data, list) else data.get("traceEvents", [])

    shapes: list[str] = []
    for ev in events:
        name = ev.get("name", "")
        if name.startswith("activation_shape:"):
            shapes.append(name)
    return shapes


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <trace.json>", file=sys.stderr)
        sys.exit(1)

    shapes = dump_activation_shapes(sys.argv[1])
    for s in shapes:
        print(s)

    print(f"\nTotal activation_shape events: {len(shapes)}", file=sys.stderr)


if __name__ == "__main__":
    main()
