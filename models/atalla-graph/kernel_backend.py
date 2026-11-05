"""Describe links from FX ops to compiler validation-kernel sources.

This module intentionally does not copy or rewrite compiler C files. The current
validation programs have standalone ``main`` functions and fixed SDMA geometry,
so a source binding is not yet a callable graph kernel. The manifest records the
selected optimized source and makes that ABI/layout gap explicit for every node.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import operator
import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node


@dataclass(frozen=True)
class KernelTemplate:
    operation: str
    source: str
    optimized: bool
    geometry: str
    limitation: str


TEMPLATES = {
    "matmul": KernelTemplate(
        "matmul", "gemm_tiled_pipelined.c", True, "4x4 compute; 8-wide matrix stride",
        "Standalone main/config ABI; expects packed matrices, not 32x32 graph tiles.",
    ),
    "add": KernelTemplate(
        "add", "add.c", False, "4 rows x 32 columns; fixed SDMA controls",
        "Standalone main/config ABI; contiguous validation rows differ from padded graph tiles.",
    ),
    "relu": KernelTemplate(
        "relu", "relu.c", False, "4 rows x 32 columns; fixed SDMA controls",
        "Standalone main/config ABI; contiguous validation rows differ from padded graph tiles.",
    ),
    "conv2d": KernelTemplate(
        "conv2d", "conv_pipelined.c", True, "fixed validation GEMM/im2col geometry",
        "Standalone main/config ABI; requires im2col input and packed weights.",
    ),
    "max_pool2d": KernelTemplate(
        "max_pool2d", "maxpool_2x2.c", True, "fixed 8x8 input to 4x4 output",
        "Standalone main/config ABI; fixed SDMA geometry is not the graph tile layout.",
    ),
}


def _shape(node: Node) -> list[int] | None:
    meta = node.meta.get("tensor_meta")
    return [int(d) for d in meta.shape] if meta is not None else None


def _classify(node: Node) -> str | None:
    target = node.target
    if target in (torch.matmul, torch.ops.aten.matmul.default):
        return "matmul"
    if target in (operator.add, torch.add, torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor):
        return "add"
    if target in (torch.relu, F.relu, torch.ops.aten.relu.default):
        return "relu"
    if target is F.conv2d:
        return "conv2d"
    if target is F.max_pool2d:
        return "max_pool2d"
    return None


def build_kernel_manifest(gm: GraphModule, compiler_root: Path) -> dict[str, Any]:
    """Return per-node bindings without materializing duplicate kernel sources."""
    bindings: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    for node in gm.graph.nodes:
        if node.op not in ("call_function", "call_module"):
            continue
        operation = _classify(node)
        if operation is None:
            unsupported.append({"node": node.name, "target": str(node.target), "reason": "no validation-kernel template mapping"})
            continue
        template = TEMPLATES[operation]
        source = compiler_root / "test_validation_kernels" / template.source
        if not source.is_file():
            raise FileNotFoundError(f"Compiler validation source not found: {source}")
        inputs = [arg for arg in node.all_input_nodes]
        bindings.append({
            "node": node.name,
            "operation": operation,
            "source": str(source),
            "optimized": template.optimized,
            "input_shapes": [_shape(arg) for arg in inputs],
            "output_shape": _shape(node),
            "template_geometry": template.geometry,
            "status": "adapter_required",
            "limitation": template.limitation,
        })
    return {
        "schema_version": 1,
        "compiler_root": str(compiler_root),
        "source_policy": "reference compiler sources; do not copy one source per graph node",
        "bindings": bindings,
        "unsupported": unsupported,
        "linkable": False,
        "link_blocker": "Validation templates are standalone programs and their fixed SDMA/layout geometry does not match runner graph storage.",
    }
