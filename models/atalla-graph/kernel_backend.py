"""Build a capability-aware binding manifest for graph kernels."""
from __future__ import annotations

import builtins
import json
import operator
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node

_VIEW_METHODS = {"reshape", "flatten", "permute", "transpose", "view"}


def _shape(node: Node) -> list[int] | None:
    meta = node.meta.get("tensor_meta")
    return [int(dim) for dim in meta.shape] if meta is not None else None


def _classify(gm: GraphModule, node: Node) -> str | None:
    target = node.target
    if target in (torch.matmul, torch.mm, torch.bmm, torch.ops.aten.matmul.default):
        return "matmul"
    if target in (operator.add, torch.add, torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor):
        return "add"
    if target in (operator.mul, torch.mul, torch.ops.aten.mul.Tensor):
        if len(node.args) >= 2 and not isinstance(node.args[1], Node):
            return "mul_scalar"
        if len(node.args) >= 2 and not isinstance(node.args[0], Node):
            return "mul_scalar"
        return "mul"
    if target in (torch.relu, F.relu, torch.ops.aten.relu.default):
        return "relu"
    if target is F.conv2d:
        return "conv2d"
    if target is torch.flatten:
        return "flatten"
    if target is F.max_pool2d:
        return "max_pool2d"
    if target is F.avg_pool2d:
        return "avg_pool2d"
    if node.op == "call_module":
        module = gm.get_submodule(str(target))
        if isinstance(module, torch.nn.MaxPool2d):
            return "max_pool2d"
        if isinstance(module, torch.nn.AdaptiveAvgPool2d):
            return "avg_pool2d"
    if node.op == "call_function" and target is operator.getitem:
        return "view"
    if node.op == "call_method":
        method = str(target).strip("'")
        if method in _VIEW_METHODS:
            return "view"
        if method == "mean":
            return "avg_pool2d"
    if node.op == "call_function" and target is builtins.getattr:
        return "view"
    return None


def _inputs_layout_compatible(inputs: list[Node]) -> bool:
    for node in inputs:
        if node.op == "call_method" and str(node.target).strip("'") in _VIEW_METHODS and not node.meta.get("materialized_view"):
            return False
        if node.op == "call_function" and node.target is operator.getitem:
            return False
    return True


def _shape_compatible(operation: str, inputs: list[list[int] | None], output: list[int] | None) -> bool:
    if output is None or not inputs:
        return False
    if operation == "conv2d":
        return (len(inputs) >= 2 and inputs[0] is not None and inputs[1] is not None
                and len(inputs[0]) == len(inputs[1]) == len(output) == 4
                and inputs[0][0] == output[0] and inputs[1][0] == output[1])
    if operation == "max_pool2d":
        return inputs[0] is not None and len(inputs[0]) == len(output) == 4 and inputs[0][:2] == output[:2]
    if operation == "flatten":
        return (inputs[0] is not None and len(inputs[0]) == 4 and len(output) == 2
                and inputs[0][0] == output[0]
                and inputs[0][1] * inputs[0][2] * inputs[0][3] == output[1])
    if operation == "matmul":
        return (len(inputs) >= 2 and inputs[0] is not None and inputs[1] is not None
                and len(inputs[0]) == len(inputs[1]) == len(output) == 2
                and inputs[0][1] == inputs[1][0]
                and output == [inputs[0][0], inputs[1][1]])
    if inputs[0] != output:
        return False
    if operation in ("relu", "mul_scalar"):
        return True
    if operation == "mul":
        return len(inputs) >= 2 and inputs[1] == output
    if operation == "add":
        if len(inputs) < 2:
            return False
        rhs = inputs[1]
        return rhs == output or (
            rhs is not None and len(rhs) == 1 and len(output) > 1 and rhs[0] == output[-1]
        )
    return False


def _parameters_compatible(gm: GraphModule, node: Node, operation: str) -> bool:
    if operation == "max_pool2d":
        if node.op == "call_module":
            module = gm.get_submodule(str(node.target))
            return (module.dilation == 1 or module.dilation == (1, 1)) and not module.ceil_mode
        dilation = node.kwargs.get("dilation", node.args[4] if len(node.args) > 4 else 1)
        ceil_mode = node.kwargs.get("ceil_mode", node.args[5] if len(node.args) > 5 else False)
        return (dilation == 1 or dilation == (1, 1)) and not ceil_mode
    if operation == "conv2d":
        groups = int(node.kwargs.get("groups", 1))
        if groups < 1:
            return False
        bias = node.args[2] if len(node.args) > 2 else node.kwargs.get("bias")
        weight_shape = _shape(node.args[1])
        return bias is None or groups == 1 or (weight_shape is not None and (weight_shape[0] // groups) % 32 == 0)
    if operation == "flatten":
        start = node.args[1] if len(node.args) > 1 else node.kwargs.get("start_dim", 0)
        end = node.args[2] if len(node.args) > 2 else node.kwargs.get("end_dim", -1)
        return start == 1 and end == -1
    return True


def build_kernel_manifest(
    gm: GraphModule, kernel_root: Path, compiler_root: Path
) -> dict[str, Any]:
    """Bind FX nodes to shared library sources and validation references."""
    capabilities_path = kernel_root / "capabilities.json"
    capabilities = json.loads(capabilities_path.read_text())
    operation_table = capabilities["operations"]
    bindings: list[dict[str, Any]] = []
    unsupported: list[dict[str, str]] = []
    library_sources: set[str] = set()

    for node in gm.graph.nodes:
        if node.op not in ("call_function", "call_module", "call_method"):
            continue
        operation = _classify(gm, node)
        if operation == "view":
            continue
        if operation is None or operation not in operation_table:
            unsupported.append({
                "node": node.name,
                "target": str(node.target),
                "reason": "no graph-kernel capability entry",
            })
            continue
        capability = operation_table[operation]
        source = capability.get("source")
        source_exists = bool(source and (kernel_root / source).is_file())
        dependencies = capability.get("dependencies", [])
        implemented = bool(capability["implemented"] and source_exists and all((kernel_root / name).is_file() for name in dependencies))
        input_nodes = list(node.all_input_nodes)
        input_shapes = [_shape(arg) for arg in input_nodes]
        output_shape = _shape(node)
        layout_compatible = _inputs_layout_compatible(input_nodes)
        storage_compatible = all(arg.meta.get("storage_layout", 0) == 0 for arg in input_nodes)
        if operation == "matmul":
            storage_compatible = len(input_nodes) >= 2 and input_nodes[0].meta.get("storage_layout", 0) == 0 and input_nodes[1].meta.get("storage_layout") == 1
        elif operation == "conv2d":
            storage_compatible = len(input_nodes) >= 2 and input_nodes[0].meta.get("storage_layout", 0) == 0 and input_nodes[1].meta.get("storage_layout") == 2 and bool(node.meta.get("gather_addr"))
        parameters_compatible = _parameters_compatible(gm, node, operation)
        if implemented:
            library_sources.add(str(source))
            library_sources.update(capability.get("dependencies", []))
        reference = capability.get("reference")
        reference_exists = bool(reference and (compiler_root.parent / reference).is_file())
        bindings.append({
            "node": node.name,
            "operation": operation,
            "abi_symbol": capability["abi_symbol"],
            "input_shapes": input_shapes,
            "output_shape": output_shape,
            "library_source": source if implemented else None,
            "validation_reference": reference if reference_exists else None,
            "implemented": implemented,
            "shape_compatible": _shape_compatible(operation, input_shapes, output_shape),
            "layout_compatible": layout_compatible,
            "parameters_compatible": parameters_compatible,
            "storage_compatible": storage_compatible,
            "supports": capability.get("supports"),
        })

    kernels_supported = not unsupported and all(
        item["implemented"] and item["shape_compatible"] and item["layout_compatible"] and item["parameters_compatible"] and item["storage_compatible"]
        for item in bindings
    )
    if not kernels_supported:
        kernel_blocker = "At least one graph operation is unsupported or incompatible with its library kernel."
    else:
        kernel_blocker = None
    return {
        "schema_version": capabilities["schema_version"],
        "tile_layout": capabilities["tile_layout"],
        "library_root": str(kernel_root),
        "compiler_root": str(compiler_root),
        "source_policy": "one shared library implementation per operation; graph nodes carry descriptors",
        "library_sources": sorted(library_sources),
        "bindings": bindings,
        "unsupported": unsupported,
        "kernel_coverage": kernels_supported,
        "linkable": False,
        "link_blocker": kernel_blocker or "Schedule emitted; run build_model.py to compile and link its ELF.",
    }
