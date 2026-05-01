#!/usr/bin/env python3
"""
Emit graph_schedule.c by consuming an FX GraphModule that already has DRAM
addresses (node.meta["dram_addr"]) and tensor metadata attached.

**Partial coverage:** this generator emits *matmul_kernel*, *add_kernel*, *relu_*,
*conv_*, *mul_*, *maxpool_*, *avgpool_* for matching ``call_function``/``call_module``
nodes. It does **not** yet lower ``F.layer_norm``, ``F.gelu``, or fused
``AtallaSdpa`` to C calls, so a ViT graph schedule still omits those ops even
when ``run_graph`` validate emulates them — see ``graph.lower_modules`` (F.linear
is lowered to matmul+add so linears appear as matmul in the schedule).
"""

from __future__ import annotations

import argparse
import builtins
import operator
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node

from graph.memoryallocator import (
    TILE_HEIGHT,
    TILE_WIDTH,
    TILE_BYTES,
    VIEW_FUNCTIONS,
    VIEW_METHODS,
)

MAX_RANK = 8

MATMUL_TARGETS = {
    torch.matmul,
    torch.ops.aten.matmul.default,
}

ADD_TARGETS = {
    operator.add,
    torch.add,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.add_.Tensor,
}

RELU_TARGETS = {
    torch.relu,
    torch.nn.functional.relu,
    torch.ops.aten.relu.default,
}

MUL_TARGETS = {
    operator.mul,
    torch.mul,
    torch.ops.aten.mul.Tensor,
}

CONV_TARGETS = {
    F.conv2d,
}

AVGPOOL_TARGETS = {
    F.avg_pool2d,
}

# Functions like builtins.getattr don't need kernels
IGNORED_FUNCTIONS = {builtins.getattr}

# Regex that matches any non-alphanumeric character
_IDENT_RE = re.compile(r"\W+")



def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


#Remove non-alphanumeric chars and ensure it doesn't start with a digit
def _sanitize(name: str) -> str: 
    cleaned = _IDENT_RE.sub("_", name)
    if not cleaned:
        cleaned = "tensor"
    if cleaned[0].isdigit():
        cleaned = f"n_{cleaned}"
    return cleaned


# Detect whether a node is just a view into another tensor.
def _view_source(node: Node) -> Optional[Node]:
    if node.op == "call_method":
        method = node.target.strip("'")
        if method in VIEW_METHODS:
            arg = node.args[0]
            if isinstance(arg, Node):
                return arg
    if node.op == "call_function" and node.target in VIEW_FUNCTIONS:
        arg = node.args[0]
        if isinstance(arg, Node):
            return arg
    return None


# Pad shape/tiles arrays out to MAX_RANK with zeros.
def _pad(values: Sequence[int]) -> List[int]:
    out = list(values)
    if len(out) > MAX_RANK:
        raise ValueError(f"Rank {len(out)} exceeds MAX_RANK={MAX_RANK}")
    out.extend([0] * (MAX_RANK - len(out)))
    return out


# Grab tensor shape from FX metadata
def _shape_from_meta(node: Node) -> List[int]:
    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is None or tensor_meta.shape is None:
        raise ValueError(f"Missing tensor_meta.shape for node {node.name}")
    shape = [int(dim) for dim in tensor_meta.shape]
    if not shape:
        raise ValueError(f"Scalar tensor encountered at node {node.name}")
    return shape


# Compute tile counts per dimension
def _tiles_for_shape(shape: Sequence[int]) -> List[int]:
    dims = list(shape)
    if len(dims) == 1:
        return [_ceil_div(max(1, dims[0]), TILE_WIDTH)]
    if len(dims) == 2:
        return [
            _ceil_div(max(1, dims[0]), TILE_HEIGHT),
            _ceil_div(max(1, dims[1]), TILE_WIDTH),
        ]
    tiles = [max(1, dim) for dim in dims[:-2]]
    tiles.append(_ceil_div(max(1, dims[-2]), TILE_HEIGHT))
    tiles.append(_ceil_div(max(1, dims[-1]), TILE_WIDTH))
    return tiles


# Multiply tile counts to get total tile slots in a tensor
def _tile_count(values: Iterable[int]) -> int:
    product = 1
    for value in values:
        product *= max(1, value)
    return product


# Normalize stride/pad args into (height, width) tuples
def _as_pair(value: Sequence[int] | int) -> Tuple[int, int]:
    if isinstance(value, Sequence):
        if len(value) == 2:
            return int(value[0]), int(value[1])
        if len(value) == 1:
            return int(value[0]), int(value[0])
    return int(value), int(value)


# Read the spatial height/width from tensor metadata
def _spatial_dims(node: Node) -> Tuple[int, int]:
    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is None or tensor_meta.shape is None or len(tensor_meta.shape) < 2:
        raise ValueError(f"Node {node.name} lacks spatial dims in tensor_meta")
    height = int(tensor_meta.shape[-2])
    width = int(tensor_meta.shape[-1])
    return height, width


# Populate AdaptiveAvgPool2d output size (None for FX node)
def _resolve_adaptive_size(
    size: Sequence[Optional[int]] | int | None,
    input_h: int,
    input_w: int,
) -> Tuple[int, int]:
    if size is None:
        return input_h, input_w
    if isinstance(size, Sequence):
        out_h = size[0] if len(size) > 0 else None
        out_w = size[1] if len(size) > 1 else None
        out_h = input_h if out_h is None else int(out_h)
        out_w = input_w if out_w is None else int(out_w)
    else:
        if size is None:
            return input_h, input_w
        out_h = out_w = int(size)
    return max(1, out_h), max(1, out_w)


#format avgpool kernel invocations
def _format_avgpool_call(
    src: str,
    dst: str,
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    global_pool: int = 0,
) -> str:
    k_h, k_w = kernel
    s_h, s_w = stride
    p_h, p_w = padding
    return (
        f"    avgpool_kernel({src}, {dst}, {k_h}, {k_w}, {s_h}, "
        f"{s_w}, {p_h}, {p_w}, {global_pool});\n"
    )


#format maxpool kernel invocations
def _format_maxpool_call(
    src: str,
    dst: str,
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    ceil_mode: int,
) -> str:
    k_h, k_w = kernel
    s_h, s_w = stride
    p_h, p_w = padding
    d_h, d_w = dilation
    return (
        f"    maxpool_kernel({src}, {dst}, {k_h}, {k_w}, {s_h}, {s_w}, "
        f"{p_h}, {p_w}, {d_h}, {d_w}, {ceil_mode});\n"
    )


@dataclass
class TensorSpec:
    identifier: str
    shape: List[int]
    tiles_per_dim: List[int]
    base_addr: int
    count: int = field(init=False)
    tiles: List[int] = field(init=False)

    def __post_init__(self) -> None:
        self.count = max(1, _tile_count(self.tiles_per_dim))
        self.tiles = [self.base_addr + i * TILE_BYTES for i in range(self.count)]


# Build a TensorSpec (tiles + metadata) for a graph node
def _tensor_spec_for(node: Node) -> TensorSpec:
    dram_addr = node.meta.get("dram_addr")
    if dram_addr is None:
        raise ValueError(f"Node {node.name} is missing dram_addr metadata")
    base_addr = int(str(dram_addr), 16)

    tensor_meta = node.meta.get("tensor_meta")
    if tensor_meta is None:
        raise ValueError(f"Missing tensor_meta for node {node.name}")
    if getattr(tensor_meta, "dtype", torch.bfloat16) != torch.bfloat16:
        raise ValueError(f"Node {node.name} has unsupported dtype {tensor_meta.dtype}")

    shape = _shape_from_meta(node)
    tiles = _tiles_for_shape(shape)

    expected_bytes = int(node.meta.get("bytes", 0))
    actual_bytes = _tile_count(tiles) * TILE_BYTES
    if expected_bytes and expected_bytes != actual_bytes:
        raise ValueError(
            f"Node {node.name} reserve mismatch: meta bytes={expected_bytes}, "
            f"derived={actual_bytes}"
        )

    return TensorSpec(
        identifier=_sanitize(node.name),
        shape=shape,
        tiles_per_dim=tiles,
        base_addr=base_addr,
    )


# Write C structs for a tensor's tiles and global descriptor.
def _render_tensor(spec: TensorSpec) -> str:
    tiles_lines = ",\n    ".join(f"{{ .base_addr = 0x{addr:08X}u }}" for addr in spec.tiles)
    tile_block = (
        f"static TileDesc32 tensor_{spec.identifier}_tiles[{spec.count}] = {{\n"
        f"    {tiles_lines}\n"
        f"}};\n"
    )
    shape_init = ", ".join(str(x) for x in _pad(spec.shape))
    tiles_init = ", ".join(str(x) for x in _pad(spec.tiles_per_dim))
    gt_block = (
        f"static GlobalTile tensor_{spec.identifier}_gt = {{\n"
        f"    .rank = {len(spec.shape)},\n"
        f"    .shape = {{{shape_init}}},\n"
        f"    .tiles_per_dim = {{{tiles_init}}},\n"
        f"    .count = {spec.count},\n"
        f"    .tiles = tensor_{spec.identifier}_tiles,\n"
        f"}};\n"
    )
    return tile_block + "\n" + gt_block


# return the pointer string of a node's tensor
def _tensor_ref(node: Node, specs: Dict[Node, TensorSpec]) -> str:
    try:
        spec = specs[node]
    except KeyError as exc:
        raise KeyError(f"Missing tensor spec for node {node.name}") from exc
    return f"&tensor_{spec.identifier}_gt"


#  _tensor_ref but emites NULL when node is None (e.g: optional bias)
def _tensor_ref_optional(node: Optional[Node], specs: Dict[Node, TensorSpec]) -> str:
    if node is None:
        return "NULL"
    return _tensor_ref(node, specs)


#  Get a  positional argument from an FX node
def _get_node_arg(node: Node, position: int, *, fallback: Optional[str] = None) -> Node:
    if len(node.args) > position and isinstance(node.args[position], Node):
        return node.args[position]
    if fallback:
        candidate = node.kwargs.get(fallback)
        if isinstance(candidate, Node):
            return candidate
    raise ValueError(f"Node {node.name} lacks argument {position}")


#  Tensor vs numeric operands for mul nodes.
def _tensor_or_scalar_arg(
    arg: object, specs: Dict[Node, TensorSpec], *, node_name: str
) -> Tuple[Optional[str], Optional[float]]:
    if isinstance(arg, Node):
        return _tensor_ref(arg, specs), None
    if isinstance(arg, (int, float)):
        return None, float(arg)
    raise ValueError(f"Mul node {node_name} has unsupported operand type {type(arg)}")


'''
Find tensor×scalar mul nodes whose only user is an add, so we can put the
scalar into the add kernel (e.g., turn `add(out, mul(residual, 0.5))` into a
single `add_kernel(out, residual, dst, 0.5)` call).
'''
def _detect_scaled_args(gm: GraphModule) -> Dict[Node, Tuple[Node, float]]:
    scaled: Dict[Node, Tuple[Node, float]] = {}
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in MUL_TARGETS:
            continue
        if len(node.args) != 2:
            continue
        tensor_arg: Optional[Node] = None
        scalar: Optional[float] = None
        for arg in node.args:
            if isinstance(arg, Node):
                tensor_arg = arg
            elif isinstance(arg, (int, float)):
                scalar = float(arg)
        if tensor_arg is None or scalar is None:
            continue
        users = list(node.users.keys())
        if len(users) != 1:
            continue
        user = users[0]
        if user.op != "call_function" or user.target not in ADD_TARGETS:
            continue
        scaled[node] = (tensor_arg, scalar)
    return scaled


# Prepare lhs/rhs/alpha for add kernels
def _add_operands(
    node: Node,
    specs: Dict[Node, TensorSpec],
    scaled_mul: Dict[Node, Tuple[Node, float]],
) -> Tuple[str, str, float]:
    lhs = _get_node_arg(node, 0)
    rhs = _get_node_arg(node, 1, fallback="other")
    alpha = float(node.kwargs.get("alpha", 1.0))

    lhs_node = lhs
    rhs_node = rhs
    lhs_scaled = lhs_node in scaled_mul
    rhs_scaled = rhs_node in scaled_mul

    if lhs_scaled and rhs_scaled:
        raise ValueError(f"Add node {node.name} has two scaled operands; unsupported.")

    if lhs_scaled:
        scaled_source, factor = scaled_mul[lhs_node]
        lhs_node, rhs_node = rhs_node, scaled_source
        alpha *= factor
    elif rhs_scaled:
        scaled_source, factor = scaled_mul[rhs_node]
        rhs_node = scaled_source
        alpha *= factor

    lhs_ref = _tensor_ref(lhs_node, specs)
    rhs_ref = _tensor_ref(rhs_node, specs)
    return lhs_ref, rhs_ref, alpha


# Emit a kernel call string for an FX node 
def _render_call(
    gm: GraphModule,
    node: Node,
    specs: Dict[Node, TensorSpec],
    scaled_mul: Dict[Node, Tuple[Node, float]],
    attr_nodes: Dict[str, Node],
) -> Optional[str]:
    if node.op == "call_module":
        module = gm.get_submodule(node.target)
        if isinstance(module, torch.nn.Linear):
            input_node = _get_node_arg(node, 0)
            weight_attr = f"{node.target}.weight"
            weight_node = attr_nodes.get(weight_attr)
            if weight_node is None:
                raise ValueError(f"Missing get_attr node for {weight_attr}")
            dst = _tensor_ref(node, specs)
            matmul_call = (
                f"    matmul_kernel({_tensor_ref(input_node, specs)}, "
                f"{_tensor_ref(weight_node, specs)}, {dst});\n"
            )
            if module.bias is None:
                return matmul_call
            bias_attr = f"{node.target}.bias"
            bias_node = attr_nodes.get(bias_attr)
            if bias_node is None:
                raise ValueError(f"Missing get_attr node for {bias_attr}")
            add_call = (
                f"    add_kernel({dst}, {_tensor_ref(bias_node, specs)}, {dst}, 1.0f);\n"
            )
            return matmul_call + add_call
        if isinstance(module, torch.nn.MaxPool2d):
            input_node = _get_node_arg(node, 0)
            kernel = _as_pair(module.kernel_size)
            stride = module.stride if module.stride is not None else module.kernel_size
            stride = _as_pair(stride)
            padding = _as_pair(module.padding)
            dilation = _as_pair(module.dilation)
            ceil_flag = 1 if module.ceil_mode else 0
            return _format_maxpool_call(
                _tensor_ref(input_node, specs),
                _tensor_ref(node, specs),
                kernel,
                stride,
                padding,
                dilation,
                ceil_flag,
            )
        if isinstance(module, torch.nn.AdaptiveAvgPool2d):
            input_node = _get_node_arg(node, 0)
            input_h, input_w = _spatial_dims(input_node)
            out_h, out_w = _resolve_adaptive_size(module.output_size, input_h, input_w)
            stride_h = max(1, input_h // out_h)
            stride_w = max(1, input_w // out_w)
            kernel_h = input_h - (out_h - 1) * stride_h
            kernel_w = input_w - (out_w - 1) * stride_w
            if kernel_h <= 0 or kernel_w <= 0:
                raise ValueError(
                    f"AdaptiveAvgPool2d at {node.name} produced non-positive kernel size"
                )
            return _format_avgpool_call(
                _tensor_ref(input_node, specs),
                _tensor_ref(node, specs),
                (kernel_h, kernel_w),
                (stride_h, stride_w),
                (0, 0),
                global_pool=0,
            )
        # Other modules should have been lowered.
        return None

    if node.op == "call_method":
        method = node.target.strip("'")
        if method in VIEW_METHODS:
            return None
        if method == "mean":
            src = _tensor_ref(_get_node_arg(node, 0), specs)
            dst = _tensor_ref(node, specs)
            return _format_avgpool_call(src, dst, (0, 0), (0, 0), (0, 0), global_pool=1)
        return None

    if node.op != "call_function":
        return None
    if node.target in IGNORED_FUNCTIONS:
        return None
    if node.target in MATMUL_TARGETS:
        lhs = _tensor_ref(_get_node_arg(node, 0), specs)
        rhs = _tensor_ref(_get_node_arg(node, 1), specs)
        dst = _tensor_ref(node, specs)
        return f"    matmul_kernel({lhs}, {rhs}, {dst});\n"
    if node.target in MUL_TARGETS:
        if node in scaled_mul:
           #already folded into an add
            return None
        if len(node.args) >= 2:
            lhs_arg = node.args[0]
            rhs_arg = node.args[1]
        else:
            lhs_arg = node.kwargs.get("input")
            rhs_arg = node.kwargs.get("other")
        if lhs_arg is None or rhs_arg is None:
            raise ValueError(f"Mul node {node.name} is missing inputs")
        lhs_ref, lhs_scalar = _tensor_or_scalar_arg(lhs_arg, specs, node_name=node.name)
        rhs_ref, rhs_scalar = _tensor_or_scalar_arg(rhs_arg, specs, node_name=node.name)
        dst = _tensor_ref(node, specs)
        if lhs_scalar is not None and rhs_scalar is not None:
            raise ValueError(f"Mul node {node.name} has two scalar operands; unsupported.")
        if lhs_scalar is not None or rhs_scalar is not None:
            scalar = lhs_scalar if lhs_scalar is not None else rhs_scalar
            tensor_ref = rhs_ref if lhs_scalar is not None else lhs_ref
            if tensor_ref is None or scalar is None:
                raise ValueError(f"Mul node {node.name} lacks tensor operand for scalar multiply")
            return f"    mul_scalar_kernel({tensor_ref}, {dst}, {scalar:.6g}f);\n"
        if lhs_ref is None or rhs_ref is None:
            raise ValueError(f"Mul node {node.name} lacks tensor operands")
        return f"    mul_kernel({lhs_ref}, {rhs_ref}, {dst});\n"
    if node.target in ADD_TARGETS:
        lhs, rhs, alpha = _add_operands(node, specs, scaled_mul)
        dst = _tensor_ref(node, specs)
        return f"    add_kernel({lhs}, {rhs}, {dst}, {alpha:.6g}f);\n"
    if node.target in RELU_TARGETS:
        src = _tensor_ref(_get_node_arg(node, 0), specs)
        dst = _tensor_ref(node, specs)
        return f"    relu_kernel({src}, {dst}, NULL);\n"
    if node.target in CONV_TARGETS:
        input_node = _get_node_arg(node, 0)
        weight_node = node.args[1]
        bias_node = node.args[2] if len(node.args) > 2 else node.kwargs.get("bias")
        stride_h, stride_w = _as_pair(node.kwargs.get("stride", 1))
        pad_h, pad_w = _as_pair(node.kwargs.get("padding", 0))
        dil_h, dil_w = _as_pair(node.kwargs.get("dilation", 1))
        groups = int(node.kwargs.get("groups", 1))
        return (
            f"    conv_kernel({_tensor_ref(input_node, specs)}, "
            f"{_tensor_ref(weight_node, specs)}, {_tensor_ref_optional(bias_node, specs)}, "
            f"{_tensor_ref(node, specs)}, {stride_h}, {stride_w}, {pad_h}, {pad_w}, "
            f"{dil_h}, {dil_w}, {groups});\n"
        )
    if node.target in AVGPOOL_TARGETS:
        input_node = _get_node_arg(node, 0)
        kernel_h, kernel_w = _as_pair(node.kwargs.get("kernel_size", 1))
        stride_h, stride_w = _as_pair(node.kwargs.get("stride", (kernel_h, kernel_w)))
        pad_h, pad_w = _as_pair(node.kwargs.get("padding", 0))
        return _format_avgpool_call(
            _tensor_ref(input_node, specs),
            _tensor_ref(node, specs),
            (kernel_h, kernel_w),
            (stride_h, stride_w),
            (pad_h, pad_w),
            global_pool=0,
        )
    return None


# Collect TensorSpec entries and lookup map for every bf16 tensor node.
def _collect_tensor_specs(
    gm: GraphModule, scaled_mul: Dict[Node, Tuple[Node, float]]
) -> Tuple[List[TensorSpec], Dict[Node, TensorSpec]]:
    tensor_specs: List[TensorSpec] = []
    specs_by_node: Dict[Node, TensorSpec] = {}
    for node in gm.graph.nodes:
        if node.op == "output" or node in scaled_mul:
            continue
        tensor_meta = node.meta.get("tensor_meta")
        if tensor_meta is None:
            continue
        if getattr(tensor_meta, "dtype", torch.bfloat16) != torch.bfloat16:
            continue
        spec = _tensor_spec_for(node)
        tensor_specs.append(spec)
        specs_by_node[node] = spec
    return tensor_specs, specs_by_node


# Walk graph nodes in order and 'render' into calls
def _render_kernel_calls(
    gm: GraphModule,
    specs_by_node: Dict[Node, TensorSpec],
    scaled_mul: Dict[Node, Tuple[Node, float]],
    attr_nodes: Dict[str, Node],
) -> List[str]:
    calls: List[str] = []
    for node in gm.graph.nodes:
        if node in scaled_mul:
            continue
        rendered = _render_call(gm, node, specs_by_node, scaled_mul, attr_nodes)
        if rendered:
            calls.append(rendered)
    return calls

#main entry point 
def emit(gm: GraphModule) -> str:
    scaled_mul = _detect_scaled_args(gm)
    attr_nodes = {
        node.target: node
        for node in gm.graph.nodes
        if node.op == "get_attr"
    }

    tensor_specs, specs_by_node = _collect_tensor_specs(gm, scaled_mul)
    calls = _render_kernel_calls(gm, specs_by_node, scaled_mul, attr_nodes)

    pieces: List[str] = [
        "#include \"kernels/kernels.h\"\n\n"
    ]
    for spec in tensor_specs:
        pieces.append(_render_tensor(spec))
        pieces.append("\n")
    pieces.append("int run_graph(void) {\n")
    pieces.extend(calls)
    pieces.append("    return 0;\n}\n")
    return "".join(pieces)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate graph_schedule.c from a serialized FX GraphModule."
    )
    parser.add_argument(
        "graph",
        nargs="?",
        help="Path to GraphModule with memory metadata",
    )
    parser.add_argument(
        "--output",
        default="graph_schedule.c",
        help="Destination C file",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.graph:
        raise SystemExit("Provide a path to a GraphModule with metadata.")
    gm: GraphModule = torch.load(args.graph)
    Path(args.output).write_text(emit(gm))


if __name__ == "__main__":
    main()
