"""Tile planner: compute tiling strategy per op and assign DRAM addresses.

Annotates each FX node with:
  node.meta['tile_config']  -- dict with tiling params per kernel type
  node.meta['dram_addr']    -- hex string of start address
  node.meta['dram_bytes']   -- allocation size in bytes
  node.meta['kernel_type']  -- one of the kernel builder names
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node

from .fx_capture import get_node_shape

VL = 32
TILE = 32
TILE_BYTES = TILE * TILE * 2  # 32x32 bf16 = 2KB
SP_SLOTS = 32


def _align(v: int, a: int) -> int:
    return int(math.ceil(v / a) * a)


def _get_module(gm: GraphModule, target: str) -> nn.Module:
    parts = target.split(".")
    mod = gm
    for p in parts:
        mod = getattr(mod, p)
    return mod


@dataclass
class TileConfig:
    kernel_type: str
    params: Dict = field(default_factory=dict)


def _plan_conv(node: Node, gm: GraphModule) -> TileConfig:
    """Plan a Conv2d -> im2col + tiled GEMM."""
    mod = _get_module(gm, node.target) if node.op == "call_module" else None
    input_shape = get_node_shape(node.args[0])  # (N, C_in, H, W)

    if mod is not None and isinstance(mod, nn.Conv2d):
        C_in = mod.in_channels
        C_out = mod.out_channels
        R, S = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
        stride = mod.stride[0] if isinstance(mod.stride, tuple) else mod.stride
        pad = mod.padding[0] if isinstance(mod.padding, tuple) else mod.padding
        H_in = input_shape[2] if input_shape else 1
        W_in = input_shape[3] if input_shape else 1
    else:
        # F.conv2d call_function
        H_in = input_shape[2] if input_shape and len(input_shape) == 4 else 1
        W_in = input_shape[3] if input_shape and len(input_shape) == 4 else 1
        weight_shape = get_node_shape(node.args[1])
        C_out = weight_shape[0] if weight_shape else 1
        C_in = weight_shape[1] if weight_shape else 1
        R = weight_shape[2] if weight_shape and len(weight_shape) >= 3 else 1
        S = weight_shape[3] if weight_shape and len(weight_shape) >= 4 else 1
        stride = node.kwargs.get("stride", 1)
        if isinstance(stride, (tuple, list)):
            stride = stride[0]
        pad = node.kwargs.get("padding", 0)
        if isinstance(pad, (tuple, list)):
            pad = pad[0]

    Ho = (H_in + 2 * pad - R) // stride + 1
    Wo = (W_in + 2 * pad - S) // stride + 1
    K_flat = R * S * C_in
    M = Ho * Wo
    N = C_out
    K = K_flat

    return TileConfig(
        kernel_type="conv",
        params=dict(
            H=H_in, W=W_in, C_in=C_in, C_out=C_out,
            R=R, S=S, stride=stride, pad=pad,
            Ho=Ho, Wo=Wo, M=M, N=N, K=K,
            M_tiles=math.ceil(M / TILE),
            N_tiles=math.ceil(N / TILE),
            K_tiles=math.ceil(K / TILE),
            has_bias=(mod.bias is not None) if mod else (len(node.args) >= 3 and node.args[2] is not None),
        ),
    )


def _plan_linear(node: Node, gm: GraphModule) -> TileConfig:
    """Plan a Linear -> tiled GEMM."""
    mod = _get_module(gm, node.target) if node.op == "call_module" else None
    if mod is not None and isinstance(mod, nn.Linear):
        K = mod.in_features
        N = mod.out_features
        has_bias = mod.bias is not None
    else:
        weight_shape = get_node_shape(node.args[1]) if len(node.args) > 1 else None
        if weight_shape and len(weight_shape) == 2:
            N, K = weight_shape
        else:
            input_shape = get_node_shape(node.args[0])
            K = input_shape[-1] if input_shape else 1
            N = 1
        has_bias = len(node.args) >= 3 and node.args[2] is not None

    M = 1
    return TileConfig(
        kernel_type="fc",
        params=dict(
            M=M, N=N, K=K,
            M_tiles=math.ceil(M / TILE),
            N_tiles=math.ceil(N / TILE),
            K_tiles=math.ceil(K / TILE),
            has_bias=has_bias,
        ),
    )


def _plan_relu(node: Node) -> TileConfig:
    shape = get_node_shape(node.args[0])
    total = 1
    for d in (shape or [1]):
        total *= d
    width = min(total, VL)
    return TileConfig(
        kernel_type="relu",
        params=dict(total_elements=total, width=width),
    )


def _plan_maxpool(node: Node, gm: GraphModule) -> TileConfig:
    mod = _get_module(gm, node.target) if node.op == "call_module" else None
    input_shape = get_node_shape(node.args[0])

    if mod is not None and isinstance(mod, nn.MaxPool2d):
        pool = mod.kernel_size if isinstance(mod.kernel_size, int) else mod.kernel_size[0]
        stride = mod.stride if isinstance(mod.stride, int) else mod.stride[0]
    else:
        pool = node.args[1] if len(node.args) > 1 else node.kwargs.get("kernel_size", 3)
        stride = node.args[2] if len(node.args) > 2 else node.kwargs.get("stride", pool)
        if isinstance(pool, (tuple, list)):
            pool = pool[0]
        if isinstance(stride, (tuple, list)):
            stride = stride[0]

    C = input_shape[1] if input_shape and len(input_shape) == 4 else 1
    H = input_shape[2] if input_shape and len(input_shape) == 4 else 1
    W = input_shape[3] if input_shape and len(input_shape) == 4 else 1

    H_out = (H - pool) // stride + 1
    W_out = (W - pool) // stride + 1
    use_numpy = (W > VL)

    return TileConfig(
        kernel_type="maxpool",
        params=dict(
            H=H, W=W, channels=C,
            pool=pool, stride=stride,
            H_out=H_out, W_out=W_out,
            use_numpy=use_numpy,
        ),
    )


def _plan_softmax(node: Node) -> TileConfig:
    shape = get_node_shape(node.args[0])
    length = shape[-1] if shape else 1
    return TileConfig(
        kernel_type="softmax",
        params=dict(length=length),
    )


def _plan_add(node: Node) -> TileConfig:
    shape = get_node_shape(node)
    total = 1
    for d in (shape or [1]):
        total *= d
    return TileConfig(
        kernel_type="add",
        params=dict(total_elements=total),
    )


def _plan_matmul(node: Node) -> TileConfig:
    lhs_shape = get_node_shape(node.args[0])
    rhs_shape = get_node_shape(node.args[1])
    M = lhs_shape[-2] if lhs_shape and len(lhs_shape) >= 2 else 1
    K = lhs_shape[-1] if lhs_shape else 1
    N = rhs_shape[-1] if rhs_shape and len(rhs_shape) >= 2 else 1
    return TileConfig(
        kernel_type="matmul",
        params=dict(
            M=M, N=N, K=K,
            M_tiles=math.ceil(M / TILE),
            N_tiles=math.ceil(N / TILE),
            K_tiles=math.ceil(K / TILE),
        ),
    )


def _plan_flatten(node: Node) -> TileConfig:
    return TileConfig(kernel_type="flatten", params={})


def _plan_adaptive_avg_pool(node: Node) -> TileConfig:
    shape = get_node_shape(node.args[0])
    C = shape[1] if shape and len(shape) >= 2 else 1
    return TileConfig(kernel_type="adaptive_avg_pool", params=dict(channels=C))


def _output_bytes(node: Node) -> int:
    shape = get_node_shape(node)
    if shape is None:
        return TILE_BYTES
    total = 1
    for d in shape:
        total *= d
    return _align(total * 2, TILE_BYTES)  # bf16


def plan_tiles(gm: GraphModule) -> GraphModule:
    """Walk nodes topologically and annotate tile configs + DRAM addresses."""
    next_addr = 0x1000  # leave space for address tables at low memory

    for node in gm.graph.nodes:
        atalla_op = node.meta.get("atalla_op")

        tc: Optional[TileConfig] = None

        if atalla_op == "conv":
            tc = _plan_conv(node, gm)
        elif atalla_op == "linear":
            tc = _plan_linear(node, gm)
        elif atalla_op == "relu":
            tc = _plan_relu(node)
        elif atalla_op == "maxpool":
            tc = _plan_maxpool(node, gm)
        elif atalla_op == "softmax":
            tc = _plan_softmax(node)
        elif atalla_op == "add":
            tc = _plan_add(node)
        elif atalla_op == "matmul":
            tc = _plan_matmul(node)
        elif atalla_op == "mul":
            tc = TileConfig(kernel_type="mul", params={})
        elif atalla_op in ("flatten", "dropout"):
            tc = _plan_flatten(node)
        elif atalla_op == "adaptive_avg_pool":
            tc = _plan_adaptive_avg_pool(node)

        if tc is not None:
            node.meta["tile_config"] = tc
            node.meta["kernel_type"] = tc.kernel_type

        if node.op == "output":
            continue

        nbytes = _output_bytes(node)
        aligned = _align(next_addr, TILE_BYTES)
        node.meta["dram_addr"] = aligned
        node.meta["dram_bytes"] = nbytes
        next_addr = aligned + nbytes

    gm.graph.lint()
    gm.recompile()
    return gm
