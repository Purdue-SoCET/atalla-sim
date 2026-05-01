"""Tile planner: compute tiling strategy per op and assign DRAM addresses.

Annotates each FX node with:
  node.meta['tile_config']  -- dict with tiling params per kernel type
  node.meta['dram_addr']    -- hex string of start address
  node.meta['dram_bytes']   -- allocation size in bytes
  node.meta['kernel_type']  -- one of the kernel builder names
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import reduce
import operator
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

    in_shape = get_node_shape(node.args[0])
    if in_shape and len(in_shape) >= 1:
        M = _prod(in_shape[:-1])
    else:
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


def _plan_gelu(node: Node) -> TileConfig:
    shape = get_node_shape(node.args[0])
    total = 1
    for d in (shape or [1]):
        total *= d
    width = min(total, VL)
    return TileConfig(
        kernel_type="gelu",
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


def _softmax_axis(node: Node, gm: GraphModule) -> int:
    """0-based axis index for softmax (default last dim)."""
    shape = get_node_shape(node.args[0])
    ndim = len(shape) if shape else 1
    if node.op == "call_module":
        mod = _get_module(gm, node.target)
        if isinstance(mod, nn.Softmax):
            dim = int(mod.dim)
        elif len(node.args) >= 2 and isinstance(node.args[1], int):
            dim = int(node.args[1])
        else:
            dim = int(node.kwargs.get("dim", -1))
    elif len(node.args) >= 2 and isinstance(node.args[1], int):
        dim = int(node.args[1])
    else:
        dim = int(node.kwargs.get("dim", -1))
    if dim < 0:
        dim += ndim
    return max(0, min(dim, ndim - 1))


def _plan_softmax(node: Node, gm: GraphModule) -> TileConfig:
    shape = get_node_shape(node.args[0])
    if not shape:
        return TileConfig(
            kernel_type="softmax",
            params=dict(num_rows=1, row_len=1),
        )
    dim = _softmax_axis(node, gm)
    row_len = int(shape[dim])
    num_rows = _prod(shape) // row_len
    return TileConfig(
        kernel_type="softmax",
        params=dict(num_rows=num_rows, row_len=row_len, softmax_dim=dim),
    )


def _plan_add(node: Node) -> TileConfig:
    shape = get_node_shape(node)
    total = 1
    for d in (shape or [1]):
        total *= d
    # 2048-wide FFN bias add: multi-tile C path may deviate in sim; set
    # ``ATALLA_FFN_ADD_NUMPY=0`` to force the AtallaC add image in validate.
    ffn_add_numpy = os.environ.get("ATALLA_FFN_ADD_NUMPY", "1") == "1"
    use_numpy = ffn_add_numpy and total > 1024
    return TileConfig(
        kernel_type="add",
        params=dict(total_elements=total, use_numpy=use_numpy),
    )


def _plan_mul(node: Node) -> TileConfig:
    shape = get_node_shape(node)
    total = 1
    for d in (shape or [1]):
        total *= d
    return TileConfig(
        kernel_type="mul",
        params=dict(total_elements=total),
    )


def _plan_sdpa(node: Node, gm: GraphModule) -> TileConfig:
    """Fused attention: Q,K,V (B, N, D) -> (B, N, D)."""
    if not node.args or not isinstance(node.args[0], Node):
        return TileConfig(
            kernel_type="atalla_sdpa",
            params=dict(B=1, N=1, D=1, use_flash=False, inv_sqrt_d=1.0),
        )
    qs = get_node_shape(node.args[0])
    if not qs or len(qs) < 2:
        return TileConfig(
            kernel_type="atalla_sdpa",
            params=dict(B=1, N=1, D=1, use_flash=False, inv_sqrt_d=1.0),
        )
    d_last = int(qs[-1])
    n_head = int(qs[-2])
    b = _prod(qs[:-2]) if len(qs) > 2 else 1
    # ATALLA_SDPA_FLASH=1 → flash_sdpa_n32d32.c (N=D=32, B=1); else NumPy ref in emitter.
    use_flash = (
        os.environ.get("ATALLA_SDPA_FLASH", "0") == "1"
        and b == 1
        and n_head == 32
        and d_last == 32
    )
    mod = _get_module(gm, node.target) if node.op == "call_module" else None
    inv = float(getattr(mod, "inv_sqrt_d", 1.0 / (d_last ** 0.5)))
    return TileConfig(
        kernel_type="atalla_sdpa",
        params=dict(
            B=b,
            N=n_head,
            D=d_last,
            use_flash=use_flash,
            inv_sqrt_d=inv,
        ),
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


def _prod(xs) -> int:
    return int(reduce(operator.mul, xs, 1))


def _plan_layernorm(node: Node, gm: GraphModule) -> TileConfig:
    """Layer norm over trailing dims ``normalized_shape``; M = product of leading dims."""
    inp_shape = get_node_shape(node.args[0])
    eps = 1e-5

    if node.op == "call_module":
        mod = _get_module(gm, node.target)
        ns = tuple(int(x) for x in mod.normalized_shape)
        eps = float(getattr(mod, "eps", 1e-5))
    else:
        ns_arg = node.args[1]
        if isinstance(ns_arg, (tuple, list)):
            ns = tuple(int(x) for x in ns_arg)
        elif isinstance(ns_arg, int):
            ns = (ns_arg,)
        else:
            ns = (inp_shape[-1],) if inp_shape else (1,)
        if "eps" in node.kwargs:
            eps = float(node.kwargs["eps"])
        elif len(node.args) > 4 and not isinstance(node.args[4], Node):
            eps = float(node.args[4])

    d_norm = _prod(ns)
    nd = len(ns)
    if inp_shape is None or len(inp_shape) < nd:
        m_groups = 1
    else:
        m_groups = _prod(inp_shape[:-nd])

    return TileConfig(
        kernel_type="layernorm",
        params=dict(M=m_groups, D=d_norm, eps=eps, normalized_dims=nd),
    )


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
            tc = _plan_softmax(node, gm)
        elif atalla_op == "add":
            tc = _plan_add(node)
        elif atalla_op == "matmul":
            tc = _plan_matmul(node)
        elif atalla_op == "mul":
            tc = _plan_mul(node)
        elif atalla_op == "atalla_sdpa":
            tc = _plan_sdpa(node, gm)
        elif atalla_op in ("flatten", "dropout", "transpose"):
            tc = _plan_flatten(node)
        elif atalla_op == "adaptive_avg_pool":
            tc = _plan_adaptive_avg_pool(node)
        elif atalla_op == "layernorm":
            tc = _plan_layernorm(node, gm)
        elif atalla_op == "gelu":
            tc = _plan_gelu(node)

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
