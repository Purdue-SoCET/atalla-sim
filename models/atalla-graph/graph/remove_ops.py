"""Graph transforms: BN folding, dropout removal, flatten/reshape handling.

Operates on a traced GraphModule *after* fx_capture.normalize_ops has run.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.fx
from torch.fx import GraphModule, Node


def _get_module(gm: GraphModule, target: str) -> nn.Module:
    parts = target.split(".")
    mod = gm
    for p in parts:
        mod = getattr(mod, p)
    return mod


def _fold_bn_into_conv(gm: GraphModule) -> GraphModule:
    """Fuse BatchNorm2d into preceding Conv2d weights."""
    graph = gm.graph
    for node in list(graph.nodes):
        if node.op != "call_module":
            continue
        mod = _get_module(gm, node.target)
        if not isinstance(mod, nn.BatchNorm2d):
            continue

        conv_node = node.args[0] if node.args else None
        if conv_node is None or conv_node.op != "call_module":
            continue
        conv_mod = _get_module(gm, conv_node.target)
        if not isinstance(conv_mod, nn.Conv2d):
            continue

        bn = mod
        w = conv_mod.weight.data
        if conv_mod.bias is not None:
            b = conv_mod.bias.data
        else:
            b = torch.zeros(w.shape[0], dtype=w.dtype, device=w.device)

        mu = bn.running_mean
        var = bn.running_var
        gamma = bn.weight if bn.weight is not None else torch.ones_like(mu)
        beta = bn.bias if bn.bias is not None else torch.zeros_like(mu)
        eps = bn.eps

        inv_std = gamma / torch.sqrt(var + eps)
        conv_mod.weight.data = w * inv_std.view(-1, 1, 1, 1)
        conv_mod.bias = nn.Parameter(beta + (b - mu) * inv_std)

        node.replace_all_uses_with(conv_node)
        graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
    return gm


def _remove_dropout(gm: GraphModule) -> GraphModule:
    """Elide dropout nodes (inference mode)."""
    graph = gm.graph
    for node in list(graph.nodes):
        is_dropout = False
        if node.op == "call_module":
            mod = _get_module(gm, node.target)
            is_dropout = isinstance(mod, nn.Dropout)
        elif node.op == "call_function":
            import torch.nn.functional as F
            is_dropout = node.target in (F.dropout,)
        if is_dropout and node.args:
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
    return gm


def _remove_adaptive_avg_pool(gm: GraphModule) -> GraphModule:
    """Replace AdaptiveAvgPool2d(1,1) with flatten (it's just global avg pooling)."""
    graph = gm.graph
    for node in list(graph.nodes):
        if node.op != "call_module":
            continue
        mod = _get_module(gm, node.target)
        if not isinstance(mod, nn.AdaptiveAvgPool2d):
            continue
        out_size = mod.output_size
        if out_size in (1, (1, 1)):
            node.meta["atalla_op"] = "adaptive_avg_pool"
    gm.graph.lint()
    gm.recompile()
    return gm


def remove_ops(gm: GraphModule) -> GraphModule:
    """Run all graph cleanup passes."""
    gm = _fold_bn_into_conv(gm)
    gm = _remove_dropout(gm)
    gm = _remove_adaptive_avg_pool(gm)
    gm.graph.lint()
    gm.recompile()
    return gm
