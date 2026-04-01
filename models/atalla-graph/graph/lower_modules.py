import operator
from typing import Optional

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node

from model.efficientformerv2 import DropPath


def _pair(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


# Search the FX graph for a get_attr node that already loads `target`.
def _find_attr_node(gm: GraphModule, target: str) -> Optional[Node]:
    for node in gm.graph.nodes:
        if node.op == "get_attr" and node.target == target:
            return node
    return None


# Insert a get_attr node for `target` right before `before`, if one is missing.
def _ensure_attr_node(gm: GraphModule, before: Node, target: str) -> Node:
    found = _find_attr_node(gm, target)
    if found is not None:
        return found

    with gm.graph.inserting_before(before):
        return gm.graph.get_attr(target)


# Turn module calls (Linear/Conv/etc.) into primitive call_function ops.
def lower_linear_modules(gm: GraphModule) -> GraphModule:
    modules = dict(gm.named_modules())

    for node in list(gm.graph.nodes):
        if node.op != "call_module":
            continue

        submod = modules.get(node.target)
        if submod is None:
            continue

        if isinstance(submod, torch.nn.Linear):
            if len(node.args) != 1:
                raise ValueError("Expected single input for nn.Linear")
            input_node = node.args[0]

            weight_target = f"{node.target}.weight"
            weight_node = _ensure_attr_node(gm, node, weight_target)
            with gm.graph.inserting_before(node):
                weight_T = gm.graph.call_method("transpose", args=(weight_node, -1, -2))
            with gm.graph.inserting_before(node):
                matmul_node = gm.graph.call_function(
                    torch.matmul, args=(input_node, weight_T)
                )

            current = matmul_node
            if submod.bias is not None:
                bias_target = f"{node.target}.bias"
                bias_node = _ensure_attr_node(gm, node, bias_target)
                with gm.graph.inserting_before(node):
                    current = gm.graph.call_function(
                        operator.add, args=(current, bias_node)
                    )

            node.replace_all_uses_with(current)
            gm.graph.erase_node(node)
        elif isinstance(submod, torch.nn.Conv2d):
            if len(node.args) != 1:
                raise ValueError("Expected single input for nn.Conv2d")
            input_node = node.args[0]
            weight_target = f"{node.target}.weight"
            weight_node = _ensure_attr_node(gm, node, weight_target)
            bias_node = None
            if submod.bias is not None:
                bias_target = f"{node.target}.bias"
                bias_node = _ensure_attr_node(gm, node, bias_target)
            stride = _pair(submod.stride)
            padding = _pair(submod.padding)
            dilation = _pair(submod.dilation)
            with gm.graph.inserting_before(node):
                conv_node = gm.graph.call_function(
                    F.conv2d,
                    args=(input_node, weight_node, bias_node),
                    kwargs={
                        "stride": stride,
                        "padding": padding,
                        "dilation": dilation,
                        "groups": submod.groups,
                    },
                )
            node.replace_all_uses_with(conv_node)
            gm.graph.erase_node(node)
        elif isinstance(submod, torch.nn.BatchNorm2d):
            if len(node.args) != 1:
                raise ValueError("Expected single input for nn.BatchNorm2d")
            input_node = node.args[0]
            running_mean = _ensure_attr_node(gm, node, f"{node.target}.running_mean")
            running_var = _ensure_attr_node(gm, node, f"{node.target}.running_var")
            weight_node = (
                _ensure_attr_node(gm, node, f"{node.target}.weight") if submod.affine else None
            )
            bias_node = (
                _ensure_attr_node(gm, node, f"{node.target}.bias") if submod.affine else None
            )
            with gm.graph.inserting_before(node):
                bn_node = gm.graph.call_function(
                    F.batch_norm,
                    args=(input_node, running_mean, running_var, weight_node, bias_node),
                    kwargs={
                        "training": submod.training,
                        "momentum": submod.momentum,
                        "eps": submod.eps,
                    },
                )
            node.replace_all_uses_with(bn_node)
            gm.graph.erase_node(node)
        elif isinstance(submod, torch.nn.LayerNorm):
            if len(node.args) != 1:
                raise ValueError("Expected single input for nn.LayerNorm")
            input_node = node.args[0]
            weight_node = (
                _ensure_attr_node(gm, node, f"{node.target}.weight") if submod.elementwise_affine else None
            )
            bias_node = (
                _ensure_attr_node(gm, node, f"{node.target}.bias") if submod.elementwise_affine else None
            )
            with gm.graph.inserting_before(node):
                ln_node = gm.graph.call_function(
                    F.layer_norm,
                    args=(input_node, submod.normalized_shape, weight_node, bias_node),
                    kwargs={"eps": submod.eps},
                )
            node.replace_all_uses_with(ln_node)
            gm.graph.erase_node(node)
        elif isinstance(submod, torch.nn.GELU):
            with gm.graph.inserting_before(node):
                gelu_node = gm.graph.call_function(
                    F.gelu,
                    args=node.args,
                    kwargs={"approximate": submod.approximate},
                )
            node.replace_all_uses_with(gelu_node)
            gm.graph.erase_node(node)
        elif isinstance(submod, torch.nn.ReLU):
            if len(node.args) != 1:
                raise ValueError("Expected single input for nn.ReLU")
            with gm.graph.inserting_before(node):
                relu_node = gm.graph.call_function(
                    F.relu, args=(node.args[0],), kwargs={"inplace": submod.inplace}
                )
            node.replace_all_uses_with(relu_node)
            gm.graph.erase_node(node)
        elif isinstance(submod, torch.nn.Dropout):
            if not node.args:
                raise ValueError("Dropout without input argument")
            node.replace_all_uses_with(node.args[0])
            gm.graph.erase_node(node)
        elif isinstance(submod, DropPath):
            if not node.args:
                raise ValueError("DropPath without input argument")
            input_node = node.args[0]
            keep_prob = 1.0 - float(submod.drop_prob)
            if keep_prob != 1.0:
                with gm.graph.inserting_before(node):
                    scaled = gm.graph.call_function(
                        operator.mul, args=(input_node, keep_prob)
                    )
                node.replace_all_uses_with(scaled)
            else:
                node.replace_all_uses_with(input_node)
            gm.graph.erase_node(node)
        elif isinstance(submod, torch.nn.AvgPool2d):
            if not node.args:
                raise ValueError("AvgPool2d without input argument")
            kwargs = {
                "kernel_size": submod.kernel_size,
                "stride": submod.stride,
                "padding": submod.padding,
                "ceil_mode": submod.ceil_mode,
                "count_include_pad": submod.count_include_pad,
                "divisor_override": submod.divisor_override,
            }
            with gm.graph.inserting_before(node):
                avg_node = gm.graph.call_function(
                    F.avg_pool2d,
                    args=(node.args[0],),
                    kwargs=kwargs,
                )
            node.replace_all_uses_with(avg_node)
            gm.graph.erase_node(node)
        elif isinstance(submod, torch.nn.Upsample):
            if len(node.args) != 1:
                raise ValueError("Upsample without input argument")
            kwargs = {
                "size": submod.size,
                "scale_factor": submod.scale_factor,
                "mode": submod.mode,
                "align_corners": submod.align_corners,
                "recompute_scale_factor": None,
            }
            with gm.graph.inserting_before(node):
                up_node = gm.graph.call_function(
                    F.interpolate,
                    args=(node.args[0],),
                    kwargs=kwargs,
                )
            node.replace_all_uses_with(up_node)
            gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
    return gm
