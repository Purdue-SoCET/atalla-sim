import math
import operator
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.fx import GraphModule, Node


TILE_HEIGHT = 32
TILE_WIDTH = 32
TILE_BYTES = TILE_HEIGHT * TILE_WIDTH * 2  # 2 KB tiles, bf16 2 bytes each

VIEW_FUNCTIONS = {operator.getitem}
VIEW_METHODS = {"reshape", "flatten", "permute", "transpose"}


def align(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def _view_source(node: Node) -> Optional[Node]:
    if node.op == "call_function" and node.target in VIEW_FUNCTIONS:
        arg = node.args[0]
        if isinstance(arg, Node):
            return arg
    if node.op == "call_method":
        method = node.target.strip("'")
        if method in VIEW_METHODS:
            arg = node.args[0]
            if isinstance(arg, Node):
                return arg
    return None


def tensor_nbytes(node: Node) -> int:
    tensor_meta = node.meta.get("tensor_meta")

    if tensor_meta.dtype != torch.bfloat16:
        raise ValueError(
            f"dtype {tensor_meta.dtype} for node {node.name} is not torch.bfloat16")

    outer = 1
    shape = tensor_meta.shape
    if len(shape) == 1:
        height = 1
        width = int(shape[0])
    else:
        for dim in shape[:-2]:
            outer *= int(dim)
        height = int(shape[-2])
        width = int(shape[-1])

    tiles_h = math.ceil(height / TILE_HEIGHT)
    tiles_w = math.ceil(width / TILE_WIDTH)
    tiles_per_plane = tiles_h * tiles_w
    total_tiles = max(1, outer) * tiles_per_plane

    return total_tiles * TILE_BYTES


def tensor_for_node(
    node: Node, gm: GraphModule, placeholder_data: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    if node.op == "placeholder":
        if node.target not in placeholder_data:
            raise ValueError(f"Missing placeholder data for {node.target}")
        return placeholder_data[node.target]

    if node.op == "get_attr":
        attr = gm
        for part in node.target.split("."):
            attr = getattr(attr, part) #must be a tensor
        return attr

    return None

def tensor_bytes(tensor: torch.Tensor, allocation_size: int, gemm_weight: bool = False) -> bytes:
    tensor = tensor.detach().cpu().contiguous()
    if tensor.ndim == 0:
        raise ValueError("Scalar tensor")
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    if tensor.ndim > 2:
        outer = 1
        for dim in tensor.shape[:-2]:
            outer *= int(dim)
        height = tensor.shape[-2]
        width = tensor.shape[-1]
        tensor = tensor.view(outer, height, width)
    else:
        tensor = tensor.unsqueeze(0)
        height = tensor.shape[-2]
        width = tensor.shape[-1]

    tiles = []
    for matrix in tensor:
        for row in range(0, height, TILE_HEIGHT):
            for col in range(0, width, TILE_WIDTH):
                tile = torch.zeros((TILE_HEIGHT, TILE_WIDTH), dtype=tensor.dtype)
                h_chunk = min(TILE_HEIGHT, height - row)
                w_chunk = min(TILE_WIDTH, width - col)
                tile[:h_chunk, :w_chunk] = matrix[row:row + h_chunk, col:col + w_chunk]
                if gemm_weight:
                    tile = tile.transpose(0, 1).flip(0).contiguous()
                tiles.append(tile)

    raw = b"".join(
        tile.view(torch.uint16).numpy().astype(np.uint16).tobytes(order="C")
        for tile in tiles
    )
    if len(raw) > allocation_size:
        raise ValueError("Tile payload larger than allocation size")
    return raw + bytes(allocation_size - len(raw))


def _write_binary_payload(file, current_size: int, start_addr: int, payload: bytes) -> int:
    if start_addr < current_size:
        raise ValueError("start_addr rewound; allocations must be monotonic")
    gap = start_addr - current_size
    if gap:
        file.write(bytes(gap))
        current_size += gap
    file.write(payload)
    return current_size + len(payload)


def assign_address(node: Node, next_addr: int, storage_bytes: Optional[int] = None) -> Tuple[int, int, int]:
    bytes_needed = tensor_nbytes(node) if storage_bytes is None else storage_bytes
    aligned_addr = align(next_addr, TILE_BYTES)
    allocation_size = align(bytes_needed, TILE_BYTES)

    node.meta["dram_addr"] = f"0x{aligned_addr:08x}"
    node.meta["bytes"] = allocation_size

    return aligned_addr, allocation_size, aligned_addr + allocation_size


def convolution_gather_bytes(node: Node) -> bytes:
    """Address-only im2col plan, reused by all output-channel tiles.

    Each group stores spatial blocks, K blocks, active rows, then 32 addresses.
    0xffffffff denotes padding. Values are gathered when the kernel executes.
    """
    source, weight = node.args[:2]
    n, channels, height, width = source.meta["tensor_meta"].shape
    out_height, out_width = node.meta["tensor_meta"].shape[-2:]
    _, in_per_group, kh, kw = weight.meta["tensor_meta"].shape
    def pair(value):
        return (value, value) if isinstance(value, int) else tuple(value)
    sh, sw = pair(node.kwargs.get("stride", 1))
    ph, pw = pair(node.kwargs.get("padding", 0))
    dh, dw = pair(node.kwargs.get("dilation", 1))
    groups = int(node.kwargs.get("groups", 1))
    base = int(source.meta["dram_addr"], 16)
    th, tw = math.ceil(height/32), math.ceil(width/32)
    total_k = in_per_group * kh * kw
    addresses = []
    for batch in range(n):
        for group in range(groups):
            for block in range(0, out_height*out_width, 32):
                rows = min(32, out_height*out_width-block)
                for k_block in range(0, total_k, 32):
                    for row in range(rows):
                        oy, ox = divmod(block+row, out_width)
                        for lane in range(32):
                            k = k_block+lane
                            address = 0xffffffff
                            if k < total_k:
                                ic, remainder = divmod(k, kh*kw)
                                ky, kx = divmod(remainder, kw)
                                iy, ix = oy*sh-ph+ky*dh, ox*sw-pw+kx*dw
                                if 0 <= iy < height and 0 <= ix < width:
                                    plane = batch*channels + group*in_per_group + ic
                                    tile = (plane*th + iy//32)*tw + ix//32
                                    address = base+tile*TILE_BYTES + ((iy%32)*32+ix%32)*2
                            addresses.append(address)
    return np.asarray(addresses, dtype="<u4").tobytes()


def allocate_memory(gm: GraphModule, bin_path: str, placeholder_data: Optional[Dict[str, torch.Tensor]] = None, base_addr: int = 0) -> GraphModule:
    placeholder_data = placeholder_data or {}
    next_addr = base_addr
    written = base_addr

    with open(bin_path, "wb") as dram_file:
        for node in gm.graph.nodes:
            if node.op == "output":
                continue
            tensor_meta = node.meta.get("tensor_meta")
            if tensor_meta is None:
                continue
            if tensor_meta.dtype != torch.bfloat16:
                # Treat non-bf16 tensors (e.g., attention bias indices) as compile-time constants.
                continue
            view_src = _view_source(node)
            materialized = None
            if view_src is not None and node.op == "call_method" and node.target == "transpose":
                source_value = tensor_for_node(view_src, gm, placeholder_data)
                if source_value is not None:
                    materialized = source_value.transpose(int(node.args[1]), int(node.args[2]))
            if view_src is not None and materialized is None and "dram_addr" in view_src.meta:
                node.meta["dram_addr"] = view_src.meta["dram_addr"]
                node.meta["bytes"] = view_src.meta["bytes"]
                continue
            tensor_value = materialized if materialized is not None else tensor_for_node(node, gm, placeholder_data)
            conv_payload = None
            if tensor_value is not None and node.users and all(
                user.op == "call_function" and user.target is torch.nn.functional.conv2d
                and len(user.args) > 1 and user.args[1] is node for user in node.users):
                group_counts = {int(user.kwargs.get("groups", 1)) for user in node.users}
                if len(group_counts) == 1:
                    groups = group_counts.pop()
                    out_per_group = int(tensor_value.shape[0]) // groups
                    chunks = []
                    for group in range(groups):
                        matrix = tensor_value[group*out_per_group:(group+1)*out_per_group].reshape(out_per_group, -1).T.contiguous()
                        packed_size = math.ceil(matrix.shape[0]/32) * math.ceil(matrix.shape[1]/32) * TILE_BYTES
                        chunks.append(tensor_bytes(matrix, packed_size, gemm_weight=True))
                    conv_payload = b"".join(chunks)
            start, size, next_addr = assign_address(node, next_addr, len(conv_payload) if conv_payload is not None else None)
            if materialized is not None:
                node.meta["materialized_view"] = True
            gemm_weight = tensor_value is not None and bool(node.users) and all(
                user.op == "call_function" and user.target is torch.matmul
                and len(user.args) > 1 and user.args[1] is node for user in node.users)
            node.meta["storage_layout"] = 2 if conv_payload is not None else (1 if gemm_weight else 0)
            node.meta["storage_tile_count"] = size // TILE_BYTES
            payload = conv_payload if conv_payload is not None else (tensor_bytes(tensor_value, size, gemm_weight=gemm_weight) if tensor_value is not None else bytes(size))
            written = _write_binary_payload(dram_file, written, start, payload)

        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target is torch.nn.functional.conv2d:
                payload = convolution_gather_bytes(node)
                start = align(next_addr, 4)
                node.meta["gather_addr"] = start
                node.meta["gather_bytes"] = len(payload)
                written = _write_binary_payload(dram_file, written, start, payload)
                next_addr = start + len(payload)

    gm.graph.lint()
    gm.recompile()

    return gm


def fake_allocate_memory(gm: GraphModule) -> GraphModule:
    """Simulate memory assignment without writing a binary file."""
    for node in gm.graph.nodes:
        if node.op == "output":
            continue
        if "tensor_meta" not in node.meta:
            continue
        node.meta["dram_addr"] = "0x00000000"
        # Setting bytes to zero skips the check in emit()
        node.meta["bytes"] = 0

    gm.graph.lint()
    gm.recompile()
    return gm
