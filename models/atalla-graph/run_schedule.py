"""Trace a PyTorch model and emit an Atalla graph schedule."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.fx import GraphModule, symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp

from graph.lower_modules import lower_linear_modules
from graph.memoryallocator import allocate_memory
from graph.fx_capture import normalize_ops
from graph.tile_planner import plan_tiles
from scripts.generate_schedule import emit
from kernel_backend import build_kernel_manifest


def load_model(name: str, scale: float = 0.01):
    if name == "basic":
        from model.basic import BasicModule
        return BasicModule(dim=32, depth=2), torch.randn(1, 32)
    if name == "alexnet_small":
        from model.alexnet_small import AlexNetSmall
        return AlexNetSmall(scale=scale, num_classes=10), torch.randn(1, 3, 32, 32)
    raise ValueError(f"Unknown model: {name}")


def build_graph(model: nn.Module, example_input: torch.Tensor) -> GraphModule:
    model = model.bfloat16().eval()
    graph_module = symbolic_trace(model)
    graph_module = lower_linear_modules(graph_module)
    ShapeProp(graph_module).propagate(example_input.bfloat16())
    return plan_tiles(normalize_ops(graph_module))


def run_schedule(graph_module: GraphModule, example_input: torch.Tensor, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    placeholder_data = {str(node.target): example_input.bfloat16().clone()
                        for node in graph_module.graph.nodes if node.op == "placeholder"}
    allocated = allocate_memory(graph_module, str(out_dir / "dram.bin"), placeholder_data, base_addr=0x80000)
    tensors = {}
    for node in allocated.graph.nodes:
        meta = node.meta.get("tensor_meta")
        if "dram_addr" in node.meta and meta is not None and meta.dtype == torch.bfloat16:
            tensors[node.name] = {
                "address": int(node.meta["dram_addr"], 16),
                "shape": list(meta.shape), "bytes": node.meta["bytes"],
                "storage_layout": node.meta.get("storage_layout", 0),
                "gather_addr": node.meta.get("gather_addr", 0),
                "gather_bytes": node.meta.get("gather_bytes", 0),
            }
    (out_dir / "tensor_manifest.json").write_text(json.dumps(tensors, indent=2) + "\n")
    code = emit(allocated)
    output_path = out_dir / "graph_schedule.c"
    output_path.write_text(code)
    repo_root = Path(__file__).resolve().parents[1]
    compiler_root = repo_root / "aihw-ppci-compiler"
    kernel_root = repo_root / "kernels"
    manifest = build_kernel_manifest(allocated, kernel_root, compiler_root)
    manifest_path = out_dir / "kernel_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    kernel_calls = [line.strip() for line in code.splitlines() if "_kernel(" in line]
    print(f"Schedule: {len(code.splitlines())} lines, {len(kernel_calls)} kernel calls -> {output_path}")
    for call in kernel_calls[:8]:
        print(f"  {call[:100]}")
    if len(kernel_calls) > 8:
        print(f"  ... ({len(kernel_calls) - 8} more)")
    if manifest["unsupported"]:
        print("Schedule emission omitted unsupported graph operations; see the manifest.")
    print(
        f"Kernel source bindings: {len(manifest['bindings'])}; "
        f"kernel coverage: {manifest['kernel_coverage']}; ELF linkable: {manifest['linkable']}"
    )
    if not manifest["linkable"]:
        print(f"Kernel capability report: {manifest_path}: {manifest['link_blocker']}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit a C schedule from an Atalla FX graph")
    parser.add_argument("--model", choices=("basic", "alexnet_small"), default="basic")
    parser.add_argument("--scale", type=float, default=0.01,
                        help="Channel scale for alexnet_small")
    parser.add_argument("--out-dir", type=Path, default=Path("out/graph"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, example_input = load_model(args.model, args.scale)
    graph_module = build_graph(model, example_input)
    run_schedule(copy.deepcopy(graph_module), example_input, args.out_dir)


if __name__ == "__main__":
    main()
