#!/usr/bin/env python3
"""Build a scheduled graph and its initialized DRAM into one Atalla ELF."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
COMPILER = ROOT / "aihw-ppci-compiler"
sys.path.insert(0, str(ROOT / "atalla-graph"))
sys.path.insert(0, str(COMPILER))

import numpy as np
import torch
from ppci import api
from ppci.binutils.objectfile import ObjectFile, Section
from ppci.format.elf.file import ElfFile
from run_schedule import build_graph, load_model, run_schedule

DRAM_BASE = 0x80000


def compile_c(source: Path, output: Path, opt_level: int = 2) -> None:
    env = dict(os.environ, PYTHONPATH=str(COMPILER))
    command = [sys.executable, "-m", "ppci", "atalla_cc", "-m", "atalla", f"-O{opt_level}", "-p", "-I", str(ROOT), str(source), "-c", "-o", str(output)]
    with output.with_suffix(".log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Compiler failed for {source}; see {output.with_suffix('.log')}")


def build(model_name: str, output_dir: Path, seed: int, scale: float) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, example_input = load_model(model_name, scale)
    graph = build_graph(model, example_input)
    schedule = run_schedule(copy.deepcopy(graph), example_input, output_dir)
    manifest_path = output_dir / "kernel_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest["kernel_coverage"]:
        raise RuntimeError(f"Graph has unsupported operations; see {manifest_path}")

    payload = (output_dir / "dram.bin").read_bytes()
    memory = ObjectFile(api.get_arch("atalla"))
    data = Section("data")
    data.add_data(payload)
    memory.add_section(data)
    image_object = output_dir / "graph_dram.o"
    with image_object.open("w") as stream:
        memory.save(stream)

    entry = output_dir / "graph_entry.c"
    entry.write_text("int run_graph(void);\nint main(void) { int status = run_graph(); atalla_halt(); return status; }\n")
    schedule_obj = output_dir / "graph_schedule.o"
    entry_obj = output_dir / "graph_entry.o"
    compile_c(schedule, schedule_obj)
    compile_c(entry, entry_obj)

    kernel_objs = []
    for source in manifest["library_sources"]:
        src = ROOT / "kernels" / source
        obj = output_dir / (src.stem + ".o")
        compile_c(src, obj, opt_level=0 if src.name in ("elementwise.c", "spatial.c") else 2)
        kernel_objs.append(obj)
    layout = output_dir / "graph_layout.mmap"
    data_capacity = (len(payload) + 0x40000 + 0xFFF) & ~0xFFF
    layout.write_text(
        "MEMORY code LOCATION=0x00001000 SIZE=0x00040000 { SECTION(code) }\n"
        f"MEMORY data LOCATION=0x{DRAM_BASE:08X} SIZE=0x{data_capacity:08X} {{ SECTION(data) }}\n"
    )
    elf = output_dir / "output.elf"
    objects = [image_object, schedule_obj, entry_obj, *kernel_objs]
    with layout.open() as layout_stream:
        linked = api.link([str(obj) for obj in objects], layout=layout_stream, entry="main")
    if linked.get_undefined_symbols():
        raise RuntimeError("ELF link left undefined symbols")
    code = linked.get_section("code")
    if code.size % 20:
        raise RuntimeError("Linked code size is not a whole number of 20-byte packets")
    for symbol in linked.symbols:
        if symbol.defined and symbol.section == "code":
            address = linked.get_symbol_id_value(symbol.id)
            if (address - code.address) % 20:
                raise RuntimeError(
                    f"Code symbol {symbol.name} at 0x{address:X} is not on a packet boundary"
                )
    api.objcopy(linked, None, "elf", str(elf))
    with elf.open("rb") as stream:
        elf_file = ElfFile.load(stream)
    data_section = elf_file.get_section("data")
    if data_section.header.sh_addr != DRAM_BASE or not data_section.data.startswith(payload):
        raise RuntimeError("Linked ELF does not contain the graph DRAM image at the scheduled address")
    manifest.update(linkable=True, link_blocker=None, elf=str(elf), dram_base=DRAM_BASE,
                    dram_bytes=len(payload), entry="main", linked_sources=[str(p) for p in objects])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Linked {elf} ({elf.stat().st_size} bytes); DRAM payload {len(payload)} bytes at 0x{DRAM_BASE:X}")
    return elf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("basic", "alexnet_small"), default="basic")
    parser.add_argument("--out-dir", type=Path, default=Path("out/graph"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=float, default=0.01)
    args = parser.parse_args()
    build(args.model, args.out_dir, args.seed, args.scale)


if __name__ == "__main__":
    main()
