# atalla-model-runner

Trace a BF16 PyTorch model, schedule 32x32 tiles, and compile its complete graph
into one Atalla ELF. The linked ELF contains both code and the initialized DRAM
image; generic kernels are compiled once per operation from `kernels/`.

The compiler submodule is pinned to the selected `atalla-arch` revision. The
runner does not generate per-node C kernel copies or modify compiler sources.

## Setup

```bash
git submodule update --init --recursive
python -m pip install torch numpy
```

## Build a model

From the repository root:

```bash
python build_model.py --model basic --out-dir out/basic
python build_model.py --model alexnet_small --scale 0.01 --out-dir out/alexnet
```

Each output directory contains `output.elf`, `dram.bin`, `graph_schedule.c`,
`kernel_manifest.json`, the generated linker layout, and compiler logs. `main`
calls `run_graph` and halts. The ELF's writable segment begins at `0x80000`;
`dram.bin` occupies its first bytes, so all scheduled tile addresses point into
the loaded ELF segment. The compiler packetizes each object with `-p` before linking, and the build checks every code symbol against the simulator's 20-byte packet boundary rule. The linker resolves all graph and library symbols.

`basic` uses tiled GEMM, bias add, ReLU and residual add. `alexnet_small` also
uses NCHW convolution, max pooling, and tiled flattening. The convolution
adapter packs im2col and weight blocks at runtime and uses the compiler's
`load_weights`/`gemm` sequence. It is a generic correctness path for compiler
contributors to optimize. The compiler's optimizer currently crashes on this
nested-loop spatial source at `-O2`, so `spatial.c` is compiled at `-O0`.

The ELF build and section placement are checked during the command. Simulator execution of `basic` now halts successfully, but its output differs
from PyTorch. The generic GEMM and elementwise adapters need numerical
validation and correction before using the ELF for inference; standalone
compiler validation does not establish graph correctness. Other graph
operators, including layer normalization, GELU, attention, and average pooling,
remain capability gated. The build stops if a graph contains unsupported nodes.

## Kernel contributor entry point

Read [kernels/README.md](kernels/README.md) for the ABI and operation map, and
[kernels/prompt.md](kernels/prompt.md) for a review workflow that propagates
compiler optimizations from validation kernels into the shared library.
