# atalla-model-runner

Graph-to-schedule frontend and shared kernel package for Atalla. The graph capture,
lowering, tile planning, DRAM allocation, and schedule emitter are ported from
`atalla-models/atalla-graph`. The runner emits `graph_schedule.c`, its tiled
`dram.bin`, and `kernel_manifest.json`. Shared callable elementwise kernels live in
`kernels/`; the manifest records implemented operations and unsupported graph cases.

The `aihw-ppci-compiler` submodule is checked out at the latest `atalla-arch` revision selected for this runner and contains the optimized validation kernels used downstream.

## Setup

```bash
git submodule update --init --recursive
python -m pip install torch numpy
```

## Emit a schedule

From the repository root:

```bash
python atalla-graph/run_schedule.py --model basic --out-dir out/basic
python atalla-graph/run_schedule.py --model alexnet_small --scale 0.01 --out-dir out/alexnet
```

Supported models are `basic` and `alexnet_small`. Schedule lowering is partial for
operators without a `generate_schedule.py` lowering. The current kernel package
implements tiled elementwise add, multiply, scalar multiply, and ReLU. Matrix,
convolution, and pooling kernels are listed as unsupported until their graph layouts
and callable implementations are complete. ELF compilation and DRAM-image
packaging are also still pending, so the manifest never reports a graph as linked.

## Kernel contributor entry point

Compiler contributors can work in `kernels/` without changing FX or graph-memory
code. Read [kernels/README.md](kernels/README.md) for the ABI and capability map, and
[kernels/prompt.md](kernels/prompt.md) for a review workflow that propagates compiler
optimizations from validation kernels into the shared library.
