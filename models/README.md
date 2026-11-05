# atalla-model-runner

Schedule-only model frontend for Atalla. The graph capture, lowering, tile planning,
DRAM allocation, and schedule emitter are ported from `atalla-models/atalla-graph`.
The runner emits the scheduled graph as `graph_schedule.c` plus its tiled `dram.bin`;
it does not generate C kernel implementations. It also writes `kernel_manifest.json`, which maps graph operations to shared validation source files without copying those files per node and records adapter/layout blockers.

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
operators that do not yet have a `generate_schedule.py` lowering. The manifest is a source-binding/preflight artifact; it does not mean the standalone validation programs can yet be linked as graph kernels.
