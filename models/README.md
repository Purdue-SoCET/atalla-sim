# atalla-model-runner

Schedule-only model frontend for Atalla. The graph capture, lowering, tile planning,
DRAM allocation, and schedule emitter are ported from `atalla-models/atalla-graph`.
The runner emits the scheduled graph as `graph_schedule.c` plus its tiled `dram.bin`;
it does not generate C kernel implementations.

The `aihw-ppci-compiler` submodule is pinned to the compiler revision used by
`atalla-models` and contains the optimized kernels used downstream.

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
operators that do not yet have a `generate_schedule.py` lowering.
