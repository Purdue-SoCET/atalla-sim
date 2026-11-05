# atalla-model-runner

Schedule-only model frontend for Atalla. The graph capture, lowering, tile planning,
DRAM allocation, and C schedule emitter are ported from `atalla-models/atalla-graph`.
This runner does not include the emulator validation pipeline.

## Requirements

Python 3.10+ with PyTorch and NumPy installed.

## Emit a schedule

From the repository root:

```bash
python atalla-graph/run_schedule.py --model basic --out-dir out/basic
python atalla-graph/run_schedule.py --model alexnet_small --scale 0.01 --out-dir out/alexnet
```

Each run writes `graph_schedule.c` and a tiled `dram.bin` into the output directory.
Supported models are `basic` and `alexnet_small`. Schedule lowering is partial for
operators that do not yet have a `generate_schedule.py` lowering.
