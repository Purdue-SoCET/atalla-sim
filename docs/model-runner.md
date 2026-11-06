# Running a model's operation plan

`atalla-sim model` runs a model, one operation after another, on atalla-sim,
and checks every result.

```bash
python examples/model/export_mlp.py -o /tmp/mlp     # weights, an input, PyTorch's output (needs torch)
atalla-sim model examples/model/mlp.plan.json \
    --inputs /tmp/mlp/mlp.inputs.npz --reference /tmp/mlp/mlp.torch.npy
```

```
examples/model/mlp.plan.json on atalla-0x01: MEISSA, 16 lanes, 4 KB 4-way dcache, fixed 6-cycle DRAM, 8 B bursts
    #  operation          where        kernel       cycles  worst BF16 steps vs numpy
    2  linalg.transpose   host         -                 -  -
    5  linalg.matmul      accelerator  gemm 8x8       1050  0.0
    6  linalg.generic     accelerator  add 2x8         303  0.0
    7  linalg.generic     accelerator  relu 2x8        283  0.0
    9  linalg.transpose   host         -                 -  -
   12  linalg.matmul      accelerator  gemm 8x8       1050  0.0
   13  linalg.generic     accelerator  add 2x3         297  0.0
5 operations on the accelerator, 2 on the host, 7 with no work; 2983 cycles
output vs numpy running the plan: worst 0.0 BF16 steps, max abs 0, rel_l2 0, cosine 1.000000
output vs the reference output: worst 1.0 BF16 steps, max abs 0.000488, rel_l2 0.00269, cosine 0.999997
```

## Status: a stub ahead of atalla-graph

The plan is what atalla-graph writes with `--plan-json` (schema
`atalla-graph-plan-v1`): the function's inputs, then every operation in
order, with operand and result ids, types, attributes, and the names of the
scalar ops inside each `linalg.generic`. Two pieces aren't in atalla-graph
upstream yet: that JSON output, and choosing a kernel for each Linalg
operation. This runner is the atalla-sim half, written against a local
atalla-graph build's plan so it is ready when they land.

`examples/model/mlp.linalg.mlir` is the atalla-graph example's MLP (Linear
4→8, ReLU, Linear 8→3, BF16), written by hand in the form torch-mlir's
linalg-on-tensors output takes. `examples/model/mlp.plan.json` is that local
atalla-graph build's `--plan-json` output for it.

## What runs where

| operation | runs | how |
|---|---|---|
| `linalg.matmul` | accelerator | the gemm kernel, square tiles up to 32 (padded); it computes C = A·Wᵀ + C, so it is given W = Bᵀ |
| `linalg.generic` over one `arith.addf` | accelerator | the add kernel; a broadcast operand (a bias) is expanded first |
| `linalg.generic` over one `arith.maximumf`, one input | accelerator | the relu kernel |
| `linalg.transpose` | host | numpy |
| `tensor.empty`, `linalg.fill`, scalar `arith.constant` | no work | |

Anything else stops the run, naming the operation and its position. The
kernels are the functional simulator's hand-written ones
(`third_party/atalla-functional-sim/kernels/build_{gemm,add,relu}.py`),
built at each operation's size. They read their tensor addresses from the
table at `0x3C`, so the runner places each operation's tensors and reads
its result back for the next operation.

## Checks

- every accelerator operation against numpy on the same inputs (the worst
  error, in BF16 steps);
- the final output against numpy running the whole plan;
- with `--reference`, the final output against the framework's own.

`--report-json` writes all of it, with the cycles of each operation and the
machine. `--config` picks the machine, as for `atalla-sim run`.

## Limits, and what lifts them

| limit | why | lifted by |
|---|---|---|
| constants other than scalars need their data in `--inputs`, as `const<id>` | the plan carries no constant data (weights) | atalla-graph exporting weights |
| `arith.maximumf` is taken as max(x, 0) | the plan names a region's ops but not their operands | atalla-graph recording region operands |
| dimensions above 32 stop the run | the kernels are single tiles; no tiling yet | a tiling pass (atalla-graph) or tiled kernels |
| `linalg.transpose` runs on the host | no transpose kernel among the team's kernels | a transpose-unit kernel |
| one operation per program, inputs placed fresh each time | the per-op execution model | a whole-model program from the compiler |
