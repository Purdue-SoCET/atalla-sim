# atalla-model-runner

Trace a BF16 PyTorch model, schedule padded 32x32 tiles, and compile its graph
into one Atalla ELF. Shared callable kernels live in `kernels/`; the runner
emits one graph schedule and embeds all initialized tensor data in the ELF.

## Build

```bash
git submodule update --init aihw-ppci-compiler atalla-sim
python -m pip install torch numpy
python build_model.py --model basic --out-dir out/basic
python build_model.py --model alexnet_small --scale 0.01 --out-dir out/graph
```

The compiler submodule is needed to build; the simulator submodule is needed
to execute and verify. Their nested submodules are optional for this workflow; initialize those separately for their own validation
tools. The simulator's native extension is optional and can be built with
`make -C atalla-sim native` for faster execution.

The output directory contains `output.elf`, `dram.bin`, `graph_schedule.c`,
`kernel_manifest.json`, `tensor_manifest.json`, a linker layout, objects, and
compiler logs. `main` calls `run_graph`, checks kernel status, and halts. Code
is packetized into four 5-byte instructions per packet; the build verifies
all code symbols are at packet boundaries. Graph data starts at `0x80000`
and occupies the first bytes of the ELF's writable segment.

`basic` uses GEMM, bias add, ReLU, and residual scaling. `alexnet_small` adds
NCHW convolution, max pooling, and flattening. Constant matrix/convolution
weights are packed into the hardware's load order when writing the ELF data.
Runtime convolution gathers im2col rows into scratchpads and uses the same
`load_weights`/`gemm` instruction sequence as the compiler kernels.

The pinned compiler is used through `kernels/compile.py`. This runner-owned
policy enables the backend's callee-save support and omits its broken CFG
cleanup pass for spatial sources. Other O2 passes and packetization remain
active. The compiler submodule has no source modifications.

## Execute and check numerical results

The pinned `atalla-sim` submodule runs the linked ELF:

```bash
python verify_model.py --model basic --out-dir out/basic
python verify_model.py --model alexnet_small --out-dir out/graph
```

The verifier builds and loads the actual ELF, runs with strict simulator
packet checks, requires status zero, and compares every scheduled result
with the same BF16 PyTorch graph. `numerical_report.json` records per-node
cosine similarity, maximum and mean absolute error, tolerances, cycle count,
and the ELF SHA256. The full spatial model takes longer; progress is printed
every 100,000 cycles. Set
`--sim-root`, `--seed`, `--scale`, or `--limit` as needed.

To run a built ELF directly:

```bash
python atalla-sim/tools/run_kernel.py out/graph/output.elf
```

Focused library tests compile small graphs, check cosine similarity (> 0.99
for nonzero outputs) and absolute error against PyTorch, and cover the reusable
kernel cases. The metric helper follows the functional simulator convention
(cosine is zero when either vector has zero norm). Run them with:

```bash
python -m pip install pytest
python -m pytest -q tests/test_kernel_numerics.py
```

`basic` has been checked with all nine scheduled results exactly matching
PyTorch. `alexnet_small` at scale 0.01 passed all 22 results at the default
tolerances (atol 0.01, rtol 0.03); its largest intermediate absolute error was
0.0078125 and final output error was 0.00048828125. That full simulation took
about 21 minutes. See the generated numerical report for a particular build. Average
pooling, layer normalization, GELU, and attention are not implemented; the
capability manifest rejects unsupported operators and weight layouts.

## Kernel contributors

Read [kernels/README.md](kernels/README.md) for the ABI and numerical contracts,
and [kernels/prompt.md](kernels/prompt.md) for propagating compiler changes.
