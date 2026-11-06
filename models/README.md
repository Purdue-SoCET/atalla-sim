# models: a PyTorch model as one Atalla ELF

Trace a BF16 PyTorch model, schedule padded 32x32 tiles, and compile its graph
into one Atalla ELF. Shared callable kernels live in `kernels/`; the runner
emits one graph schedule and embeds all initialized tensor data in the ELF.

## Build

From the atalla-sim repository root:

```bash
git submodule update --init third_party/aihw-ppci-compiler
pip install torch
atalla-sim models build basic                  # writes models/out/basic/
atalla-sim models build alexnet_small          # --scale 0.01 by default
```

or, from this folder, `python build_model.py --model basic --out-dir out/basic`.
This folder uses the repository's compiler submodule
(`third_party/aihw-ppci-compiler`) and the simulator around it; `make native`
at the repository root makes the simulation faster.

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

atalla-sim runs the linked ELF:

```bash
atalla-sim models verify basic
atalla-sim models verify alexnet_small --limit 20000000
```

The verifier builds and loads the actual ELF, runs with strict simulator
packet checks, requires status zero, and compares every scheduled result
with the same BF16 PyTorch graph. `numerical_report.json` records per-node
cosine similarity, maximum and mean absolute error, tolerances, cycle count,
and the ELF SHA256. The full spatial model takes longer; progress is printed
every 100,000 cycles. Set
`--seed`, `--scale`, or `--limit` as needed (`verify_model.py` also takes
`--sim-root`, another atalla-sim checkout).

To run a built ELF directly:

```bash
atalla-sim run models/out/basic/output.elf --profile 10 --sdma-log -
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

## Reference programs for atalla-sim

`python freeze.py basic` stores a built ELF as a program image in
`tests/equivalence/programs/model_basic.c.in.gz`, with every checked result's
PyTorch value in `tests/models/model_basic.expected.json`. atalla-sim's tests
then run and check the model without torch or the compiler, and `atalla-sim
bench` and the gallery include it. The compiler doesn't emit the same code on
every run (cycle counts vary by about 1.5% between builds), so refreeze, and
`atalla-sim bench --update`, only on purpose.

## Kernel contributors

Read [kernels/README.md](kernels/README.md) for the ABI and numerical contracts,
and [kernels/prompt.md](kernels/prompt.md) for propagating compiler changes.
