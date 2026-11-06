# A PyTorch model as one ELF

`atalla-sim models` takes a PyTorch model all the way to a program the
simulator runs: it traces the model, schedules it onto a library of graph
kernels, compiles and links everything, data included, into one ELF, runs
that ELF and checks every intermediate result against PyTorch.

```bash
pip install -e '.[models]'           # torch, and pydot for the compiler
atalla-sim models verify basic
```

```
PASS matmul: cos 1.000000; max abs 0; mean abs 0
PASS add: cos 1.000000; max abs 0; mean abs 0
...
PASS add_3: cos 1.000000; max abs 0; mean abs 0
basic: PASS; 40,622 cycles; status 0
```

`build` stops after writing the ELF; `verify` builds, runs and checks. The
output (`models/out/<model>/`) holds `output.elf`, the DRAM image
`dram.bin`, the generated `graph_schedule.c`, the kernel and tensor
manifests, and `numerical_report.json` with each result's cosine
similarity and errors. The ELF is an ordinary program for the rest of
atalla-sim:

```bash
atalla-sim run models/out/basic/output.elf --profile 10 --check --sdma-log -
```

The code lives in `models/` (its [README](../models/README.md) and
[kernel README](../models/kernels/README.md) have the full contracts).
It needs the compiler submodule, `third_party/aihw-ppci-compiler`, but not
the functional simulator.

## The models

| model | operations | result |
|---|---|---|
| `basic` | Linear (GEMM + bias), ReLU, residual scaling | all 9 results equal BF16 PyTorch exactly; about 41k cycles, 25 s |
| `alexnet_small` (`--scale 0.01`) | adds NCHW convolution, max pooling, flatten | all 22 results within atol 0.01 / rtol 0.03 of BF16 PyTorch; about 5.2M cycles, about 20 min |

Average pooling, layer normalization, GELU and attention aren't
implemented: the capability table (`models/kernels/capabilities.json`)
rejects a graph that needs them, or a weight layout it can't take, before
anything is built.

`basic` is also frozen as a reference program (`model_basic.c` in
`tests/equivalence/programs`, with PyTorch's values in
`tests/models/model_basic.expected.json`), so the test suite checks the
whole model on every run without torch or the compiler, and `atalla-sim
bench` and the gallery include it.

## How a model becomes an ELF

1. **Trace and plan** (`models/atalla-graph/`, the FX-based graph
   frontend with the graph scheduler): the model goes to BF16, is traced
   with torch.fx, Linear is split into matmul + add, shapes are
   propagated and every tensor gets padded 32 x 32 BF16 tiles in DRAM.
2. **Allocate**: tensors are placed one after another from `0x80000`
   (no reuse of dead tensors yet). Constant weights are stored in the
   order the systolic array loads them: a GEMM tile's columns as rows,
   last column first, since `lw.vi` shifts each vector in at column 0.
3. **Schedule**: `graph_schedule.c` declares a descriptor per tensor
   (shape, tile grid, each tile's DRAM address) and calls one shared
   kernel per operation (`matmul_kernel`, `add_kernel`, ...).
4. **Compile and link**: the schedule, a `main` that calls it and halts,
   the kernels in `models/kernels/src/` and the DRAM image as a data
   object are compiled (packetized) and linked: code at `0x1000`, data at
   `0x80000`. The build checks every code symbol sits on a packet
   boundary and that the ELF's data starts with the DRAM image.
5. **Verify**: the ELF runs with strict packet checks; `run_graph`'s
   status (x10) must be 0; each result is read back from DRAM and compared
   with PyTorch running the same graph in BF16.

## Things to know

- **The stack.** atalla-sim, like the functional simulator's loader,
  starts the stack pointer (x2) one to two pages above the highest data
  address, and nothing reserves or checks a stack region. A program whose
  calls need more than about 4 KB of stack would overwrite the top of its
  data if the stack grows down. The models here stay well inside it.
- **Builds aren't reproducible.** The compiler doesn't emit the same code
  on every run, even for the same source and `PYTHONHASHSEED`: `basic`
  varies between about 40.6k and 41.2k cycles from build to build, always
  with the same results. Compare cycles between frozen images, not
  between builds.
- **Compiler workarounds** in `models/kernels/compile.py` and the
  kernels, which the compiler should eventually make unnecessary:
  - the Atalla backend's callee-save set is empty, so values live across a
    call were corrupted; the build sets x9, x11 and x18-x27;
  - the CFG cleanup pass drops a loop's phi edge in the convolution code
    (only the last K block's contribution survived); `spatial.c` is built
    with the other O2 passes;
  - `&global` as an argument passed the global's first word, not its
    address; the schedule passes pointers through a global instead;
  - a `float` argument is lowered to BF16 bits, while the vector-scalar
    port reads FP32 bits; scalars travel as an int holding the FP32 bits.
- **DMA ordering.** A `vreg.ld` waits for a `scpad.ld` only through the
  register the DMA holds (see `--check`); the kernels read that register
  before each vector load, so `atalla-sim run --check` finds nothing in
  these ELFs. Scalar stores sit in the write-back data cache, where the
  scratchpad DMAs can't see them, so data a DMA reads is staged through
  vector stores.

## Relation to `atalla-sim model`

`atalla-sim model` ([model-runner.md](model-runner.md)) is the other path:
it reads an operation plan in the JSON form planned for the MLIR-based
atalla-graph and runs one operation at a time on the functional
simulator's hand-written kernels. `atalla-sim models` is a whole model in
one program, through the C compiler. Both stay until the MLIR frontend
can feed one of them.
