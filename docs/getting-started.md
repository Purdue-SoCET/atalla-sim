# Getting started

Atalla-sim simulates the Atalla accelerator one clock cycle at a time: the
VLIW scheduler, the scalar execute units and data cache, the vector core
(32-element BF16 vectors on 16 lanes), the MEISSA systolic array, four
0.5 MB scratchpad pads with their DMA engines, and DRAM. Give it a program and
it reports how many cycles the program took, where they went, how busy each
unit was, and what ended up in the registers and memory.

Use it to ask "what if": change a parameter or a unit, rerun the same
program, and compare the reports.

## 1. Install

```bash
git clone --recurse-submodules git@github.com:Purdue-SoCET/atalla-sim.git
cd atalla-sim
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pytest   # and matplotlib for the plotting tools
make native          # optional, needs g++: about 5x faster, identical results
pytest               # about two minutes
```

Python 3.9 or newer. Two submodules live in `third_party/`:

| submodule | what it is | without it |
|---|---|---|
| `aihw-ppci-compiler` (branch `atalla-shaunak`) | the C compiler, which packs instructions into VLIW packets, and the C kernels with their test inputs | C programs can't be built |
| `atalla-functional-sim` | an instruction-level simulator (no timing), the reference for values, and the hand-written assembly kernels | no `--golden` checks, no assembly kernels |

Both are private Purdue-SoCET repositories: you need access and an SSH key on
your GitHub account. Without them atalla-sim still runs assembled programs
(`.in` files) and the harness experiments, and the tests that need them are
skipped, not failed.

`pytest` leaves out the hour-long 1024 × 1024 GEMM test; run it with
`pytest -m slow`.

## 2. Run a program

```bash
python tools/run_kernel.py relu.c --golden
```

`relu.c` is one of the compiler's kernels: it zeroes the negative values of a
4 × 32 BF16 tile. The report, annotated:

```
relu.c: 830 cycles

where the cycles went:                       every cycle in exactly one bucket
  frontend:icache             313   37.7%    the instruction cache starts empty
  halt:drain                  188   22.7%    waiting for the last stores at the end
  frontend:refill              96   11.6%    refetching after a taken branch
  issue                        86   10.4%    a packet issued
  stall:hazard:scalar_load     74    8.9%    the next packet waited on a load's result
  ...

issue: 86 packets, slots 28.5% full          how much of each 4-slot packet was used

unit busy:                                   cycles each unit had work
  dram_channel        312   37.6%
  dcache              304   36.6%
  sdma0               118   14.2%
  ...

DRAM: 2496 bytes in 312 bursts, 3.01 B/cycle of 8 peak
  icache      216 bursts     1728 bytes  waited 11 cycles
  scpad0       64 bursts      512 bytes  waited 77 cycles    contention for the shared channel

reference result: PASS -- matches the numpy reference
against the functional sim (lanes reduction order):
  identical: registers and DRAM
```

- A unit near 100% busy is the bottleneck; a hazard bucket names the unit
  the program kept waiting on.
- `reference result` compares DRAM with the kernel's numpy answer.
- `--golden` also runs the functional sim on the identical program and
  compares every register and DRAM halfword.

What `run_kernel.py` runs:

```bash
python tools/run_kernel.py relu.c                 # a C program, compiled on the fly
python tools/run_kernel.py gemm                   # an assembly kernel from the functional sim
python tools/run_kernel.py prog.in                # an already assembled program
python tools/run_kernel.py relu.c --save relu.in  # keep the program image that ran
python tools/run_kernel.py relu.c --perf-json relu.json --timeline   # the report as JSON
```

A C program is found by path, or by name among the compiler's
`test_validation_kernels/` (kernels) and `test_validation/` (unit tests). The
compiler's register allocation is not deterministic, so cycle counts can move
a little between compiles; for comparisons between designs, `--save` the
image once and run that `.in` every time.

Not every kernel that ships with the compiler and the functional sim computes
what it claims. `docs/architecture/scheduler.md`, *Which kernels compute what
they claim*, lists each one; the correct ones are a safe starting point.

## 3. Explore a design

### From the command line

| flag | default | |
|---|---|---|
| `--lanes N` | 16 | vector lanes; must divide 32 |
| `--array meissa\|tpu` | `meissa` | the systolic array |
| `--dram-latency N` | 6 | cycles from a DRAM request to its data |
| `--burst-bytes N` | 8 | bytes per DRAM request, for every DRAM master; 8 is the RTL's 64-bit interface |
| `--warm-icache` | off | start with the program in the instruction cache |
| `--limit N` | 5,000,000 | stop after this many cycles |

### From Python

Everything `build_atalla_platform` takes, for example a smaller data cache:

```python
import sys
sys.path[:0] = ["src", "tools"]

import c_kernels
from atalla.atalla_platform import build_atalla_platform
from scheduler.dcache import DCacheConfig

path = c_kernels.KERNELS / "relu.c"
program, data, regs = c_kernels.build(path)          # compile, assemble, inputs

plat = build_atalla_platform(
    program.instr, data, init_sregs=regs,
    lane_count=8,                                    # 8 lanes of 4 elements
    dram_latency=20,
    dcache_config=DCacheConfig(size_bytes=2048, ways=2),
)
cycles = plat.run_until_done()
print(cycles, c_kernels.check(path, plat))
print(plat.perf.report()["cycle_breakdown"])
```

| parameter | default | defined in |
|---|---|---|
| `systolic_array` | `"meissa"` | `src/atalla/atalla_platform.py` |
| `lane_count`, `dram_latency`, `burst_bytes` | 16, 6, 8 | `build_atalla_platform` |
| `dcache_config` | 4 KB, 4-way, 64 B lines, 8 MSHRs | `DCacheConfig`, `src/scheduler/dcache.py` |
| `ex_latency` | e.g. `{"div.s": 66, "mul.s": 2}` | `DEFAULT_LATENCY`, `src/scheduler/execute.py` |
| `lsu_depth` | 4 | load/store queue entries |
| `perf_timeline` | off | keep a per-cycle record in `plat.perf.timeline` |

Scratchpad sizes are constants in `src/atalla/sysarr_tpu_system.py`
(`SPAD_TOTAL_BYTES`, `SPAD_NUM_PADS`, ...).

### Larger studies

The harness-driven platform runs GEMMs too big to write as a program, and
the sweep tools vary one parameter at a time: see
[harnesses/tiled-1024.md](harnesses/tiled-1024.md) and
[sweeps/sysarr-tpu.md](sweeps/sysarr-tpu.md).

### Changing a unit

Each module in `src/` opens with a docstring saying what it models and how to
extend it. [architecture/base-classes.md §8](architecture/base-classes.md#8-writing-a-component)
walks through writing a component. After a change, run the tests:
`tests/equivalence` compares every cycle count and statistic of 34 fixed runs
with a snapshot, so it tells you exactly what your change moved. When the
change is intended, regenerate the entries it moved:

```bash
python tests/equivalence/make_snapshot.py kernel_gemm tiled_cosim_64
```

## 4. Write a kernel

Kernels are C with Atalla intrinsics. The compiler's `README.md` lists them;
the core ones:

| intrinsic | instruction | what it does |
|---|---|---|
| `scpad_load(sp, dram, ctl)` | `scpad.ld` | DMA a tile from DRAM into a scratchpad |
| `scpad_store(sp, dram, ctl)` | `scpad.st` | DMA a tile back to DRAM |
| `vector_load(addr, row, cols - 1, sid)` | `vreg.ld` | a scratchpad row into a vector register |
| `vector_store(v, addr, row, cols - 1, sid)` | `vreg.st` | a vector register into a scratchpad row |
| `a + b`, `vec_op_masked(op, a, b, mask)` | `add.vv`, `mul.vs`, ... | element-wise ops; `"RSUM"`, `"RMAX"`, `"RMIN"` reduce |
| `make_mask(op, a, b, mask)` | `mgt.mvv`, ... | compare, giving a 32-bit lane mask |
| `load_weights(v)` | `lw.vi` | shift a vector into the systolic array's weights |
| `gemm(a, acc, mask)` | `gemm.vv` + `add.vv` | multiply a row by the weights, add `acc` |

The DMA control word (`ctl`) packs: bits [31:30] the scratchpad, [29:25]
rows − 1, [24:20] columns − 1, [19:0] the matrix's row width − 1, in
elements. For 4 rows of 32 BF16 from a 32-wide matrix into scratchpad 0:
`(0 << 30) | (3 << 25) | (31 << 20) | 31` = `133169183`.

Rules the hardware enforces, which kernels have broken before:

- **Data reaches vector registers only through a scratchpad.** `scpad_load`
  first, then `vector_load`. The functional sim also lets `vector_load` read
  DRAM directly; the hardware, and atalla-sim, read the scratchpad.
- **`gemm.vv` computes `out[j] = x · (j-th weight vector)`.** `load_weights`
  shifts each vector into column 0 and moves the others right, so to compute
  `x @ W`, load W's *columns*, last first. Loading W's rows computes `x @ Wᵀ`.
- **A vector load waits for its DMA only through a register.** An
  `scpad.ld` holds its first register (`rs1`) until the data is in the
  scratchpad, and a `vreg.ld` that reads that register waits for it. In
  assembly, use the same register for both. In C the compiler doesn't
  guarantee it, even when both calls use the same variable (it may put the
  address in a new register), so a load right after a DMA can read stale
  data: check with `--golden` and a reference result.
- **Don't rely on masked-off lanes.** What a masked op leaves in the lanes its
  mask switches off is not defined by the intrinsics.
- **Immediates have limits.** `addi.s` takes a signed 12-bit immediate
  (−2048..2047); atalla-sim's assembler rejects anything larger rather than
  wrapping it.

To add a kernel with a checked result: put it in the compiler's
`test_validation_kernels/`, and give it a `seed_<name>(words)` (its DRAM
inputs) and a `validate_<name>(out_mem)` (a numpy check) in that folder's
`run_kernel_validation.py`, registered in `KERNEL_REGISTRY`. For a one-feature
test, put a `.c` file in `test_validation/` whose header says
`EXPECT_X10: <value>`; the result is `main`'s return value.

## 5. Troubleshooting

| symptom | meaning |
|---|---|
| `no halt after N cycles` | the program never reached `halt`: an endless loop, or waiting on something that never comes. The growing stall bucket names what. |
| `packets the RTL would mangle` | a packet uses more of some unit than the hardware has in a cycle; the real scheduler would drop part of it. With C, a compiler bug. |
| `reference result: FAIL` with *a known compiler bug* | the compiler team already lists this kernel as failing. |
| tests skipped: *functional sim unavailable* | the submodule is missing: `git submodule update --init`. |
| `atalla_cc failed` | the compiler rejected the program; its message follows. Run it alone from `third_party/aihw-ppci-compiler`: `python -m ppci.cli.atalla_cc --machine atalla -O 2 -p -S k.c -o k.s` |
| slow runs | `make native`. Kernels run at roughly 2,000–4,000 simulated cycles a second. |

## Where next

- [architecture/scheduler.md](architecture/scheduler.md): what the scheduler
  models, the platform, the performance monitor, kernel status.
- [README.md](README.md): the map of every document.
