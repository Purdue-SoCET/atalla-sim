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
pip install -e '.[test]'   # numpy, pytest and the `atalla-sim` command; '.[test,plots]' adds matplotlib
make native          # optional, needs g++: about 5x faster, identical results
make ramulator       # optional, needs a C++20 g++ and CMake: DDR4 timing (--dram ramulator)
atalla-sim doctor    # checks all of the above and says how to fix what's missing
pytest               # about two minutes
```

`atalla-sim` is the command for everything below (`atalla-sim -h` lists the
subcommands); `atalla-sim run` is `python tools/run_kernel.py`, and the
examples use either.

Python 3.9 or newer. Two submodules live in `third_party/`:

| submodule | what it is | without it |
|---|---|---|
| `aihw-ppci-compiler` (branch `atalla-arch`) | the C compiler, which packs instructions into VLIW packets, and the C kernels with their test inputs | C programs can't be built |
| `atalla-functional-sim` | an instruction-level simulator (no timing), the reference for values, and the hand-written assembly kernels | no `--golden` checks, no assembly kernels |
| `ramulator2` | Ramulator 2, a cycle-level DDR4 model, built by `make ramulator` | DRAM is fixed-latency only |

Both are private Purdue-SoCET repositories: you need access and an SSH key on
your GitHub account. Without them atalla-sim still runs assembled programs
(`.in` files) and the harness experiments, and the tests that need them are
skipped, not failed.

`pytest` leaves out the hour-long 1024 × 1024 GEMM test; run it with
`pytest -m slow`.

## 2. Run a program

```bash
python tools/run_kernel.py relu.c --golden --report
```

`relu.c` is one of the compiler's kernels: it zeroes the negative values of a
4 × 32 BF16 tile. Without `--report` a run prints only its cycle count: the
performance monitor that writes the report is attached only when asked for,
so plain runs and sweeps pay nothing for it. The report, annotated:

```
relu.c: 768 cycles

where the cycles went:                       every cycle in exactly one bucket
  frontend:icache             318   41.4%    the instruction cache starts empty
  halt:drain                  154   20.1%    waiting for the last stores at the end
  frontend:refill              98   12.8%    refetching after a taken branch
  issue                        86   11.2%    a packet issued
  stall:hazard:scalar_load     52    6.8%    the next packet waited on a load's result
  ...

issue: 86 packets, slots 28.5% full          how much of each 4-slot packet was used

unit busy:                                   cycles each unit had work
  dram_channel        312   40.6%
  dcache              235   30.6%
  sdma0               105   13.7%
  ...

DRAM: 2496 bytes in 312 bursts, 3.25 B/cycle of 8 peak
  icache      216 bursts     1728 bytes  waited 12 cycles
  scpad0       64 bursts      512 bytes  waited 68 cycles    contention for the shared channel

reference result: PASS -- matches the numpy reference
against the functional sim (lanes reduction order):
  identical: registers and DRAM
```

- A unit near 100% busy is the bottleneck; a hazard bucket names the unit
  the program kept waiting on.
- `reference result` compares DRAM with the kernel's numpy answer.
- `--golden` also runs the functional sim on the identical program and
  compares every register and DRAM halfword. Before that it names the
  *first* register write where the two part ways -- which write to which
  register, at which cycle, made by which packet -- so a wrong result points
  at the instruction that went wrong, not just at the final state. A
  difference that never reaches the output still shows here (a load that
  read a row before its DMA landed, then was overwritten).

What `run_kernel.py` runs:

```bash
python tools/run_kernel.py relu.c                 # a C program, compiled on the fly
python tools/run_kernel.py gemm                   # an assembly kernel from the functional sim
python tools/run_kernel.py prog.in                # an already assembled program
python tools/run_kernel.py prog.elf               # a program the compiler linked
python tools/run_kernel.py relu.c --elf           # compile and link it, then load the ELF
python tools/run_kernel.py relu.c --save relu.in  # keep the program image that ran
python tools/run_kernel.py relu.c --report       # where the cycles went
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

### See which units work when

```bash
python tools/run_kernel.py gemm --activity gemm.html
```

writes one self-contained page (open it in a browser, no server needed):
a drawing of the chip as the model builds it -- fetch, decode, the
register-file banks, scalar execute, the vector lanes, GSAU and load/store
units, the systolic array, the scratchpads with their crossbars and DMA
engines, the DRAM channel -- with every block coloured by what it was doing
at the chosen cycle: idle, active (moving work) or stalled (holding work it
can't move: a busy DRAM channel, a bank conflict). A slider and a play
button move through the run; a window of 10, 100 or 1000 cycles shades each
block by how much of that span it was busy instead.

The percentage on a block is the share of the run's cycles in which it had
work, active or stalled: 30% on the GSAU means it had rows to send or
results to return on 30% of the cycles. Hovering a block splits that into
active and stalled cycles and says what each state means for it. Below the
drawing, a timeline has a row per block and, on top, the performance
monitor's cycle buckets; click it to jump there.

`--activity gemm.json` saves the recording instead;
`python tools/activity_view.py gemm.json` draws it later. The blocks and
what "active" means for each are in `src/atalla/activity.py`.

### Find the packets that cost the most

```bash
python tools/run_kernel.py softmax.c --profile 8 --save-asm softmax.s
```

charges every cycle to the packet responsible -- the one issuing, the one
decode 2 holds while it stalls, the one being fetched while the frontend
waits -- and prints the packets that took the most, with what they waited on:

```
where the cycles went, by packet (top 8 of 105 packets; 657 cycles):
  cycles      %   issued  pc                        packet / waits
     133  20.2%        0  halt                      (after halt.s: draining work in flight and dcache writebacks)
      37   5.6%        1  0xf0 (asm line 50)        scpad.ld x11, x10, x9
                                                      stall:hazard:scalar_load 28, frontend:icache 7, frontend:refill 1
      17   2.6%        2  0x118 (asm line 59)       addi.s x11, x0, 1
                                                      stall:hazard:sdma 14, frontend:icache 1
```

The wait names are the monitor's cycle buckets (`--report` explains them).
For a C program the line numbers refer to the compiler's assembly, which
`--save-asm` keeps; the same `asm line N` appears in the hang report and the
dependency check. `--profile-json PATH` writes every packet's row.

### Check a program's dependencies

```bash
python tools/run_kernel.py add.c --check          # report; --check=error also exits 1
```

reports the accesses nothing in the hardware orders, so a program gets them
right only by timing:

- a `vreg.ld` reading a pad row an in-flight `scpad.ld` is still filling: the
  hardware makes the load wait only if it reads the register the DMA holds
  busy (the DMA's `rs1`);
- a `scpad.st` reading a row a `vreg.st` hasn't written yet: the vreg.st
  reservations order only an SDMA on the same `rs1`;
- the same rows from the other side (a `scpad.ld` over rows a `vreg.st` still
  writes, a `vreg.st` into rows an SDMA still uses);
- scalar loads and stores at addresses that aren't a multiple of 4 (the word
  path aligns them down);
- packets the hardware can't run as one packet.

```
dependency check: 1 finding at 1 pc
  scpad.ld -> vreg.ld: a vreg.ld read a row an in-flight scpad.ld was still filling
    0x44c, first at cycle 343: vreg.ld of pad 1 row 0 (rs1 x10) while the scpad.ld at
    0x3ac (rs1 x11) still fills rows 0-3; the load doesn't read x11, so nothing makes it wait
      packet: vreg.ld v1, x10, x9
```

A finding can still compute the right answer in this run (here row 0 lands
first); it breaks when the timing changes. The checker only reads, so a
checked run takes the same cycles. Rules and limits: `src/atalla/check.py`.

## 3. Explore a design

### Machine configurations

A design point is a file in `configs/`:

```bash
python tools/run_kernel.py gemm --config atalla-0x01        # the default machine
python tools/run_kernel.py gemm --config atalla-0x01-ddr4   # with the AXI bus and Ramulator DDR4
python tools/run_kernel.py gemm --config my-design.toml --lanes 8   # flags override the file
```

`configs/atalla-0x01.toml` lists every key with its default and what it
means: the lanes, the array, the data cache, the load/store queue, scalar
execute latencies, the DRAM model and its parameters, the AXI bus's sizes.
Your own file needs only the keys it changes; an unknown key is an error, so
a typo can't leave a parameter at its default unnoticed. `--report` prints
the resolved machine on one line, and `--perf-json`, `--profile-json` and
`--activity` record all of it next to the results. `atalla-sim --version`
says which simulator, ISA and RTL produced them.

When comparing with an RTL simulation or the functional simulator, read
[known-differences.md](known-differences.md) first.

### From the command line

| flag | default | |
|---|---|---|
| `--lanes N` | 16 | vector lanes; must divide 32 |
| `--array meissa\|tpu` | `meissa` | the systolic array |
| `--dram-latency N` | 6 | cycles from a DRAM request to its data |
| `--dram fixed\|ramulator` | `fixed` | DRAM timing: `--dram-latency`, or Ramulator 2's DDR4 model (needs `make ramulator`) |
| `--axi`, `--no-axi` | on with Ramulator | the `memory_FA26` AXI bus between the DRAM masters and DRAM |
| `--dram-config rtl\|ddr4-1600\|ddr_cntrl` | `rtl` | with Ramulator: `memory_FA26`'s DDR4-3200W wrapper, or the DDR controller's DDR4-1600 (by the spec, or with its own timings) |
| `--dram-param NAME=VALUE` | | with Ramulator: change one value of the config or of the DDR4 model, e.g. `read_buffer=64`, `preset=DDR4_2400R`, `nCL=22`; repeatable |
| `--burst-bytes N` | 8 | bytes per DRAM request, for every DRAM master; 8 is the RTL's 64-bit interface |
| `--warm-icache` | off | start with the program in the instruction cache |
| `--limit N` | 5,000,000 | stop after this many cycles |
| `--hang-after N` | 50,000 | stop once nothing has moved for N cycles, with a report of what everything waits on (0: never) |

### Your own inputs and outputs

```bash
python tools/run_kernel.py add.c --input a.npy@0x1000 --input b.npy@0x2000 \
    --output 0x3000:4x32=c.npy --expect c_ref.npy@0x3000 --ulps 1
```

`--input FILE@ADDR[:DTYPE]` puts an array (a `.npy` file, or numbers in a
text file) into DRAM at byte address `ADDR` before the run, after the
program's own data; `--output ADDR:SHAPE[:DTYPE][=FILE]` reads one back,
prints a summary and saves it; `--expect FILE@ADDR[:DTYPE]` compares DRAM
with a reference and exits 1 when any element is more than `--ulps` BF16
steps off. Types: `bf16` (the default), `f32`, `i32`, `u32`, `i16`, `u16`,
`i8`, `u8`. A C program's stack is placed above every input and output.

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
    perf=True,                                       # attach the performance monitor
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
| `perf` | off | attach the performance monitor as `plat.perf` |
| `perf_timeline` | off | keep a per-cycle record in `plat.perf.timeline` (implies `perf`) |
| `history` | off | keep the scheduler's per-packet lists (`core.issued`, `core.issued_to_d1d2`, `core.scalar_writes`); off, only their counts |

Anything that records is off unless asked for, so a plain run pays only for
the model.

Scratchpad sizes are constants in `src/atalla/sysarr_tpu_system.py`
(`SPAD_TOTAL_BYTES`, `SPAD_NUM_PADS`, ...).

### A kernel in a test

`atalla.kernel.Kernel` runs one program with numpy inputs and returns numpy
outputs, on atalla-sim and, with `golden=True`, on the functional sim from
the same image:

```python
import numpy as np
from atalla.kernel import Kernel
from atalla.tensors import assert_close, bf16_round

a = bf16_round(np.random.default_rng(0).normal(size=(4, 32)))
k = Kernel(c=open("my_kernel.c").read())    # or asm=..., image="prog.in", program=(instr, data)
k.write(0x1000, a)                          # bf16 unless dtype= says otherwise
k.output(0x3000, (4, 32))                   # keeps the C stack clear of it
r = k.run(golden=True, check=True)          # keywords go to build_atalla_platform
assert_close(r.read(0x3000, (4, 32)), 2 * a, ulps=1)          # within 1 BF16 step
assert_close(r.golden.read(0x3000, (4, 32)), 2 * a, ulps=1)
print(r.cycles, r.plat.check.summary())
```

`atalla.tensors.compare(got, want)` gives max abs error, rel_l2, cosine and
the worst error in BF16 steps. Cosine alone hides scale errors; the BF16-step
bound is the one to gate on.

### Sweep a parameter

```bash
atalla-sim sweep gemm --param lanes=4,8,16,32 --param dram_latency=6,50 -j 8 \
    -o sweep.csv --plot sweep.png
```

runs one program on every combination, in parallel, and prints a row per
point (the parameters, cycles, packets issued; `--perf` adds the monitor's
cycle buckets). A parameter is any key of the machine configuration: a
dotted path (`dcache.size_bytes`, `ramulator.nCL`), a key name that appears
once (`lane_count`), or `lanes`, `array`, `dram`, `latency`. `--config`
picks the base machine. A C program is compiled once, so every point runs
the same image.

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
| `hung: nothing moved for N cycles` | nothing in the machine did any work for `--hang-after` cycles (50,000 by default): decode 2 waits on something that will never come. The report under it names the packet, what it waits on (a busy register, a busy unit, a reservation), the registers held busy and every block still holding work. A "broke the hardware's packet rules" line means the RTL would drop part of a packet, typically losing a register write. |
| `no halt after N cycles` | the program never reached `halt` within `--limit`, but kept doing work: usually an endless loop. The same report follows. |
| `packets the RTL would mangle` | a packet uses more of some unit than the hardware has in a cycle; the real scheduler would drop part of it. With C, a compiler bug. |
| `reference result: FAIL` with *a known compiler bug* | the compiler team already lists this kernel as failing. |
| tests skipped: *functional sim unavailable* | the submodule is missing: `git submodule update --init`. |
| `atalla_cc failed` | the compiler rejected the program; its message follows. Run it alone from `third_party/aihw-ppci-compiler`: `python -m ppci.cli.atalla_cc --machine atalla -O 2 -p -S k.c -o k.s` |
| slow runs | `make native`. Kernels run at roughly 2,000–4,000 simulated cycles a second. |

## Where next

- [architecture/scheduler.md](architecture/scheduler.md): what the scheduler
  models, the platform, the performance monitor, kernel status.
- [README.md](README.md): the map of every document.
