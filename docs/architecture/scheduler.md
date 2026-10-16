# Scheduler

`src/scheduler/` — a cycle-level model of Atalla's control processor,
transcribed from the RTL on atalla's `scheduler_integration_SP26_joshklug`.
One `SchedulerCore` is the clocked object; every stage inside it is an
`RTLModule`, driven each cycle as readiness, then data and next state, then
one commit, so no stage sees another's value early.

```
 PC ─► I$ + BTB ─► IF/D1 ─► decode 1 ─► D1/D2 ─► decode 2 ─► D2/EX ─► EX1–EX5 ─► scalar WB ─► EX/WB
  ▲                                                │                      │   │                      │
  └──────────────── redirect, halt ────────────────┼──────────────────────┘   dcache                 │
                     scoreboard · scalar register file · control units ◄──────────────────────────────┘
```

| stage | file | status |
|---|---|---|
| 1 | `isa.py`, `golden.py` | ISA and packet decode, checked against the functional sim |
| 2 | `fetch.py`, `icache.py`, `decode1.py` | front end |
| 3 | `control.py`, `decode2.py` | decode 2: operands, hazards, issue |
| 4 | `semantics.py`, `execute.py`, `dcache.py` | scalar execute, writeback, a non-blocking load/store unit, the data cache |
| 5 | `vector.py` | vector and DMA dispatch into the vector core's units, vector writeback |
| 6 | `atalla/atalla_platform.py` | real kernels end to end on the scheduler-driven platform |

`SchedulerCore(vector_core=..., backends=...)` drives a vector core and the
scratchpads' DMA backends (stage 5); `atalla/atalla_platform.py` builds the
whole machine around it (see Platform). Without a vector core, the vector side's signals (unit
readiness, VLSU readiness, scratchpad busy, and `writeback_at(cycle, ...)` for
vector, mask and SDMA writebacks) are inputs that tests set.
`SchedulerCore(execute=False)` turns the execute stage off too, and makes its
signals (redirect, halt, unit readiness, scalar writebacks) inputs: the
stage 2 and 3 tests drive them by hand.

A program runs with `run_until_done()`: until execute has halted and the data
cache has written back its dirty lines. `scalar_reg(r)` and `memory` hold the
final state; on the platform, `AtallaPlatform.run_until_done()` ticks every
component.

## Decode 2

A packet issues as a whole, in a cycle where all of these hold:

- **No hazard.** No source and no destination of any instruction is busy in
  the scoreboard (read-after-write, write-after-write).
- **Every unit it needs is ready:**
  - scalar EX1–EX5;
  - the vector ALU, MUL and reduction lanes, the GSAU, and the move-to-scalar unit;
  - the VLSU of each scratchpad it names;
  - the scratchpad an SDMA's `rs3[31:30]` names. That's a register *value*, so it's read first.
- **The scalar register file has served every read.**

The scoreboard is a busy bit per register — 256 scalar, 256 vector, 16 mask —
set when the instruction that writes it issues and cleared when its
writeback reaches the register file. Cleared and set on one edge, it stays
set. A dependent packet issues the cycle after its producer's writeback.

Decode 2 owns three register files and reads every operand at issue, so the
units downstream get data, not register numbers, and write-after-read is
safe for every kind of register:

| file | registers | banks | read ports | register 0 |
|---|---|---|---|---|
| scalar | 256 × 32 bits | 4, by `reg[1:0]` | 4 | reads 0 |
| vector | 256 × 32 BF16 | 4, by `reg[1:0]` | 4 | reads all zeros |
| mask | 16 × 32 bits | 2, by `reg[0]` | 2 | reads all ones: `m0` is "every lane" |

Writes to register 0 are dropped in all three. The vector file's storage is
the vector core's Veggie banks, so the registers exist once.

Each file is the same `reggie`: two reads of one bank in a packet put it into
its conflict FSM, which serves one read per bank per cycle, and the packet
issues only when all three files are ready:

| reads on the busiest bank | issue delay |
|---|---|
| 0 or 1 | none |
| *k* ≥ 2 | *k* cycles |

The FSM only starts once the packet's dependencies are free, so a packet
waiting on a writeback pays for its conflict afterwards. Register 0 still
occupies bank 0.

### Read and write sets

Taken from the control units, not the encodings:

| instruction | reads | writes |
|---|---|---|
| `beq.s` … `ble.s` | rs1, rs2 | rs1 (`rs1 += incr7`) |
| `sw.s`, `shw.s` | rs1, the data register in [14:7] | — |
| `rcp.bf`, `sqrt.bf`, `stbf.s`, `bfts.s` | rs1 | rd |
| `jal`, `lui.s` | — | rd |
| `gemm.vv` | vs1 | vd |
| `lw.vi` | vs1, mask | — |
| `vreg.ld` | rs1, rs2 | vd |
| `vreg.st` | rs1, rs2, the vector register in [14:7] | — |
| `mv.stm` | rs1 | mask `rd[3:0]` |
| `scpad.ld`, `scpad.st` | rs1, rs2, rs3 | holds rs1 until the scratchpad is done |

### Packets the RTL cannot execute

The RTL does not stall on these; it drops the extra instruction or zeroes
the extra operand. They are a compiler contract, so decode 2 raises
`PacketContractError` (or counts them with `strict=False`):

- one instruction per scalar EX unit — **EX1 runs ALU and control ops**;
- 4 scalar, 4 vector and 2 mask register reads;
- one GSAU op, one move-to-scalar op, two lane ops;
- one VLSU op and one SDMA per scratchpad.

The functional sim's packet checker allows one ALU op *and* one control op
per packet; the RTL can run only one of them.

## Where the model differs from the RTL

It models what the RTL is meant to do, not its bugs:

| RTL | model |
|---|---|
| A flush clears only D1/D2's scalar slots; wrong-path vector and SDMA ops issue | flush clears all of D1/D2 |
| Decode 2 has no flush input: a packet in decode 2 on a redirect still reserves its registers, then is squashed, and nothing clears them | only a packet that issues reserves |
| `lw.vi` marks its `vd` busy, but a weight load writes no register, so it stays busy forever | `lw.vi` writes nothing |
| `sqrt.bf` goes to EX2, but decode 2's structural check forgets it | `sqrt.bf` waits for EX2 |
| Mask writes are tracked for vector slots 0–1 only, and `mv.stm` is never WAW-checked | every mask write is tracked |

`li.s` is a pseudo-instruction (`addi.s`, or `lui.s` then `addi.s`). The
RTL has no case for it, and neither does the model: the assemblers expand
it (see Kernels).

## Execute and writeback

A packet that decode 2 issues is in the D2/EX latch the next cycle, its EX
cycle. The crossbar sends each scalar op to its unit. Timing is from the
RTL; values are computed by `semantics.py`.

| unit | ops | result | next op |
|---|---|---|---|
| EX1 | ALU, branches, `jal`, `jalr` | EX cycle | every cycle |
| EX2 | `rcp.bf`, `sqrt.bf` (11); `div`, `mod` (66); `bfts.s`, `stbf.s`, `mv.stm` (1) | EX + L | EX + L + 2 |
| EX3 | BF16 add, sub, mul, slt | EX + 1 | EX + 3 |
| EX4 | `mul.s` | EX + 2 | EX + 4 |
| EX5 | loads, stores | see below | when its queue has room |

- **EX1** is combinational. A branch or jump that the BTB mispredicted
  redirects fetch in its EX cycle; the packets behind it are flushed.
- **EX2–EX4** hold their ready low from the EX cycle until the writeback
  arbiter takes the result.
- **Writeback.** A result offered in cycle v and granted is in the EX/WB
  latch in v + 1, which writes the register and clears its busy bit on that
  edge. A dependent packet issues in v + 2. The arbiter grants one write
  per register-file bank (`rd[1:0]`) per cycle by fixed priority EX5, EX1,
  EX4, EX3, EX2; the loser holds its result. `mv.stm` goes to the mask
  registers.
- **Halt.** A packet holding `halt.s` stops fetch and decode when it reaches
  EX. Execute halts once no register is busy and EX5 is idle; the data cache
  then writes back its dirty lines.

### EX5: a non-blocking load/store unit

The RTL's `ld_st_unit` is blocking: it waits out every miss and replays it,
so the data cache's MSHRs are never used. The model's EX5 is non-blocking,
so the cache can work as designed:

- Ops queue in EX5 (4 entries by default, `lsu_depth`); its ready is "the
  queue will have room". An op arriving to an empty queue goes to the cache
  in its EX cycle, as the RTL's does.
- Accesses go to the cache in program order, one lookup at a time. A retry
  (no MSHR free) goes back to the front of the queue.
- A load hit writes back from the hit cycle. A load miss waits in its MSHR
  while EX5 carries on; it writes back when its fill completes. Finished
  loads share EX5's one writeback port.
- A store is done when the cache takes it: on a hit at the SRAM write, on a
  miss at once (the MSHR holds the data).
- Register hazards still come from the scoreboard: a load's `rd` is busy
  until it writes back. Memory ordering comes from the cache: accesses
  reach it in program order, and an access to a block with a miss
  outstanding joins that MSHR in order.

So packets without memory ops keep issuing under a miss, loads that hit are
answered under a miss, and several misses can be outstanding. The cache has
one fill engine, so fills complete one after another, 28 cycles apart.

### Values

Where the RTL and the functional sim disagree, the model follows the ISA
document and the assembler, and otherwise the functional sim:

| | model | RTL |
|---|---|---|
| load/store address | `rs1 + imm` | `imm` |
| `jal`, `jalr` offset | bytes | × 4 |
| branch increment | unsigned 0..127 | sign-extended |
| `blt`/`bge`/`bgt`/`ble` | signed (the functional sim compares unsigned) | signed |
| scalar BF16 values | fp32 layout, BF16 in the upper half | lower half |
| `lhw.s`, `shw.s`, `mod.s` | as the functional sim | differs |
| divide by zero | `rcp.bf` gives ±∞; `div`/`mod` follow RISC-V (−1, the dividend) | not checked (the functional sim raises) |

## Vector and DMA dispatch

`vector.py` takes the vector and SDMA slots of the packet in the D2/EX latch
to the vector core's units (`vector_core/`) and the scratchpad backends
(`memory/backend.py`), following `scheduler_core.sv`'s dispatch:

| ops | unit |
|---|---|
| `add`/`sub`/`mul` `.vv` and `.vs`, compares, `expi.vi`, reductions | the lane datapath, at most 2 a packet |
| `gemm.vv`, `lw.vi` | the GSAU and the systolic array |
| `vreg.ld`, `vreg.st` | the VLSU of scratchpad `sid`: row `rs1 / 64 + rs2` |
| `vmov.vts`, `mv.mts` | move-to-scalar: combinational, through scalar writeback |
| `scpad.ld`, `scpad.st` | the backend of scratchpad `rs3[31:30]`: rows, columns and DRAM row stride from `rs3` |

- **Readiness** to decode 2 is per unit: the lanes' ALU, MUL and EXP, the
  reduction, the GSAU, move-to-scalar, each VLSU, and each scratchpad's busy
  flag while an SDMA runs on it.
- **Writeback.** One vector register write per bank (`vd[1:0]`) per cycle,
  by fixed priority VLSU 0–3, GSAU, reduction, lanes; the loser holds its
  result. Compares write the mask file. A write offered in cycle v reaches
  the register file in v + 1. An SDMA's completion clears its `rs1`, which
  is how a later `vreg.ld` waits for the data it brings in.
- **Halt** also waits for the vector side to drain: lanes, GSAU, VLSUs and
  DMA.

Where the RTL is wrong this does what was meant: all four VLSUs write back,
write ports follow the register file's bank `vd[1:0]`, and `lw.vi` writes no
register.

**Values** come from the units; the scheduler only routes operands to them
and formats what comes back for writeback.

- **The lane datapath** (`vector_core/vector_lanes.py`) does the element ops:
  BF16 operands, the op, the result rounded to BF16 (through fp32, as the
  functional sim rounds).
- **Reductions** are the lane model's: each lane sums (or compares) the
  active elements of its slice, a pairwise tree combines the lanes that had
  one, and the result is placed by `imm[6:5]`: broadcast, alone at
  `imm[4:0]`, or at `imm[4:0]` over `vs1`. The sum leaves the lanes
  unrounded; the 16-bit register file rounds it to BF16.
- **The systolic array** computes `gemm.vv` (see Kernels, Weights).
- **Writeback formatting** is the scheduler's part. It packs a compare's
  1/0 lanes into a mask. Masked-off elements and mask bits keep the
  destination's old value, as the functional sim does; the RTL's result
  collector writes them as 0. `.vs` operands are the scalar register's fp32
  bits, and `vmov.vts` moves an element's fp32 bits.

The functional sim sums a reduction sequentially in fp32 and keeps the
result unrounded in the register. The 16-bit register file can't do that,
and the lanes don't add in that order. So the golden comparisons run the
functional sim with its reductions swapped for the lane model's order
(`golden.run_golden_state(..., reductions="lanes")`).

**Transpose.** `tpus.vi` and `tpop.vi` (opcodes 79 and 78, from the ISA
sheet; neither the RTL nor the functional sim decodes them yet) go to the
vector core's transpose unit, a functional unit of its own:

| instruction | semantics | what the scheduler does |
|---|---|---|
| `tpus.vi vs1` | `transpose_unit <= vs1` | `transpose.push(vs1)`: one row in |
| `tpop.vi vs1` | `vs1 <= transpose_unit` | the next transposed column into the register in the `vs1` field |

The unit drains a whole tile off one `pop()` request, so the first `tpop.vi`
of a tile starts the drain and each `tpop.vi`, that one included, takes one
column, in order. Requests wait in a two-entry queue in front of the unit,
in program order. Its results write back after the GSAU's.

## Platform

`build_atalla_platform(program, data, systolic_array="meissa")` builds
`AtallaPlatform` (`src/atalla/atalla_platform.py`) from the same parts as
the harnesses' `TPUPlatform`, arranged as `rtl/modules/system.sv` arranges
them: the vector core's units, four scratchpad pads with their DMA backends,
DRAM, the VLSU-to-scratchpad bridges, the GSAU-to-array bridge, and the
systolic array (MEISSA, as in the RTL, or `"tpu"`). The scheduler core takes
the vector core's place in the tick order. It drives the units through
their own calls and keeps none of their timing itself:

| op | call |
|---|---|
| lane ops | `vc.datapath.enqueue(...)` |
| `gemm.vv`, `lw.vi` | `vc.gsau.issue(...)` |
| `vreg.ld`, `vreg.st` | the VLSU of scratchpad `sid` |
| `scpad.ld`, `scpad.st` | `backends[sid].driver_to_backend_start_load/store(...)` |

and reads their readiness for decode 2 and their results for writeback.

**Memory.** In the RTL the icache, the dcache and the four scratchpad
backends share one memory (`sim_ram_rr`, in the testbench), round robin.
Here they are six masters on one DRAM channel: one burst launch a cycle
across all of them, `burst_bytes` each, back `dram_latency` (6) cycles
later, taking turns in a round-robin ticker. A burst is 8 bytes by default
(`memory.dram.DRAM_BURST_BYTES`): the RTL's 64-bit DRAM request, four BF16
elements, so the channel peaks at 8 bytes a cycle. A DMA row narrower than a
burst writes only its own elements, as the RTL's lane mask does. The caches reach it through
bus ports (`memory/bus.py`): a line fill goes out as bursts and each 64-bit
beat is taken when its burst is back; written beats gather into bursts in a
small buffer. A cache line fill therefore waits behind the scratchpads'
DMA, and they behind it. The icache starts cold (`warm_icache=True` for
tests that time something else). The ports' `stats` count bursts, bytes,
cycles busy and cycles lost to contention.

## Performance monitor

`build_atalla_platform` attaches a `PerfMonitor` (`src/atalla/perf_monitor.py`)
as `platform.perf`: a read-only observer, ticked last every cycle, that
gathers metrics for workload and system analysis. `tools/run_kernel.py`
prints its summary; `--perf-json PATH` writes the report, and `--timeline`
adds the per-cycle record.

- **Where the cycles go.** Every cycle goes in one bucket, which together
  add up to the run: `issue` (a packet issued); `stall:hazard:<unit>` (decode
  2 waited on a register, put down to the unit producing it: `gemm`,
  `vreg_load`, `sdma`, `vector_lane`, a scalar class...); `stall:<unit>` (the
  unit the packet needs was busy); `stall:regfile` (a register-file bank
  conflict); `frontend:icache` (nothing to issue while the icache fills);
  `frontend:refill` (after a flush, or at start); `halt:drain` (halted,
  waiting for work in flight and the dcache's writebacks).
- **Unit busy:** cycles each unit had work (EX2-4, the load/store unit, the
  dcache, the lanes, the GSAU, the array, each VLSU, the transpose unit,
  each scratchpad's DMA, the DRAM channel), and the fraction of the run.
- **Issue and mix:** packets and ops per cycle, slot use, and the retired
  instruction mix by class.
- **Memory:** bursts and bytes per DRAM master (icache, dcache, scratchpads
  0-3) and the cycles each waited for the channel; channel occupancy,
  bandwidth used against peak; icache fills, the dcache's counters, the
  scratchpad pads' reads, writes and stalls.
- **Work:** FLOPs from the lanes and the array, FLOP/cycle, the array's MAC
  utilisation, and arithmetic intensity against the bytes actually moved
  over DRAM.

On gemm, for example, a third of the cycles are decode 2 waiting for
`gemm.vv` results and another sixth for `vreg.ld`s; on softmax, half are
the icache filling straight-line code.

## Kernels

`tools/run_kernel.py` runs a kernel through the whole machine: the
instruction stream goes to the scheduler, which drives the vector core's
units, the systolic array, the scratchpads with their DMA, and DRAM.

```
tools/run_kernel.py gemm              # build kernels/build_gemm.py and run it
tools/run_kernel.py prog.in           # an already assembled program
tools/run_kernel.py softmax --golden  # also run the functional sim and compare
```

`--golden` compares against the functional sim with the lane model's
reduction order; add `--functional-reductions` to compare against its own fp32 sums.
`--warm-icache` starts with the program in the icache.

A kernel name is assembled by the functional sim's own build script
(`third_party/atalla-functional-sim/kernels/build_<name>.py`), in its `.in`
format. From Python, `scheduler.core.load_program_text(text)` reads an
`.in` and `atalla_platform.build_atalla_platform(instr, data)` builds the
machine around it.

Every kernel below runs to halt and matches the functional sim exactly at
the end: every register, and DRAM as the DMA reads it. That shows the two
simulators agree, not that the kernel is right; see *Which kernels compute
what they claim* below.

| kernel | cycles |
|---|---:|
| `add`, `relu`, `sigmoid` | 2,172 / 2,105 / 441 |
| `layernorm_param`, `maxpool` | 4,034 / 1,415 |
| `gemm`, `gemm_tiled` | 2,584 / 1,397 |
| `gemms`, `gemms_function`, `gemms_pipelined`, `…_loop_unroll` | 9,552 / 8,679 / 8,661 / 8,829 |
| `conv`, `conv_tiled`, `conv_pipelined`, `conv_unrolled_pipelined` | 3,160 / 1,604 / 3,181 / 3,195 |
| `softmax`, `attention`, `flash_attention` | 2,694 / 65,654 / 20,795 |

Cycle counts are with the default platform: the MEISSA array, 16 lanes,
DRAM latency 6, 8-byte DRAM requests, and a cold instruction cache fetching
over the shared DRAM channel. Against the functional sim's own sequential fp32
sums, `softmax`'s outputs differ by at most 2 BF16 steps.

**C programs.** They go straight from the compiler to atalla-sim:

    C --atalla_cc -p -S--> packetized assembly --src/scheduler/assembler.py--> image

`third_party/aihw-ppci-compiler` (branch `atalla-shaunak`) packetizes: its
output is already four-slot packets, padded with `nop`, a packet per label,
within the scheduler's slot limits. The assembler only encodes and lays out
what it is given (a `jal x1, main; halt.s` bootstrap first, `li.s` expanded),
and its encodings match the functional sim's assembler word for word.
`tools/run_kernel.py add.c` (a path, or a file in the compiler's
`test_validation_kernels/` or `test_validation/`) compiles, assembles and
runs one, seeds DRAM and checks the result with the compiler's own
validation harness (numpy references; `x10` for the unit tests), and
`--golden` also runs the functional sim on the identical image. The pieces
are in `tools/c_kernels.py`. It runs with the C loader's stack registers set
(`c_entry_sregs`: `x2` and `x33` above the data, as the functional sim's
loader does).

Against the functional sim (`tests/scheduler/test_kernels.py` lists each case):

| | programs |
|---|---|
| identical | add, relu, maxpool, maxpool_2x2, softmax, conv_*, gemm_tiled_pipelined, gemm_tiled_pipelined_unrolled, scalar_sum_loop, sqrt_perfect_square |
| differ: a DMA race | scpad_load_roundtrip: its vector load reads the scratchpad before its DMA has filled it (the functional sim has no timing, so it never races) |
| differ: `vector_load` on a DRAM address | 9 unit tests: only the functional sim allows it; the hardware's `vreg.ld` reads the scratchpad |
| differ: `num_cols` | gemm_tiled_baseline, in a vector register only (DRAM matches): `vreg.ld`/`vreg.st` move 32 lanes where the functional sim moves `num_cols + 1` |
| differ: misaligned `sw.s` | layernorm (stack only): aligned down here, written at the byte address there |

### Which kernels compute what they claim

Checked against numpy on each kernel's own inputs (read back from its
program image), and by reading the kernel against the instruction and
intrinsic semantics. A kernel can match the functional sim and still be
wrong: both simulators then run the same wrong program.

| kernel | result | problem |
|---|---|---|
| C `add`, `scalar_sum_loop`, `sqrt_perfect_square` | correct | |
| C `scpad_load_roundtrip` | a race, from a compiler bug | the kernel passes the same address variable to `scpad_load` and `vector_load`, which is all C can say. But a `vreg.ld` waits for a DMA only by reading the register the DMA holds (its `rs1`, which the RTL's decode 2 marks busy until the DMA is done), and the compiler doesn't model that: `scpad_load` returns nothing in its IR, the address is put in a new register for the load, and the packetizer assumes a fixed 13-cycle DMA. The load reads stale data unless the DMA finishes first, as it did with 32-byte bursts. `relu` loads its zero vector the same way, harmlessly |
| C `relu`, `maxpool` | correct, by luck | they rely on a masked op leaving its masked-off lanes holding the destination's old value, which the intrinsics don't define; it works because the compiler happens to pick the kept value's register as the destination. `maxpool` also reads its zero vector before loading the scratchpad, and pools vertically only (2×1). `relu` writes −0.0 for negative inputs |
| C `make_mask_eq_vector` | passes without testing anything | it compares a vector with itself, loaded from a DRAM address with no DMA (and from row 1: the second argument of `vector_load` is a row offset, not a row count) |
| C `softmax`, `layernorm`, `maxpool_2x2` | wrong | known compiler bugs (reductions write lane 0 only; masked-op merge) |
| C `conv_*`, `gemm_tiled_*` | wrong | weight rows loaded first to last, so output columns are reversed |
| asm `relu`, `softmax` | correct | `softmax` uses inputs 0..1023, which BF16 can't hold exactly, at an unaligned address (`0xcafa`) |
| asm `maxpool` | correct for a vertical 2×1 pool | the name and `--pool 2` suggest a 2×2 pool; the width is never pooled |
| asm `sigmoid` | computes `exp(-x)`, not a sigmoid | |
| asm `add` | the arithmetic is right, the test isn't | A, B and C are placed 128 bytes apart but are 2 KB each, so they overlap and C's zero fill overwrites most of A and B |
| asm `gemm`, `gemm_tiled` | compute `A @ Wᵀ` | their own reference is `A @ W`: they push W's rows into the array, where `gemm.vv` needs its columns |
| asm `gemms` | wrong | it advances its input pointer with `addi.s $2, $2, 2048`; the immediate is 12-bit signed, so 2048 is encoded as −2048 and tiles 1 and 2 are read from the wrong addresses (atalla-sim's own assembler rejects this) |
| asm `conv`, `conv_tiled` | wrong | they push W's rows (27×4) where the columns are needed, so only part of W reaches the array; `conv` also builds its lane mask as `0x07FFFF7F`, leaving lane 7 out |
| asm `layernorm_param` | wrong | its DMA control word puts the matrix width (32) in the stride field, which takes width − 1, so every row is read and written one element out of place and one element per row is never normalized; it also normalizes the whole tile, not each row, has no γ/β, and its ε is 0 |

These are problems in the kernels, which come from the compiler and the
functional sim repositories, not in atalla-sim: report them there.

**Weights.** `lw.vi` shifts its vector into the systolic array's weights as
column 0, moving every column one to the right, as the array is built
(MEISSA `mul_grid.sv`). Kernels load their weight rows last to first, so
column j ends up holding `w_j`, and `gemm.vv` computes `out[j] = vs1 · w_j`
with the functional sim's rounding (BF16 operands, fp32 dot products, a BF16
result). To compute `x @ W`, push W's columns, last first. The systolic array model computes the
values: its weight path shifts each vector in at column 0, and
`ArrayValueBridge` (`atalla_platform.py`) hands its BF16 rows back to the
GSAU, which pairs them with `vd`.

**`li.s`** is a pseudo-instruction the hardware doesn't decode. The
assemblers (`build.py` for assembly kernels, `src/scheduler/assembler.py`
for C programs) expand every one: `addi.s rd, $0, imm` when it fits 12
bits, otherwise `lui.s` then `addi.s`.

## Data cache

`dcache.py` is the scalar core's data cache: a lockup-free cache with miss
status holding registers. It is what the RTL's `rtl/modules/scheduler/dcache/`
sets out to be, without its known bugs, and its geometry is configurable (`DCacheConfig`):

| parameter | default (the RTL's) |
|---|---|
| size, ways, line | 4 KB, 4, 64 B (16 sets) |
| MSHRs, targets per MSHR | 8, 8 |
| SRAM read / write latency | 2 / 4 (`sram_bank`) |
| memory | 64-bit beats; on the platform, bursts on the shared DRAM channel (see Platform); alone, first-beat and write latency are parameters, 0 by default |

- **Policy:** write-back, write-allocate; an invalid way, else tree
  pseudo-LRU.
- **Lookups** take one request at a time over a valid/ready port. Answers:
  load hit in 4 cycles, store hit in 9, store miss at once, load miss
  `miss` then `fill` with the data, `retry` when no MSHR or target slot is
  free.
- **Fills** serve the MSHRs oldest first: read the set, invalidate the
  chosen way, burst the line in, write back a dirty victim, write the line
  with the MSHR's stores applied in order, answer its loads. 28 cycles from
  the miss, 36 with a dirty victim.
- Lookups have priority on the SRAM, but a fill gets it in the cycle a
  lookup's read completes, so it can't be starved.

## Tests

`tests/scheduler/test_decode2.py` derives every expected cycle by hand:
- issue timing, read-after-write, write-after-write, SDMA holds, mask tracking, flush;
- bank conflicts of every size;
- each structural stall.

It also checks the control units' bit slices against the ISA layout on
random words.

`tests/scheduler/test_execute.py` checks stage 4:
- hand-derived timing for each unit, writeback bank conflicts, branches and
  loops;
- EX5 with the cache: hit and miss latency, issue continuing under misses,
  hits under a miss, overlapping misses, a store miss followed by a load of
  the same word, retries when the MSHRs are full, a full EX5 queue, and
  halt draining stores;
- random scalar programs with loads and stores, run on the model and the
  functional sim, with every register and memory word compared at the end,
  on the default cache and on a small 2-way cache with 2 MSHRs. Every run
  also checks that accesses reached the cache in program order.

`tests/scheduler/test_dcache.py` tests the data cache alone.

`tests/scheduler/test_kernels.py` builds kernels with the functional sim's
build scripts and runs them on both: `add`, `relu`, `sigmoid`, `layernorm`,
`maxpool`, `softmax`, `gemm`, `gemm_tiled`, `gemms`, `conv` and `conv_tiled`
must match exactly against the functional sim with the lane model's reduction
order, and `softmax` within 2 BF16 steps of its own fp32 sums.

`tests/scheduler/test_vector.py` checks stage 5 on the scheduler-driven
platform:
- vector and mask register-file bank conflicts, `v0` and `m0`;
- SDMA holding its `rs1`, dependent vector ops waiting for writeback, two
  lane ops and a VLSU op in one packet, halt draining the vector side, the
  GSAU path completing;
- directed and random vector programs (lane arithmetic, `.vs`, compares and
  masked ops, `expi`, reductions, moves to scalar, SDMA and VLSU traffic)
  run on the model and the functional sim, comparing every scalar, vector
  and mask register and the DRAM it wrote.
