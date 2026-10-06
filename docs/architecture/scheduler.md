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
| 5 | `vector.py`, `platform.py` | vector and DMA dispatch into the vector core's units, vector writeback |
| 6 | — | platform wiring in `build_tpu_platform`, end-to-end kernels |

`SchedulerCore(vector_core=..., backends=...)` drives a vector core and the
scratchpads' DMA backends (stage 5); `scheduler/platform.py` builds the whole
machine around it. Without a vector core, the vector side's signals (unit
readiness, VLSU readiness, scratchpad busy, and `writeback_at(cycle, ...)` for
vector, mask and SDMA writebacks) are inputs that tests set.
`SchedulerCore(execute=False)` turns the execute stage off too, and makes its
signals (redirect, halt, unit readiness, scalar writebacks) inputs: the
stage 2 and 3 tests drive them by hand.

A program runs with `run_until_done()`: until execute has halted and the data
cache has written back its dirty lines. `scalar_reg(r)` and `memory` hold the
final state; on the platform, `SchedulerPlatform.run_until_done()` ticks every
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

`li.s` is a pseudo-instruction (`lui.s` then `addi.s`). The RTL has no case
for it, correctly, and neither does the model. The functional sim's
assembler currently emits it as a real instruction (opcode 47) and runs it,
so a program built that way would lose it on the hardware.

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
| load/store address | `rs1 + imm` | `imm` (bug 20) |
| `jal`, `jalr` offset | bytes | × 4 (bug 21) |
| branch increment | unsigned 0..127 | sign-extended (bug 22) |
| `blt`/`bge`/`bgt`/`ble` | signed (the functional sim compares unsigned) | signed |
| scalar BF16 values | fp32 layout, BF16 in the upper half | lower half |
| `lhw.s`, `shw.s`, `mod.s` | as the functional sim | see the bug note |

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

Where the RTL is wrong this does what was meant: all four VLSUs write back
(bug 4), write ports follow the register file's bank `vd[1:0]` (bug 5), and
`lw.vi` writes no register (bug 3).

**Values** follow the functional sim's lane rules: operands rounded to BF16,
the op in fp32, the result rounded to BF16; compares on the raw values;
`.vs` operands are the scalar register's fp32 bits; masked-off lanes keep the
old destination; reductions sum or compare the active lanes in order and
place the result by `imm[6:5]`. Two differences, both because the vector
registers are 16-bit:

- a reduction's result is rounded to BF16; the functional sim writes its
  fp32 sum into the register unrounded;
- the lane datapath model only times the ops: its BF16 cast is FP16 while
  numpy has no bfloat16, so the values are computed in `vector.py`.

**Open:** `gemm.vv` is timed through the GSAU and the systolic array model,
but its values don't follow the ISA yet. The array takes each `lw.vi` as a
weight row where the ISA loads a column (`out[j] = vs1 · w_j`), and its BF16
is FP16 too.

## Data cache

`dcache.py` is the scalar core's data cache: a lockup-free cache with miss
status holding registers. It is what the RTL's `rtl/modules/scheduler/dcache/`
sets out to be, without its bugs (8–10, 15, 17–19 in the bug note), and its
geometry is configurable (`DCacheConfig`):

| parameter | default (the RTL's) |
|---|---|
| size, ways, line | 4 KB, 4, 64 B (16 sets) |
| MSHRs, targets per MSHR | 8, 8 |
| SRAM read / write latency | 2 / 4 (`sram_bank`) |
| memory | 64-bit beats; first-beat and write latency are parameters, 0 by default |

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
