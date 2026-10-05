# Scheduler

`src/scheduler/` — a cycle-level model of Atalla's control processor,
transcribed from the RTL on atalla's `scheduler_integration_SP26_joshklug`.
One `SchedulerCore` is the clocked object; every stage inside it is an
`RTLModule`, driven each cycle as readiness, then data and next state, then
one commit, so no stage sees another's value early.

```
 PC ─► I$ + BTB ─► IF/D1 ─► decode 1 ─► D1/D2 ─► decode 2 ─► D2/EX ─► (execute: stage 4)
                                                   │
                     scoreboard · scalar register file · control units
```

| stage | file | status |
|---|---|---|
| 1 | `isa.py`, `golden.py` | ISA and packet decode, checked against the functional sim |
| 2 | `fetch.py`, `icache.py`, `decode1.py` | front end |
| 3 | `control.py`, `decode2.py` | decode 2: operands, hazards, issue |
| 4 | — | execute, writeback, dcache |
| 5 | — | vector and DMA dispatch into `VectorCore` |
| 6 | — | platform wiring |

Signals from stages not built yet are inputs on the core that tests set:
execute-unit readiness, vector-unit readiness, scratchpad busy, and
`writeback_at(cycle, ...)` for what the EX/WB latch delivers.

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

The scalar register file is four banks by `reg[1:0]`, read combinationally.
Two reads of one bank in a packet put `reggie` into its conflict FSM, which
serves one read per bank per cycle:

| reads on the busiest bank | issue delay |
|---|---|
| 0 or 1 | none |
| *k* ≥ 2 | *k* cycles |

The FSM only starts once the packet's dependencies are free, so a packet
waiting on a writeback pays for its conflict afterwards. `x0` reads 0, still
occupies bank 0, and ignores writes.

Vector and mask registers live in the vector core's Veggie, which models
their storage and bank arbitration; decode 2 tracks them only in the
scoreboard.

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

## Tests

`tests/scheduler/test_decode2.py` derives every expected cycle by hand:
- issue timing, read-after-write, write-after-write, SDMA holds, mask tracking, flush;
- bank conflicts of every size;
- each structural stall.

It also checks the control units' bit slices against the ISA layout on
random words.
