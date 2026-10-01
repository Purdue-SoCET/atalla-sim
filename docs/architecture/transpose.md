# Transpose Unit

`src/vector_core/transpose.py` — a `VEC_LEN x VEC_LEN` tile transpose.
Rows go in one vector per push; columns come out one vector per pop.

Cycle-matched to the RTL: `rtl/modules/vector/transpose_unit.sv` on atalla's
`transpose_integration` branch (b1ba35ff), with the `sram_bank` and `clos` it
instantiates. The storage scheme comes from the architecture doc,
`docs/src/architecture/transpose.md` on `documentation_update_branch`; the
timing does not — see [Checked against the RTL](#checked-against-the-rtl).

## What it is

VEC_LEN single-port SRAM banks with a Clos network in front of them. Vectors
arrive a row at a time, so the unit can never see the whole matrix before it
starts storing it. It skews each row instead:

```
  write row r:   element i  ->  bank (i + r) % N,  address r
  read column c: bank (r + c) % N, address r  ->  output position r
```

The write rotation run backwards is the read rotation, so column `c` comes
back whole. Neither direction touches a bank twice, so there is no conflict
either way.

Both rotations are permutations of a vector across banks, which is exactly
what `memory.crossbar.Xbar` models, so the Clos network **is** an `Xbar`: the
per-input destination index is its shift mask and its delay is the network's
depth: a 3-stage pipe — input, center and output modules, with two latches
between them. Its counters (`total_submitted`, `total_completed`) cover the
transpose for free.

```
  a 4-wide tile, pushed row by row        where the banks end up
                                          (row = address)
   row 0   a0 a1 a2 a3                          B0  B1  B2  B3
   row 1   b0 b1 b2 b3                    0     a0  a1  a2  a3
   row 2   c0 c1 c2 c3                    1     b3  b0  b1  b2
   row 3   d0 d1 d2 d3                    2     c2  c3  c0  c1
                                          3     d1  d2  d3  d0

  column 1 is [a1 b1 c1 d1]: B1@0, B2@1, B3@2, B0@3 -- one element per
  bank, on a diagonal, so the whole column is read in a single access.
```

Banks start at 0.0, and M x 32 tiles work without knowing M up front: push M
rows, pop, and the rows never pushed read back as whatever the banks held —
zeros on a fresh unit, the previous tile's rows otherwise, since a pop rewinds
the shared row/column counter but clears nothing. The RTL behaves the same.

## Cycles

The RTL handles one row or one column at a time, and every step waits on a
counter or a bank's done flag:

```
  push   IDLE (accept) -> WAIT_CLOS_WRITE x3 -> BUSY_WRITE x5 -> IDLE         9
  pop    IDLE (pop seen) -> POPPING -> WAIT_SRAM x3 -> WAIT_CLOS_READ x3 -> DONE
                              ^                                            |
                              +----------- next column, once taken --------+   8
```

| step | cycles | why |
|---|---|---|
| accept | 1 | a push is taken in IDLE (`ready_in = state == IDLE`) |
| `WAIT_CLOS_*` | 3 | `lat_count` 0..2: the vector crosses the 3-stage Clos pipe; it leaves on the write enable's edge |
| `BUSY_WRITE` | 5 | wait for the banks' write done |
| `POPPING` | 1 | read enable |
| `WAIT_SRAM` | 3 | wait for the banks' read done |
| `DONE` | 1+ | `valid_out` held until the consumer takes the column |

The bank waits come from `sram_bank.sv` — modelled once, in
`memory/sram_bank.py`, and shared with the scratchpad — which raises done one
cycle after the enable for a latency of 0 or 1, and `latency + 1` cycles after
for anything longer. `transpose_unit.sv` instantiates its banks without overriding the
defaults — read 2, write 4 — hence 3 and 5. Those defaults are the model's too;
whether the RTL means them is an open question for the RTL.

So a row costs **9** cycles, a column **8**, and one pop drains the tile:
1 cycle for IDLE to see the pop, then 32 columns — **257** cycles to the end of
the last `DONE`, the last column handed over 256 cycles after the pop is
taken. An M x 32 tile is 9M + 257 cycles in the unit. All three latencies are
constructor parameters; `push_cycles`, `column_cycles` and `drain_cycles` give
the costs for whatever they are set to.

`DONE` is the stall state. There is no output queue: if the consumer cannot
take the column, the FSM stays in `DONE` holding it, and the drain stretches
by every cycle it waits.

## API

```python
unit = TransposeUnit(vec_len=32)

unit.push(vec)           # feed one row; False when the unit is not idle
unit.pop(dsts=None)      # start a drain of all VEC_LEN columns
unit.tick(cycle)         # advance one cycle

unit.can_pop_writeback() # a transposed column is valid (FSM in DONE)
unit.pop_writeback()     # {"col": c, "data": [...], "dst": ...}; moves on
                         # at the next tick

unit.ready_in            # accepting a request this cycle (IDLE)
unit.valid_out           # holding a column the consumer has not taken
unit.busy                # FSM is not IDLE
```

The timing above holds when the caller does what `VectorCore` does each cycle:
make requests before `tick()`, take writebacks after it.

`dsts` optionally names a destination register per column; it rides along into
the writeback entries. `next_wake()` returns `None` whenever the unit is idle
with nothing requested, so it costs no ticks when unused.

## As a functional unit

The transpose is a peer of the GSAU and the four VLSUs: its own VLIW slot
group, its own issue path, its own writeback source.

```
  VLIW packet     gsau x1 | vlsu x4 | transpose x1 | datapath x2
                                          |
                    _issue_transpose  ----+
                                          |
                    TransposeUnit  -> writeback buffer -> Veggie
```

```python
vc.enqueue_scheduler_instruction(
    {"unit": "transpose", "kind": "push", "src": 5})       # a register, or a vector
vc.enqueue_scheduler_instruction(
    {"unit": "transpose", "kind": "pop", "dst": 128})      # 128, 129, ... 159
vc.enqueue_scheduler_instruction(
    {"unit": "transpose", "kind": "pop", "dst": [20, 10, 30, 12]})
```

One slot per packet, like the GSAU. Its ready is the RTL's `ready_in`: idle.
A push is refused while the unit is busy and stays in its packet, so the
scheduler backpressures naturally — pushes issue every 9 cycles at best. A pop is
one instruction for the whole tile — the unit drains all VEC_LEN columns off
a single request, so there is no per-column handshake to issue against; the
destinations ride along and come back attached to each column's writeback.

`PACKET_UNITS` in `vector_core.py` is the slot list. Adding a unit is an entry
there plus an `_issue_<unit>` and a writeback candidate.

## The ISA spec disagrees

The Atalla ISA spreadsheet defines `tpus.vi` (79, `transpose_unit <= vs1`) and
`tpop.vi` (78, `vs1 <= transpose_unit`), and its `tpop.vi` writes **one**
vector register per instruction. The RTL has neither opcode, and its
`transpose_unit.sv` drains the whole tile off a single `pop_req`. This model
follows the RTL: one pop instruction, 32 destination registers. The scheduler
model cannot issue either instruction until the RTL decodes them.

## Feeding it

The unit is a functional unit fed from the VLIW bundles: a push takes a row
from the register file, a pop writes columns back to it. The RTL reads
`vec_in` for the whole push and relies on its producer to hold it, so the
issue path needs an input register; the model latches the row when it
accepts it.

## Checked against the RTL

`tests/vector_core/test_transpose_rtl_trace.py` replays a Questa run of the
RTL's own unit testbench, `tb/unit/vector/transpose_unit_tb.sv` — 47,126
cycles: every tile height 1..32, with and without output backpressure. It
drives the model with the trace's inputs and, before every clock edge, checks
the RTL's state, row/column counter, `lat_count`, `ready_in`, `valid_out`, the
bank enables and done flags, and every transposed column. They agree on every
cycle. A one-cycle change to any latency fails at the first cycle it touches.
`tests/vector_core/data/README.md` says how to regenerate the trace.

That testbench asserts `pop_req` once per column. The RTL drains the whole
tile off one request, so the per-column requests are redundant, and the one
still high when the FSM returns to IDLE starts a second, unchecked drain — 63
of the 64 tests do the whole drain twice. The data checks still pass; the
next test just waits ~257 cycles for the extra drain.

## What the model leaves out

- **Clos port ordering.** The RTL reverses indices within each output module
  and cancels the reversal again at the bank write and at the unit's output.
  It has no effect on what the unit produces, so it is not modelled.
