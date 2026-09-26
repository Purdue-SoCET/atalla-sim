# Transpose Unit

`src/vector_core/transpose.py` — a `VEC_LEN x VEC_LEN` tile transpose.
Rows go in one vector per push; columns come out one vector per pop.

Modelled from the Atalla RTL repo's architecture doc:
`docs/src/architecture/transpose.md` on `documentation_update_branch`.

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
per-input destination index is its shift mask, the network depth is its delay,
and its tail backpressure is the FSM's `DONE`. Its counters
(`total_submitted`, `total_retire_stalls`) cover the transpose for free.

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

Banks reset to 0.0, so an M-row matrix (M < N) pops back with zeros in the
rows that were never pushed — that is the point of the design: M x 32 tiles
work without knowing M up front.

## Cycles

`clos_latency + sram_latency` per vector, each direction — the network traversal
plus one bank access, and nothing else.

> The architecture doc specifies 2 cycles for the Clos network, giving 3 per
> vector. `CLOS_LATENCY` in `transpose.py` is currently **3**, matching the RTL's
> `lat_count == 2` comparison (which counts 0, 1, 2), so the model costs 4.
> Set it to 2 for the doc's figure. The tests derive from the constant either way.

```
  push   IDLE -> WAIT_CLOS_WRITE (2) --------------> IDLE
                      \__ BUSY_WRITE, only if a bank write takes >1 cycle

  pop    IDLE -> POPPING (1) -> WAIT_CLOS_READ (2) -> column out
                    ^                                     |
                    +-------- next column ----------------+
```

A 1-cycle bank takes the row on the same cycle it leaves the network, so
`BUSY_WRITE` is skipped; it exists to hold a multi-cycle bank.

One pop request drains the whole tile — all VEC_LEN columns — with no
per-column request.

`DONE` is the stall state: if the consumer has no room, the column stays in
the crossbar's tail with `valid_out` high until it is taken.

The three latencies are constructor parameters, so a different Clos depth or
a multi-cycle bank changes the cost without touching the FSM.

## API

```python
unit = TransposeUnit(vec_len=32)

unit.push(vec)           # feed one row; False when the unit is not idle
unit.pop(dsts=None)      # start a drain of all VEC_LEN columns
unit.tick(cycle)         # advance one cycle

unit.can_pop_writeback() # a transposed column is waiting
unit.pop_writeback()     # {"col": c, "data": [...], "dst": ...}

unit.ready_in            # accepting a request this cycle
unit.valid_out           # holding a column the consumer has not taken
unit.busy                # FSM is not IDLE
```

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

One slot per packet, like the GSAU. A push is refused while the unit is busy
and stays in its packet, so the scheduler backpressures naturally. A pop is
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

## What the model leaves out

- **Clos port ordering.** The RTL reverses indices within each output module
  and cancels the reversal again at the bank write and at the unit's output.
  It has no effect on what the unit produces, so it is not modelled.
- **The RTL's real bank latency.** The doc's 3 cycles assume a 1-cycle SRAM
  access, which is this model's default. `sram_bank.sv` actually takes 4
  cycles to write and 2 to read, and the FSM burns one more cycle per vector
  entering and leaving IDLE. Give this model those latencies and a push costs
  7 cycles instead of 3 — the difference is the bank, not the network.
