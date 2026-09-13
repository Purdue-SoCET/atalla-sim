# Vector Core: old vs new

The vector core kept its top level and its units. What changed is how a vector
is cut up across the lanes, how many scratchpad ports it has, and whether
operands actually travel through the register file's read ports.

`old` is the core as it stood before this change. `new` is the same core after
it. The slicer and the 4-VLSU count come from the SystemVerilog on the
`Vector_S26_L1_TB` branch of the atalla RTL repo (`NUM_SCPADS = 4`).

---

## Top level — unchanged

Both are one `VectorCore`, a `Clocked` object the platform registers.

```
   VectorCore                                   <- the Clocked object
   │
   ├── scheduler          VLIW packet: gsau | vlsu | transpose | datapath
   │     vliw_q
   │
   ├── VectorDatapath     the SIMD side
   │     ├── lane0 .. laneN
   │     └── ResultCollector ── GlobalReductionUnit
   │
   ├── VLSU x N           scratchpad load/store
   │     issue_q, req_q, rsp_q, wb_q, load_dst_fifos
   │
   ├── GSAU               systolic array ingress/egress
   │
   ├── TransposeUnit      Clos crossbar + SRAM banks, added since
   │
   ├── Veggie             vector register file, 4 banks x 64
   │
   └── WBBuffer           staging before the VRF
```

Every box but the last is in both; the wiring between three of them is not.
The transpose unit is new to this side — see [transpose.md](transpose.md).

---

## What changed

```
            OLD                                  NEW
            ───                                  ───

  scheduler                            scheduler
      │                                    │
      │  operands read straight            │  operands requested from the
      │  out of veggie.data_banks[]        │  VRF's read ports
      │                                    ▼
      │                                 Veggie ── one read per bank
      │                                    │      per cycle
      │                                    ▼
      │                                 OpBuffer ── holds a partial pair
      │                                    │        until both land
      ▼                                    ▼
  VectorDatapath                       VectorDatapath
      │                                    │
      │ round-robin stripe                 │ slicer: contiguous
      │ lane = e % lane_count              │ lane = e // slice_w
      ▼                                    ▼
   lane0..laneN                         lane0..laneN
      │                                    │
      ▼                                    ▼
  ResultCollector                      ResultCollector
   idx = lane + i*lane_count            idx = lane*slice_w + i
      │                                    │
      ▼                                    ▼
   WBBuffer ──► Veggie                  WBBuffer ──► Veggie
                                        (unchanged)

  VLSU x2                              VLSU x4
```

| | old | new |
|---|---|---|
| top level | `VectorCore` | same |
| lanes | `lane_count`, silently clamped to `min(lane_count, vector_len)` | `lane_count`, must divide the vector or it raises |
| slice width | n/a | `slice_w = vector_len // lane_count` |
| element → lane | round-robin stride | contiguous slice |
| FUs per lane | 5 | 5 — unchanged |
| VLSUs | 2 | **4**, one per scratchpad |
| operand path | direct `data_banks[]` read | VRF read ports → `OpBuffer` |
| bank conflicts | not modelled | cost a cycle |
| writeback | `WBBuffer` | same |

The FUs are untouched: `alu` 4 cycles, `sqrt` 8, `exp` 14, `div` 11, `shift` 3,
one pipeline each per lane. (The RTL model has only 2 FUs per lane, ALU and
MUL, both 2 cycles — that difference was not carried over.)

---

## What the slicer does

It decides which lane computes which element of the vector. That is the whole
job, and it is the single biggest behavioural difference between the two.

```
  A 8-element vector across 4 lanes.

  OLD — round-robin stride,  lane = e % lane_count
    lane0 │ e0      e4
    lane1 │ e1      e5
    lane2 │ e2      e6
    lane3 │ e3      e7
           └─ each lane hops by lane_count

  NEW — contiguous slice,    lane = e // slice_w,  slice_w = 8//4 = 2
    lane0 │ e0  e1
    lane1 │ e2  e3
    lane2 │ e4  e5
    lane3 │ e6  e7
           └─ each lane owns an adjacent run
```

Three pieces have to agree on this, or results land in the wrong element:

```
   slicer                    lane                     collector
   lane_slice_indices()  ->  computes its slice  ->  lane*slice_w + i
```

`slice_to_lane(e, slice_w)` is the forward map, `lane_slice_indices(lane,
slice_w)` the reverse. Both live in `vector_lanes.py` and are pinned by tests
that check every element lands in exactly one lane and that the map inverts.

Why it matters: it changes which elements share a functional unit. Anything
non-uniform across a vector — a mask, a data-dependent stall — now falls on a
different set of lanes than it used to.

---

## What the operand collector does

Before, a datapath instruction read its sources by indexing storage directly:

```python
raw = self.veggie.data_banks[bank][addr]     # no ports, no arbitration
```

Now the sources are requested at the register file's read ports, and the
collector gathers them:

```
   instruction needs v1, v2
        │
        ▼
   read_reqs ──► Veggie        one read granted per bank per cycle;
        │                      the loser is re-driven next cycle
        ▼
   dvalid/vreg ──► OpBuffer    slot i owns ports 2i and 2i+1;
        │                      holds a partial pair until both arrive
        ▼
   slot_ready(i) ──► issue to the datapath
```

`bank = reg % 4`, so two sources in the same bank serialise:

| sources | banks | cycles to writeback |
|---|---|---|
| `v0, v1` | 0, 1 | 8 |
| `v0, v4` | 0, 0 | 9 |

Immediates never touch a bank, so they are handed to the collector directly
via `present()` — the slot still needs both halves before it can issue.

Two stubs had to go for this to be real. `OpBuffer`'s ready line was
`any(dready) and any(mready)`, carrying a comment reading *"For this test"* —
one arrived operand marked every slot ready. And `Veggie` pushed conflicting
requests onto a `pending_reqs` list that nothing ever read, so a request that
lost its bank was silently dropped rather than retried.

---

## What the writeback buffer does

Unchanged, and it was already correct — it is the staging point between the
units and the register file.

```
   lanes ─┐
   VLSUs ─┼─► WBBuffer ──► Veggie
   GSAU  ─┘   depth-limited,     one commit
              one entry per      per cycle
              VRF bank in flight
```

Two rules, both now pinned by tests: a second write to a bank already queued is
refused until the first commits, and the queue is FIFO.

---

## Related

- [vector-core.md](vector-core.md) — the core's shape and API
- [transpose.md](transpose.md) — the transpose unit
