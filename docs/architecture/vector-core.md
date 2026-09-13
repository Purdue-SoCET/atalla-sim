# Vector Core

A cycle-level model of the vector core. It executes VLIW packets: each packet
carries one slot group per functional unit, and each group feeds that unit.

For what changed recently and why, see
[vector-core-old-vs-new.md](vector-core-old-vs-new.md).

## Shape

```
   scheduler ── one VLIW packet: gsau x1 | vlsu x4 | transpose x1 | datapath x2
      │
      ├─ vlsu ──────► VLSU x4, one per scratchpad
      │
      ├─ datapath ──► slicer ─► lane0 .. laneN
      │                           └─► ResultCollector ─► reduction tree
      │
      ├─ transpose ─► TransposeUnit ─► Clos xbar + 32 SRAM banks
      │
      └─ gsau ──────► GSAU ─► systolic array

   operands   Veggie (VRF) ─► read ports ─► OpBuffer ─► issue
   results    units ─► WBBuffer ─► Veggie
```

| | |
|---|---|
| lanes | `lane_count`, must divide the vector |
| elements per lane | `slice_w = vector_len // lane_count`, contiguous |
| FUs per lane | 5: alu, sqrt, exp, div, shift |
| FU latency | 4 / 8 / 14 / 11 / 3 cycles |
| VLSUs | 4, one per scratchpad |
| transpose | 1, a `vector_len` square tile — [transpose.md](transpose.md) |
| vector registers | 256, over 4 banks |
| vector length | `veggie_size // 16` |

## The cycle

`VectorCore` is the `Clocked` object the platform registers. Each tick:

```
1. issue      take a VLIW packet, claim units that can accept
2. operands   drive VRF reads, collect them in the OpBuffer
3. advance    datapath, VLSUs, GSAU, transpose
4. collect    unit results into the WBBuffer
5. commit     one writeback into the VRF
```

Operands are read through the register file's ports, so two sources in the same
bank (`bank = reg % 4`) serialise and cost an extra cycle.

## API

```python
vc = VectorCore(veggie_size=32 * 16, lane_count=4, vls_count=4, dtype="fp16")
```

Enqueue work — each returns False when the scheduler queue is full:

```python
vc.enqueue_scheduler_instruction({"unit": "datapath", "op": "add",
                                  "dst": 3, "src0": 1, "src1": 2})
vc.enqueue_memory({"kind": "load", "vls": 0, "dst": 5, "addr": 0})
vc.enqueue_scheduler_instruction({"unit": "transpose", "kind": "push", "src": 5})
vc.enqueue_vliw_packet({"datapath": [...], "vlsu": [...], "gsau": [...],
                        "transpose": [...]})
```

A source is a register index or an inline vector. Ops: `add sub mul max min
and or xor sqrt exp div shl shr`.

Registers, for setup and observation:

```python
vc.write_vreg(r, data, dtype="fp16");  vc.read_vreg(r)
```

Run and read results:

```python
vc.tick(cycle)
vc.wb_valid          # something committed this cycle
vc.last_wb           # what committed
```

Ports for the platform bridges:

```python
vc.pop_scratchpad_request(vls_id);  vc.push_scratchpad_response(vls_id, rsp)
vc.pop_systolic_request();          vc.push_systolic_response(rsp)
```

## Related

- [vector-core-old-vs-new.md](vector-core-old-vs-new.md) — what changed, with diagrams
- [base-classes.md](base-classes.md) — the tick/cycle model
- [queue-glossary.md](queue-glossary.md) — the FIFOs
