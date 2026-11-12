# Vector Core v2 (M×N): Work Log

What was changed, how it was tested, and what got stuck, in order. The design is in [`vector_core_v2_change_plan.md`](vector_core_v2_change_plan.md). This log records what was actually built, and every place the build differs from the plan.

Branch: `atalla_sim_mxn`, from `8e956f9`.

Diagram of the core as built: [`vector_core_v2_diagram.html`](vector_core_v2_diagram.html). Open it in a browser.

Proposal for what the rest of atalla needs for software-pipelined activation kernels: [`vector_core_v2_system_proposal.md`](vector_core_v2_system_proposal.md).

End-to-end flow, from C kernel to the units, for v1 today and the v2 brainstorm: [`atalla_flow_v1_v2.html`](atalla_flow_v1_v2.html). Open it in a browser.

Plan for RELU as a CISC instruction (`relu.vk`) driven by a kernel FSM: [`relu_cisc_plan.md`](relu_cisc_plan.md).

## Baseline (2026-11-08, before any change)

| Suite | Result |
|---|---|
| `tests/vector_core`, `tests/scheduler/test_vector.py`, `tests/equivalence` | 99 passed |
| Full suite (`pytest`) | 633 passed, 1 deselected, 27 xfailed (67 s) |

## Scope of the first version (Q33)

Run today's kernels on the M×N core in single mode: milestones M0–M4 + M7, on `build_atalla_platform`, BF16 only, with `read_mode="compat"` (Q34).

---

## Log

### 1. The v2 package: config, FU groups, banked VRF, writeback, core (M0 config, M1–M4)

**New files** (v1 is untouched; everything is under `src/vector_core/v2/`):

| File | What it holds |
|---|---|
| `config.py` | `FUGroupSpec`, `VectorCoreConfig` (every default is a plan decision), `make_vector_core` (v1 for `arch="lanes"`, v2 for `"fu_groups"`) |
| `fu_group.py` | `FUGroup` (latency L, II, stall/freeze), `ALUGroup` with its `ReductionTree` (separate `fu:rtree` output, Q7 revised), the element and reduction numerics |
| `vrf_banks.py` | `BankedVRF` (8 banks × 32, `bank = reg % 8`, per-element write enable), `BankedVRFStorage` (decode 2's storage adapter), `SlotContractError` |
| `writeback.py` | `WBArbiter` (≤ 4 grants per cycle, fixed priority, 1 mask write per mask bank), `WBBufferV2` (per-bank FIFOs, depth 4, 1 write port, `t_wbbuf` = 1) |
| `slots.py` | `SlotOp` (what one slot fires: destination, registers, kind) and its per-round state |
| `core.py` | `VectorCoreV2`: rounds, slots, bank and mask read ports, per-destination arbitration, single-mode scoreboard, writeback, plus the v1 VLSUs, GSAU and transpose unit |

**How a cycle works** (`VectorCoreV2.step`): every decision is made from the state at the start of the cycle (bank FIFO drains, writeback grants, slot reads, slot fires), and only then applied. Nothing depends on the order units are looked at. The VLSUs, GSAU and transpose are the v1 classes, ticked after `step` as v1 does.

**Timing as built:** a slot reads and fires in cycle c → the FU output is offered at c+L+1 → the bank FIFO writes at the end of c+L+2 → a dependent slot reads at c+L+3. So D = L + 3, as Q10 decided.

**Tests:** `tests/vector_core_v2/`, 52 tests, all passing.
- `test_v2_fu_group.py` (M1):
  - output at c+L+1 for L = 1…14
  - II honoured, one token per cycle at II = 1
  - stall freezes without losing or duplicating a token
  - reductions leave through the tree at c+L+5+1
  - a plain op and a reduction finish together on separate outputs
  - a frozen tree blocks reductions only
  - element values equal v1's `_op_lut` + BF16 rounding for 10 ops
  - full-mask reductions equal v1's tree and `golden.lanes_reduce`
- `test_v2_vrf_wb.py` (M2, M3):
  - bank map, BF16 rounding, per-element enable
  - untimed reads equal v1's `VectorCore`
  - grant limit and priority, full FIFO refusal, two grants to one bank
  - mask ports and a blocked mask bank
  - FIFO drain after `t_wbbuf`, one write per bank per cycle
- `test_v2_core.py` (M4):
  - **D = L + 3 exactly** for ALU (4), MUL (4), EXP (17) and reduction (9)
  - values end to end
  - 4 slots fire in one cycle when ≤ 2 masks on different banks
  - two unmasked ops share mask bank 0 (Q16a), or don't with `m0_uses_port=False`
  - two slots on one FU group: the loser fires 1 cycle later
  - strict mode rejects an out-of-pair source
  - compat mode charges +1 cycle
  - same-bank sources cost +1 cycle
  - compat bank contention: the owning slot reads first
  - WAR inside a round
  - two banks written in one cycle
  - the watchdog fires on a stuck round

**Where the build differs from the plan** (all to revisit):
- **Location:** files live in `src/vector_core/v2/`, not `src/vector_core/`. That keeps v1 and v2 apart.
- **Reading before the destination is ready:** a slot may read its operands while its destination is not ready yet, and hold them; the plan reads and fires in the same cycle. Values can't go stale: hazards are checked before reading, and a later write to a source already read doesn't matter.
- **Writeback priority:** fixed order only. Oldest-tag-first (section 7) is not built yet. In single mode, with one round at a time, the fixed order almost always gives the same answer.
- **ALU freeze:** a stuck ALU output freezes reductions too, since they share the ALU stage. Only a stuck tree is reduction-specific.
- **Reductions with every element masked off:** v2 gives the identity (0, +inf, −inf), where v1 leaves the destination's old value/seed. Masked-off elements in a reduction are an edge case no shipped kernel hits.

### 2. Wiring it into the scheduler and the platform (M0 plumbing, M7 packet adapter)

**Changed files** (each change is off by default, so v1 runs exactly as before):

| File | Change |
|---|---|
| `src/scheduler/decode2.py` | New `vector_operands_at_issue` flag (default True). When False (v2): no vector or mask reads at issue, no reggie bank-conflict timing for those two files, and `Scoreboard.track_vector = False`. Scalar and **mask** busy bits stay |
| `src/scheduler/core.py` | When the vector core is v2: decode 2 gets the banked VRF as storage, the vector side is `VectorSideV2`, and the cycle's scalar-side mask-write banks (`mv.stm`) are passed to the vector core |
| `src/scheduler/vector_v2.py` (new) | `VectorSideV2`: each issued packet becomes one round, with packet word k → slot V_k. Reuses v1's SDMA and `vreg.st` release code. `MoveToScalarV2` exposes the vector core's hold register to scalar writeback |
| `src/atalla/sysarr_tpu_system.py` | `build_tpu_platform(vector_core_factory=None)` |
| `src/atalla/atalla_platform.py` | `build_atalla_platform(vector_arch="lanes", vector_cfg=None)`. `"fu_groups"` builds the M×N core and defaults to `read_mode="compat"` (Q33/Q34). The activity view raises for v2 (not drawn yet) |
| `tools/run_kernel.py` | `--vector-arch {lanes,fu_groups}`, `--read-mode`, `--wb-grants`, `--m0-port/--no-m0-port` |
| `tools/compare_vector_arch.py` (new) | Builds each program once and runs the same image on both cores. Reports cycles, the reference result, an optional golden diff, and the M×N slot stalls, compat reads and refused writebacks. `--wb-sweep 1,2,4,8` sweeps Q11 |

**How to run it:**
```bash
python tools/run_kernel.py relu.c --vector-arch fu_groups --golden
python tools/compare_vector_arch.py relu.c softmax.c layernorm_param gemm --golden
python tools/compare_vector_arch.py layernorm_param --wb-sweep 1,2,4,8
```

**Bugs found and fixed along the way:**
- **`.vs` scalar rounding (`layernorm.c` was 1 ulp off).** v1 rounds the broadcast scalar of a `.vs` op to BF16 before the op (`VectorDatapath.enqueue` casts both operands). The first v2 used the fp32 scalar as is. v2 now rounds it too, and `layernorm.c` matches the functional sim.
- **Round-queue overflow on `vmov.vts`.** `SchedulerCore` overwrites decode 2's "movement" readiness with the move-to-scalar unit's own `ready()`, and v2's first version always said yes, so a `vmov.vts` packet could issue into a full round queue (`layernorm_param`). `MoveToScalarV2.ready` now reports the round-queue room.

**Tests:** `tests/vector_core_v2/test_v2_kernels.py` (M7) runs on the M×N core:
- the 11 assembly kernels of `tests/scheduler/test_kernels.py`
- every C program, against the functional sim and against its reference result
- v1's known-failure tables, imported as strict xfails, so a change in behaviour would show up
- a check that the platform builds the M×N core with compat reads

**Results** (`tools/compare_vector_arch.py --golden`, 2026-11-08, default config: 4 slots, 4 writebacks per cycle, compat reads, m0 uses a mask port). Every run matches the functional sim **bit for bit**, and the C programs pass their numpy references:

| Program | v1 lanes | M×N | Change | Where the M×N slots waited (cycles summed over slots) |
|---|---|---|---|---|
| relu.c | 664 | 667 | +0.5% | compat 8 |
| softmax.c | 657 | 658 | +0.2% | compat 2 |
| layernorm.c | 1552 | 1552 | 0.0% | compat 42 |
| maxpool.c | 760 | 766 | +0.8% | compat 10 |
| maxpool_2x2.c | 1411 | 1411 | 0.0% | compat 42 |
| add (asm) | 2106 | 1824 | −13.4% | raw 383, compat 32 |
| relu (asm) | 2038 | 1941 | −4.8% | raw 240, compat 97, waw 29 |
| sigmoid (asm) | 395 | 355 | −10.1% | raw 108, compat 12 |
| layernorm_param (asm) | 4007 | 2341 | **−41.6%** | raw 336, compat 222 |
| maxpool (asm) | 1360 | 1283 | −5.7% | raw 137, compat 49 |
| softmax (asm) | 2656 | 2612 | −1.7% | compat 192 |
| gemm (asm) | 2502 | 2336 | −6.6% | raw 1172, compat 80 |
| gemm_tiled (asm) | 1341 | 1247 | −7.0% | raw 472, compat 24 |
| gemms (asm) | 9398 | 8680 | −7.6% | raw 4890, compat 320 |
| conv (asm) | 3061 | 2946 | −3.8% | raw 1620, compat 100 |
| conv_tiled (asm) | 1537 | 1333 | −13.3% | raw 438, compat 35 |

**What the numbers say:**
- **Compiled C kernels barely change.** They put one vector op in each packet (`relu.in`, for example), so there is never more than one slot op per round, and their cycles go to instruction fetch, scalar code and DRAM.
- **The assembly kernels gain from four things:**
  - log-depth reductions (9 cycles vs 36), which is most of `layernorm_param`'s 42%
  - add and mul on separate groups
  - a full vector per cycle into each group, instead of 2 beats on 16 lanes
  - several writebacks per cycle
- **Compat reads cost about 1 cycle per out-of-pair read.** Today's register allocation puts most operands outside their slot's pair. That is the cost a v2-aware compiler would remove.
- **Neither result shows loop mode.** Software pipelining across rows needs loop mode (M5) or a kernel FSM. These are single-mode numbers only.

**Sweeps** (Q11 writebacks per cycle; Q16a m0 port), on `layernorm_param` and `gemms`:
- **Writebacks per cycle:** 1, 2, 4 and 8 give identical cycles. At wb = 1, `layernorm_param` sees 5 refused writebacks but no extra cycles.
- **m0 port:** turning it off changes nothing.
- **Why:** today's programs put about one vector op in each round, so neither writeback bandwidth nor mask ports is ever the limit. Both sweeps only become meaningful with denser rounds: loop mode, a kernel FSM, or a compiler that fills more slots.

**Full test suite after these changes:** 718 passed, 1 deselected, 54 xfailed (90 s). That is the 633-test baseline plus 85 new v2 tests. The 27 new xfails are the v1 kernel tables reused for v2. No v1 test changed result, and `tests/equivalence` (v1's cycle-exact oracle) still passes unchanged.

### 3. Known limits and deviations of the first version

These are things to fix or decide later; none blocks running kernels:

1. **Masks are still tracked by decode 2** (busy bits at issue), not by the vector scoreboard as Q20 decided. Decode 2 issues a mask consumer only after the mask has landed, so compare → masked-op waits a little longer than the plan's D = 4. Moving masks into the vector core needs `mv.stm` (a scalar op) to reserve the mask there. It is needed before loop mode.
2. **On a mask-bank clash, `mv.stm` wins and the compare waits one cycle.** Q21 decided the reverse. The scalar side's mask writes are decided first in the cycle, and making `mv.stm` retry would mean changing the scalar writeback arbiter.
3. **Move-to-scalar is offered one cycle after its slot fires.** Q19 decided the same cycle. Scalar writeback is arbitrated before the vector core's cycle runs, so getting the same cycle would mean reordering `SchedulerCore._execute_eval`.
4. **Writeback priority is fixed-order only** (section 1).
5. **GSAU ops never read a mask.** v1's decode 2 counted a mask read for `lw.vi` (it sets `use_vms`), but the GSAU never used it.
6. **Round queue depth 2 includes the D2/EX latch.** Decode 2 issues a vector packet only when no round is queued in front of the slots. That is the simplest way to never overflow, since readiness is computed before dispatch adds the latch's round.
7. **PerfMonitor:**
   - "lanes" means "any FU group busy", and `lane_ops` counts FU ops (so FLOPs are right).
   - Vector RAW waits now happen in the slots, not at decode 2, so `stall:hazard:vector_lane` mostly disappears from the cycle buckets. The waits show up in `vc.stats["stall"]` instead, which `compare_vector_arch.py` prints.
   - Wiring the slot stalls into the monitor's buckets is part of M6.
8. **Not supported on v2 yet:**
   - the activity view (raises a clear error)
   - the harness platform (`sysarr_tpu_system` experiments, which stay on v1)
   - FP16
   - loop mode (M5)
9. **Strict read mode can't run today's programs**, by design. Their registers aren't allocated for bank pairs.
10. **`add.c` fails its reference on both cores.** It is a known compiler issue, already an xfail in `tests/scheduler/test_kernels.py`: its `vreg.ld` doesn't wait for the DMA that fills its scratchpad.

### 4. Stuck items / questions for the team

- **Nothing is blocking.** The first version runs every kernel.
- **Compat penalty.** Should an out-of-pair read cost +1 cycle (the default) or more? It only matters for comparing today's code. A v2-aware register allocator removes those reads, and `--read-mode strict` shows the ideal.
- **Next milestones.** The natural next step is M5 (loop mode via `issue_round` from a loop controller or a kernel FSM), plus moving mask tracking into the vector core (limit 1). Q12, Q22 and Q23 need answers first.
