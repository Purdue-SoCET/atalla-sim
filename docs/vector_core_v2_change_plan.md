# Vector Core v2: Change Plan

Status: plan only. No source has been changed. Line numbers refer to commit
`8e956f9` (main). Q1–Q10 were answered on 2026-11-06, and Q15–Q19
(with Q16a and Q17b) on 2026-11-07/08; see [Decisions](#decisions-q1q10).
Q7 was revised and Q11 decided on 2026-11-08 after the diagram review (section 13).
Section 14 (CISC kernels and data movement outside the vector core) is under
discussion; its answers (Q24, Q25, Q28) are provisional.
Q20, Q21, Q33 and Q34 were decided on 2026-11-08 for the first running version
(section 10). Still open: Q12–Q14, Q22–Q23, and Q24–Q32 from the CISC kernel
direction (section 14). None of them blocks the first version (M0–M4 + M7).

The redesign replaces the 16-lane datapath with j full-width FU groups. It
also adds S instruction slots (V1..V4), each wired to a pair of register-file
banks through its own crossbar, and supports loop-resident instructions that
software-pipeline a loop in hardware. Scope here is the vector core. The
scheduler, compiler and functional sim are covered briefly in section 12.

Notation used throughout:
- **S**: number of slots (4).
- **j**: number of FU groups (5).
- **B**: number of banks (2S = 8).
- **N**: loop trip count.
- **D(edge)**: dependent-instruction delay, from producer operand read to the earliest consumer operand read.

---

## Decisions (Q1–Q10)

| Q | Decision | What it fixes in the model |
|---|---|---|
| Q1 | 256 vector registers; `bank = reg % 8`, `addr = reg // 8` (32 per bank) | `BankMap("mod")`: vd[2:0] is the bank, vd[7:3] the address (section 4) |
| Q2 | src0 comes from the pair's even bank, src1 from the odd bank; every bank is 1R1W | 1-cycle slot read; both sources in one bank cost +1 cycle; a source outside the pair is a `SlotContractError` |
| Q3 | A value needed in another slot's banks gets there by **writeback multicast** | Multicast is part of the base design, not an option (sections 4, 7) |
| Q3a | Multicast is encoded as `vd` unchanged plus a 4-bit **pair mask in bits [38:35]** of VV/VS/VI instructions | Copies land at the same address and the same even/odd side in each selected pair; `vreg.ld` (VM) can't multicast |
| Q4 | Transpose and move-to-scalar become **crossbar outputs**; each crossbar is **2 × (j+7)** | Destinations: j groups, vlsu0..3, gsau, transpose, mts, so 12 at j=5 (section 6) |
| Q5 | `.vlsu_out` and `.gsau_out` feed the WB arbiter; the WB buffer writes only the banks | There is no path from the WB buffer to a VLSU or the GSAU |
| Q6 | ALU 1, MUL 1, EXP 14, SQRT 8, RECIP 11; all II=1, fixed latency; EXP = `math.exp`, RECIP = `1/x`, rounded to BF16 | `FUGroupSpec` defaults (section 2) |
| Q7 | RTree: 5 levels × 1 cycle, pipelined, masked elements count as identity. **Revised 2026-11-08 to follow the diagram:** the RTree has **its own WB arbiter input**, separate from the ALU's | Two sources, `fu:alu` (plain ALU and compare results) and `fu:rtree` (reductions). They can't collide, so the Q18 guard is dropped (section 2) |
| Q11 | The WB arbiter grants up to **4 writebacks per cycle** by default (today's scheduler path also does up to 4), and the model **sweeps 1/2/4/8**. Each bank keeps 1 write port (Q2); per-bank FIFO depth 4 and oldest-first priority stay as the defaults | `wb_grants_per_cycle = 4`. Each extra write per cycle is another 512-bit write bus in hardware, so the sweep decides whether 8 banks' worth of write paths is worth it (section 7) |
| Q8 | Lockstep round barrier by default, decoupled mode kept as an option; the vector core holds a fixed trip count set at loop start | `LoopController(round_mode="lockstep")` (section 3) |
| Q9 | Fixed register names; hardware stalls the producer (WAR checked when a slot fires); one writer slot per register per loop | `war_policy` fixed to issue side; write-hold dropped (section 5) |
| Q10 | `t_read=1, t_xbar=0, t_arb=0, t_wbbuf=1, t_write_visible=1`; a read in the cycle a write lands sees the old value | **D = L + 3** (section 5) |
| Q10b | Banks have a **per-element write enable** for masked writes | No read-modify-write and no read-port use (section 7) |
| Q15 | A loaded value read by slots in two bank pairs is **loaded twice** (two `vreg.ld` into registers in different pairs) | `vreg.ld` stays unchanged; this costs a scratchpad read and a VLSU issue rather than an ALU fire (section 4) |
| Q16 | One **shared mask file** for all slots, kept as today: 16 × 32 bits, 2 banks (`bank = m % 2`), **one read port per bank**, so at most 2 mask reads per cycle and at most 1 per bank | A masked slot that loses the mask read waits with new stall reason **`mask`**; mask reads are granted in `xbar_policy` order (section 4) |
| Q16a | An unmasked op's **m0 read uses mask bank 0's port**, as today's `RegFile` does | Per cycle, at most one op on m0 or an even mask plus one on an odd mask can fire. That caps mask-reading ops (ALU, MUL, EXP, SQRT, RECIP, reductions, compares, `mv.mts`) at **2 fires per cycle**. VLSU, `gemm.vv`, transpose and `vmov.vts` read no mask and aren't limited (section 4) |
| Q17 | Multicast copies keep the **same side** as the home copy (`vd[0]`); bit 39 stays unused | A value read as src0 in one slot and src1 in another needs an explicit copy instruction (section 4) |
| Q17b | As today, **`vd` is always written**; the pair mask only adds copies, and vd's own pair bit is ignored | A zero mask is a plain write, so existing encodings (bits [38:35] = 0) keep their meaning. The diagram changes needed for multicast are in section 7 |
| Q18 | ~~ALU/RTree port collisions are avoided at issue~~ **Superseded 2026-11-08 by the revised Q7:** with separate WB inputs there is no collision to avoid | No issue guard and no `alu.rtree_blocks` stat (section 2) |
| Q20 | The vector scoreboard **tracks the 16 mask registers like vector registers**: versions, RAW/WAR/WAW, one writer slot per mask per loop | Compare → masked consumer D = L_alu + 3 = 4 (section 5) |
| Q21 | **1 write port per mask bank.** A compare and an `mv.stm` on the same mask bank in one cycle: the compare wins, and `mv.stm` retries the next cycle | Mask writes are no longer always granted, unlike today's `VectorSide.arbitrate` (section 4) |
| Q33 | **First running version** = run today's kernels on the new core in straight-line (single) mode: milestones **M0–M4 + M7**, on the scheduler platform (`build_atalla_platform`), BF16 only. Loop mode (M5) comes next | Section 10, "First version" |
| Q34 | **Compat read mode** for existing programs: strict bank-pair rule by default, plus `read_mode="compat"`, in which a slot may read any bank, an out-of-pair read costs +1 cycle, and reads of one bank are served one per cycle | Existing kernels, register-allocated for v1, can run and the cost of breaking the rule is counted (section 4) |
| Q24 | *Provisional; kernels are still under discussion.* The kernel sequencer is a **hand-written FSM per kernel** in the RTL | The simulator models one `KernelFSM` subclass per opcode on a shared IDLE/LOAD/RUN/RELEASE base, so each kernel can carry its own timing like its RTL FSM. Each kernel's phase list is still checked against the slot rules when it loads (section 14.4) |
| Q25 | *Provisional; data movement outside the vector core is still under discussion.* While a kernel runs, **scalar and SDMA packets keep issuing** (subject to hazards); vector packets wait for `kernel_ready` | A kernel owns all 4 slots; tile hazards go through the base-register scoreboard and the destination reservation (section 14.5) |
| Q28 | *Provisional; scratchpad use is still under discussion.* Kernels longer than S ops run as **phases through the scratchpad**: intermediate rows are stored at the end of a phase and loaded by the next | No new slots. `loop_done` must also wait for the phase's scratchpad stores to be written to SRAM. Softmax takes 5 phases (section 14.3) |
| Q19 | Move-to-scalar works **as today**: the value is offered to the scalar writeback in the cycle it leaves the crossbar, and held while refused. It does not go through the vector WB arbiter | `t_mts = 0`. While a result is held, the `mts` destination is not ready, so the next move-to-scalar waits with stall `xbar` (section 6) |

What this gives for dependent-op delay. v2 measures from a slot's operand read to the dependent slot's read; today's numbers are decode-2 issue to dependent issue. The reference points differ, so compare the trends rather than single cycles.

| Producer | v2 D | Today |
|---|---|---|
| ALU (add, sub, compare) | 1 + 1 + 1 + 1 = **4** | 5 |
| MUL | **4** | 5 (mul runs on the ALU today) |
| EXP | **17** | 18 |
| SQRT | **11** | — (no vector sqrt today) |
| RECIP | **14** | — (no reciprocal today) |
| Reduction (ALU + 5-level RTree) | 1 + (1 + 5) + 1 + 1 = **9** | 36 |

---

## 0. Where the code differs from the brief

Some of these change the plan, so they come first.

| # | Brief says | Code does | Where |
|---|---|---|---|
| C1 | 32-element vectors split into two 16-element halves on 16 lanes | Each lane gets a **contiguous** slice of `vector_len // lane_count` elements and feeds **one element per FU per cycle**. With the platform's 16 lanes that is 2 elements (2L, 2L+1) per lane, so it is 2 beats: beat 0 = even elements, beat 1 = odd. That is not two halves. The harness platform defaults to `lane_count=4`, which is 8 beats. `docs/architecture/platform-uml.md:21` says "striped round-robin", which is also wrong. | `vector_lanes.py:29-37`, `:515-536`; `atalla_platform.py:134`; `sysarr_tpu_system.py:587` |
| C2 | Each lane holds its own FUs | True, but the FU set is `alu, sqrt, exp, div, shift`. There is **no MUL unit** (`mul` runs on the ALU) and **no reciprocal** (`div` is the closest). From the ISA only add/sub/mul/compares/exp/reductions are reachable. `sqrt`, `div` and `shift` have no vector opcode. | `vector_lanes.py:53-85`, `:447-453`; `scheduler/vector.py:98-105`; `isa.py:106-134` |
| C3 | FU latencies from the report | The doc table says ALU 4. The platform overrides the **ALU to 1 cycle** (`fu_latencies={"alu": 1}`), and the equivalence snapshot encodes that. The code has no LUT exponential and no pipelined reciprocal: `exp` is a 14-cycle delay line around `math.exp`. | `sysarr_tpu_system.py:620`; `vector-core.md:29` |
| C4 | Packet issue: ≤2 datapath, 1 GSAU, ≤4 VLSU | This describes the **harness path** only (`PACKET_UNITS` widths 1 GSAU / 4 VLSU / 1 transpose / 2 datapath). On the **scheduler path** a packet has 4 vector words, and decode 2 allows 2 lane ops, 1 GSAU, 1 move-to-scalar, and 1 VLSU op **per scratchpad**. On both paths the datapath accepts **one instruction per cycle** (`issue_width=1`) from an in-order FIFO with head-of-line blocking. | `vector_core.py:35`, `:317-320`; `decode2.py:52-55`, `:152-172`; `vector_lanes.py:754-760` |
| C5 | Per-FU result collectors | There is **one** `ResultCollector` for the whole datapath (plus a per-lane `pending_outputs` FIFO). It releases at most **one finished vector per cycle**. | `vector_lanes.py:236`, `:794-797` |
| C6 | At most one architectural writeback per cycle | True on the **harness path** (`_commit_one_writeback`). On the **scheduler path**, writeback is `VectorSide.arbitrate`: **one write per bank (`reg % 4`) per cycle**, so up to 4, with fixed priority VLSU0-3 > GSAU > transpose > reduction > lanes. The lanes still deliver ≤1 result/cycle. | `vector_core.py:843-872`; `scheduler/vector.py:186-187`, `:256-273` |
| C7 | Operand read path in the vector core | On the **scheduler path** (the real machine), **decode 2 reads all vector operands at issue** (`RegFile`, 4 banks, 4 read ports, reggie conflict FSM) and hands data to the units. The vector core's `Veggie`/`OpBuffer` port model is used only by the harness path. `Veggie.dwrite_ports` is never used, and neither path writes through `Veggie.tick`: writes go straight into `data_banks`. | `decode2.py:296-372`, `:537-557`; `core.py:96-112`; `veggie_file.py:20-87` |
| C8 | (implied) hazards tracked | The scheduler path has a busy-bit scoreboard in decode 2. The **harness path has no RAW checking at all**. The harness path's GSAU, transpose and VLSU-store reads are **untimed** (they bypass the ports), and a VLSU store reads its register when the VLSU dequeues it, not at issue. | `decode2.py:185-225`; `vector_core.py:609`, `:662`; `vector_load_store.py:192-194` |
| C9 | (implied) reductions are a tree | The **values** come from a pairwise tree. The **latency** is `(n-1) × alu_latency` with n = active elements, so it is **linear**: 31 cycles for 32 elements at ALU=1. `scheduler/vector.py:43` says "every step rounded to BF16", but the code accumulates in Python floats and rounds once, at the register-file write, which is what `golden.lanes_reduce` assumes. | `vector_lanes.py:347-394`; `core.py:107-112`; `golden.py:46-66` |
| C10 | FUs are pipelined | Within one instruction, yes. Across instructions, a lane's FU context is held until **all of that instruction's elements have left the pipeline**, so two ops on the same FU type never overlap. Decode 2 also treats EXP as always ready (an RTL TODO). | `vector_lanes.py:560-564`, `:574-582`; `decode2.py:470-471` |
| C11 | Item 3: WB Buffer → VLSU (`.vlsu_out`), GSAU (`.gsau_out`) | In the diagram, `.vlsu_out` and `.gsau_out` are **inputs to** the WB arbiter (load results and array results), and the WB buffer feeds only the banks. **Resolved (Q5): the diagram is right.** | diagram |
| C12 | Crossbar size 2×(j+5) | The diagram label reads "each 2 x j + 5". With j=5 that is ambiguous between 2×10=20 and 2·5+5=15. **Resolved (Q4):** 2 inputs, and with transpose and move-to-scalar added, **j+7 outputs**, so 2×(j+7). | diagram |
| C13 | Destinations: j FU groups + 4 VLSU + GSAU | Today the core also has the **transpose unit** (`tpus.vi`/`tpop.vi`), **move-to-scalar** (`vmov.vts`, `mv.mts`) and the **mask register file** (compares write masks; ops read masks). None of these appear in the diagram. **Resolved (Q4):** transpose and move-to-scalar become crossbar outputs; masks stay in a separate file. | `vector_core.py:316`; `scheduler/vector.py:139-171`; `decode2.py:446` |

### Baseline measurements

These are on today's scheduler platform: `build_atalla_platform` defaults, 16 lanes, ALU=1, EXP=14, warm icache, source registers chosen to avoid bank conflicts. Cycles are counted from the producer's decode-2 issue. They are the numbers v2 will be compared against.

| Pair | Producer result in RF | Consumer issues / 2nd result in RF |
|---|---|---|
| `add.vv` → dependent `add.vv` | +4 | issues +5 |
| `expi.vi` → dependent `add.vv` | +17 | issues +18 |
| `rsum.vi` → dependent `add.vv` | +35 | issues +36 |
| `add`, `add` independent (same ALU) | +4 | 2nd result +6 (2-beat lanes) |
| `add`, `mul` independent (mul shares ALU) | +4 | +6 |
| `expi`, `expi` independent | +17 | +32 (no overlap on one FU type) |
| `add`, `expi` independent | +4 | +18 |

Method: instrument `SchedulerCore.writeback_at` and read `core.issued` on small programs built with the encoders in `tests/scheduler/test_vector.py`. Measuring a source pair in the same bank (v6, v10) adds +2 cycles from reggie's conflict FSM.

---

## 1. Current-state map

There are two drive modes, explained in the module docstring at `vector_core.py:1-18`:

- **Harness path**: `VectorCore.tick()` runs packing, operand fetch, the units and writeback itself. Used by `sysarr_tpu_experiment`, `transpose_bench`, and the tiled and blocked GEMM tests.
- **Scheduler path**: `VectorCore.tick_units()` only advances the units (`vector_core.py:896-908`). Decode 2 reads the operands, `scheduler/vector.py` dispatches them, and `scheduler/vector.py` arbitrates writeback. This is the machine `build_atalla_platform` builds.

### Issue

**Harness path:**
- `enqueue_scheduler_instruction` packs instructions into `_build_packet` and then into `vliw_q` (`vector_core.py:399-446`).
- `_try_issue_scheduler` peeks the head packet and tries every unit group in the order GSAU, VLSU, transpose, datapath (`:682-699`).
- Instructions that don't issue stay in the packet, so a packet can issue partially over several cycles.
- A datapath instruction takes a free operand-collector slot (`_issue_datapath`, `:480-492`). There is one slot, because `issue_width=1`.

**Scheduler path:**
- `VectorSide.dispatch` routes each valid vector op from the D2/EX latch (`scheduler/vector.py:289-312`):
  - lane ops go to `vc.datapath.enqueue` (`:322-347`)
  - GSAU ops go to `vc.gsau.issue` (`:402-410`)
  - VLSU ops go to `vls_units[sid].enqueue_issue` (`:415-428`)
- Decode 2's readiness for lane ops is `pending_issue empty AND every lane's FU free` (`scheduler/vector.py:219-228`). It is computed from the previous cycle's state, so a second op can slip in one cycle later. Both of the independent-pair measurements above issued 1 cycle apart.

**Both paths:**
- `VectorDatapath.tick` moves at most `issue_width` (1) instructions from `pending_issue` (a 128-deep FIFO) into the lanes.
- The head waits until `_can_issue_to_all_lanes(fu)` (`vector_lanes.py:738-787`). That is in-order issue with head-of-line blocking.

### Operand read

**Harness path:**
- `_operand_cycle` drives read requests into `Veggie.tick` (`vector_core.py:494-581`).
- Veggie grants one read per bank per cycle, with `bank = reg % 4` (`veggie_file.py:46-87`, `vector_core.py:341-348`). Losing reads are re-driven the next cycle.
- `OpBuffer` holds operands until both have arrived (`veggie_file.py:89-172`).

**Scheduler path:**
- Decode 2 reads `vs1`, `vs2` and `vms` at issue (`decode2.py:537-557`).
- Bank conflicts are reggie's FSM: k reads on the busiest bank cost k cycles (`decode2.py:331-363`).
- The storage behind decode 2 is Veggie's `data_banks`, through `VeggieStorage` (`core.py:96-112`).

### Lane and FU execution

**Splitting and execution:**
- `VectorDatapath` creates `lane_count` `VectorLane`s and one `ResultCollector` (`vector_lanes.py:605-658`).
- Each lane runs its slice through a `FunctionalUnitPipeline` per FU type (`:426-468`). That class is a delay line with `remain` counters and capacity 8 (`:88-127`).
- Stage 1 sends one element per FU per cycle and computes the value immediately with `_op_lut` + `cast_scalar`. The pipeline only delays it (`:515-564`).

**Collecting results:**
- Stages 3-4 pair results with metadata and forward them to the collector (`:573-602`).
- `ResultCollector._try_complete` reassembles the vector. For reductions it runs `GlobalReductionUnit.reduce_tree` over the lane partials and schedules completion `(n-1)*alu_latency` cycles later (`:336-394`).

### Latency model

| Unit | Latency | Where set |
|---|---|---|
| ALU | 4 by default; platform overrides to 1 | `vector_lanes.py:447-455`; `sysarr_tpu_system.py:620` |
| SQRT | 8 | `vector_lanes.py:447-455` |
| EXP | 14 | `vector_lanes.py:447-455` |
| DIV | 11 | `vector_lanes.py:447-455` |
| SHIFT | 3 | `vector_lanes.py:447-455` |
| Reduction completion | (n-1) × ALU latency | `vector_lanes.py:378-379` |

**Instruction cost:** an instruction takes `slice_w` feed cycles plus the FU latency. A second instruction on the same FU type waits for the first to drain (C10).

**VLSU:** one issue and one response per cycle (`vector_load_store.py:265-274`). Pad timing belongs to the scratchpad.

**GSAU:** `rd_queue` depth is 64 × 3 = 192. Array results are paired in order (`vector_core.py:116-249`).

### Writeback

**Harness path:**
- `_collect_results_to_wb` takes candidates in priority order: VLSU0, other VLSUs, GSAU/transpose, reductions, lanes (`vector_core.py:701-841`).
- They go into `WBBuffer`, which refuses an entry whose bank is already queued or reserved this cycle (`:45-95`, `:61-75`).
- `_commit_one_writeback` writes one entry per cycle, merging masked writes (`:843-872`).

**Scheduler path:**
- `VectorSide.offers` and `arbitrate` allow one write per bank (`reg % 4`, hard-coded at `scheduler/vector.py:266`).
- Granted writes go through `writeback_at(cycle + 1)` (`core.py:294-302`), reach the RF on that cycle's edge, and clear the busy bit.
- A dependent packet issues the next cycle.

### Hazard handling

- **Scheduler path:** decode 2's `Scoreboard` keeps a busy bit per vector register, set at issue and cleared at writeback, so RAW and WAW are covered (`decode2.py:185-225`). WAR can't happen because operands are read at issue (`decode2.py:23-28`).
- **`vreg.st` → SDMA ordering:** `VstReservations` (`decode2.py:236-279`).
- **Harness path:** no RAW or WAW checks.

### Stats

- `VectorSide.stats`: `lane_ops`, `gsau_ops`, `vlsu_loads/stores`, `vector_writes`, `mask_writes`, `wb_conflict_cycles` (`scheduler/vector.py:213-216`).
- Per lane: `total_ops`, `op_counts`. Reduction unit: `reduce_ops`.
- `PerfMonitor`: lane FLOPs are `lane_ops × vector_len` (`perf_monitor.py:241`); "lanes" busy unit (`:180-181`); instruction mix by packet (`:57-70`, `:158-165`).
- `ActivityProbe`: one block per lane and per decode-2 VRF bank (`activity.py:100-104`, `:121-122`, `:255-256`).
- `sysarr_tpu_experiment`: queue-depth snapshot of `_build_packet`, `vliw_q` and `wb_buffer` (`sysarr_tpu_experiment.py:225-238`), and lane op counts (`:260-265`).

---

## 2. Structural changes: lanes become FU groups

**Keep, unchanged, for the v1 config:** `VectorLane`, `ResultCollector`, `VectorDatapath`, `Veggie`, `OpBuffer`, `WBBuffer`. Nothing in v1 is edited (see section 10).

**Add** (new files, so the v1 classes don't have to carry v2 flags):

| New file / class | Replaces (v2 only) | Purpose |
|---|---|---|
| `vector_core/fu_group.py`: `FUGroup` | `VectorLane` + `FunctionalUnitPipeline` + `LaneFUContext` | One W=32-wide pipelined unit. A whole vector enters in one cycle. |
| `vector_core/fu_group.py`: `ReductionTree` (RTree) | `GlobalReductionUnit` latency + `ResultCollector` reduction scheduling | Log-depth pipelined tree, attached to the ALU group only. |
| `vector_core/fu_group.py`: `ALUGroup(FUGroup)` | — | ALU ops plus the `lanes_out` path through the RTree. |

### `FUGroup` model

**Parameters:**
- `name`, `ops` (the subset of `_op_lut` it executes)
- `latency` L
- `ii` (initiation interval; 1 = fully pipelined)
- `width` (32)
- `stall_mode`: `"stall"` (the default: the pipeline freezes when the output isn't granted) or `"skid"` (an output FIFO of depth `out_depth`)

**State:**
- `stages`: a list of length L of `Optional[Token]`
- `out`: an `Optional[Token]` output register
- `last_accept_cycle`

A `Token` holds `(op, values[32], mask, dst, tag, src_slot, iteration)`.

**Handshake** (three-phase `RTLModule`, `base/rtl_module.py`):
- `ready = (cycle - last_accept_cycle >= ii) and not frozen`
- `frozen = out is not None and not granted_last_cycle` in `"stall"` mode
- `valid_out = out is not None`

**Numerics:** compute when the token enters, using the same `_op_lut` and `cast_scalar`/`cast_vector` as v1. Element values then stay identical to v1, and only timing changes.

**Why three-phase:** with S slots, a per-destination crossbar arbiter and a WB arbiter all deciding in one cycle, a plain `tick()` would make the result depend on tick order. `base-classes.md` §4 warns about exactly this. The scheduler's stages already use `RTLModule` for the same reason.

### RTree on the ALU group

- **Input:** `lanes_if.vector_output`, the ALU group's 32 element results. For a reduction the element op is a pass-through (today's code uses `add` with src1 = 0, `scheduler/vector.py:337`).
- **Structure:** `levels = ceil(log2(W)) = 5`, latency `levels × rtree_level_latency`, pipelined at II=1. Pairs are adjacent, `(e0,e1),(e2,e3),…`, the same pairing as `GlobalReductionUnit.reduce_tree` (`vector_lanes.py:184-204`).
- **Matches v1 when every element is active:** the first tree level equals today's per-lane fold of 2 contiguous elements, and the levels above equal today's tree across 16 lanes.
- **Masked elements** get the identity (0, +inf, -inf), so the tree shape stays fixed as in hardware. v1 instead **skips** inactive elements and lanes (`vector_lanes.py:349-354`), which changes the tree shape. Sums are formed in double precision and rounded to BF16 once, so results should almost never differ. `golden.lanes_reduce` needs a `tree32` mode for an exact check (section 12).
- **Output placement** (broadcast, partial_zero, partial_passthru from `imm`): reuse `GlobalReductionUnit.format_output_vector` (`vector_lanes.py:206-233`).
- **Output ports (decided, Q7 as revised to follow the diagram):** the ALU group and the RTree have **separate WB arbiter inputs**.
  - `fu:alu` carries plain ALU and compare results. `fu:rtree` (`lanes_out`) carries reductions.
  - The ALU's output is steered by op: a reduction's element results go into the RTree, and everything else goes to `fu:alu`. No result goes to both.
  - The two outputs can't collide, so there is no issue guard. Q18 is superseded.
  - **Backpressure:** each output has its own output register and freezes on its own when not granted (`stall` mode). A frozen RTree stops accepting, so in that case the ALU group's `ready` is false **for reductions only**; plain ops still issue. The other way round, a frozen `fu:alu` blocks only plain ops. The slot waits with stall reason `fu_busy`.
  - The WB arbiter has one more source (`fu:rtree`), so 13 sources at j = 5.

### FU latency mapping (v1 → v2 defaults)

Decided (Q6): use the v1 platform numbers below, all fixed latency and II=1, until RTL numbers exist.

| v2 group | Ops | v1 source | v1 latency (doc / platform) | v2 default L, II | Notes |
|---|---|---|---|---|---|
| 32xALU | add, sub, and, or, xor, not, gt, lt, eq, ne, min, max; reduction input | lane `alu` | 4 / **1** | L=1, II=1 | 1 keeps the platform's current ALU timing. |
| RTree | rsum, rmin, rmax | collector, `(n-1)·L_alu` | 31 at ALU=1 | 5 levels × 1, II=1 | Linear becomes log depth, so reductions get much faster than baseline. |
| 32xMUL | mul (and the mul half of `.vs` forms) | lane `alu` (shared) | 4 / 1 | L=1, II=1 | **New structural resource**: add and mul no longer contend. |
| 32xEXP | exp | lane `exp` (`math.exp`) | 14 | L=14, II=1 | Numerics: `math.exp` rounded to BF16 (Q6). Revisit if the RTL's LUT gives different values. |
| 32xSQRT | sqrt | lane `sqrt` | 8 | L=8, II=1 | No vector opcode yet (C2). |
| 32xRECIP | rcp | none (`div`=11 is closest) | — | L=11, II=1 | Pipelined. Reuse the scalar `rcp.bf` numerics from `semantics.py` so scalar and vector agree. |
| (dropped) | shl, shr | lane `shift` | 3 | → ALU, or unsupported | Not reachable from the ISA today. |
| (dropped) | div | lane `div` | 11 | unsupported; MUL × RECIP in software | Not reachable from the ISA today. |

A full vector enters a group in one cycle, so v2 group throughput is 1 vector/cycle per group (II=1). v1 manages one per `slice_w + L` cycles on the same FU type.

---

## 3. Slot model (V1..VS)

**New file** `vector_core/slots.py`: `VectorSlot`, `SlotSpec`, `LoopController`.

### `SlotSpec`

This is what the scheduler loads, and it is static while the slot is resident:
- `op`, `group`: the destination FU group, VLSU id, or GSAU
- `srcs`: up to 2 vector registers
- `mask_reg`
- `dst`: a vector or mask register
- `imm`, `dtype`
- for VLSU: `sid`, `base_row`, `row_stride_per_iter`, `vl`
- for `.vs` forms: a scalar operand value or source (Q12)
- `resident: bool`

### `VectorSlot` state (registers)

| Field | Meaning |
|---|---|
| `spec` | `Optional[SlotSpec]`. Empty when None. |
| `bank_pair` | `(2k-2, 2k-1)` for slot k (1-based). Fixed wiring, from `config.slot_banks(k)`. |
| `iter` | Next iteration this slot will fire (1..N). |
| `fired_in_round` | Lockstep mode: whether this slot has fired in the current round. |
| `state` | One of `EMPTY`, `IDLE` (loaded, not active this round), `WAIT` (has a stall reason), `FIRE` (operands read and destination granted this cycle), `DONE`. |
| `stall_reason` | One of `raw`, `war`, `bank`, `mask`, `xbar`, `fu_busy`, `round`, `none`. |
| `counters` | Fires, cycles per state, and stall cycles per reason. |

### `LoopController` (one per vector core)

**State:**
- `trip` N
- `round` t
- `active_slots` S_used
- `mode`: `"single"` or `"loop"`
- `round_mode`: `"lockstep"` or `"decoupled"`

**Firing rule:** slot k is eligible for iteration `i_k = t - (k-1)` when `1 ≤ i_k ≤ N`. This gives fill (rounds 1..S-1), steady state, and drain (the last S-1 rounds) with no prologue or epilogue code. The loop has `N + S_used - 1` rounds.

**What a round is** (this answers brief item 8):
- In `lockstep` mode (the default, decided in Q8) a round is a **barrier**, not a cycle count.
- Round t begins in the cycle after the last eligible slot of round t-1 fired.
- Within the round, each eligible slot fires **once**, in the first cycle in which every condition in section 5 holds (RAW, WAR, bank ports, mask port, crossbar grant, FU ready).
- The round ends when every eligible slot has fired. Its length is ≥ 1 cycle.
- "Fired" means the slot read its operands and its crossbar grant was accepted. Completion is not required: results land later and are tracked by the scoreboard.
- Consequence: with no forwarding, V_{k+1}(i) in round i+k waits for V_k(i)'s write, and V_k(i) fired at the start of round i+k-1 at the earliest. So the **steady-state round length is ≥ max over dependent edges of D(edge)**, plus structural stalls. The model must not shorten this (section 5 has the formula and a test).
- In `decoupled` mode (a research option) there is no barrier. Slot k may fire iteration i once slot k-1 has fired iteration i and the RAW/WAR checks pass. This bounds how much the barrier costs. With fixed register names, WAR keeps it close to lockstep anyway (section 5).
- `"single"` mode is trip = 1, with the slot freed after firing. It carries straight-line code, so existing kernels can run on v2 without loop support (milestone M7).

**Loading and draining:**
- `load_slots(specs, trip)` writes the slot specs. The cost of loading (cycles, and which packet does it) is a scheduler matter (Q13); the vector core only exposes `can_load()`.
- `done` is raised when the last slot has fired iteration N **and** every write tagged with this loop has landed.
- Slots are freed at `done`, or only on an explicit unload for loop-resident reuse.

**Termination (decided, Q8):** the vector core's `LoopController` holds a fixed trip count, set at loop start from a scalar register value handed over with the load. Data-dependent exits are out of scope.

### Rounds are the vector core's primitive (added 2026-11-08)

The team's draft RELU FSM (section 14.10) runs the software pipeline **from decode**: each FSM state names the ops of one round (PROLOGUE_0..2, BODY_LOOP, EPILOGUE_0..2), and decode checks `i == ROWS`. That puts the round counter in decode, not in the vector core as Q8 assumed. Kernels are still under discussion, so the simulator should support both without choosing:

- **The primitive** is `issue_round(ops)`: up to S `(slot, SlotSpec, iteration)` entries, one per slot. Each entry fires once, when its section 5 conditions hold. The round ends when all have fired (the lockstep barrier). `round_ready()` tells the driver when the next round may be issued.
- **Driver 1, the `LoopController`** (as above): generates the rounds itself from resident specs and a trip count. Fill and drain come out of the `1 ≤ i_k ≤ N` rule.
- **Driver 2, an external sequencer** (a decode-side kernel FSM, section 14): calls `issue_round` once per FSM state, with the slot ops and iteration indices that state names. Prologue, body and epilogue are just different rounds.
- **Driver 3, single mode:** one packet's vector ops become one round with trip = 1 (M7).

Both loop drivers produce the same schedule: N + S_used − 1 rounds, each at least as long as the largest edge D. They differ only in where the round counter lives. The scoreboard, crossbar arbiters and writeback don't care which driver is used. M5 tests both on the same 4-slot chain and checks that cycles and final state match.

---

## 4. Banked register model (VEGGIE v2)

**New file** `vector_core/vrf_banks.py`: `BankedVRF`. v1's `Veggie` stays.

### Shape

- B = 2S = 8 banks.
- Each entry is one full W=32 vector (BF16 values as floats, plus a dtype tag).
- `regs_per_bank = R` (32 if there are still 256 registers; Q1).

### Register-to-bank map

This is a policy object, `BankMap`:
- `mod` (**decided, Q1**): 256 registers, `bank = reg % 8`, `addr = reg // 8`, 32 per bank. vd[2:0] is the bank and vd[7:3] the address. This keeps today's low-bit interleave (`vector_core.py:341-348` uses `reg % 4`).
- `block`: `bank = reg // R`, so slot k sees a contiguous register window. Kept only for experiments.

### Slot read rule

- Slot k reads only its pair `(2k-2, 2k-1)`, with **src0 from the even bank and src1 from the odd bank** (decided, Q2). That is one access per bank and a 1-cycle read.
- Both sources in one bank of the pair cost **2 reads (+1 cycle)**, unless `bank_read_ports ≥ 2`.
- A source outside the pair is a **contract violation**: `SlotContractError` in strict mode (like decode 2's `PacketContractError`), counted otherwise.
- Slots never compete for read ports with each other, because each slot owns its banks. The only read conflicts are (a) intra-slot same-bank reads and (b) read vs. write collisions.

### Compat read mode for existing programs (decided, Q34)

Today's programs were register-allocated for v1 and mostly break the bank-pair rule. For example, `relu.in` puts every vector op in packet word 0 (slot V1, banks 0–1) but reads v4 (bank 4). To run them on v2 for a first comparison:
- `read_mode="strict"` (the default) keeps the rule above.
- `read_mode="compat"` lets a slot read **any** bank:
  - A read inside the slot's own pair costs as in strict mode.
  - A read **outside** the pair costs **+1 cycle** (`compat_read_penalty = 1`). That models an extra path the real hardware doesn't have, so its cost is charged rather than hidden.
  - Each bank still serves one read per cycle, so two slots reading one bank in one cycle are served one after the other. The bank's owning slot goes first, then `xbar_policy` order. The loser stalls with reason `bank`.
  - Counted as `compat.cross_pair_reads` and `compat.cross_pair_cycles`, per slot.
- Compat mode applies only to single mode (Q33). Loop mode and kernels stay strict.

### Ports

Per bank:
- `bank_read_ports` (default 1)
- `bank_write_ports` (default 1)
- `bank_port_mode`: `"1R1W"` (decided, Q2: independent read and write ports) or `"1RW"` (a write takes the bank's only port; kept for experiments)

In `1RW` mode, `rw_priority` decides which goes first: `"write"` (the default: the read retries next cycle, stall reason `bank`) or `"read"`.

### Write-to-read visibility (no bypass)

- A write committed on the edge ending cycle w is readable from cycle `w + t_write_visible` (default 1). A read in cycle w sees the **old** value.
- There is no path from the WB buffer or arbiter to a slot's read. The register's `readable_at` field (section 5) enforces this, so it does not depend on tick order.

### Steering a result to the consumer's bank pair (decided, Q3)

**Home copy.** The register number chooses the home bank. A producer's `vd` is allocated in the bank pair of one consumer. With `mod`, if V2 reads it as src0, `vd % 8 == 2`. This is a register-allocation contract (section 12).

**Further consumers** get the value by writeback multicast:
- VV, VS and VI instructions carry a 4-bit **pair mask in bits [38:35]**. These bits are unused today (`isa.py:154-159`). Bit 35+p selects pair p, which is slot V_{p+1}.
- For each selected pair, the value is written at the same address `vd[7:3]` and the same side `vd[0]` (even or odd bank) as the home copy. The registers written are `vd` plus `8·(vd >> 3) + 2p + (vd & 1)` for every selected p.
- Each copy is a real register write. The scoreboard versions and tracks every target register, and each copy uses one write port on its bank.
- A sibling register that held a live value is overwritten. The compiler must not keep live values there. The WAR check in section 5 covers every target, so a multicast into a register that still has an unread value stalls the producer.

**What can and can't multicast:**
- `vreg.ld` (VM format) has only bits [39:38] free, so **loads can't multicast**. Decided (Q15): a loaded value read by slots in two bank pairs is **loaded twice**, with two `vreg.ld` into registers in different pairs. That costs one more pad read and VLSU issue rather than an ALU fire.
- **Copies keep the home copy's side** (decided, Q17). A value read as src0 in one slot and src1 in another can't be served by one multicast. The compiler adds an explicit copy instruction into the other side. That costs an ALU fire, a slot and one more D of latency. Bit 39 stays unused.
- **The home copy is always written** (decided, Q17b, following today's design, where `vd` is always written). The pair mask only adds copies. vd's own pair bit is ignored, since the home copy is written anyway. A zero mask is a plain write, so existing encodings, which have bits [38:35] = 0, keep their meaning.
- GSAU results (`gemm.vv`, VV) and reductions (VI) can multicast. Compares write the mask file and don't.

**Checks when slots are loaded.** Every source of slot k must map into slot k's pair, either as a home copy or as a multicast copy. Anything else is a `SlotContractError`.

### Untimed API

`read_vreg`, `write_vreg` and `read_vreg_with_dtype` stay, with the same signatures as `vector_core.py:359-397`, so harness setup, golden comparison and `VeggieStorage` keep working. They go through `BankMap`.

### Masks

The diagram leaves out the mask file (C13). Decided (Q16): one `MaskFile` shared by all slots, the same shape as today's:
- 16 × 32 bits, 2 banks, `bank = m % 2` (as `decode2.py:446`, `RegFile` with `MASK_BANKS = 2`).
- **One read port per bank**, so at most 2 mask reads per cycle across all 4 slots, and at most 1 per bank.
- **Arbitration:** in `eval_ready`, each slot that will fire a masked op this cycle requests its mask's bank. Each bank grants one requester in `xbar_policy` order (oldest iteration first by default). A slot that loses waits a cycle with stall reason **`mask`** (section 5, condition 4).
- The mask read happens in the same cycle as the slot's vector bank read (`t_read`), so a granted mask read adds no latency and D is unchanged.
- **Today's code differs:** a packet with more than 2 mask reads is a `PacketContractError` (`decode2.py:162-166`), and a same-bank pair costs a reggie FSM cycle. In v2 a third mask read, or a second read on one bank, is a **stall**, not an error. Slots fire independently, so the compiler can't always keep them apart.
- **Mask writes (decided, Q21):** 1 write port per mask bank. Compare results go from the WB arbiter's mask output (`fu:alu` produces at most one per cycle). `mv.stm` writes come from the scalar side. If both hit the same mask bank in one cycle, the compare wins and `mv.stm` retries the next cycle. Today every mask write is granted (`scheduler/vector.py:262-265`), and `RegFile` asserts on two writes to one mask bank (`decode2.py:338-340`).
- **Mask hazards (decided, Q20):** masks are tracked by `VectorScoreboard` like vector registers (section 5).
- `mv.mts` competes for a mask read port like any masked op, then sends the mask to the crossbar's move-to-scalar output (Q4).
- Masks are not banked per slot, so compares need no multicast.

**m0 takes bank 0's port (decided, Q16a), as today.** Every VV/VS/VI op sets `use_vms`, so an unmasked op reads m0 (`control.py:292`, `:303`, `:316`), and `RegFile` banks that read onto bank 0 even though m0 is hardwired to all ones (`decode2.py:325-329`, `:371-372`). v2 keeps that rule.

What that costs:
- **Which ops read a mask:** ALU, MUL, EXP, SQRT, RECIP, reductions, compares (`.mvv`/`.mvs`) and `mv.mts`.
- **Which don't:** `vreg.ld`/`vreg.st`, `gemm.vv`, `tpus.vi`/`tpop.vi` and `vmov.vts` (`control.py:293-295`, `:307-332`).
- **Per cycle:** at most one mask-reading op on m0 or an even mask, plus one on an odd mask. So at most **2 mask-reading fires per cycle**, and at most **1 unmasked** one.
- **Effect on loops:** small in steady state. A lockstep round is already ≥ D ≥ 4 cycles (section 3), and dependent slots fire at different cycles within it. It matters for independent work: four independent unmasked ops need 4 cycles, not 1.
- **Compiler rule:** put real masks in odd registers (m1, m3, …, bank 1), so they don't compete with unmasked ops on bank 0 (section 12).
- `m0_uses_port` stays a config flag (default `True`), so the cost can be measured by turning it off.

---

## 5. Dependency and readiness tracking (no forwarding)

**New file** `vector_core/vscoreboard.py`: `VectorScoreboard`, owned by the vector core. It replaces decode 2's vector busy bits for slot-issued work (section 12).

### Per-register state

This applies to the 256 vector registers **and the 16 mask registers** (decided, Q20). A compare in one slot that writes a mask read by another slot is an ordinary RAW edge, with D = L_alu + 3 = 4. A fixed mask name reused every iteration gets the same WAR check as a vector `dst`. m0 is never written, so it is always ready.

- `committed_ver`: the version (iteration tag) of the value currently in the bank.
- `readable_at`: the cycle the committed value becomes readable.
- `inflight`: an ordered list of `(ver, writer_slot)` that have fired and not yet landed.
- `readers_left[ver]`: how many slot reads of version `ver` have yet to fire.

### What each source needs

When slots are loaded, the vector core works out what each source requires:

| Source register is… | Needed version for iteration i | Edge kind |
|---|---|---|
| `dst` of an earlier slot p < k | `i` | intra-iteration RAW (p → k) |
| `dst` of slot p ≥ k (including k itself) | `i - 1` | loop-carried RAW |
| Not written by any resident slot | `pre` (loop-invariant: written before the loop) | invariant |

### Fire conditions for slot k, iteration i, in cycle c

All registered state, evaluated in `eval_ready`:

1. **RAW:** for each source r, `committed_ver(r) == needed(i)` and `readable_at(r) ≤ c`. A loop-carried source at i = 1 needs `pre`.
2. **WAR, issue side** (decided, Q9): for `dst` and every multicast target, `readers_left[committed_ver or latest inflight ver] == 0`. A producer cannot fire a write to r while any slot still has to read r's previous version.
3. **WAW:** writes to r land in version order. One writer slot per register per loop body is a load-time contract (`SlotContractError`). A slot's own writes are in order because a slot always uses one FU group.
4. **Ports and destination:** the bank read is granted (section 4), the mask read is granted if the op is masked (section 4, Q16; failing this one gives stall reason `mask`), the crossbar destination is granted (section 6), and the destination is `ready`.
5. **Round** (lockstep): `i == t - (k-1)` and `not fired_in_round`.

The stall reason is the first failing condition, in that order. It feeds the stats.

### Write-hold was considered and dropped (Q9)

The alternative to stalling the producer was to let it fire and hold its write in the WB buffer until the readers had read. That is more aggressive, but it can **deadlock**:
1. A write held for WAR fills the WB buffer.
2. The full buffer back-pressures an FU group.
3. That stalls the reader slot, whose crossbar destination is that group.
4. The reader therefore never reads, and the held write is never released.

Q9 chose the producer stall, so write-hold is not modeled. A progress watchdog stays as a general safety net: with no fire, grant or write for `watchdog_cycles` while work is pending, it raises with a state dump.

### Round-trip delay the model must reproduce

```
D(p→k) = t_read + t_xbar + L_group(p) [+ L_rtree] + t_arb + t_wbbuf + t_write_visible
         ^ producer reads at c0          consumer may read at c0 + D(p→k)
```

All terms are config parameters. Decided (Q10): `t_read=1, t_xbar=0, t_arb=0, t_wbbuf=1, t_write_visible=1`, so **D = L + 3**, plus 5 RTree levels for a reduction. A read in the cycle a write lands sees the old value. The values per FU group are in the [Decisions](#decisions-q1q10) table. The M4 unit test asserts that the measured consumer read cycle minus the producer read cycle equals **exactly** this sum, and that nothing can read earlier.

### What fixed register names cost

With the same `dst` every iteration:
- WAR condition 2 makes V_k(i+1) wait until every reader of V_k(i)'s value has fired. In lockstep both are in round i+k, so the producer fires later in the round than the consumer.
- That costs no throughput: the round is already ≥ D long because of RAW.
- Getting more than one iteration per round trip would need rotating registers (`dst + (i mod R_rot)`). Q9 chose fixed names, so v2 runs at most one iteration per round trip and doesn't model rotation. `BankMap` leaves room to add it later.

### Single mode

`ver` becomes a global write sequence number and `readers_left` is always 0. This is the busy-bit scoreboard decode 2 has today, so straight-line code keeps today's RAW and WAW behaviour, with operand reads now in the vector core. Since reads now happen after issue, WAR is checked too.

---

## 6. Crossbar model

**New file** `vector_core/vector_xbar.py`: `SlotCrossbar`, `DestArbiter`. (`memory/crossbar.py` already exists for the scratchpad, so use a different name.)

### Crossbar shape

- One `SlotCrossbar` per slot.
- 2 inputs: operand A from the even bank, operand B from the odd bank.
- j+7 outputs (decided, Q4): `fu:<group>` × j, `vlsu0..3`, `gsau`, `transpose`, `mts`. That is 2×(j+7) switch points, 2 × 12 at j=5.
- A slot sends at most one destination per cycle:
  - binary ops send A and B
  - GSAU, VLSU stores, `tpus.vi` and `vmov.vts` send A
  - VLSU loads and `tpop.vi` send no vector data, only the request
  - `mv.mts` sends the mask it was granted from the shared mask file (Q16)

  So there is no conflict inside a crossbar.

### Structural hazards

These happen at destinations, which are shared by S slots:
- An S × (j+7) request matrix is formed in `eval_ready`.
- `DestArbiter[d]` grants **at most one slot per destination per cycle**, and only if destination d is `ready` (the FU group's `ready`; `vls.can_accept_issue()`; GSAU `to_systolic`/`rd_queue` not full).
- A loop body in which two slots use the same group type (say two adds) collides every round. With II=1 the loser fires one cycle later; with II>1 it waits II cycles.

### Arbitration policy

`xbar_policy`:
- `"oldest"` (the default): lowest iteration first. In lockstep that is the highest slot index, which drains the pipeline and avoids starving the critical path.
- `"fixed"`: by slot index.
- `"rr"`: round-robin.

### Latency

`t_xbar` defaults to 0, meaning combinational in the read cycle. Set it to 1 to register the crossbar output.

### Paths to the VLSUs and GSAU

- **VLSU:** the granted slot calls `vls_units[sid].enqueue_issue(op)` with **inline data** for stores. The store data comes from the slot's bank read, so it is timed. This removes the v1 `read_vreg_cb` untimed read (C8).
- **GSAU:** `gsau.issue({"vdata": A, …, "dst", "meta": {"tag": (reg, pair_mask, ver)}})`.
- **Transpose:** `tpus.vi` pushes operand A as a row. `tpop.vi` asks for the next column, which comes back through the WB arbiter (source `transpose`) into `vd`. The destination is ready when the unit's `ready_in` is set and its request queue has room (today `TRANSPOSE_QUEUE = 2`, `scheduler/vector.py:189`).
- **Move-to-scalar (decided, Q19: as today):**
  - `vmov.vts` sends operand A (element `imm`, as fp32 bits) and `mv.mts` sends its granted mask.
  - The result is a scalar register write. It is offered to the scheduler's scalar writeback **in the cycle it leaves the crossbar** (`t_mts = 0`), as `MoveToScalar` does today in EX (`scheduler/vector.py:139-171`).
  - It does **not** go through the vector WB arbiter or WB buffer.
  - If the scalar writeback refuses it, the result is held in a one-entry register. While that is full, the `mts` destination is not ready, so a second move-to-scalar waits with stall reason `xbar`.
  - Scalar-consumer delay: the slot's `t_read` (1) plus today's scalar writeback timing. Decode 2 keeps a busy bit on the scalar `rd` from slot load until that write lands (section 12).

---

## 7. Writeback

**New file** `vector_core/writeback.py`: `WBArbiter`, `WBBufferV2`.

### Sources

- one per FU group (`fu:alu`, `fu:mul`, `fu:exp`, `fu:sqrt`, `fu:recip`)
- `fu:rtree`, the RTree's own input, as drawn in the diagram (Q7 revised)
- `vlsu0..3` (each VLSU's `wb_q` head)
- `gsau` (`gsau.writebacks` head)
- `transpose` (Q4)

### Two levels, as in the diagram

1. **Arbiter → WB buffer.** Each cycle, up to `wb_grants_per_cycle` source heads are moved into the buffer. The default is **4** (decided, Q11), the same as today's scheduler path (one per bank of 4). The model sweeps 1/2/4/8; 1 reproduces the harness path's single writeback. A grant needs room in that entry's bank FIFO. A multicast entry needs room in **every** target bank's FIFO; it is granted as one entry and placed in each. An ungranted head stays at its source. In `"stall"` mode that freezes the FU group, which is the backpressure.
2. **WB buffer → banks.** Use **per-bank FIFOs** of `wb_fifo_depth` (default 4), not one shared queue. v1's `WBBuffer` refuses any entry whose bank is already queued (`vector_core.py:61-75`), which effectively makes it one-deep per bank and blocks the head. Each cycle each bank drains up to `bank_write_ports` entries, after `t_wbbuf` cycles in the buffer. A drain must respect the per-register version order (WAW). WAR is already settled when the producer fires (Q9).

### Priority (`wb_priority`, a configurable list)

- The default is **oldest tag first**, with ties broken by a fixed order: `vlsu0..3 > gsau > transpose > fu:rtree > fu:alu > fu:mul > fu:exp > fu:sqrt > fu:recip`. `fu:rtree` sits before `fu:alu`, as `reduction` sits before `lanes` in today's `VLSU_PRIORITY`.
- The fixed part mirrors today's `VLSU_PRIORITY` (`scheduler/vector.py:186-187`), so single-mode runs are easy to compare.
- Oldest-first keeps a late reduction or EXP result from being starved by a stream of young ALU results.

### Masked and mask writes

Decided (Q10b): banks have a **per-element write enable**. A masked write writes only its active elements, with no read of the old value and no read-port use. The old-value merge in `VectorSide._collect_lanes` (`scheduler/vector.py:349-367`) and `_commit_one_writeback` is not carried over. Mask results go to `MaskFile` on its own port.

### Why the brief's bottleneck goes away

**Supply.** In loop mode each slot fires once per round, and a round with a dependent chain lasts at least D ≥ 4 cycles. That is at most 4 slot results per 4+ cycles, **about 1 per cycle on average**, plus VLSU loads and GSAU rows. In straight-line code Q16a caps arithmetic fires at 2 per cycle. Results with different latencies can still land together (an EXP fired 14 cycles ago and an ALU op fired 1 cycle ago), and the per-bank FIFOs absorb those bursts.

**Capacity.** 8 banks with one write port each could take 8 writes per cycle, but only to 8 different banks. Results bunch up in the banks of the slots that consume them, and multicast copies add writes.

**Hardware cost.** Each write is a 512-bit vector (32 × BF16). Every extra write per cycle is another 512-bit bus from the WB buffer to the banks, plus another grant in a 13-source arbiter. The diagram's single WB Buffer → veggie arrow is 1 write per cycle.

**Decision (Q11).** Default 4 grants per cycle, which matches today's machine and keeps v1/v2 comparisons fair. Sweep 1/2/4/8 with `tools/compare_vector_arch.py` and report cycles, `wb.denied[src]` and `wb.fifo_occupancy` at each point, to find where extra write paths stop paying off. Watch loops that multicast heavily, since each copy uses a bank write port.

### What the diagram needs for multicast (Q3, Q3a, Q17, Q17b)

The brief's diagram has one path per result: WB Arbiter → WB Buffer → its bank. Multicast needs these changes:

1. **The pair mask travels with the result.** Decode takes bits [38:35] into `SlotSpec`, and the 4-bit mask rides with the token through `vector_if`, the crossbar, the FU group (or the GSAU `rd_queue` tag) and the WB arbiter. Today none of these carry it.
2. **The arbiter checks every target.** A multicast result is granted only when the home bank's FIFO **and** the same-side bank FIFO of every selected pair have room. Otherwise the whole result waits at its source; there are no partial grants.
3. **The WB buffer fans out.** The diagram's single WB Buffer becomes **per-bank FIFOs** (already planned above). One granted entry is placed in up to 4 FIFOs: the home bank, plus bank `2p + vd[0]` of each selected pair p. Each copy drains on its own bank's write port, at its own time.
4. **The banks need no new port.** A copy is an ordinary write: the same address `vd[7:3]`, on one bank's existing write port. The 8 banks keep 1R1W.
5. **The scoreboard tracks every target.** Each copy register gets its own version, `readable_at` and `readers_left`. Each becomes readable when its own bank's write lands.
6. **Masks don't multicast.** The mask file is shared (Q16), so compare results go only to `MaskFile`, as today.

Nothing changes on the read side: a slot still reads only its own pair (Q2), so the slot crossbars stay 2 × (j+7).

### Scoreboard updates

When an entry drains, set `committed_ver`, `readable_at = now + t_write_visible`, and `readers_left[ver] = static consumer count`, then pop `inflight`. For a multicast entry, each target register is updated when its own bank's write lands.

---

## 8. Interfaces and queues

### Keep the bridge-facing API unchanged

These methods stay, with the same signatures:
- `pop_scratchpad_request(vls_id)`, `push_scratchpad_response(vls_id, rsp)`
- `vls_units[i].req_q`
- `pop_systolic_request()`, `push_systolic_response(rsp)`
- `vector_len`
- `read_vreg` / `write_vreg`

`VLSFrontendBridge` (`sysarr_tpu_system.py:106-229`), `GSAUTPUBridge` and `ArrayValueBridge` (`sysarr_tpu_system.py:296-411`, `atalla_platform.py:76-86`) then need **no change**. Version tags ride in `meta`, which both bridges already carry through.

### VLSU and GSAU

`VLSU` and `GSAU` objects are reused as they are. In v2 the VLSU is always given inline store data. The GSAU `rd_queue` entry carries `(dst, pair_mask, ver)`, so a returning array row is written to every target and clears the right versions.

### `vector_if` / `lanes_if` equivalents

Model them as explicit, typed signal bundles between `RTLModule`s rather than ad-hoc dicts:

```python
@dataclass
class VectorIf:            # one per slot: slot -> crossbar -> destination
    valid: bool; dest: str; op: str
    a: Optional[list]; b: Optional[list]; mask: Optional[list]
    dst: int; ver: int; slot: int; iteration: int; meta: dict
    ready: bool            # driven back by the destination arbiter

@dataclass
class LanesIf:             # inside the FU-group section
    lanes_in: Dict[str, VectorIf]       # granted token per group this cycle
    vector_output: Dict[str, Token]     # each group's output register
    lanes_out: Optional[Token]          # RTree output
```

- `VectorIf.ready` is computed in `eval_ready` from registered state only.
- `valid` and data move in `eval_data`.
- Grants and state change in `commit`.

### `VectorCoreV2` shell

`vector_core/vector_core_v2.py`, with the same outward surface as `VectorCore`:
- `tick`, `tick_units`, `vector_len`, `vls_units`, `gsau`, `transpose`
- plus `load_slots`, `loop_done`, `reg_ready(r)`, and the stats

It owns operand read and writeback in **both** drive modes. v2 therefore has one path, not v1's two (the harness `tick()` and the scheduler `tick_units()`).

---

## 9. Stats and metrics

### New counters (`VectorCoreV2.stats`)

| Metric | Definition |
|---|---|
| `slot[k].fires`, `slot[k].occupancy` | fires; cycles with a resident spec |
| `slot[k].stall[reason]` | cycles in WAIT per reason: `raw`, `war`, `bank`, `mask`, `xbar`, `fu_busy`, `round` |
| `mask[b].reads`, `mask[b].conflicts` | mask reads per mask bank; conflicts = cycles with more than one requester on that bank (Q16) |
| `raw_stall_by_edge[(p,k)]`, `raw_stall_by_group[g]` | RAW wait cycles, by producer slot and by producer FU group |
| `group[g].accepts`, `.busy_cycles`, `.util` | utilisation = accepts × II / cycles; `busy` = any stage occupied |
| `group[g].frozen_cycles` | output not granted (WB backpressure) |
| `rtree.reductions`, `rtree.busy_cycles` | |
| `xbar.requests[d]`, `xbar.conflicts[d]`, `xbar.grants[d]` | per destination; conflicts = cycles with more than one requester |
| `bank[b].reads`, `.writes`, `.read_conflicts`, `.rw_collisions` | intra-slot same-bank reads; 1RW collisions |
| `wb.grants[src]`, `wb.denied[src]`, `wb.fifo_occupancy[b]` (histogram) | arbiter and buffer pressure |
| `wb.multicast_copies`, `slot[k].stall["war"]` on multicast targets | write-port cost of multicast; producer stalls caused by sibling registers still being read |
| `loop.rounds`, `loop.round_len` (histogram), `loop.fill_cycles`, `loop.steady_ii`, `loop.drain_cycles` | `steady_ii` = median round length over full rounds |
| `timeline` (optional) | per cycle: per-slot state and iteration, per-group busy, for the activity view |

### Existing metrics that break or change meaning under v2

| Metric | Why |
|---|---|
| `VectorSide.stats["lane_ops"]` (`scheduler/vector.py:347`) and `PerfMonitor` lane FLOPs = `lane_ops × n` (`perf_monitor.py:241`) | Counted at dispatch. A resident instruction is dispatched once and fires N times. Count at **fire** instead. |
| `PerfMonitor.mix` and `packets` (`perf_monitor.py:158-165`) | Counts per issued packet, so it undercounts loop work. |
| `wb_conflict_cycles` (`scheduler/vector.py:266-269`) | Hard-codes `reg % 4`. |
| `PerfMonitor._busy_units` "lanes" (`perf_monitor.py:180-181`) | Reads `_lane_ops` / `_lane_results`, which don't exist in v2. Becomes one entry per group. |
| `ActivityProbe` `lane0..N` and `vrf_b0..3` (`activity.py:100-104`, `:121-122`, `:255-256`) | Lanes and 4 decode-2 banks become slots, groups and 8 banks. |
| `sysarr_tpu_experiment` queue snapshot (`_build_packet`, `vliw_q`, `wb_buffer`) and lane op counts (`sysarr_tpu_experiment.py:225-238`, `:260-265`) | v1 internals. |
| Decode 2 `stall_reasons["unit"]` with `blocked_by in {alu, mul, reduction}` (`decode2.py:466-477`) | Lane readiness no longer gates decode 2 the same way (section 12). |

---

## 10. Implementation order and regression strategy

### Config flag (milestone M0)

```python
# vector_core/config.py
@dataclass(frozen=True)
class FUGroupSpec:
    name: str; ops: Tuple[str, ...]; latency: int; ii: int = 1; rtree: bool = False

@dataclass(frozen=True)
class VectorCoreConfig:
    arch: str = "lanes"                  # "lanes" (v1, default) | "fu_groups" (v2)
    # v1 (today's arguments, unchanged)
    lane_count: int = 16; fu_latencies: Optional[dict] = None
    # v2
    num_slots: int = 4
    fu_groups: Tuple[FUGroupSpec, ...] = DEFAULT_V2_GROUPS     # j = len(fu_groups)
    regs_per_bank: int = 32; bank_map: str = "mod"
    bank_read_ports: int = 1; bank_write_ports: int = 1; bank_port_mode: str = "1R1W"
    t_read: int = 1; t_xbar: int = 0; t_arb: int = 0; t_wbbuf: int = 1; t_write_visible: int = 1
    rtree_level_latency: int = 1
    wb_grants_per_cycle: int = 4; wb_fifo_depth: int = 4; wb_priority: Tuple[str, ...] = (...)  # Q11; sweep 1/2/4/8
    xbar_policy: str = "oldest"; round_mode: str = "lockstep"   # or "decoupled" (Q8)
    stall_mode: str = "stall"; strict: bool = True
    multicast_bits: Tuple[int, int] = (38, 35)  # pair mask in VV/VS/VI (Q3)
    rtree_shares_alu_port: bool = False         # Q7 revised: RTree has its own WB input (fu:rtree)
    masked_write: str = "enable"                # per-element write enable (Q10b)
    mask_banks: int = 2; mask_read_ports_per_bank: int = 1   # shared mask file (Q16)
    m0_uses_port: bool = True                   # unmasked ops read m0 on mask bank 0, as today (Q16a)
    mask_write_ports_per_bank: int = 1          # compare beats mv.stm on a conflict (Q21)
    read_mode: str = "strict"                   # or "compat" for v1-allocated programs (Q34)
    compat_read_penalty: int = 1                # extra cycles for an out-of-pair read in compat mode
    multicast_side: str = "same"                # copies keep vd[0] (Q17)
    multicast_writes_home: bool = True          # vd always written; mask only adds copies (Q17b)
    t_mts: int = 0                              # move-to-scalar offered to the scalar WB the same cycle (Q19)
    # DEFAULT_V2_GROUPS (Q6): alu L=1 with rtree, mul 1, exp 14, sqrt 8, recip 11; all II=1
    # crossbar outputs (Q4): the j groups, vlsu0..3, gsau, transpose, mts
    watchdog_cycles: int = 10_000
    @property
    def num_banks(self): return 2 * self.num_slots
```

- Add `make_vector_core(cfg, veggie_size, vls_count, dtype)`. It returns `VectorCore` for `"lanes"` and `VectorCoreV2` for `"fu_groups"`.
- Thread `vector_cfg` through `build_tpu_platform` (`sysarr_tpu_system.py:583-622`) and `build_atalla_platform` (`atalla_platform.py:133-163`). Today the latter forwards only `lane_count`.
- Add `tools/run_kernel.py --vector-arch {lanes,fu_groups}`.
- **Pass condition:** the whole suite and `tests/equivalence` stay **byte-identical** with the default config. Today that is 99 passed for `tests/vector_core`, `tests/scheduler/test_vector.py` and `tests/equivalence`.

### First version (decided, Q33)

The first running version runs **today's kernels** on the new core:
- **Milestones:** M0 → M1 → M2 → M3 → M4 → M7, skipping M5 (loops) and M6 (full stats and activity view) for now.
- **Platform:** the scheduler platform only (`build_atalla_platform`, `tools/run_kernel.py --vector-arch fu_groups`). The harness platform (`sysarr_tpu_system`) stays on v1 until later.
- **Data type:** BF16 only (Q14 deferred).
- **Mode:** single mode. Each packet's vector words become one round, with packet word k → slot V_k (the M7 rule).
- **Register reads:** `read_mode="compat"` (Q34), since today's programs aren't allocated for bank pairs. Pair masks are always zero in today's code, so there is no multicast.
- **Masks:** tracked by the vector scoreboard (Q20), with 1 write port per mask bank (Q21).
- **Done when:** `tools/run_kernel.py <kernel> --vector-arch fu_groups --golden` passes for `relu.c`, `add.c` and `softmax.c`, and `tools/compare_vector_arch.py` prints v1 vs v2 cycles, the stall breakdown and `compat.cross_pair_reads` for each.
- **Basic stats for this version:** total cycles, `slot[k].stall[reason]`, `group[g].accepts`, `wb.denied`, and the compat counters. The rest of section 9 comes with M6.

### Milestones

Each is small and has its own tests under `tests/vector_core_v2/`.

| # | Deliverable | Tests (all exact-cycle) |
|---|---|---|
| M1 | `FUGroup`, `ALUGroup` + `ReductionTree` (standalone `RTLModule`s) | push at c, valid at c+L; II honoured; with the output not granted nothing is lost or duplicated and `ready` drops; RTree depth = 5; element values equal v1 `_op_lut`+cast for every op; full-mask reductions equal `GlobalReductionUnit.reduce_tree` and `golden.lanes_reduce`; a plain op and a reduction can finish in the same cycle on `fu:alu` and `fu:rtree` with no stall; a frozen RTree blocks only reductions, and a frozen `fu:alu` blocks only plain ops (Q7 revised) |
| M2 | `BankedVRF` + `BankMap` | same-bank pair costs +1; a write in cycle w is invisible to a read in w and visible in w+1; a per-element write enable leaves masked elements unchanged; untimed API parity with `VectorCore.read_vreg`/`write_vreg` |
| M3 | `WBArbiter` + `WBBufferV2` | ≤ `bank_write_ports` per bank per cycle; priority order; a full FIFO back-pressures its source; per-register version order; a multicast entry waits for room in every target FIFO, and each copy becomes readable when its own bank's write lands |
| M4 | Slots + crossbar + scoreboard, **single mode** | `D(p→k) = L + 3` exactly for each group (ALU 4, EXP 17, reduction 9); a consumer in another pair reads a multicast copy at the same D; `vreg.ld` with a pair mask raises; a multicast with a zero mask writes only `vd`, and one with vd's own pair bit set writes `vd` once; 4 slots fire in one cycle when at most 2 read a mask, on different mask banks (e.g. an unmasked add, a MUL on m1, a `vreg.ld` and a `gemm.vv`); two slots on one group: the loser fires 1 cycle later (II=1); source outside the bank pair raises `SlotContractError`; two unmasked arithmetic slots: one waits 1 cycle with stall `mask` (both read m0 on bank 0); three mask-reading slots: at most two fire; with `m0_uses_port=False`, unmasked slots never wait on a mask port; `vmov.vts` → dependent scalar op: the scalar write lands at today's delay + `t_read`, and a refused move-to-scalar holds the `mts` destination |
| M5 | `issue_round` primitive + `LoopController`, **loop mode** (lockstep, then decoupled) | the same 4-slot chain (e.g. ALU→MUL→EXP→ALU) run by the `LoopController` and by an explicit prologue/body/epilogue driver gives identical cycles and final state; over N iterations final registers equal a sequential reference; rounds = N+S-1; `steady_ii` = max edge D; WAR: fixed `dst` never loses a value; loop-carried accumulator; the watchdog fires on a constructed stuck state |
| M6 | Stats, `PerfMonitor`/`ActivityProbe` v2 blocks | counters add up (fires = Σ slots × N); the activity JSON renders in `tools/activity_view.py` |
| M7 | Compatibility adapter: run today's packets in single mode (packet vector slot k → V_k) | `vector_program_0..2` and the assembly/C kernels: final state equals golden (with `reductions="tree32"`); print cycle deltas against v1 |

### Regression strategy

- **v1 is untouched.** All v2 code is in new files, and the only edits to existing files are the factory and config plumbing.
- `tests/equivalence/snapshot.json` is the v1 oracle. Don't regenerate it while doing v2 work. Add v2 scenarios as separate keys only once M5 is stable.
- New `tools/compare_vector_arch.py` runs one program under both archs and reports:
  - (a) whether final registers and DRAM are identical, which they must be
  - (b) cycles, vector-core utilisation and RAW stall cycles side by side, using the baseline table in section 0 as the first check
- Tests that inspect v1 internals (`test_vector_lane.py`, `test_vector_datapath.py`, the activity test and the experiment snapshot keys) stay v1-only. Shared behavioural tests, such as golden comparisons, are parametrized over `arch`.

---

## 11. Open questions

**Q1–Q11 and Q15–Q19 (with Q16a and Q17b) are decided**; Q7 was revised and Q18 superseded after the diagram review. See [Decisions](#decisions-q1q10). The original wording of Q1–Q10 is kept below for reference. Still open: Q12–Q14, Q22–Q23 from the diagram review (section 13), and Q24–Q32 from the CISC kernel direction (section 14). Q20, Q21, Q33 and Q34 are decided. **Nothing open blocks the first running version (M0–M4 + M7).** Q12, Q22 and Q23 are needed before loop mode (M5). Section 14 also answers part of Q12 (scalar operands are read once at issue) and most of Q13 (the kernel sequencer loads the slots). None of them blocks M1–M3. Q24, Q25 and Q28 have provisional answers. Q24–Q30 belong to the kernel and data-movement design, which is still under discussion, and should be settled before M8. Q20 and Q21 should be settled before M4. Q12, Q13, Q22 and Q23 should be settled before M5 (loop mode) and M7 (the packet adapter).

### Decided (for reference)

1. **Registers:** how many vector registers in total, and per bank (32 × 8 = 256?). Map `reg % 8` or block? Is the bank part of the architectural register number (compiler allocates), or does hardware steer the result to the consumer's pair? If hardware steers, how does it know the consumer?
2. **Operand-to-bank rule:** must src0 come from B_{2k-1} and src1 from B_{2k}? Can either operand be in either bank of the pair? How many read ports per bank? Is a register-file bank 1R1W or 1RW?
3. **Cross-slot reads:** can a slot read a register outside its pair at all (loads, GSAU results, loop-invariant constants)? If a value has several consumers, is it copied (by the compiler) or multicast (by WB)? Where do masks live, and how are they read per slot?
4. **Crossbar:** is it 2×(j+5) or 2j+5 (C12)? Are the transpose unit and move-to-scalar removed, or extra destinations (C13)?
5. **Writeback direction:** do `.vlsu_out` and `.gsau_out` feed the arbiter (diagram) or come out of the WB buffer (text) (C11)?
6. **FU groups:** latency and II for each group, especially the LUT EXP (numerics too: must equal `math.exp`→BF16 or not?) and the pipelined RECIP. Is latency fixed, or data-dependent (for example SQRT or RECIP early-out)? Where do `shl`/`shr` and `div` go?
7. **RTree:** latency per level; is it pipelined? Does it take data from the ALU output (`lanes_if.vector_output`) or straight from `lanes_in` (the dashed line in the diagram)? Masked elements: identity or skip? Does it share the ALU group's WB port?
8. **Rounds:** lockstep barrier or decoupled? Who owns the trip count and iteration counter (vector core or scheduler)? Data-dependent exit? How a reduction fits the skewed schedule: is a loop-carried reduction (an accumulator over iterations) allowed, and does its result go out once, after drain?
9. **Register reuse:** fixed `dst` per iteration (WAR/WAW as in section 5) or rotating/iteration-indexed registers? Is WAR enforced by hardware or a compiler contract?
10. **Timing terms:** `t_read`, `t_xbar`, `t_arb`, `t_wbbuf`, `t_write_visible` from the RTL. Can a bank be read and written in one cycle, and which value does the read see? Masked writes: read-modify-write or per-element write enable?

### Still open

11. **Decided: 4 grants per cycle by default, swept over 1/2/4/8.** Bank write ports stay at 1 (Q2); FIFO depth 4 and oldest-first priority stay as the defaults (section 7).
12. **Per-iteration operands:** how do loop-resident `vreg.ld`/`vreg.st` get per-iteration addresses (`base + i·stride`?), and `.vs` forms their scalar? Are masks fixed for the loop?
13. **Slot loading:** does V_k correspond to packet vector-word k (`PACKET_SIZE = 4`, `isa.py:36`)? What does loading cost, and can straight-line vector code issue while a loop is resident?
14. **Data type:** BF16 only (scheduler path), or must v2 also support the harness path's FP16?
15. **Decided: load it twice.** `vreg.ld` has only bits [39:38] free, so it can't carry a 4-bit pair mask; a value needed in two bank pairs is loaded once into each.
16. **Decided: one shared mask file, 2 banks × 1 read port, as today.** A slot that loses a mask read stalls a cycle with reason `mask` (section 4).
    - **16a. Decided: m0 takes bank 0's port, as today.** At most one unmasked op fires per cycle, and at most 2 mask-reading ops (section 4).
17. **Decided: copies keep the same side** (`vd[0]`); bit 39 stays unused. A value needed as src0 in one slot and src1 in another needs an explicit copy instruction.
    - **17b. Decided: as today, `vd` is always written.** The pair mask only adds copies, and vd's own pair bit is ignored. The diagram changes for multicast are in section 7.
18. **Superseded.** It was decided to avoid ALU/RTree collisions at issue, but Q7 was revised so the RTree has its own WB input, and there is no collision left to avoid.
19. **Decided: as today.** A move-to-scalar result is offered to the scalar writeback in the cycle it leaves the crossbar, held if refused, and never goes through the vector WB arbiter (section 6).
20. **Decided: masks are tracked by `VectorScoreboard` like vector registers** (section 5).
21. **Decided: 1 write port per mask bank**; on a conflict the compare wins and `mv.stm` retries the next cycle (section 4).
22. **`vreg.st` → SDMA ordering in loop mode:** today's per-`rs1` count at decode-2 issue undercounts a resident store that fires N times. Proposed: an SDMA on that `rs1` waits for `loop_done`; single mode keeps today's count.
23. **Scratchpad ordering between slots:** is a store and a later load of the same row inside one loop a compiler contract (distance ≥ S rounds), or does hardware order it? Proposed: a compiler contract, checked in strict mode.

From the CISC kernel direction (section 14):

24. **Provisional: a hand-written FSM per kernel** in the RTL. The simulator has one `KernelFSM` subclass per kernel (section 14.4). Kernels are still under discussion.
25. **Provisional: scalar and SDMA packets keep issuing** during a kernel, subject to hazards. Vector packets wait for `kernel_ready`. Data movement outside the vector core is still under discussion.
26. **Kernel-internal registers:** proposed that v240–v255 and m14–m15 are reserved and clobbered by kernels (ABI). The alternative is naming them in the instruction, but VK has no bits left.
27. **One path into the vector core:** do plain vector instructions also go through the sequencer as one-op kernels? Proposed: yes.
28. **Provisional: phases through the scratchpad.** Softmax takes 5 phases and 2 intermediate tiles, and a phase's `done` waits for its stores to reach SRAM (section 14.3).
29. **Kernel parameters beyond the VK fields** (slopes, epsilons, gamma/beta addresses, and now the **scratch tile for multi-phase kernels**): a parameter block in scalar registers, a descriptor in memory, or a second instruction word?
30. **First kernel set and its numerics:** which kernels first (RELU, add, maxpool, softmax, layernorm), and their exact BF16 results (e.g. RELU = `max(x, +0.0)`, which differs from today's −0.0 for negative inputs)?
31. **Masked-off elements** (from the draft RELU FSM, section 14.10): keep vd's old value (today), or take src0 ("merge from src0")? Merge lets a masked op write a fresh register in another slot's pair, with no in-place writes. Does it apply only to micro-ops, or to the ISA too?
32. **Micro-op loads may multicast:** Q15 forbids load multicast because `vreg.ld` has no free bits. Should loads issued by a kernel FSM (as `SlotSpec`s, not 40-bit words) be allowed a pair mask? Proposed: yes. RELU's compare + mul form needs it to fit in 4 slots.

From planning the first running version:

33. **Decided: the first version runs today's kernels** in single mode (M0–M4 + M7), scheduler platform, BF16 (section 10).
34. **Decided: compat read mode** for v1-allocated programs, with +1 cycle per out-of-pair read (section 4).

---

## 12. Downstream impacts (brief)

### Scheduler (`src/scheduler/`)

- **Decode 2 stops reading vector operands at issue** (`decode2.py:550-551`), because slots read their own banks every iteration. It passes register numbers instead.
- The vector part of the scoreboard (`decode2.py:185-225`) becomes a query to `VectorScoreboard.reg_ready`. This matters for scalar consumers of vector results (move-to-scalar) and for decode-2-level ordering.
- reggie's vector bank-conflict FSM (`VECTOR_BANKS=4`) is replaced by the slot bank-pair rule.
- `contract_violations`' lane limit (`LANE_ISSUE_W=2`) is replaced by slot and contract rules.
- `VectorSide`'s readiness, dispatch, offers, arbitrate and retire (`scheduler/vector.py:219-312`) shrink to a slot-loading adapter. Writeback arbitration and the `reg % 4` rule move into the vector core, so `writeback_at` stops carrying vector writes.
- `idle` and halt must also wait for `loop_done` and an empty WB buffer.
- Move-to-scalar results now come from the vector core's crossbar output, not from a decode-2 operand read. `MoveToScalar`'s hold-until-granted behaviour moves with it, and the timing is unchanged apart from the slot's 1-cycle read (Q19). Decode 2's scalar scoreboard keeps `rd` busy from slot load until the scalar write lands.
- New control: load slots, set the trip count, start the loop, wait for it to finish, then drain. These need encodings, decode, and stall reasons.

### Harness path

`VectorCore`'s packet builder (`build_packet`/`vliw_q`, `vector_core.py:399-446`, `:682-699`) is v1-only. For v2, harness instructions map to single-mode slot loads. The GEMM harnesses mostly use the VLSUs and GSAU and should port with little change. Their queue-depth snapshot keys change.

### ISA and compiler (`third_party/aihw-ppci-compiler`)

- New opcodes: vector sqrt and reciprocal (if they become instructions), and the loop/slot control above.
- A 4-bit multicast pair-mask field in bits [38:35] of VV/VS/VI. That touches `isa.FIELDS` (`isa.py:149-171`), `control.decode_vector`, and the assembler's operand syntax.
- Register allocation under bank-pair constraints: a home bank per value, pair masks for extra consumers, no live values in multicast sibling registers, and one `vreg.ld` per bank pair for loaded values read by several slots.
- An explicit copy instruction when a value is src0 in one slot and src1 in another, since multicast copies keep the home side (Q17).
- Mask allocation: each mask bank serves one read per cycle, and unmasked ops already use bank 0 through m0 (Q16, Q16a). So put real masks in odd registers (m1, m3, …), and schedule independent unmasked ops so that no more than one fires per cycle where possible.
- Modulo scheduling of a loop body onto S slots and j groups without same-group collisions.
- No prologue or epilogue code (hardware fills and drains).
- Iteration-indexed addressing for loop VLSU ops.

### Functional sim (`third_party/atalla-functional-sim`)

- The new opcodes, multicast copies, and loop semantics.
- A reduction order matching a fixed 32-wide RTree. Add `golden.lanes_reduce(..., order="tree32")`. Today `golden.py:46-66` hard-codes 16 lanes.
- The current LUT EXP and RECIP numerics, if the hardware's differ from `math.exp` and `1/x`.

### Tools and docs

- `PerfMonitor`, `ActivityProbe` and `tools/activity_view.py` blocks.
- `docs/architecture/vector-core.md`, which should also correct C1, C3 and C9 for v1.
- A new `docs/architecture/vector-core-v2.md` once M5 lands.

---

## 13. Diagram review: required updates and overlooked decisions

This compares the MxN block diagram (Scheduler → V1..V4 → veggie B1..B8 → crossbars → `vector_if_vX` → FU lanes / GSAU / VLSU → Write Back Arbiter → WB Buffer → veggie) with decisions Q1–Q19.
- 13.1 covers elements the diagram already has but draws differently from what was decided.
- 13.2 covers blocks it is missing.
- 13.3 covers design decisions that neither the diagram nor Q1–Q19 cover. They are listed as Q20–Q23 in section 11.

### 13.1 Elements to change

| # | Element | The diagram shows | Change to | Why |
|---|---|---|---|---|
| D1 | Bank labels | B1..B8, 1-based | Relabel **B0..B7**, or annotate each pair with "src0 / src1" | The plan uses `bank = reg % 8`, so V_k's pair is banks 2k-2 and 2k-1. "src0 from the even bank" means B1, B3, B5, B7 in the diagram's labels, which invites off-by-one mistakes (Q1, Q2) |
| D2 | Slot → bank arrows | Two arrows from each V_k into its pair | Label them as **read address/enable** (src0 → even bank, src1 → odd bank), and add a third request from each slot to the mask file (13.2) | Data flows bank → crossbar; the slot only addresses the banks (Q2) |
| D3 | Crossbar label | "each 2 x j + 5" | **"each 2 × (j+7)"**, with the outputs listed: j groups, vlsu0..3, gsau, transpose, mts. Each crossbar also carries the slot's **mask** with its operands | Q4. The mask goes with the token to the FU group, where it drives the per-element write enable (Q10b) |
| D4 | `vector_if_vX` | One shared bus with `.gsau`, `.vlsu` and `.lanes_in` taps | **One `vector_if` per slot** (4). Where the 4 meet, add a **4 → 1 arbiter in front of every destination**, with `ready` going back to the slots. The bundle carries `op, a, b, mask, dst, pair_mask, ver, iteration, imm`, and the scalar operand (Q12) | Destinations are shared and granted once per cycle (section 6, `DestArbiter`). The pair mask must travel with the result (Q3a) |
| D5 | `.lanes_in` into MUL/EXP/SQRT/RECIP | Dashed | **Solid.** Each group has its own arbitrated input | Every group is a real destination with its own arbiter and `ready` |
| D6 | Dashed `.lanes_in` → RTree | A bypass into the RTree | **Remove it** | Q7: the RTree takes only the ALU output. A reduction passes through the ALU as `add` with src1 = 0 |
| D7 | ALU and RTree outputs | **Two separate inputs** to the WB arbiter: the ALU output directly, and the RTree's `.lanes_out` | **Keep as drawn** (resolved 2026-11-08). Label the inputs `fu:alu` and `fu:rtree`. Optionally note that the ALU output is steered by op: reductions go into the RTree, everything else goes to `fu:alu` | The diagram originally conflicted with Q7 (shared input). Q7 was revised to match the diagram, and Q18's issue guard was dropped (section 2) |
| D8 | RTree output label | `vector_if_vX` | **`fu:rtree`** (its own WB arbiter input) | `vector_if` is the slot → destination interface; reusing the name for a result path is misleading |
| D9 | WB Arbiter | One output | **Two outputs:** vector results to the WB buffer, and **compare results to the mask file** | Compares write masks, not vector registers (section 4) |
| D10 | WB Buffer → veggie | One arrow into the bank array | **8 per-bank FIFOs** (depth 4), each draining through its bank's single write port. One multicast entry **fans out to up to 4 banks** (home bank, plus bank `2p + vd[0]` for each selected pair p). Writes use a per-element enable. Annotate **"≤ 4 writes/cycle (Q11)"** on the arbiter → buffer path, i.e. 4 write buses of 512 bits, not 8 | Sections 4 and 7; Q3, Q10b, Q11, Q17, Q17b. A single arrow suggests one write per cycle |
| D11 | VLSU "× 4" | One box | **Four VLSUs**, each with its own arbitrated `vlsu_in` and its own `vlsu_out` input to the WB arbiter. Annotate that store data arrives inline from the crossbar | Each VLSU is a separate destination and WB source (section 6). Inline store data replaces v1's untimed register read (C8) |
| D12 | Notes box | "lane = FU group …" | Add: compares run on the ALU group and write masks; shift and div are dropped; every FU group has II = 1 | Q6, section 2 |

### 13.2 Blocks to add

| # | Block | Connections | Decision |
|---|---|---|---|
| A1 | **Mask file** (16 × 32 bits, 2 banks, 1 read port per bank) with a **mask read arbiter** | In: a mask-read request from each slot, a write from the WB arbiter's mask output, and a write from the scheduler (`mv.stm`, Q21). Out: the granted mask, into that slot's crossbar | Q16, Q16a: at most 2 mask reads per cycle, at most 1 per bank, and m0 uses bank 0 |
| A2 | **Transpose unit** | In: the crossbar `transpose` output (`tpus.vi` pushes A; `tpop.vi` sends a request). Out: columns into the WB arbiter as source `transpose` | Q4 |
| A3 | **Move-to-scalar** | In: the crossbar `mts` output. Out: a one-entry hold register, then the scheduler's **scalar writeback**. It does not go through the WB arbiter | Q19 |
| A4 | **Vector scoreboard** | In: fire events from the slots (version, reader counts) and drain events from the WB buffer. Out: RAW/WAR `ready` to each slot, and `reg_ready` to the scheduler | Section 5, Q9. Without it nothing in the diagram stops a slot reading too early |
| A5 | **Loop controller** | In: slot specs, trip count and start from the scheduler. Out: per-slot eligible iteration, round barrier, `loop_done` back to the scheduler | Section 3, Q8 |
| A6 | **Scheduler ↔ vector core control** | Scheduler to core: slot specs, trip count, scalar operands (Q12). Core to scheduler: `loop_done`, `reg_ready`, `vreg.st` row-done (Q22) | The diagram's scheduler arrows only show instructions going down |
| A7 | **Backpressure (`ready`) arrows** | Destination arbiter → slot; WB arbiter → each FU group (freezes it in `stall` mode); full bank FIFO → WB arbiter | Sections 2, 6 and 7. The diagram shows only forward data |

Updated dataflow (text sketch; redraw in the real diagram):

```
Scheduler ──slot specs, trip, scalars──►  LoopController + VectorScoreboard  ──loop_done, reg_ready──► Scheduler
                                              │ eligible / ready per slot
                    V1          V2          V3          V4
                    │rd addr    │           │           │           mask read req ──► MaskFile (2 banks × 1R)
              [B0 B1]     [B2 B3]     [B4 B5]     [B6 B7]   (8 banks, 1R1W, per-element write enable)
                    │           │           │           │                ▲ granted mask (into each crossbar)
              XBAR 2×(j+7) per slot, carrying A, B, mask, pair_mask, ver
                    └───────────┴─────┬─────┴───────────┘
                   one 4→1 arbiter per destination (ready goes back to the slots)
  ┌─────┬─────┬──────┬───────┬──────────────┬──────┬───────────┬──────────────────────────┐
 ALU   MUL   EXP   SQRT   RECIP       VLSU0..3   GSAU     TRANSPOSE          MTS → hold → scalar WB
  ├─► RTree ──► fu:rtree                  │       │ ↔ SA       │
  └──────────── fu:alu   (two separate inputs, as drawn; Q7 revised)
  fu:alu  fu:rtree  fu:mul  fu:exp  fu:sqrt  fu:recip  vlsu_out×4  gsau_out  transpose
  └─────────────────────────── Write Back Arbiter ───────────────────────────┘
          │ vector results                                │ compare results
  WB buffer = 8 per-bank FIFOs, multicast fan-out ≤ 4     └──► MaskFile write  (◄── mv.stm from scheduler)
          └──► bank write ports ──► scoreboard (committed_ver, readable_at)
```

### 13.3 Decisions the diagram and Q1–Q19 don't cover

Each has a proposed default so M1–M4 can proceed. They are listed as Q20–Q23 in section 11.

- **Q20. Mask hazards.** Compares in one slot write a mask that a later slot reads. In loop mode that is the same RAW/WAR/WAW problem as for vector registers, including loop-carried masks. Today decode 2 keeps a busy bit per mask (`decode2.py:185-210`). The `VectorScoreboard` in section 5 tracks only vector registers.
  - *Proposed:* track the 16 mask registers in `VectorScoreboard` with the same per-register state and fire conditions, with one writer slot per mask per loop.
  - D for compare → masked consumer = L_alu + 3 = 4.
- **Q21. Mask write ports and `mv.stm`.** Today `VectorSide.arbitrate` grants every mask write (`scheduler/vector.py:262-265`), and `RegFile` asserts if two land on one mask bank in a cycle (`decode2.py:338-340`). There are two mask writers: compares (vector side) and `mv.stm` (a scalar-side op, `decode2.py:138-141`). In v2 the vector side produces at most one compare result per cycle, since compares come only from `fu:alu`. But `mv.stm` can land on the same mask bank in the same cycle.
  - *Proposed:* one write port per mask bank. A compare and an `mv.stm` on the same bank in one cycle are resolved in the WB arbiter, with the compare winning (it is older, since it fired from a slot) and `mv.stm` retrying the next cycle.
- **Q22. `vreg.st` → SDMA ordering.** `VstReservations` (`decode2.py:236-279`) counts outstanding `vreg.st` per `rs1` **at decode-2 issue** and releases a count when the row is written. An SDMA on that `rs1` waits. A loop-resident `vreg.st` is issued once but writes N rows, so counting at issue undercounts.
  - *Proposed:* while a loop with a resident `vreg.st` on `rs1` is loaded, an SDMA on that `rs1` waits for `loop_done`. In single mode, keep today's per-store count, reported by the vector core when the store fires and when its row is written.
- **Q23. Ordering through the scratchpad.** The register scoreboard doesn't see a `vreg.st` and a later `vreg.ld` of the same scratchpad row, for example V4 storing row r in iteration i and V1 loading it in iteration i+1. A row lives in one scratchpad and so goes through one VLSU, which is in order. The two still fire from different slots, though, and in lockstep their order within a round depends on stall timing.
  - *Proposed:* a compiler contract. A loop body doesn't load a row it stores unless the distance is at least S rounds. In strict mode the model checks this at run time with a per-row last-store-iteration table and raises `SlotContractError`.

Q12 (per-iteration operands) also needs a diagram change: draw the scalar path from the scheduler into the slots (A6). `.vs` forms, VLSU row addresses and the trip count all come from scalar registers.

---

## 14. Atalla-wide changes: CISC kernel instructions and the kernel sequencer

> **Follow-up:** [`vector_core_v2_system_proposal.md`](vector_core_v2_system_proposal.md) proposes the system-wide changes for software-pipelined activation kernels: the round-engine scheduling algorithm, rotating registers, FU modes, store on writeback, and a kernel-by-kernel analysis.
>
> **Status: under discussion (2026-11-08).** The CISC kernels and all data movement outside the vector core (SDMA, scratchpad use, hazards with the scheduler) are still being worked out by the team. Everything in this section is a working proposal. Answers recorded for Q24, Q25 and Q28 are **provisional**. Milestones M0–M7 must not depend on this section. The vector-core pieces it asks for (phases in `LoopController`, `done` waiting for scratchpad stores) are kept as optional extensions until it settles.

**The direction (2026-11-08).** Atalla will get CISC instructions for whole kernels, starting with RELU. When one reaches decode, an FSM (the **kernel sequencer**) expands it, follows the kernel, and drives the vector core: it loads the slots, sets the trip count, starts the loop and waits for it.

The diagram's **Scheduler → V1..V4** arrows are where that sequencer sits. It is also the answer to the open questions about who loads the slots (Q13) and where per-iteration operands come from (Q12).

This section covers what has to change across atalla for that to work. Q24–Q30 in section 11 are the decisions it raises.

### 14.1 What a kernel instruction does

- **What one instruction covers:** a whole tile-level kernel. For example, "RELU every row of the tile at scratchpad 0, rows `base .. base+n-1`, into scratchpad 1."
- **What the sequencer turns it into:** one or more **phases**. Each phase is up to S slot specs plus a trip count, run by the `LoopController` (section 3). Phases are separated by a `loop_done` barrier.
- **Micro-ops are internal.** The sequencer writes `SlotSpec`s directly, so a micro-op can use any FU op the hardware has, including ones with **no ISA opcode**: `max`/`min` (already in `_op_lut`, `vector_lanes.py:58-59`), `sqrt` and `recip`, and the RTree's broadcast modes. The ISA doesn't need new opcodes for these (compare section 12).
- **Scalar operands are read once.** Base rows, row count and column count are read by decode 2 when the kernel issues, as every scalar operand is today (`decode2.py:537-557`). They stay fixed for the whole kernel. A per-iteration address is `base + i·stride`, computed in the VLSU slot (`row_stride_per_iter`, section 3). That settles the scalar-source part of Q12.

### 14.2 Worked example: RELU

**Today** (`relu.c`, assembled in `third_party/atalla-functional-sim/tests/relu.in`): each row is 5 vector ops in a serial chain (`vreg.ld`, a compare into a mask, zero `v1`, a masked add, `vreg.st`), with one instruction per packet and a scalar loop around them. In the getting-started report, 41% of `relu.c`'s 768 cycles are instruction-cache misses on those packets.

**v2: one instruction, one phase, 3 slots.** Register numbers follow the reserved-register proposal in 14.6.

| Slot | Micro-op | Reads | Writes | Group |
|---|---|---|---|---|
| V1 | `vreg.ld` row `src_base + i` from `src_sid` | — | v250 (bank 2: V2's src0 bank) | vlsu[src_sid] |
| V2 | `max v252 ← v250, 0.0` (internal op, unmasked) | v250 | v252 (bank 4: V3's src0 bank) | ALU |
| V3 | `vreg.st` v252 to row `dst_base + i` of `dst_sid` | v252 | scratchpad | vlsu[dst_sid] |

- **Trip count** = n rows, so the phase takes n + 2 rounds.
- **Steady state:** about 1 row per max(D(load → max), D(max → store) = 4) cycles. D(load → max) is the VLSU load latency + `t_wbbuf` + `t_write_visible`.
- **Masks:** none are used apart from m0, and only V2 reads a mask, so Q16a costs nothing here. V4 is idle.
- **Numerics:** `max(x, 0)` gives **+0.0** for a negative x, where today's masked multiply-by-zero gives −0.0. The compiler's validator uses `np.maximum` with `atol=1e-6` (`run_kernel_validation.py:270-278`), so both pass, but a bit-exact `--golden` check against the functional sim would see the sign bit differ. The functional sim's RELU must be defined as `max(x, +0.0)` (Q30).

### 14.3 Worked example: softmax shows the limit of S = 4

Per row, `softmax.c` does `vreg.ld` → `rmax` (broadcast) → `sub` → `exp` → `rsum` (broadcast) → `vmov.vts` + scalar `1/x` → `mul.vs` → `vreg.st`.

**In v2:**
- The scalar round trip goes away: a RECIP micro-op on the broadcast sum replaces `vmov.vts` + scalar divide.
- The loaded row `v` has two consumers (`rmax` and `sub`) in different slots. Loads can't multicast (Q15), so it is **loaded twice**.
- `exp_v` has two consumers (`rsum` and `mul`), so it is **multicast**.
- The body is therefore 9 slot ops: 2 loads, `rmax`, `sub`, `exp`, `rsum`, `recip`, `mul`, store. That is more than S = 4.

**Provisional (Q28): phases through the scratchpad.** More slots and multi-op slots were not chosen. Every phase must fit in 4 slots, and a phase may only pass a value to the next phase by storing it to the scratchpad.

**Softmax works out to 5 phases**, each over all n rows:

| Phase | Slots | Reads | Writes |
|---|---|---|---|
| P1 | `vreg.ld x` → `rmax` (broadcast) → `vreg.st` | x | M (each row's max, broadcast) |
| P2 | `vreg.ld x`, `vreg.ld M` → `sub` → `vreg.st` | x, M | D = x − max |
| P3 | `vreg.ld D` → `exp` → `vreg.st` | D | E = exp(D) |
| P4 | `vreg.ld E` → `rsum` (broadcast) → `recip` → `vreg.st` | E | R = 1 / Σ |
| P5 | `vreg.ld E`, `vreg.ld R` → `mul` → `vreg.st` | E, R | output |

- P2 and P5 use two loads because `sub` and `mul` need both operands in their own bank pair, and loads can't multicast (Q15).
- P3 and P4 can't merge: `exp` → `rsum` → `recip` → store plus the load is 5 ops, and E is needed again in P5.

**What phases cost:**
- **Rounds:** each phase takes n + S_used − 1 rounds, and each round is at least that phase's largest edge D. P3 is limited by D(EXP) = 17, so about 17 cycles per row. P1 and P4 are limited by the reduction, D = 9.
- **Scratchpad traffic:** 8 row loads and 5 row stores per input row, against 1 and 1 for a single-phase kernel.
- **Scratchpad space:** intermediate tiles M, D, E and R. With ping-pong reuse (D can reuse M's rows, and R can reuse D's), that is **2 intermediate tiles the size of the input**, plus the output. One can be the output tile itself. Where the other lives has to be passed in somehow, but VK has no field left, so it joins Q29.

**Ordering between phases:** P2 loads rows that P1 stored. The `LoopController`'s `done` (section 3) must therefore also wait until **every scratchpad store of the phase is written to SRAM**. That is the event `VstReservations` waits for today (`decode2.py:36-46`). Only then does the next phase start. Within a phase, Q23's rule still applies.

Short kernels (RELU, add, scale, maxpool rows) fit in one phase.

### 14.4 The kernel sequencer

**Where it starts: in EX, not decode 2.** Decode 2 has no flush input, and a wrong-path packet is squashed at the D2/EX latch (`core.py:351-359`, `decode2.py:60-63`). A kernel that started at decode 2 could write the scratchpad on a mispredicted path, and that can't be undone. So:
- decode 2 issues it like any packet
- the sequencer accepts it from the D2/EX latch, as `VectorSide.dispatch` does today (`scheduler/vector.py:289-312`)

**States:**
- **IDLE**
- **LOAD:** write phase p's slot specs, with scalar operands filled in, and the trip count.
- **RUN:** the `LoopController` runs; wait for `loop_done`.
- Then the next phase if there is one, otherwise **RELEASE** (free the scratchpad reservations, 14.5), then **IDLE**.

**Occupancy:** one kernel at a time, since there is one `LoopController`. Decode 2 sees a new unit-ready signal, `kernel_ready` = sequencer IDLE (optionally a one-entry queue), used like `vlsu_ready` (`decode2.py:454-482`).

**Representation (provisional, Q24): one hand-written FSM per kernel in the RTL.**
- **In the simulator:** one `KernelFSM` subclass per opcode (`ReluFSM`, `SoftmaxFSM`, …), on a shared base that provides IDLE/LOAD/RUN/RELEASE and talks to the `LoopController`. Each subclass lists its phases: slot templates (op, group, internal registers, addressing from operand fields, immediates) and where each trip count comes from.
- Each subclass can override per-state timing (e.g. LOAD cycles) to match its RTL FSM, since hand-written FSMs need not share timing.
- When a kernel class is registered, the simulator checks its phase list against the slot rules (bank pairs, one writer per register, mask ports, ≤ S slots per phase), so a mistake in a kernel shows up before any program runs.
- **Cost of the choice:** every new kernel is new RTL control logic and a new simulator class. Not a table entry.

### 14.5 Front-end, decode and hazard changes

| Area | Change | Today |
|---|---|---|
| **ISA** (`isa.py`) | New opcodes from the free range **80–127** (48 free; 1–79 are all used). New instruction type **VK**, laid out like VM: `[6:0]` opcode, `[14:7]` rs3 = row count, `[22:15]` rs1 = source base row, `[30:23]` rs2 = destination base row, `[35:31]` num_cols, `[37:36]` source sid, `[39:38]` destination sid. That fills all 40 bits, so extra parameters (a leaky-RELU slope, a layernorm epsilon, gamma/beta addresses) need another source (Q29) | `isa.py:49-171` |
| **Decode 1** (`decode1.py`) | `classify` gets a `KERNEL` class for the new opcodes, routed to the vector slots. Without it they classify as `NONE` and are dropped | Vector = opcodes 51–77 plus transpose (`decode1.py:56-70`) |
| **Packet contract** | At most one kernel instruction per packet, and no other vector op beside it. Scalar and SDMA ops may share the packet | `contract_violations`, `decode2.py:152-183` |
| **Decode 2** | Decode VK in `control.decode_vector`, read its 3 scalar operands, gate on `kernel_ready` | `control.py:280-340` |
| **Input hazard** | None needed. An SDMA holds its `rs1` busy until the scratchpad finishes (`decode2.py:33-34`), and the kernel reads the same base register as `rs1`, so the scoreboard already makes it wait for the tile to arrive, as `vreg.ld` does today | — |
| **Output hazard** (Q22) | The kernel **reserves its destination base register** (rs2) in a `VstReservations`-style table at issue, and releases it at RELEASE. A later `scpad.st` / `scpad.ld` on that register waits. That is today's `vreg.st` rule (`decode2.py:36-46`), applied once per kernel, so the "resident store fires N times" problem in Q22 goes away for kernels | `VstReservations`, `decode2.py:236-279` |
| **What keeps issuing during a kernel** (provisional, Q25) | Scalar and SDMA packets keep issuing, subject to the hazards above. Vector packets wait for `kernel_ready`, because the slots belong to the kernel. So the scalar loop bookkeeping and the next tile's `scpad.ld` (into a different tile) overlap with the kernel | — |
| **Halt and idle** | Halt waits for the sequencer to be IDLE and the reservation table to be empty | Halt waits for the vst table (`decode2.py:46`) |

### 14.6 Registers a kernel uses internally (Q26)

A kernel needs vector registers in specific banks: each value lives in its consumer's bank pair, on the right side (Q2). The VK format has no bits left to name them.

**Proposal:** an ABI rule that reserves **v248–v255** (address 31 in every bank, so `v248 + b` is in bank b, exactly one per bank) and **v240–v247** as a second set, plus **m14–m15**, as **clobbered by any kernel instruction**. The compiler keeps no live values there across a kernel. Table templates name these registers directly, and the load-time check verifies them. The RELU example in 14.2 uses v250 and v252.

### 14.7 Vector core changes beyond sections 2–9

- `LoopController` gets **phases**: `load_slots` is called once per phase, and `loop_done` is the barrier between them. For a phase, `done` also waits for every scratchpad store of that phase to reach SRAM (Q28), so the next phase's loads see the stored rows.
- `VectorScoreboard` treats the kernel's reserved registers as `pre` (written before the loop) at phase start.
- The values a phase leaves in the reserved registers are visible to the next phase, so a later phase can read an earlier phase's results.
- Plain vector instructions (Q27): proposed to go through the same sequencer as **one-op kernels** (single mode, trip = 1). That gives the vector core one way in, and M7's packet adapter becomes "decode 2 hands each vector op to the sequencer".

### 14.8 Compiler, functional sim and tools

- **Compiler** (`third_party/aihw-ppci-compiler`):
  - An intrinsic per kernel, e.g. `relu_tile(src_sid, src_base, dst_sid, dst_base, rows, cols)`.
  - Treat v240–v255 and m14–m15 as clobbered.
  - For code written as kernel instructions, the compiler needs **no modulo scheduling and no bank-aware allocation**: the sequencer's templates already do both. Section 12's harder compiler work is needed only for loops still written as plain vector code.
- **Functional sim** (`third_party/atalla-functional-sim`): each kernel opcode as a function over scratchpad rows, with the same per-op BF16 rounding as the micro-op sequence, so `--golden` stays bit-exact. RELU is `max(x, +0.0)` (14.2).
- **Simulator:** a new `src/scheduler/kernel_seq.py` (the `KernelFSM` base, one subclass per kernel, and a registry from opcode to class), wired in `core.py` beside `VectorSide`.
- **Stats:** `kernel.issued[op]`, `kernel.cycles[op]`, `kernel.phases`. FLOPs are counted per micro-op fire (section 9). PerfMonitor's instruction mix counts a kernel as one packet, so report kernel micro-ops separately.
- **New milestone M8** (after M5, and only once the kernel design settles): the sequencer plus `relu` as a kernel instruction.
  - Tests: the output equals `relu.c`'s within the validator's tolerance, and equals the functional sim bit for bit.
  - The phase takes n + 2 rounds.
  - A wrong-path `relu` is squashed before it touches the scratchpad.
  - `scpad.st` of the output waits for RELEASE.
  - Report cycles against `relu.c` on v1.

### 14.10 The team's draft RELU FSM (2026-11-08): discussion notes

The draft FSM has these states:
- RELU_START
- PROLOGUE_0: `load[0]`
- PROLOGUE_1: `load[1]`, `compare[0]`
- PROLOGUE_2: `load[2]`, `compare[1]`, `mul_zero[0]`
- BODY_LOOP: `load[i]`, `compare[i-1]`, `mul_zero[i-2]`, `store[i-3]`
- CHECK_DONE: `i == ROWS?`
- EPILOGUE_0..2: drain the last 3 rows
- back to RELU_START

Note on the drawing: "if rows < 4, don't use relu intrinsic".

**How it maps to the plan:**
- Each state is one lockstep round. load → V1, compare → V2, `mul_zero` → V3, store → V4.
- 3 prologue states + (N − 3) body states + 3 epilogue states = **N + 3 = N + S − 1 rounds**, exactly what the `LoopController`'s fill/drain rule gives. The simulator can run it either way (section 3, "Rounds are the vector core's primitive").

**Points to discuss:**

1. **A state lasts at least D cycles, not 1.** There is no forwarding, so `compare[i-1]` in BODY_LOOP can't read row i−1 until that row's load has landed. The same holds for each later stage. The FSM must **hold each state until all its ops have fired**, so it needs a "round done" / stall input from the vector core. A steady-state round lasts at least max(D(load → compare), D(compare → mul) = 4, D(mul → store) = 4) cycles.
2. **v has two consumers in different slots.** `compare` (V2) and `mul_zero` (V3) both read the loaded row as src0, so it must be in both V2's and V3's bank pairs.
   - Under Q15, loads can't multicast, so it would be loaded twice. That is 5 ops, which doesn't fit 4 slots.
   - But Q15's reason was encoding: `vreg.ld` has no free bits for a pair mask. An FSM micro-op is a `SlotSpec`, not a 40-bit word, so **a micro-op load can carry a pair mask** and multicast to V2's and V3's pairs (Q32).
3. **What `mul_zero` keeps in masked-off elements.** Today a masked op keeps the old value of `vd` in masked-off elements. For RELU those elements must be v. The ways to get that:
   - **In-place (vd = v):** two writers of one register per iteration, which breaks Q9. And `vd` sits in V3's pair while the store (V4) reads V4's pair; a multicast copy with a per-element write enable would leave stale elements in V4's copy.
   - **Merge from src0:** masked-off elements take src0 (v) instead of vd's old value. The FU writes every element, into a fresh register in V4's pair (Q31).
   - **Use `max(v, 0)` instead of compare + mul:** 3 ops and no mask (section 14.2).
4. **The mask is a cross-slot dependency.** `compare` (V2) writes a mask that `mul_zero` (V3) reads, and with a fixed mask name the next iteration's compare overwrites it. That is RAW and WAR on a mask register, so **Q20 (masks tracked by the scoreboard) is required** for this FSM.
5. **"rows < 4 → don't use the intrinsic".** That limit comes from the explicit prologue, which always loads rows 0–2, and from BODY_LOOP running at least once. Guarding each op with `0 ≤ index < ROWS` (skip it otherwise) makes any ROWS ≥ 1 work and removes the compiler special case. That is what the `LoopController`'s rule does.
6. **Off-by-one at CHECK_DONE.** EPILOGUE_0 issues `compare[i]` and `store[i-2]`, so the last body state must have loaded row i = ROWS − 1. Either the check is `i == ROWS − 1`, or i is incremented after the check. The drawing should say which.
7. **When the FSM can go back to RELU_START.** After EPILOGUE_2 issues `store[i]`, the stores are still in flight. The FSM can accept a new kernel, but the output tile's reservation (section 14.5) is released only when the last store reaches SRAM.

### 14.9 Diagram additions for the kernel path

- **Between the Scheduler and V1..V4:** a **Kernel Sequencer** block containing the micro-program table and the phase FSM. It sends slot specs, the trip count and phase start to the slots and loop controller, and takes `loop_done` back.
- **From the Scheduler into the sequencer:** the decoded kernel instruction plus its 3 scalar operand values, entering at EX (after the D2/EX latch).
- **Back to the Scheduler:** `kernel_ready` (unit ready) and the release of the destination reservation.
- Annotate the reserved registers (v240–v255, m14–m15) on veggie.
