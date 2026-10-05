# Scheduler RTL bugs found while modelling it

Found while transcribing the scheduler into atalla-sim (`src/scheduler/`), for
whoever is verifying the SystemVerilog. Everything here is against atalla
branch `scheduler_integration_SP26_joshklug` at `1dfe2e35` unless it says
otherwise; paths are under `rtl/`.

The simulator models what the RTL is meant to do, not these bugs, so it can
serve as the reference for them. Each entry says what it models instead.

**How each was found**
- **Reading:** found by reading the RTL. It hasn't been run in simulation yet, so a directed test should confirm it.
- **Questa:** seen in a Questa run.

**Severity**
- **High:** wrong results or a hang in ordinary programs.
- **Medium:** needs a specific instruction or condition.
- **Contract:** the RTL silently drops work that the compiler is supposed to never produce. It needs an assertion, or a guarantee from the compiler.
- **Doc:** a stale comment or a misleading name.

## Summary

| # | Severity | Where | Bug |
|---|---|---|---|
| 1 | High | `scheduler_core.sv` | Flush clears only the D1/D2 scalar slots; wrong-path vector and SDMA ops issue |
| 2 | High | `decode_2.sv`, `dependency_checker.sv` | A packet squashed on a redirect still reserves its registers, which then stay busy forever |
| 3 | High | `vector_control_unit.sv` | `lw.vi` marks `vd` busy, but nothing ever writes it back |
| 4 | High | `v_wb_arbiter.sv` | Writebacks from VLSU 2 and 3 are never arbitrated |
| 5 | High | `v_wb_arbiter.sv`, `reggie.sv` | Vector WB ports and VRF banks use different register bits; a resulting write conflict loses or corrupts the write |
| 6 | Medium | `dependency_checker.sv` | Mask writes from vector slots 2–3 are never tracked; `mv.stm` is never WAW-checked |
| 7 | Medium | `decode_2.sv` | `sqrt.bf` can issue into a busy EX2 |
| 8 | Medium | `dcache/cache_bank.sv` | A store miss to words 0–1 of a block shifts the fill by one beat when memory is busy |
| 9 | Medium | `dcache/` | 1-bit miss tag for 8 MSHRs; merges overwrite the tag; the LSU ignores it |
| 10 | Medium | `dcache/cache_bank.sv` | The FSMs and `sram_bank` disagree on when an SRAM access is taken: stray accesses, a fill can latch another set, and lookups can starve a fill |
| 11 | Medium | `decode_2.sv` | The EXP unit's readiness is ignored |
| 12 | Contract | `xbar_4x5_exec_comb.sv` | Two ops for one scalar EX unit: the second is dropped; EX1 is both ALU and control |
| 13 | Contract | `source_reg_allocator.sv` | Reads beyond the ports are zeroed, and escape the dependency check |
| 14 | Contract | `scheduler_core.sv` | Extra lane, GSAU, VLSU or SDMA ops in a packet are dropped or overwritten |
| 15 | Low | `dcache/cache_bank.sv` | A fill starts a RAM burst even when it needs no RAM data |
| 16 | Doc | several | Stale comments and inverted names |
| 17 | Medium | `dcache/cache_bank.sv`, `ld_st_unit.sv` | A request made while the dcache SRAM is busy is never answered |
| 18 | Medium | `dcache/cache_mshr_buffer.sv` | A miss with the MSHR buffer full is reported as a miss but never recorded |
| 19 | Medium | `dcache/` | A miss merged after its entry's fill has started is never applied |
| 20 | High | `execute_stage.sv` | Load and store addresses ignore the base register: the address is the immediate alone |
| 21 | High | `control.sv` | `jal` and `jalr` scale their byte offset by 4 |
| 22 | Medium | `scalar_control_unit.sv` | A branch's increment is sign-extended; the assembler encodes 0..127 |

Bugs 9, 10 and 17–19 can't show up today, because the load/store unit (EX5) is
blocking. It never has more than one miss outstanding, and it never sends a
request while a fill is using the SRAM. In today's configuration the
blocking LSU is correct, just not lockup-free. These bugs are in the cache's
lockup-free machinery, and they block making the LSU non-blocking. Of the
dcache bugs, only 8 can show up today. They become real as
soon as the LSU is made non-blocking, which is what the cache was built for.

## Details

### 1. Flush clears only the scalar slots of D1/D2 — High, Reading

**Where:** `modules/scheduler/scheduler_core.sv:80-81`, the "DEC1 outputs to latch" `always_comb`.

**What:** On `redirect_valid || internal_halt`, only
`n_D1_D2_latch.scalar_instrs = NOP_PACKET` is assigned. The vector and SDMA
slots, `pc`, `valid` and the prediction fields aren't assigned on that path,
so the `always_comb` infers a latch and the register keeps the wrong-path
packet. `decode_2` has no flush input, so the next cycle it issues the
wrong-path packet's vector and SDMA ops.

**Effect:** After a mispredict, vector loads and stores, GEMMs and SDMAs
from the fall-through packet execute. That changes results, not just timing.
The synthesizer may also warn about the inferred latch.

**Check:** Put a mispredicted branch in a packet followed by one containing a
`vreg.st` or `scpad.st`, and assert that nothing from that packet reaches
`D2_EX_vec_sdma_latch`. Alternatively, assert that after any redirect cycle,
`D1_D2_latch.valid == 0` or the latch holds the redirect target's packet.

**Sim:** flush clears every slot and drops valid.

### 2. Squashed packets reserve their registers forever — High, Reading

**Where:** `modules/scheduler/decode2/decode_2.sv:118` (`assign dcif.ready = d2if.ready`) and `decode2/dependency_checker.sv:77` (`if (dc_if.ready)`); the D2/EX latch in `scheduler_core.sv` zeroes the packet on `redirect_valid || internal_halt`.

**What:** The dependency checker sets a destination's busy bit whenever
decode 2 is ready. On a redirect (or halt) cycle, the packet in decode 2 is
dropped at D2/EX, but its busy bits are set anyway. Nothing ever writes those
registers back, so the bits never clear.

**Effect:** Any later instruction that reads or writes one of those registers
stalls forever. `*_halt_ready` stays high, so halt never completes.

**Check:** Assert that every bit set in a dependency table belongs to an
instruction that entered D2/EX. Or test it directly: a branch that
mispredicts while the next packet (writing `r5`) is in decode 2, then read
`r5`, then halt. It should finish.

**Sim:** only a packet that actually issues reserves its registers.

### 3. `lw.vi` reserves a register nothing writes — High, Reading

**Where:** `modules/scheduler/decode2/vector_control_unit.sv:171-184` (`LW_VI` sets `vector_reg_write = 1` with `vd = I[14:7]`).

**What:** A weight load writes the systolic array's weights, not a vector
register. `gsau_control_unit` only raises `sa_weight_en`, with no FIFO push
and no writeback. The scoreboard marks `vd` busy and nothing clears it.

**Effect:** Whatever register the encoding happens to name in `[14:7]` is
busy forever: a hang the next time anything touches it, and halt never
completes.

**Check:** Run `lw.vi` with `[14:7] = 5`, then `add.vv v6, v5, v5`. It hangs.
Also assert that `vector_halt_ready` falls after a `lw.vi`-only program.

**Sim:** `lw.vi` writes no register.

### 4. VLSU 2 and 3 writebacks are never arbitrated — High, Reading

**Where:** `modules/scheduler/writeback/v_wb_arbiter.sv:9` (the TODO "Add two more vector inputs from the load/store units") and `:83-106`. Only `vlsu_out.wb[0]` and `wb[1]` are handled.

**What:** There are four VLSUs, one per scratchpad. Loads through scratchpads
2 and 3 produce `wb[2]` and `wb[3]`, which the arbiter never looks at. Their
`vlsu_wb_ready` keeps its default of 1, so the VLSU believes its writeback
was taken.

**Effect:** A `vreg.ld` with `sid` 2 or 3 never writes its register, and its
busy bit never clears, so anything that reads the register hangs.

**Check:** `vreg.ld v5, ..., sid=2`, then read `v5`. Assert that each
`vlsu_out.wb[i].valid` is eventually followed by a write to `wb[i].vdst`.

**Sim:** all four VLSUs write back.

### 5. Vector writeback port vs register-file bank — High, Reading

**Where:** `modules/scheduler/writeback/v_wb_arbiter.sv:85, 98, ...` choose the write port from `vdst[7:6]`. `decode2/regfile/reggie.sv:45` banks writes by `vd[1:0]`. Then:
- `reggie.sv:113`: a conflict only moves to CONFLICT when `dependencies_ready`;
- `reggie.sv:189`: a pending write takes `rif.vd` and `rif.vdata` from the current cycle.

**What:** The arbiter guarantees one write per *port*, assuming port equals
bank. But the VRF banks by the low bits, so two writebacks in one cycle with
equal `vd[1:0]` and different `vd[7:6]` (say `v4` and `v68`) both get
through, then conflict inside `reggie`:
- **Dependencies not ready:** the FSM stays in READY, and the second write is lost when the next cycle recomputes requests from fresh inputs.
- **Dependencies ready:** the pending write is performed a cycle later using *that* cycle's `vd` and `vdata`, which by then belong to a different writeback.

**Effect:** A vector register write is lost or lands with the wrong data.
For scalars, the scalar arbiter's port does equal the bank (`rd[1:0]`), so
scalar writes never conflict.

**Check:** Get two vector writebacks into one cycle, for example a GSAU
result to `v4` and a VLSU load to `v68`. Assert each register holds its
value. Also assert that `reggie` never sees two write requests on one bank.

**Sim:** vector registers are the vector core's register file, which serves
one write per bank per cycle and retries the other. Nothing is lost.

### 6. Mask writes from vector slots 2–3 are untracked — Medium, Reading

**Where:** `modules/scheduler/decode2/dependency_checker.sv:92` (setting) and `:124` (the WAW check). Both loops run to `MASK_WRITE_PORTS = 2` and `MASK_READ_PORTS = 2`, and index by vector *slot*.

**What:**
- **Mask writes:** a mask write from vector slot 2 or 3 (`mgt.mvv` and the like) never sets its busy bit, and is never WAW-checked.
- **`mv.stm`:** its mask write (`scalar_m_WEN`) is only set for slots 0–1, and is never WAW-checked at all.

**Effect:** A later instruction reads the mask before it's written (RAW), or
two mask writes complete out of order (WAW).

**Check:** Put `mgt.mvv m5, ...` in vector slot 3 and `add.vv ..., m5` in the
next packet, with a slow mask writeback. The reader must wait for it.

**Sim:** every mask write is tracked and checked.

### 7. `sqrt.bf` skips EX2's structural check — Medium, Reading

**Where:** `modules/scheduler/decode2/scalar_control_unit.sv:145` (`fu_enable = sqrt_valid`) and `decode2/decode_2.sv:195`. The EX2 list has `bf_div`, `s_div`, `s_mod`, `BF_to_int` and `int_to_BF`, but not `sqrt_valid`.

**What:** The execute crossbar routes `sqrt_valid` (`4'b1111`) to EX2, but
decode 2 never asks whether EX2 is ready for it.

**Effect:** A `sqrt.bf` right behind a divide issues into a busy EX2 and is
dropped or corrupts the divide.

**Check:** Issue `div.s` then `sqrt.bf` in the next packet. The sqrt must
wait for EX2.

**Sim:** `sqrt.bf` waits for EX2.

### 8. A store miss to words 0–1 shifts the fill — Medium, Reading

**Where:** `modules/scheduler/dcache/cache_bank.sv:233` (the word counter's advance condition) and `:352-363` (BLOCK_PULL).

**What:** In BLOCK_PULL, a word pair holding a pending store's word advances
the counter *without waiting for* `ram_mem_complete`, and takes the pair's
other word from `ram_mem_data` as it stands. That's only right if that pair's
beat is on the bus in the same cycle. `sim_ram_rr` serves a granted
data-cache burst's first beat in the request cycle and the other seven on the
next seven, so with the bus free the beats line up and nothing goes wrong.
When the burst's start is delayed, for example by a scratchpad holding the
`sim_ram_rr` bus lock or by any memory with latency, a store miss to word 0
or 1 advances the counter before the first beat arrives. The pair's other
word gets whatever is on the bus, and every later pair takes the previous
pair's beat.

**Effect:** For a store miss to the first pair of a block, when the memory
is busy or slow, the block is filled shifted by one beat and is corrupt. The
last pair is exempt (its condition excludes `BLOCK_SIZE - 2`).

**Check:** Store to offset 0 of an uncached block *while a scratchpad DMA
holds the memory*, then load offsets 2–15 and compare with memory.

**Sim:** the fill takes each beat when memory delivers it (the memory's
first-beat latency is a parameter), then applies the MSHR's stores on top.

### 9. The miss tag can't tell misses apart — Medium, Reading

**Where:**
- `include/scheduler/dcache/cache_types_pkg.svh:9, 11`: `MSHR_BUFFER_LEN = 8`, `UUID_SIZE = 1`;
- `modules/scheduler/dcache/cache_mshr_buffer.sv:29, 84`;
- `modules/scheduler/execution_units/ld_st_unit.sv:136`.

**What:**
- **Tag width:** tags are 1 bit, for up to 8 outstanding misses.
- **Merging:** a secondary miss merged into an existing entry overwrites that entry's tag with the new requester's tag (`next_buffer[i].uuid = uuid`) and advances the tag counter, so the first requester's tag never completes.
- **LSU:** the load/store unit wakes on any fill completion (`block_status`) and never compares `uuid_block` with its own `mem_out_uuid`.

**Effect:** None today, because the LSU has at most one miss outstanding.
With a non-blocking LSU, requesters wake on the wrong fill or never wake.

**Check:** Have enough tag bits for the MSHR depth. Assert that a merge
returns the existing entry's tag, and that each requester completes on its
own tag.

**Sim:** `src/scheduler/dcache.py` is a lockup-free cache with MSHRs. Every
request carries the requester's own id and is answered with it: loads on a
hit or when their fill completes, stores on a hit or as soon as the MSHR
holds them.

### 10. The FSMs and the SRAM disagree on when an access is taken — Medium, Reading

**Where:** `modules/scheduler/dcache/cache_bank.sv`:
- `:83-87`: the SRAM enables are driven from the grant alone, without `!sram_busy`;
- `:170, :200, :466, :481, :500`: each FSM only advances on `grant && !sram_busy`;
- `:96`: `main_sram_rdone = sram_rdone && !hc_sram_rdone`;
- `:295-296`: the miss FSM loads `sram_rdata` into its set copy on any `main_sram_rdone`, in every state.

`modules/scheduler/dcache/sram_bank.sv`:
- `:32`: `busy = r_busy || w_busy`;
- `:42, :68`: each channel starts an access whenever *that channel* is idle;
- `:91-95`: `rdata` is loaded and the array written on *every* enable, busy or not.

**What:** An FSM waiting for the SRAM keeps its enable high while `busy` is
high. If the busy channel is the *other* one, the SRAM performs the access,
but the FSM hasn't advanced, so it doesn't know. Then:
- **Stray accesses:** the access is repeated every time its channel frees up.
- **Done stolen:** a read the hit-check FSM issued from HC_IDLE completes
  while that FSM is not in HC_WAIT_READ. `:96` then gives the done to the
  miss FSM, which overwrites its copy of the victim set with the *other*
  set's contents. FINISH_WRITE (`:380-382`, `:447`) writes that copy back to
  the victim set: lines of one set appear in another under their own tags,
  and the set's real lines, dirty ones included, are lost.
- **Lookup reads the wrong set:** while the hit-check FSM waits in
  HC_WAIT_READ, it isn't requesting, so the miss FSM is granted. If the
  miss FSM is in ADDRESS_READ (or FLUSH_READ), its `ren` reaches the SRAM
  every cycle, and `:91-92` reloads `rdata` with the miss FSM's set. The
  hit-check FSM latches that in its done cycle (`:178`) and evaluates the
  request against the wrong set. A load can get a false miss, or a false
  hit that returns another line's data. The typical case is the first
  lookup after a miss: the miss FSM enters ADDRESS_READ the cycle after the
  miss, the same cycle the next lookup is taken.
- **Starvation:** with strict hit priority (`:83`), a lookup and a fill keep
  each other's channel busy in turn, so `!sram_busy` is never true when the
  fill asks. A steady stream of lookups stops a fill in FINISH_WRITE (or any
  other SRAM state) indefinitely.

**Effect:** None while the LSU is blocking, because a lookup and a fill never
overlap. With hits under a miss, data is corrupted and fills stall.

**Check:** Assert `sram_ren`/`sram_wen` imply the issuing FSM advances in
the same cycle. Assert every `sram_rdone` arrives while the FSM that issued
the read is in its wait state. Test: a miss, then back-to-back loads that
hit in another set until the fill completes, with a bound on the fill time.

**Sim:** each FSM raises an enable only when it is taken, and a done goes to
the FSM that issued the read. Lookups have priority, and the fill gets the
SRAM in the cycle a lookup's read completes, so it can't be starved.

### 11. The EXP unit's readiness is ignored — Medium, Reading

**Where:** `modules/scheduler/decode2/decode_2.sv:243` (`(~need_vector_exp | 1)`, with a TODO).

**What:** Decode 2 treats the EXP unit as always ready.

**Effect:** An `expi.vi` issues into a busy EXP unit.

**Check:** Two `expi.vi` in back-to-back packets with a slow EXP unit.

**Sim:** follows the RTL for now. EXP is always ready.

### 12. One instruction per scalar EX unit, silently — Contract, Reading

**Where:** `modules/scheduler/execution_units/xbar_4x5_exec_comb.sv:36-51`, an `if / else if` priority mux per destination.

**What:** Two slots routed to the same EX unit: the first wins and the other
is dropped without any signal. EX1 handles both ALU and control ops. But the
compiler's packet checker (`atalla-functional-sim/src/misc/packet_checker.py`,
`fu_limits`) allows one `ALU.S` and one `CONTROL` per packet. So a packet the
compiler considers legal, such as `add.s` with `beq.s`, loses an instruction.

**Check:** Assert at decode 2 that no two valid scalar ops share an EX unit.
Fix either the RTL or the packet checker so they agree.

**Sim:** raises `PacketContractError`.

### 13. Register reads beyond the ports are zeroed — Contract, Reading

**Where:** `modules/scheduler/decode2/source_reg_allocator.sv:103` onward. Ports are handed out first-come; there are 4 scalar, 4 vector and 2 mask read ports.

**What:** A read that gets no port reads as 0. It is also missing from the
dependency checker's inputs, so it isn't hazard-checked. A single SDMA uses 3
scalar ports, so an SDMA plus any 2-read scalar op overflows.

**Check:** Assert that the reads a packet needs fit in the ports.

**Sim:** raises `PacketContractError`.

### 14. Extra vector or SDMA ops in a packet are dropped — Contract, Reading

**Where:** `modules/scheduler/scheduler_core.sv`:
- `:374`: lane ops beyond `LANE_ISSUE_W = 2`;
- `:356`: GSAU, where the last op wins;
- the VLSU `sched_req[sid]` and `:431` SDMA `scpad_in[sid]` writes, where the last op per scratchpad wins.

**Check:** Assert at most 2 lane ops, 1 GSAU op, 1 move-to-scalar op, and one
VLSU op and one SDMA per scratchpad, per packet.

**Sim:** raises `PacketContractError`.

### 15. A fill always starts a RAM burst — Low, Reading

**Where:** `modules/scheduler/dcache/cache_bank.sv:365`. `ram_mem_REN = 1`, with its gating (`!latched_mshr_hit && ...`) commented out.

**What:** When the missing block turns out to be in the cache already
(`latched_mshr_hit`), the fill still requests a burst it ignores.

**Effect:** Wasted memory bandwidth, and a stray burst that could overlap
the next fill's.

**Sim:** a fill whose block is already present skips the burst.

### 16. Stale comments and inverted names — Doc

| Where | Says | Is |
|---|---|---|
| `include/scheduler/atalla_isa_types.vh:13` | `PACKET_W` is 192 bits | 160 (4 × 40) |
| `execution_units/control.sv:37` | `pc_plus4` | `pc + PACKET_BYTE_W` = +20, correctly |
| `decode2/scalar_control_unit.sv:242` | "Branch ops — no reg_write" | they set `reg_write` and write rs1 (`rs1 += incr7`) |
| `decode2/dependency_checker.sv:26` | `*_halt_ready = \|table` | high while something is still *busy* |
| `scheduler_core.sv:260` | TODO: connect STM to the vector WB | the scalar-to-mask path looks connected; confirm |

### 17. A request made while the SRAM is busy is never answered — Medium, Reading

**Where:** `modules/scheduler/dcache/cache_bank.sv:165-173` (HC_IDLE only
leaves on `hc_grant && !sram_busy`); `modules/scheduler/execution_units/ld_st_unit.sv:79`
(`mem_in_valid` is high for one cycle, in `start`), `:89-130` (`latch` waits
for `hit` or `miss`). Nothing in `ld_st_unit` reads `ld_st_if.stall`.

**What:** The cache only takes a request in a cycle where the SRAM is free;
otherwise it raises `stall` and expects the request to be held. The LSU
presents it for one cycle and ignores `stall`, so the request vanishes and
the LSU waits in `latch` for an answer that never comes.

**Effect:** None today: with a blocking LSU the SRAM is always free when it
asks. With a non-blocking LSU, any load or store issued while a fill uses the
SRAM hangs EX5.

**Check:** Assert that `mem_in` with `stall` high is held until `hit` or
`miss`. Or drive one-cycle requests while a fill is in ADDRESS_WAIT.

**Sim:** the request port is a valid/ready handshake; the requester holds a
request until it is taken.

### 18. A miss with the MSHR buffer full is not recorded — Medium, Reading

**Where:** `modules/scheduler/dcache/cache_mshr_buffer.sv:28, 69, 87` (the
entry is only written on `miss && !stall`); `cache_bank.sv:211-214` (the miss
is reported regardless).

**What:** With 8 entries outstanding and the bank busy, the buffer's `stall`
is high, so a new primary miss is dropped. But the hit-check FSM still raises
`miss`, which tells the requester its fill is coming.

**Effect:** None while the LSU is blocking. With more than 8 misses
outstanding, the ninth requester waits forever.

**Check:** Nine misses to distinct blocks without waiting for fills. Assert
nine fills, or that `miss` is never raised while the buffer drops it.

**Sim:** a miss that can't get an MSHR entry is answered *retry*, and the
requester presents it again.

### 19. A merge after the fill has started is never applied — Medium, Reading

**Where:** `modules/scheduler/dcache/cache_bank.sv:302-306, 386-392` (the
bank copies the entry into `latched_mshr_entry` in START or FINISH and never
reads it again); `cache_mshr_buffer.sv:78-86` (a merge only updates the
buffer's copy, which stays until FINISH_WAIT).

**What:** A secondary miss (same block) merges into the buffer's entry,
which is correct. But once the bank has started the fill, it works only from
its own copy, `latched_mshr_entry`. A merge after that point updates the
buffer's copy, which nothing reads; it is popped and discarded at
FINISH_WAIT. The first miss is filled correctly. Only the merged request's
contribution is dropped.

**Effect:**
- A merged load still finds its word in the line, but FINISH reports the
  first request's `uuid`, so a requester waiting for its own tag never
  wakes (on top of bug 9).
- A merged store's word is not written into the line. That is only harmless
  if the requester replays the store after the fill, as today's LSU does.

**Check:** Miss on a block, wait until the bank leaves START, store-miss to
another word of the block, and don't replay it. Assert the word holds the
stored value after the fill.

**Sim:** an entry accepts new targets until its line is written, and every
target is applied in order. A miss after that gets a new entry.

### 20. Load and store addresses ignore the base register — High, Reading

**Where:** `modules/scheduler/execution_units/execute_stage.sv:150`
(`assign unit5_if.addr = post_xbar_ex5.imm;`). `ld_st_unit.sv` uses that
as the address unchanged.

**What:** `lw.s`, `sw.s`, `lhw.s` and `shw.s` address `rs1 + imm` (the ISA
document, the assembler and the functional sim all agree). Decode 2 reads
`rs1` and the D2/EX latch carries it, but nothing adds it: the address is the
12-bit immediate alone.

**Effect:** Every scalar load and store with a nonzero base register goes to
the wrong address. Only accesses within ±2 KB of address 0 through `x0` work.

**Check:** `addi.s x1, x0, 0x400` then `lw.s x2, 4(x1)`. It must read
`0x404`.

**Sim:** the address is `rs1 + imm`.

### 21. `jal` and `jalr` scale their offset by 4 — High, Reading

**Where:** `modules/scheduler/execution_units/control.sv:80` (`jal`:
`pc + (imm << 2)`) and `:94` (`jalr`: `rs1 + (imm << 2)`).

**What:** The assembler encodes `jal`'s immediate as a byte offset
(`build.py`: `imm = labels[target] - pc`), and the functional sim jumps to
`pc + imm` and `rs1 + imm`. The RTL shifts both left by 2. (Branches are
different: their encoding is a word offset, and both sides shift it.)

**Effect:** Every `jal` and `jalr` jumps four times too far.

**Check:** `jal x1, +40` from pc 0 must land on pc 40 with `x1 = 20`.

**Sim:** byte offsets, as the assembler encodes them.

### 22. A branch's increment is sign-extended — Medium, Reading

**Where:** `modules/scheduler/decode2/scalar_control_unit.sv:252`
(`incr7 = {{25{I[13]}}, I[13:7]}`).

**What:** The assembler accepts increments 0..127 (`build.py`
`parse_incr_imm7`), and the functional sim adds them unsigned. The RTL
sign-extends bit 13, so an increment of 64..127 becomes −64..−1.

**Effect:** Loops that step a counter by 64 or more count down instead.

**Check:** `beq.s x1, x2, off, 100` must add 100 to `x1`.

**Sim:** unsigned.

## Where the RTL and the functional sim disagree

These are not clearly RTL bugs: the ISA document doesn't say which is
meant. The compiler targets the functional sim, so the model follows it, and
a decision is needed either way.

| | RTL | functional sim (and the model) |
|---|---|---|
| Scalar BF16 values | in the **low** half of the register: EX3 takes `rs1_value[15:0]` and writes `{16'b0, bf_out}` (`execute_stage.sv:117-118, 124`) | fp32 bit patterns, BF16 in the **upper** half; a register holding only 16 bits is read as a raw BF16 constant (`scalar_reg_as_fp32_for_bf16_r_op`) |
| `lhw.s` | loads the halfword at the address, zero-extended (`ld_st_unit.sv:29-30`) | loads the word and shifts it up 16 |
| `shw.s` | read-modify-write of the halfword (`ld_st_unit.sv:170-177`) | stores `rs >> 16` as a whole word |
| `mod.s` with a negative operand | remainder takes the dividend's sign (`socetlib_shift_test_restore_divider.sv:81, 139-140`) | takes the divisor's sign (numpy `%`) |

## Not bugs

- **The BTB indexes on a 4-byte granule** (`PC[7:2]`) with 20-byte packets.
  It looks wrong but wastes nothing: 64 consecutive packets use all 64
  entries.
- **`li.s` (opcode 47) isn't decoded.** It's a pseudo-instruction (`lui.s`
  then `addi.s`). But the functional sim's assembler emits it as a real
  opcode 47 and runs it, and kernels use it (softmax). Code built that way
  loses those instructions on the hardware. That needs fixing in the
  assembler, not the RTL.
- **`gemm.vv` reads no `vs2`.** It's matmul-only by design; partial sums are
  accumulated with `add.vv`.

## Found outside the scheduler

| Where | What | How found |
|---|---|---|
| `tb/unit/vector/transpose_unit_tb.sv` (branch `transpose_integration`, `b1ba35ff`) | Raises `pop_req` once per column, but the unit drains the whole tile per request; the request still high when the FSM returns to IDLE starts a second, unchecked drain in 63 of 64 tests | Questa |
| `modules/vector/transpose_unit.sv` | Instantiates `sram_bank` without setting latencies, so it gets read 2 / write 4 (9 cycles a row, 8 a column). Are these intended? Comments still say "1 SRAM + 2 Clos" and "wait 2 cycles" | Questa |
| `tb/unit/vector/perf_monitor.sv` (same PR) | The transpose active-cycle counter reads registered state and misses the first cycle: reports 799 for an 800-cycle transpose | Reading |
| `include/memory/scratchpad/scpad_params.svh` | `SCPAD_SIZE_BYTES` is 1 MB per pad; the intended size is 0.5 MB (4 × 0.5 MB = 2 MB) | Reading |
| `atalla-functional-sim` `src/functional_sim.py:326` onward | `blt.s`, `bge.s`, `bgt.s`, `ble.s` compare the registers unsigned (they are stored as `& 0xFFFFFFFF`); the ISA document says signed, as the RTL does | Reading |
| `atalla-functional-sim` `src/components/scalar.py:190` | `sll.s`/`srl.s`/`sra.s` raise `OverflowError` under numpy 2 when the shift-amount register is 2³¹ or more | Seen in the model's golden tests |
| `modules/systolic_array/sysarr_MEISSA_top.sv`, `pipelined_adder_tree.sv` | The psum path is half-removed: the skew buffer and the adder tree's final psum add are commented out, but the GSAU still drives `sa_partial_en` and `sa_array_in_partials` | Reading |
