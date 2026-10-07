# atalla-sim modelling rules

Decisions made for atalla-sim while modelling the Atalla scheduler and
putting the whole machine together. Read this before changing how the
simulator models the hardware. Each rule says what to do and why.

## What atalla-sim is

- **It models the RTL's intended behaviour, never its bugs.** Where the RTL
  is wrong, model what was meant, say so in the module's docstring, and
  record the bug in `docs/scheduler-rtl-bugs.md`. Don't add switches that
  reproduce a bug. First make sure it really is a bug: some odd-looking RTL
  is harmless (the BTB's 4-byte granule, for example).
- **The functional simulator is the golden reference for values**
  (`third_party/atalla-functional-sim`, through `scheduler/golden.py`).
  Kernels are checked against it register for register, and DRAM halfword
  for halfword.

## The RTL

- **Read the RTL; don't change it.** Never modify, fix or commit to the
  atalla RTL repository, and don't run its testbenches unless asked to.
  Bugs found by reading go in `docs/scheduler-rtl-bugs.md`, which feeds the
  team's SystemVerilog verification.
- **The scheduler reference** is atalla `scheduler_integration_SP26_joshklug`.
- **Outside the scheduler, atalla-sim's own models are the reference:** the
  vector core, lanes, VLSUs, GSAU, transpose unit, scratchpad, DMA
  backends, DRAM and systolic arrays. Keep them as they are. Where the
  scheduler branch's non-scheduler RTL disagrees with them, that is the
  branch's problem, not atalla-sim's. Notes on it can go in the bug note,
  but the models don't follow it.

## The scheduler and the units

- **The scheduler drives the unit models; it doesn't re-implement them.** It
  sends control the way the RTL sends signals: function calls that enqueue
  jobs into the units (`vc.datapath.enqueue`, `vc.gsau.issue`, the VLSUs'
  `issue`, the backends' `driver_to_backend_start_load/store`, the transpose
  unit's push and pop). It reads their readiness and takes their results
  back through writeback. It routes and arbitrates; it doesn't keep their
  timing.
- **The units compute the values.** The lanes do the element ops and
  reductions, and the systolic array does `gemm.vv`. The scheduler only
  formats results for writeback (packing a compare's lanes into a mask,
  merging masked-off elements).
- **Changes to the unit models are limited to hooks:** the calls and
  adapters the scheduler needs (`VectorCore.tick_units`, extra lane ops,
  `reduce_index`, `dram_stride`, `ArrayValueBridge`), not their timing or
  arithmetic.

## The platform

- **`AtallaPlatform`** (`src/atalla/atalla_platform.py`) is `TPUPlatform`
  rebuilt with the scheduler in front: the same parts, arranged as
  `system.sv` arranges them. The systolic array is MEISSA by default; `tpu`
  is selectable. (`TPUPlatform`'s name is historical: its array may be MEISSA
  too.)
- **One DRAM, shared.** The icache, the dcache and the four scratchpad
  backends are masters on the same DRAM channel and contend for the same
  bandwidth. The caches issue AXI-style burst requests like the scratchpad
  backends (`memory/bus.py`). The icache starts cold.
- **The data cache is a real lockup-free cache with MSHRs**, configurable
  (`DCacheConfig`), and EX5 is a non-blocking load/store unit that uses it.
- **The performance monitor gathers metrics for workload and system
  analysis**: where cycles go, how busy each unit is, memory traffic and
  contention. It isn't a copy of the RTL's counters.

## ISA decisions

- **`lw.vi` shifts its vector into weight column 0**, moving every column
  one to the right, as the systolic array is built. Kernels load weight rows
  last to first, so column j ends up holding `w_j`. This was a change to the
  ISA and the kernels, not to the modelled hardware.
- **`li.s` is a pseudo-instruction.** The assemblers expand it (`addi.s`, or
  `lui.s` then `addi.s`); the hardware never sees it.
- **The transpose unit is a VLIW functional unit**, a peer of the VLSUs, the
  GSAU and the reduction tree, reading rows from and writing columns to the
  vector register file. The atalla PR that puts it inside VLSU port 0 is
  wrong: don't model it, compare against it or document it. Its timing is
  `transpose_unit.sv`'s (9 cycles a row pushed, 8 a column popped). Its
  instructions:

  | mnemonic | type | name | unit | semantics | next PC | description | opcode |
  |---|---|---|---|---|---|---|---|
  | `tpop.vi` | vector immediate | vector lane transposition pop | vector core (transpose unit) | `vs1 <= transpose_unit` | PC + 4 | push the transposed vector into the Veggie | `1001110` |
  | `tpus.vi` | vector immediate | vector lane transposition push | vector core (transpose unit) | `transpose_unit <= vs1` | PC + 4 | push the vector from the Veggie into the transpose | `1001111` |

## Toolchain

- **C kernels** come from the `aihw-ppci-compiler` submodule
  (`third_party/aihw-ppci-compiler`, branch `atalla-models`, which has the
  kernels) and are packetized by the functional sim's `build_compiler.py`.
- **The packetizer must produce packets the hardware's scheduler can run:**
  one instruction per scalar EX unit, 4/4/2 scalar/vector/mask register
  read ports, one SDMA (`HW_PACKET_RESOURCES` in `build_compiler.py`).
  The scheduler model flags any packet that breaks these
  (`decode2.violations`), because the RTL would silently drop part of it.
- **Kernels follow the ISA decisions above** whether written in assembly or
  C: weight rows are loaded last to first.

## Timing

- **Unit models match the RTL cycle for cycle where that has been
  established** (the transpose unit against a Questa trace, the scratchpad's
  SRAM latencies). Hand counts and document figures have been wrong before;
  a trace settles it.
- **Model what the RTL's control does, not a guess at it.** For example, a
  `gemm.vv` result reaches the GSAU 46 cycles after dispatch, per
  `sysarr_MEISSA_top.sv`'s shift register.

## Working on the repository

- **Work in the main `atalla-sim` tree**, not `atalla-sim-sram` or other
  checkouts.
- **Never commit `.claude/`.**
- **Pushes need the user's SSH key.** Give them the `git push` command to run.
- **Commits carry no co-author trailer.**
