# Where atalla-sim differs from the RTL and the functional simulator

atalla-sim models what the hardware is meant to do. Where the RTL has a bug,
the model does the intended thing and the difference is listed here; where
the model is built differently from the chip diagram, that is listed too.
Read this before comparing atalla-sim's cycles or values with an RTL
simulation or with the functional simulator. `atalla-sim --version` names the
RTL branch each part follows.

## Scheduler: RTL bugs the model doesn't reproduce

Details and the RTL lines: [architecture/scheduler.md, *Where the model
differs from the RTL*](architecture/scheduler.md#where-the-model-differs-from-the-rtl).

- A flush clears all of the D1/D2 latch. The RTL clears only its scalar
  slots, so wrong-path vector and SDMA ops issue.
- Only a packet that issues reserves registers. In the RTL, a packet in
  decode 2 on a redirect reserves its registers, is squashed, and nothing
  clears them.
- `lw.vi` writes no register. The RTL marks its `vd` busy forever.
- `sqrt.bf` waits for EX2. The RTL's structural check forgets it.
- Every mask write is tracked. The RTL tracks vector slots 0–1 only, and
  never WAW-checks `mv.stm`.

## Instruction semantics

Where the RTL, the functional simulator and the ISA document disagree, the
model follows the ISA document and the assembler, then the functional
simulator (`src/scheduler/semantics.py`):

| | model (ISA document) | RTL | functional sim |
|---|---|---|---|
| load/store address | `rs1 + imm` | the immediate alone | as the model |
| `blt` / `bge` / `bgt` / `ble` | signed | | unsigned |
| branch increment `[13:7]` | unsigned 0..127 | sign-extended | |
| branch / `jal` offset | `pc + imm * 5` (instructions are 5 bytes) | shifted by 2 (4-byte steps) | |
| `jalr` target | `rs1 + imm`, in bytes | `imm` shifted left by 2 | |
| scalar BF16 | fp32 bit pattern, BF16 in the upper half; `lhw.s` / `shw.s` move the upper half of a word | BF16 in the lower half; halfword loads and stores | as the model |
| `mod.s` sign | the divisor's | the dividend's | as the model |
| divide by zero | `rcp.bf` gives ±infinity; `div.s` / `mod.s` follow RISC-V | not checked | raises |

Blank: not established, or the same as the model.

## Memory system

- **AXI bus** (`src/memory/axi.py`, memory_FA26): the model takes one AR or W
  beat whenever it can. It leaves out two RTL behaviours: the wrapper's
  registered ARREADY / WREADY with accept-on-valid (a refused AR can be
  popped, W beats taken twice), and its 256-bit beats packed onto the 64-bit
  bus. B returns once the wrapper has taken the beats, as in the RTL.
- **DRAM** is a fixed latency unless you choose Ramulator 2
  (`--dram ramulator` or `--config atalla-0x01-ddr4`). There is no DDR4
  controller model besides Ramulator's.

## How the model is built compared with the chip diagram

These are design choices, not bugs. The activity view draws the model's
version.

- **Vector lanes:** 16 lanes of 2 elements by default (`lane_count`), not 32.
- **Transpose unit:** a VLIW functional unit, a peer of the VLSUs, the GSAU
  and the reduction tree. It reads rows from and writes columns to the vector
  register file. Its timing is `transpose_unit.sv`'s: 9 cycles a row pushed,
  8 a column popped.
- **Scratchpads:** four pads, one per VLSU, not two.
- **Scalar execute:** multiply (EX4) and divide (EX2) are separate units, and
  int↔BF16 conversion shares EX2 with divide and square root.
- **MEISSA**, the default systolic array, has no partial-sum input FIFO.

## Compared with the functional simulator

`--golden` and `Kernel.run(golden=True)` run the functional simulator on the
same image. Known differences:

- **DMAs finish instantly** in the functional simulator. A program that
  reads a pad row before its `scpad.ld` lands is still right there and can
  be wrong here. `--check` reports such dependencies, and `--golden` shows
  the first register that got the stale value.
- **Reduction order.** Sums are compared in the lane model's order by
  default; `--functional-reductions` uses the functional simulator's own
  sequential fp32 sum, which can differ in the last bits.
- **Masked compares** (`mgt.mvv` and friends under a partial mask): the
  model writes 0 for masked-off lanes. The functional simulator can leave
  other bits set, e.g. `0xff0000f1` where the model gives `0x000000f1`
  (maxpool.c). Final DRAM still matches.
- **Misaligned words.** The functional simulator keys memory by exact
  address, so a 32-bit access at an address that isn't a multiple of 4 gets
  its own entry. Here, as in the RTL's word path, it is aligned down and
  overlaps the word below. `--check` reports misaligned accesses.
