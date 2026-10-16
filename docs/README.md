# Atalla-Sim Documentation

```
   docs/
   ├── getting-started.md  install, run, read the report, explore a design
   ├── architecture/     what is modelled, and how to change it
   │   ├── scheduler.md       the scheduler, running programs, kernel status
   │   ├── vector-core.md     the vector core: shape, cycle, API
   │   ├── meissa.md          the MEISSA systolic array
   │   ├── transpose.md       the transpose unit: banks, rotation, cost
   │   ├── scratchpad-pads.md 4 pads, 4 VLSUs, and DRAM bandwidth
   │   ├── queue-glossary.md  every FIFO, and what it carries
   │   ├── platform-uml.md    runtime dataflow and class views
   │   └── base-classes.md    clocking, ticking, writing a component
   │
   ├── harnesses/        how experiments drive the platform
   │   ├── tiled-1024.md      the 1024x1024 tiled GEMM harness
   │   └── arch-comparison.md old vs new architecture, and how to run it
   │
   ├── results/          measurements from specific runs
   │   ├── gemm-32.md              single 32x32 tile, end to end
   │   ├── gemm-1024-tiled.md      1024x1024 tiled, with Gantt
   │   └── arch-comparison.md      old vs new architecture, 1024x1024
   │
   ├── sweeps/           parameter studies
   │   ├── sysarr-tpu.md           tile / DRAM latency / queue depth
   │   └── blocked-mn-roofline.md  reuse policy roofline
   │
   ├── notebooks/        interactive models
   │   └── smart_tile_scheduling.ipynb  (+ smart_tile_scheduling_outputs/)
   │
   └── animations/       generated GIFs
```

New here? Start with [getting-started.md](getting-started.md): install,
run a program, read the report, explore a design.

## Start here

| If you want to… | Read |
|---|---|
| install, run a first program, read its report | [getting-started.md](getting-started.md) |
| run a C or assembly program, and read its report | [architecture/scheduler.md](architecture/scheduler.md) |
| change the machine's parameters (lanes, caches, DRAM, array) | [architecture/scheduler.md §Platform](architecture/scheduler.md#platform) |
| know which kernels give correct results | [architecture/scheduler.md §Kernels](architecture/scheduler.md#kernels) |
| understand the tick/cycle model | [architecture/base-classes.md](architecture/base-classes.md) |
| add or change a component | [architecture/base-classes.md §8](architecture/base-classes.md#8-writing-a-component) |
| trace data through the platform | [architecture/platform-uml.md](architecture/platform-uml.md) |
| know what a queue in the stats means | [architecture/queue-glossary.md](architecture/queue-glossary.md) |
| run or modify the big GEMM | [harnesses/tiled-1024.md](harnesses/tiled-1024.md) |
| compare the two architectures | [harnesses/arch-comparison.md](harnesses/arch-comparison.md) |
| see the comparison results | [results/arch-comparison.md](results/arch-comparison.md) |
| interpret a harness stats report | [results/gemm-32.md](results/gemm-32.md) |
| run a parameter sweep | [sweeps/sysarr-tpu.md](sweeps/sysarr-tpu.md) |
| use the vector core | [architecture/vector-core.md](architecture/vector-core.md) |
| transpose a tile | [architecture/transpose.md](architecture/transpose.md) |
| know how memory is laid out | [architecture/scratchpad-pads.md](architecture/scratchpad-pads.md) |
| use the MEISSA array | [architecture/meissa.md](architecture/meissa.md) |

## The model in one picture

```
   C program ── atalla_cc -p ──► packets ── assembler ──┐      assembly kernel (.in)
                                                        ▼              │
   ┌───────────────────────────────────────────────────────────────────▼───────┐
   │ Scheduler: fetch ─► icache ─► decode ─► scoreboard/issue ─► EX1–EX5 ─► WB │
   │               (one 4-slot VLIW packet a cycle)          │       │         │
   │                                                         │     dcache      │
   └─────────────────────────────────────────────────────────┼───────┼─────────┘
                       vector and DMA dispatch               │       │
   ┌─────────────────────────────────────────────────────────▼───┐   │
   │ Vector core:  lanes (16)  │  VLSU x4  │  GSAU  │  transpose │   │
   │               register file (Veggie)                        │   │
   └──────────────────────┬──────────────┬───────────┬───────────┘   │
                          │ vreg.ld/st   │ gemm.vv   │               │
                          ▼              ▼ lw.vi     │               │
          Scratchpad: 4 pads of 0.5 MB   Systolic array (MEISSA,     │
          one per VLSU; DMA backends     or the TPU-style array)     │
                          │ scpad.ld/st                              │
                          ▼                                          ▼
                        DRAM: one shared burst channel (icache, dcache, 4 DMA backends)
```

The scheduler-driven platform is `src/atalla/atalla_platform.py`; the harness
platforms (`src/atalla/sysarr_tpu_system.py`) drive the same vector core,
scratchpad and array from a Python harness instead of a program.

The vector core has **two** compute paths, not one. It runs elementwise SIMD and
reductions on its own lanes, *and* feeds the systolic array through the GSAU.
These are separate slots of the same VLIW packet, so they can be occupied in the
same cycle — they are not stages of one pipeline. The transpose unit is a third
slot beside them, turning a tile of rows into a tile of columns
([architecture/transpose.md](architecture/transpose.md)).
[architecture/vector-core.md](architecture/vector-core.md) has the detail.

Every box is a `Clocked` object advanced once per simulated cycle. Boxes are
connected only by `SimQueue` FIFOs, so backpressure propagates on its own
rather than being modeled explicitly.

## Conventions

- **A tick is a cycle.** `ClockDomain(period=1.0)` means time `t` is cycle `t`.
- **Order within a cycle matters.** Components tick in a declared phase order;
  a consumer sees a producer's output in the same cycle only if it ticks later.
- **Idle components are skipped**, and that must never change a result. Run
  `ATALLA_SCHED=legacy` to tick everything unconditionally and compare.
