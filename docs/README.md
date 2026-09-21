# Atalla-Sim Documentation

```
   docs/
   ├── architecture/     how the simulator is built
   │   ├── base-classes.md    clocking, ticking, scheduling, composition
   │   ├── platform-uml.md    runtime dataflow and class views
   │   ├── queue-glossary.md  every FIFO, and what it carries
   │   ├── vector-core.md    the vector core: shape, cycle, API
   │   ├── vector-core-old-vs-new.md  what changed, and why
   │   ├── transpose.md       the transpose unit: banks, rotation, cost
   │   ├── scratchpad-pads.md 4 pads, 4 VLSUs, and DRAM bandwidth
   │   ├── meissa.md          the MEISSA systolic array
   │   └── gem5-roadmap.md    what else is worth taking from gem5
   │
   ├── harnesses/        how experiments drive the platform
   │   ├── tiled-1024.md      the 1024x1024 tiled GEMM harness
   │   └── arch-comparison.md old vs new architecture, and how to run it
   │
   ├── results/          measurements from specific runs
   │   ├── gemm-32.md              single 32x32 tile, end to end
   │   ├── gemm-1024-tiled.md      1024x1024 tiled, with Gantt
   │   ├── arch-comparison.md      old vs new architecture, 1024x1024
   │   └── presentation-graphs.md  figure-by-figure notes
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

## Start here

| If you want to… | Read |
|---|---|
| understand the tick/cycle model | [architecture/base-classes.md](architecture/base-classes.md) |
| add a new component | [architecture/base-classes.md §8](architecture/base-classes.md#8-writing-a-component) |
| trace data through the platform | [architecture/platform-uml.md](architecture/platform-uml.md) |
| know what a queue in the stats means | [architecture/queue-glossary.md](architecture/queue-glossary.md) |
| run or modify the big GEMM | [harnesses/tiled-1024.md](harnesses/tiled-1024.md) |
| compare the two architectures | [harnesses/arch-comparison.md](harnesses/arch-comparison.md) |
| see the comparison results | [results/arch-comparison.md](results/arch-comparison.md) |
| interpret a stats report | [results/gemm-32.md](results/gemm-32.md) |
| run a parameter sweep | [sweeps/sysarr-tpu.md](sweeps/sysarr-tpu.md) |
| use the vector core | [architecture/vector-core.md](architecture/vector-core.md) |
| transpose a tile | [architecture/transpose.md](architecture/transpose.md) |
| know how memory is laid out | [architecture/scratchpad-pads.md](architecture/scratchpad-pads.md) |
| use the MEISSA array | [architecture/meissa.md](architecture/meissa.md) |
| decide what to build next | [architecture/gem5-roadmap.md](architecture/gem5-roadmap.md) |

## The model in one picture

```
        DRAM
          │ bursts
        Backend
          │
        Scratchpad     frontends ──► crossbars ──► SRAM banks
        2 MB           4 pads of 0.5 MB, one per VLSU
          ▲ store   │ load
          │         ▼
   ┌──────┴───────────────────────────────────────────────────────────────────┐
   │ Vector Core                                                              │
   │                                                                          │
   │   scheduler ─ one VLIW packet: gsau 1 | vlsu 4 | transpose 1 | datapath 2│
   │      │            │                   │                    │             │
   │      ▼ vlsu       ▼ datapath          ▼ transpose          ▼ gsau        │
   │    VLSU x4      VectorDatapath      TransposeUnit        GSAU            │
   │  1/scratchpad   slicer ─► lanes     Clos xbar +            │             │
   │      │          ResultCollector      32 SRAM banks         │             │
   │      │            │                   │                    │             │
   │      └────────────┴───────────────────┘                    │             │
   │                   ▼                                        │             │
   │    Veggie (VRF, 4 banks) ─► OpBuffer ─► operands           │             │
   │              WBBuffer ◄───┘  stages writeback              │             │
   └────────────────────────────────────────────────────────────┼─────────────┘
                                                                ▼
                                                       SystolicArrayTPU
                                                       grouped MAC cells
```

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
