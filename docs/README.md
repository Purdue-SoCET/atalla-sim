# Atalla-Sim Documentation

```
   docs/
   ├── architecture/     how the simulator is built
   │   ├── base-classes.md    clocking, ticking, scheduling, composition
   │   ├── platform-uml.md    runtime dataflow and class views
   │   └── queue-glossary.md  every FIFO, and what it carries
   │
   ├── harnesses/        how experiments drive the platform
   │   └── tiled-1024.md      the 1024x1024 tiled GEMM harness
   │
   ├── results/          measurements from specific runs
   │   ├── gemm-32.md              single 32x32 tile, end to end
   │   ├── gemm-1024-tiled.md      1024x1024 tiled, with Gantt
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
| interpret a stats report | [results/gemm-32.md](results/gemm-32.md) |
| run a parameter sweep | [sweeps/sysarr-tpu.md](sweeps/sysarr-tpu.md) |

## The model in one picture

```
   DRAM ──► Backend ──► Scratchpad ──► VLSU ──► Veggie (VRF)
                          ▲  banks                  │
                          │  xbars                  ▼
                          │  frontends            GSAU
                          │                         │
                          │                         ▼
                          └────────────  SystolicArrayTPU
                                          (grouped MAC array)
```

Every box is a `Clocked` object advanced once per simulated cycle. Boxes are
connected only by `SimQueue` FIFOs, so backpressure propagates on its own
rather than being modeled explicitly.

## Conventions

- **A tick is a cycle.** `ClockDomain(period=1.0)` means time `t` is cycle `t`.
- **Order within a cycle matters.** Components tick in a declared phase order;
  a consumer sees a producer's output in the same cycle only if it ticks later.
- **Idle components are skipped**, and that must never change a result. Run
  `ATALLA_SCHED=legacy` to tick everything unconditionally and compare.
