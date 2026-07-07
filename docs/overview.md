# atalla-sim Overview

## What is atalla-sim?

atalla-sim is a **cycle-accurate hardware simulator** for a heterogeneous SoC accelerator developed by Purdue SoCET. It models the full datapath of a tensor processing system, from DRAM all the way through to a systolic array compute engine, with realistic cycle-by-cycle timing for every component in between.

The simulator is gem5-inspired: all hardware components inherit from a common `Clocked` base class and execute one `tick()` per clock cycle, driven by a global event queue.

**Primary use:** measure performance of GEMM and tensor workloads — cycle counts, FLOP rates, memory bandwidth utilization, and queue depths — to identify bottlenecks and evaluate scheduling strategies.

## Architecture at a Glance

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                        DRAM (off-chip)                           │
  └──────────────────────────┬───────────────────────────────────────┘
                             │ burst transfers
  ┌──────────────────────────▼───────────────────────────────────────┐
  │                  Backend (DRAM↔Scratchpad bridge)                │
  └──────────────────────────┬───────────────────────────────────────┘
                             │ row writes / row reads
  ┌──────────────────────────▼───────────────────────────────────────┐
  │      Scratchpad  [tile 0 | tile 1]  (XOR-swizzled SRAM banks)    │
  └────────────┬─────────────────────────────────────────────────────┘
               │ read/write via Frontends
  ┌────────────▼─────────────────────────────────────────────────────┐
  │                        VectorCore                                │
  │  ┌──────────┐   ┌────────────────┐   ┌──────────────────────┐    │
  │  │   VLSU   │   │ VectorDatapath │   │        GSAU          │    │
  │  │(load/str)│   │  (ALU lanes)   │   │  (SA controller)     │    │
  │  └──────────┘   └────────────────┘   └──────────┬───────────┘    │
  │      Veggie register file ◄──────── WBBuffer ◄──┘                │
  └────────────────────────────────────────┬─────────────────────────┘
                                           │ weights + activations
  ┌────────────────────────────────────────▼─────────────────────────┐
  │               SystolicArrayTPU  (e.g. 32×32 MACs)                │
  │          grouped MAC cells, pipelined mul/add, output FIFO       │
  └──────────────────────────────────────────────────────────────────┘
```

**Key design insight:** the system is queue-centric. All inter-component state lives in `SimQueue` FIFOs. Backpressure propagates naturally when queues fill. See [sysarr_tpu_queue_glossary.md](sysarr_tpu_queue_glossary.md) for the full queue inventory.

## Core Simulation Framework (`src/base/`)

===============================================================================================
| Class         | File                | Role                                                  |
|---------------|---------------------|-------------------------------------------------------|
| `EventQueue`  | `eventq.py`         | Heap-based priority queue; the global clock           |
| `ClockDomain` | `clock_domain.py`   | Groups `Clocked` objects; fires `tick()` every period |
| `Clocked`     | `clocked_object.py` | Base class for all cycle-aware components             |
| `Sim`         | `sim.py`            | Top-level driver; calls `EventQueue.run_until()`      |
| `Core`        | `core.py`           | Owns clock domains; used for initialization           |
| `SimQueue`    | `queue.py`          | Generic fixed-capacity FIFO used everywhere           |
| `DType`       | `dtype.py`          | Numeric precision: `INT8`, `FP16`, `BF16`             |
===============================================================================================

**How time flows each cycle:**

```
Sim.run()
  └─► EventQueue pops next event (time, callback)
        └─► ClockDomain.tick(time)
              └─► for each Clocked object:
                    obj.tick(time)   ← all hardware logic lives here
              └─► schedule next tick at time + period
```

Every hardware component overrides `tick(time)`. The `_consume_tick()` guard on `Clocked` ensures a component only executes once per cycle even if ticked multiple times.

## Components

### SystolicArrayTPU (`src/systolic_array/systolic_array_tpu.py`)

A pipelined weight-stationary matrix multiply engine.

- **Topology:** `size × num_groups` array of `TPUCell4Input` cells (e.g. 32×8 for `size=32, group_size=4`)
- **Each cell** holds a `group_size`-wide dot product: activations × weights → accumulation
- **Pipeline stages:** mul (latency configurable) → add4 → add2 → output FIFO
- **Control signals:** `weight_en` (shift weights down), `mac_shift` (shift activations left), `start` (begin multiply)
- **Key interfaces:**
  - `enqueue(activations, dtype)` — push one activation row
  - `enqueue_weights(weights, dtype)` — push one weight column
  - `load_weights(matrix)` — bulk-load weight matrix
  - `psum_output_fifo_bottom` — dequeue completed output rows
- **Metrics tracked:** `mac_ops`, `mul_ops`, `add_ops`, `active_pe_sum`, `valid_mac_cycles`, arithmetic intensity

### VectorCore (`src/vector_core/vector_core.py`)

The processor that orchestrates all data movement and issues instructions to the systolic array.

===========================================================================================================================================
| Subcomponent   | Class                           | Role                                                                                 |
|----------------|---------------------------------|--------------------------------------------------------------------------------------|
| VLSU           | `VectorLoadStoreUnit`           | Translates vector load/store ops into scratchpad requests                            |
| VectorDatapath | `VectorDatapath` + `VectorLane` | Per-lane pipelined ALU (shift, mux, reduce, etc.)                                    |
| GSAU           | `GSAU`                          | Queues instructions for the systolic array; matches outputs to destination registers |
| Veggie         | `Veggie`                        | Vector register file; tracks dtype per register                                      |
| WBBuffer       | `WBBuffer`                      | Shared writeback arbitration; routes results back to Veggie                          |
===========================================================================================================================================

Instruction scheduling uses a VLIW packet model (`vliw_q`). The scheduler bundles VLSU, GSAU, and datapath operations into packets to reduce front-end overhead.

### Scratchpad (`src/memory/scratchpad.py`)

Software-managed on-chip SRAM with two independent tiles for double buffering.

- **Layout:** 2 tiles × `num_banks` × `bank_size` rows of SRAM
- **Access:** XOR-based bank swizzle (`_xor_bank`) distributes vector lanes across banks to avoid conflicts
- **Crossbar:** per-tile read and write crossbars add a configurable routing delay (`xbar_delay`)
- **Frontends:** `Frontend` objects handle per-port read/write requests from VectorCore
- **Connection:** each tile connects to one `Backend` for DRAM↔SRAM transfers

### Backend (`src/memory/backend.py`)

Handles burst transfers between DRAM and Scratchpad.

- `load(base_sp, base_dram, rows, cols)` — fetch a 2D tile from DRAM into scratchpad
- `store(base_sp, base_dram, rows, cols)` — write a tile from scratchpad back to DRAM
- Manages outstanding burst queues (`_dram_pending`, `_dram_ready_heap`) with configurable burst bandwidth

### DRAM (`src/memory/dram.py`)

Sparse, lazily-allocated main memory. Unread addresses return zero. No timing model of its own — latency is modeled in `Backend`.

## Data Flow (End-to-End)

### Load phase
```
Backend.load() → burst DRAM rows → Scratchpad.frontend_write() per row
```

### Compute phase
```
VectorCore.enqueue_memory(load_req)
  └─► VLSU issues scratchpad read → response arrives in vlsu_rsp_q
        └─► writeback to Veggie register
              └─► scheduler issues GSAU instruction
                    └─► GSAU enqueues to gsau_to_systolic
                          └─► GSAUTPUBridge feeds weights/activations to SA
                                └─► SystolicArrayTPU.tick() → psum_output_fifo_bottom
                                      └─► GSAUTPUBridge reads outputs → VectorCore.push_systolic_response()
                                            └─► WBBuffer writeback → Veggie
```

### Store phase
```
VectorCore.enqueue_memory(store_req) → VLSU → Scratchpad write → Backend.store() → DRAM
```

### Bridges

Both bridges live in `src/atalla/sysarr_tpu_system.py` and are `Clocked`:

- **`VLSFrontendBridge`** — converts VectorCore load/store requests into scratchpad frontend calls; handles uint16↔float lane encoding
- **`GSAUTPUBridge`** — pops instructions from `VectorCore.gsau.to_systolic`, feeds rows to `SystolicArrayTPU`, reads outputs and pushes them back as `push_systolic_response()`

## System Integration & Entry Points

===================================================================================================================
| Symbol                   | File                       | Purpose                                                 |
|--------------------------|----------------------------|---------------------------------------------------------|
| `build_tpu_platform()`   | `sysarr_tpu_system.py`     | Instantiate and wire all components                     |
| `SysArrTPUSystem`        | `sysarr_tpu_system.py`     | Owns `EventQueue`, `ClockDomain`, all components        |
| `TiledTPUCosim`          | `sysarr_tpu_experiment.py` | Tiled GEMM harness (tile K-loop, double-buffer preload) |
| `MNReuseBlockedTPUCosim` | `sysarr_tpu_experiment.py` | Blocked M×N weight-reuse harness                        |
===================================================================================================================

The `SysArrTPUSystem` constructor wires together:

```python
eq = EventQueue()
clk = ClockDomain(period=1.0, eq=eq)
clk.add_clocked([vc, vls_bridge, sysarr_bridge, spad, harness, ...])
sim.run()
```

## Running Tests & Experiments

```bash
# Run full test suite
pytest tests/

# Key integration tests
pytest tests/test_scratchpad_vector_core_sysarr_tpu_tiled_1024.py
pytest tests/test_scratchpad_vector_core_sysarr_tpu_tiled_1024_blocked_mn.py

# Parameter sweeps
python tools/run_sysarr_tpu_sweeps.py
python tools/run_blocked_mn_reuse_sweeps.py

# Visualization
python tools/plot_sysarr_tpu_sweeps.py
python tools/plot_blocked_mn_reuse_roofline.py
python tools/plot_tiled_sysarr_tpu_gantt.py
```

Logs are written to `logs/`: `stats.log`, `schedule.log`, `gemm_cycles.log`, `gantt.log`.

## Key Performance Findings (32×32 array, 1024×1024 GEMM)

- **Single tile GEMM** (32×32): ~1092 cycles; bottleneck is `gsau_rd_queue` (result retirement), not scheduling
- **Tiled 1024×1024**: ~21M cycles; bottleneck is **preload latency** (~804 cyc/tile) vs. compute (~441 cyc/tile) — the array starves even with double buffering
- **Reuse policies** (blocked M×N): weight-stationary reuse improves arithmetic intensity; best configs expose 900+ ready GEMMs once weights are resident
- **MAC utilization:** ~42–44% on average for a single-tile GEMM

## Where to Go Next

| Document | What it covers |
|---|---|
| [sysarr_tpu_queue_glossary.md](sysarr_tpu_queue_glossary.md) | Every queue/FIFO in the system with payload definitions |
| [sysarr_tpu_system_uml.md](sysarr_tpu_system_uml.md) | Mermaid runtime flowchart and class diagram |
| [understanding-core-classes.md](understanding-core-classes.md) | Clock/tick/cycle model in detail |
| [harness_at_test_scratchpad_vector_core_sysarr_tpu_tiled_1024.md](harness_at_test_scratchpad_vector_core_sysarr_tpu_tiled_1024.md) | Full tiled harness walkthrough |
| [blocked_mn_reuse_roofline_sweeps.md](blocked_mn_reuse_roofline_sweeps.md) | Reuse policy comparison and roofline methodology |
| [metrics_at_test_scratchpad_vector_core_sysarr_tpu.md](metrics_at_test_scratchpad_vector_core_sysarr_tpu.md) | Single-tile GEMM metric walkthrough |
| [smart_tile_scheduling.ipynb](smart_tile_scheduling.ipynb) | Interactive scheduling model and bandwidth sweep |
