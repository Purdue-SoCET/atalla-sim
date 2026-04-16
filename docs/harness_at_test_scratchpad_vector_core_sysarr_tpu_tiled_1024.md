# Tiled TPU Harness Guide

This note explains how
[test_scratchpad_vector_core_sysarr_tpu_tiled_1024.py](/home/asicfab/a/socet149/atalla-sim/tests/atalla/test_scratchpad_vector_core_sysarr_tpu_tiled_1024.py)
works.

## Purpose

The harness runs a full tiled GEMM:

- matrix size: `1024 x 1024`
- tile size: `32 x 32`
- output tiles: `32 x 32 = 1024`
- tile GEMMs: `32 x 32 x 32 = 32768`

Each tile GEMM is one `32 x 32 x 32` kernel identified by `(ti, tj, tk)`.

## Reused TPU System Pieces

The harness does not build the TPU platform from scratch anymore.

It reuses shared helpers from
[sysarr_tpu_system.py](/home/asicfab/a/socet149/atalla-sim/src/atalla/sysarr_tpu_system.py):

- `build_tpu_platform(...)`
  - creates the shared VC/SPAD/DRAM/backend/VLS platform
- `build_tpu_compute_path(...)`
  - creates a fresh TPU + GSAU bridge for one kernel stream

It also reuses pieces from
[sysarr_tpu_experiment.py](/home/asicfab/a/socet149/atalla-sim/src/atalla/sysarr_tpu_experiment.py):

- `MetricsVLSFrontendBridge`
- `QUEUE_NAMES`

So the tiled flow shares the same machine blocks and tracing style as the single-tile TPU experiment path.

For a high-level class view of the assembled simulator stack, see
[sysarr_tpu_system_uml.md](sysarr_tpu_system_uml.md).

## Dataflow

Modeled path:

`dram -> backend -> scratchpad -> vlsu -> veggie -> gsau -> tpu -> gsau -> vector datapath(add) -> vlsu -> scratchpad accumulator slot -> backend -> dram`

Per output tile `(ti, tj)`:

1. Clear one accumulator tile in scratchpad.
2. Walk `tk = 0..31`.
3. For each `tk`, stage:
   - one `A[ti, tk]` tile
   - one `B[tk, tj]` tile
4. Run one TPU kernel for that tile pair.
5. Accumulate the partial result into the scratchpad accumulator tile.
6. After all `tk` finish, drain the accumulator tile to DRAM.

## Double Buffering

Two preload slots are used:

- slot `0`
- slot `1`

Each slot has:

- one activation tile staging region
- one weight tile staging region

While the current tile GEMM is computing, the next tile pair can already be moving from DRAM to scratchpad through the backend.

Only one tile GEMM computes at a time, but preload of the next job overlaps with current compute.

## What Runs in the TPU

Within one tile GEMM:

1. VC issues VLS loads for weight rows from scratchpad.
2. Returned weight vectors are issued into GSAU as TPU weight commands.
3. VC issues VLS loads for activation rows.
4. Returned activation vectors are issued into GSAU as TPU activation commands.
5. GSAU streams into TPU.
6. TPU produces one partial output row at a time.
7. GSAU returns those rows back to VC.

## Where Accumulation Happens

Accumulation is done in the vector core datapath, not in Python.

For each returned partial output row:

1. Load the current accumulator row from scratchpad into `ACC_REG`.
2. Take the TPU partial row from the GSAU response path.
3. Issue a vector `add` in the VC datapath into `SUM_REG`.
4. Store the summed row back into the scratchpad accumulator tile.

That means the final `C[ti, tj]` tile is built up by VC-side vector adds across all `tk`.

## Why Scratchpad Accumulator Rows Exist

The TPU kernel computes one partial tile for one `tk`.

The full output tile needs:

`C[ti, tj] = sum over tk of A[ti, tk] * B[tk, tj]`

So scratchpad holds the running tile accumulator between successive TPU kernels.

## Kernel Tags and Gantt Paths

Every tile GEMM gets a stable tag:

- `tiXX_tjXX_tkXX`

That tag is attached to traced subpaths such as:

- `sdma_act`
- `sdma_wgt`
- `vls_wgt`
- `gsau_wgt`
- `vls_act`
- `gsau_act`
- `systolic_array`
- `gsau_rsp`
- `vls_acc_load`
- `datapath_add`
- `vls_acc_store`
- `kernel_total`
- `prefetch_window`
- `compute_window`

These paths are written to:

- [gantt.log](/home/asicfab/a/socet149/atalla-sim/logs/sysarr_gemm_tpu_tiled_1024/gantt.log)

## Logs

The harness writes debug-style logs with `base/debug.py`:

- `stats.log`
- `schedule.log`
- `gemm_cycles.log`
- `sdma_load_cycles.log`
- `gantt.log`
- `harness.log`

`harness.log` is the shortest high-level explanation of the run.

## Plotting

Use:

```bash
/home/asicfab/a/socet149/sc_env_new/bin/python \
/home/asicfab/a/socet149/atalla-sim/tools/plot_tiled_sysarr_tpu_gantt.py \
  --input /home/asicfab/a/socet149/atalla-sim/logs/sysarr_gemm_tpu_tiled_1024/gantt.log \
  --ti 0 --tj 0 --include-envelopes
```

That reads `gantt.log` directly and renders one output-tile Gantt.

For the slide-oriented presentation plots, use:

```bash
/home/asicfab/a/socet149/sc_env_new/bin/python \
/home/asicfab/a/socet149/atalla-sim/tools/plot_tiled_sysarr_tpu_gantt.py \
  --input /home/asicfab/a/socet149/atalla-sim/logs/tiled_1024_m8_n32/gantt.log \
  --tj 0 --tk 0 --presentation-set
```

That writes the block overview, one-weight flow, compute-detail, and reuse
balance PNGs next to the input log.

See
[presentation_graphs_at_test_scratchpad_vector_core_sysarr_tpu_tiled_1024_blocked_mn.md](presentation_graphs_at_test_scratchpad_vector_core_sysarr_tpu_tiled_1024_blocked_mn.md)
for notes on what each presentation graph is meant to communicate.

## What Is Still Simplified

This harness still makes a few simplifying choices:

- one tile GEMM computes at a time
- accumulator drain to DRAM is done after a full output tile completes
- tile scheduling is deterministic and single-stream over `(ti, tj)`

Those are fine for studying overlap, queue pressure, and kernel timing, but they are still a harness-level execution policy, not a full program/runtime model.
