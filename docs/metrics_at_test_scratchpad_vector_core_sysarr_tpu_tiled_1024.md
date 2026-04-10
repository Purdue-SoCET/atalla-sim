# Tiled TPU GEMM Stats Report

This note explains the metrics emitted by `stats.log` for
`tests/atalla/test_scratchpad_vector_core_sysarr_tpu_tiled_1024.py`.

It is the tiled counterpart to `metrics_at_test_scratchpad_vector_core_sysarr_tpu.md`.

Current modeled path:

`dram -> backend -> scratchpad -> vlsu -> veggie -> gsau -> tpu -> gsau -> veggie datapath(add) -> vlsu -> scratchpad accumulator -> backend -> dram`

The numbers below match the current 1024 x 1024 tiled run in
`logs/sysarr_gemm_tpu_tiled_1024/`.

## Tagged Microkernel Gantt

The tiled harness now assigns every `32 x 32` tile GEMM a stable tag:

- `tiXX_tjXX`

That tag is attached to the per-kernel traffic the harness launches through:

- SDMA tile staging
- VLS activation / weight / accumulator traffic
- GSAU ingress and GSAU response egress
- systolic-array compute
- vector-datapath accumulation adds

The run writes one debug log with the envelope span of each tagged path:

- `logs/sysarr_gemm_tpu_tiled_1024/gantt.log`

Log columns:

- `tag, ti, tj, tk, slot`
- `path`
- `start_cycle, end_cycle, duration_cycles`
- `touches`

For poster or debug visualization, plot one output tile at a time:

```bash
/home/asicfab/a/socet149/sc_env_new/bin/python \
/home/asicfab/a/socet149/atalla-sim/tools/plot_tiled_sysarr_tpu_gantt.py \
  --input logs/sysarr_gemm_tpu_tiled_1024/gantt.log \
  --ti 0 --tj 0 --include-envelopes
```

That emits a row-per-microkernel Gantt where each row is one `tk` job and each
color is one path. It makes three things visually obvious:

- how much SDMA preload overlaps across the two slots
- where the systolic array is busy versus starved waiting for the next tagged job
- how much accumulation traffic extends beyond the core systolic compute window

## Workload Shape

- Matrix size is `1024 x 1024`.
- Tile size is `32 x 32`.
- `num_tiles = 32` along each dimension.
- There are `1024` output tiles total because `32 x 32 = 1024` tile positions.
- There are `32768` tile-pair GEMMs total because each output tile accumulates across `32` values of `tk`, so `32 x 32 x 32 = 32768`.
- There are `1,048,576` final scalar outputs because `1024^2 = 1,048,576`.

- A row in `gemm_cycles.log` is one tile GEMM, indexed by `(ti, tj, tk)`.
- A completed output tile is one final `32 x 32` block of `C`.
- The whole matrix contains `1,048,576` scalar dot products, but the scheduler operates on `32768` tile GEMMs.

## Tiled Schedule

The outer schedule is row-major over output tiles:

- `ti` is the outer loop.
- `tj` is the inner loop.
- One output tile `C[ti, tj]` is fully completed before `C[ti, tj + 1]` begins.

Inside one output tile:

1. The accumulator tile in scratchpad is cleared.
2. Two prefetch slots are used for double buffering over `tk`.
3. The scheduler launches the next available `tk` into any free slot.
4. Only one `tk` job computes at a time.
5. When multiple jobs are ready, the smallest ready `tk` is chosen.
6. A slot is not reusable until its current `tk` job completes.
7. After the job completes, that freed slot immediately launches the next `tk`.
8. After all `tk = 0..31` jobs finish, the accumulated output tile is drained to DRAM.

The accumulation itself is done inside the vector core

1. TPU produces one partial output row through GSAU.
2. The current accumulator row is loaded from scratchpad into `ACC_REG`.
3. The vector datapath issues `add(partial_row, accumulator_row)` into `SUM_REG`.
4. The summed row is stored back into the scratchpad accumulator tile.
5. After all `32` `tk` contributions arrive, that accumulator tile is the final `C[ti, tj]` tile.

For a focused explanation of the full tiled harness, see:

- [harness_at_test_scratchpad_vector_core_sysarr_tpu_tiled_1024.md](/home/asicfab/a/socet149/atalla-sim/docs/harness_at_test_scratchpad_vector_core_sysarr_tpu_tiled_1024.md)

Using the first output tile in `schedule.log` to make the pattern explicit:

- cycle `0`: launch `tk=0` into slot `0` and `tk=1` into slot `1`
- cycle `807`: `tk=0` becomes preload-ready and starts compute
- cycle `813`: `tk=1` becomes preload-ready while `tk=0` is computing
- cycle `1247`: `tk=0` completes, slot `0` is reused for `tk=2`, and `tk=1` starts compute
- cycle `1687`: `tk=1` completes
- cycle `2056`: `tk=2` becomes preload-ready and starts compute

`1687 -> 2056` gap: the reused slot is not ready yet, so compute stalls waiting for preload

## Summary

- End-to-end matrix latency is `21,072,896` cycles.
- Every output tile takes exactly `20,579` cycles in this deterministic run.
- Every tile GEMM takes about `441.4` cycles on average.
- Every SDMA preload transaction takes about `803.6` cycles on average.
- Every tile GEMM opens a systolic-array compute window of `76` cycles, with `75` valid-MAC cycles.
- Correctness is clean in this run:
  - `fp16_saturation_count = 0`
  - `fp16_overflow_count = 0`
  - `max_abs_error = 0.0`
  - `mean_abs_error = 0.0`
- Peak instantaneous PE activity is still below full mesh occupancy:
  - `max_active_pes_in_any_cycle = 864`

## Cycle Metrics

### Whole Matrix

- `matrix_total_cycles = 21072896`
  - Full run latency across all `1024` output tiles.
  - Includes preload warmup, compute, accumulation, inter-job bubbles, and final tile drains.

### Tile GEMM Timing

- `gemm_total_cycles_summary = {'count': 32768, 'total': 14464000, 'min': 440, 'max': 443, 'avg': 441.40625}`
  - One counted GEMM means one `(ti, tj, tk)` tile-pair job.
  - The timer starts at `compute start` and ends after `_drain_compute_path()` for that job.
  - This does not include time spent waiting for the next preload to finish before the job can start.

- `32` tile GEMMs per output tile -- the measured tile-GEMM work sums to:
  - `32 * 441.40625 = 14125` cycles per output tile on average.

### SDMA Preload Timing

- `sdma_load_cycles_summary = {'count': 65536, 'total': 52662272, 'min': 800, 'max': 809, 'avg': 803.5625}`
  - Each recorded SDMA load is one backend transaction that stages either one activation tile or one weight tile from DRAM to scratchpad.
  - There are two such loads per tile GEMM, so `2 * 32768 = 65536` total transactions.
  - These latencies overlap with compute and with each other, so their summed total is much larger than matrix end-to-end latency.

### Systolic Array Timing

- `systolic_array_compute_cycles_summary = {'count': 32768, 'total': 2490368, 'min': 76, 'max': 76, 'avg': 76.0}`
  - This is the TPU compute window for one tile GEMM.
  - It includes ramp and drain behavior of the array-level compute episode.

- `systolic_array_valid_mac_cycles_summary = {'count': 32768, 'total': 2457600, 'min': 75, 'max': 75, 'avg': 75.0}`
  - Cycles where the TPU is doing valid MAC work.
  - The one-cycle gap between `76` and `75` reflects the difference between the compute window and valid arithmetic occupancy.

## Per-Output-Tile Latency Breakdown

Whole run is uniform:

- `21072896 / 1024 = 20579` cycles per output tile

First tile confirms this:

- `cycle 0`: first prefetches launched
- `cycle 20467`: `tk=31` compute done
- `cycle 20579`: output tile complete

Per-output-tile picture:

- `20579` cycles total per output tile
- `14125` cycles are inside the measured tile-GEMM windows
- `807` cycles are the initial two-slot preload warmup before `tk=0` can start
- `112` cycles occur after the last `tk` compute to drain the finished accumulator tile to DRAM
- The remaining `5535` cycles are mostly steady-state bubbles waiting for a reused prefetch slot to finish loading -- `15 x 369`, which matches the schedule pattern where every other job waits for the reloaded slot to become ready.
    - a reused SDMA preload takes about `802-805` cycles
    - the opposite-slot compute lasts about `440-443` cycles
    - the reused slot therefore comes up short by about `369` cycles

-- Machine is preload-limited even though it uses two prefetch slots.

## Compute Metrics

- `flops_micro = 9861595136`
  - Modeled floating-point work counted inside the machine.
  - Includes TPU arithmetic and vector accumulation adds.

- `throughput_float_operations_per_cycle = 467.98`
  - Computed as `flops_micro / matrix_total_cycles`.
  - This is a whole-run throughput metric, so preload and drain overhead remain in the denominator.

- `mac_utilization = 0.4179`
  - `active_pe_sum / (valid_mac_cycles * 32 * 32)`
  - When the TPU is in valid MAC cycles, about `41.8%` of the `32 x 32` mesh is active on average.

- `avg_active_pes_when_active = 427.95`
  - Average active PE count over valid-MAC cycles only.

- `avg_active_pes_during_compute_window = 422.32`
  - Average active PE count over the full TPU compute window.

- `max_active_pes_in_any_cycle = 864`
  - Peak active PE count in one cycle.

### Vector Accumulation Readout

- `vec_total_ops = 33554432`
- `vec_op_counts['add'] = 33554432`
- `vec_reduce_ops = 0`

- each tile GEMM contributes a `32 x 32 = 1024` partial output tile
- each of those `1024` elements is accumulated by one vector add
- `32768 x 1024 = 33554432` total vector adds

Vector datapath is doing pure accumulation work here, not reductions or general ALU traffic (as expected).

## Bandwidth and Intensity

- `bytes_transmitted = 268435456`
  - VLS-visible movement across the scratchpad / vector-core boundary.
  - In this tiled flow that includes activation loads, weight loads, accumulator loads, and accumulator stores.

- `bytes_internal = 4700014592`
  - Useful internal TPU traffic only.

- `arithmetic_intensity_internal = 2.0982`
  - `flops_micro / bytes_internal`
  - Lower than the single-GEMM report because the tiled flow adds repeated accumulator traffic and repeated tile staging overhead.

- `external_bandwidth_avg_bytes_per_cycle = 12.74`
  - Average VLS-visible bandwidth over the full run.

- `external_bandwidth_active_bytes_per_cycle = 71.23`
  - Average only over cycles where the VLS bridge is active.

- `internal_bandwidth_bytes_per_cycle = 223.04`
  - Useful internal TPU traffic per cycle over the full run.

### Tiled-Traffic Interpretation

Each activation or weight tile is `32 x 32 x 2 = 2048` bytes.

Across `32768` tile GEMMs:

- activation tile fetches account for `32768 x 2048 = 67108864` bytes
- weight tile fetches account for `32768 x 2048 = 67108864` bytes

That is `134217728` bytes total for A and B tile fetches.

Since `bytes_transmitted = 268435456`, the other `134217728` bytes come from accumulator-row traffic used by vector-core accumulation.

So in this tiled design, accumulation traffic is as large as the raw activation-plus-weight tile fetch traffic seen at the VLS boundary.

## Reuse

- `reuse_weight_internal_over_external = 16.5`
- `reuse_act_internal_over_external = 31.4375`
- `reuse_psum_internal_over_external = 675.1416`


- partial sums move internally many times across TPU and scratchpad-backed accumulation
- the final output tile is only drained once after all `32` `tk` contributions finish

Final external output traffic is tiny compared with the total amount of internal psum movement required to build that final tile

## Queue and Scheduler Analysis

These metrics are sampled once per simulated cycle in the test harness.

### Main Observations

- `gsau_rd_queue` is still the most visibly occupied queue:
  - `max = 15`
  - `avg = 2.2392`

- `vlsu_dst_fifo` also carries meaningful steady pressure:
  - `max = 10`
  - `avg = 2.5408`

- Scheduler packet pressure stays shallow:
  - `scheduler_packets: max 1, avg 0.0031`
  - `scheduler_packet_vlsu: max 1, avg 0.0031`
  - `wb_buffer: max 1, avg 0.0171`

- The datapath is active in op counts, but it does not build persistent packet backlog:
  - `scheduler_build_datapath: max 0, avg 0.0`
  - `scheduler_packet_datapath: max 0, avg 0.0`

### Interpretation

The packet scheduler is not the bottleneck in this tiled run.

The stronger limit is the mismatch between:

- SDMA tile preload latency of about `803.6` cycles
- tile-GEMM service time of about `441.4` cycles

Double buffering helps, but it cannot fully hide a producer that is slower than the consumer. That is why the schedule exhibits repeated preload-wait bubbles and why end-to-end tile latency is much larger than the summed tile-GEMM compute windows.

## Correctness and Numerical Behavior

- `fp16_saturation_count = 0`
- `fp16_overflow_count = 0`
- `max_abs_error = 0.0`
- `mean_abs_error = 0.0`

This run is numerically clean under the current reference model.

## Current Performance Conclusion

The tiled machine is still movement-limited rather than scheduler-limited.

- Output tiles are processed in a simple row-major order and fully drained before the next output tile begins.
- Within one output tile, the machine uses a two-slot alternating `tk` schedule.
- The systolic array itself is busy for only `76` compute-window cycles per tile GEMM.
- But each reused slot needs about `803` preload cycles, so the compute path repeatedly waits for data.
- Vector-core accumulation is functioning correctly and at scale, but it also adds significant scratchpad traffic.

The main bottleneck is not packet formation and not datapath ALU throughput. The dominant cost is repeated tile staging plus accumulator traffic, with preload latency still large enough to leave visible holes between successive `tk` jobs.
