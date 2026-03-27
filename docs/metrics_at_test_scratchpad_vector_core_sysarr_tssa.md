# TSSA GEMM Stats Report

This note explains the metrics emitted by `stats.log` for
`tests/atalla/test_scratchpad_vector_core_sysarr_tssa.py`.

Current modeled path:

`dram -> backend -> scratchpad -> vlsu -> veggie -> gsau -> tssa -> gsau -> veggie -> vlsu -> scratchpad -> backend -> dram`

The numbers below match the current backend-wired run in
`logs/sysarr_gemm_tssa/stats.log`.

## Summary

- End-to-end latency is `574` cycles.
- Algorithmic GEMM arithmetic intensity is `10.67` for a `32 x 32` FP16 GEMM.
- Correctness is clean:
  - `fp16_saturation_count = 0`
  - `fp16_overflow_count = 0`
  - `max_abs_error = 0.0`
  - `mean_abs_error = 0.0`
- The array reaches high but not full instantaneous occupancy in this run:
  - `max_active_pes_in_any_cycle = 896`

## Compute Metrics

- `flops_micro = 652288`
  - Modeled floating-point work performed inside the simulated machine.
  - Includes PE arithmetic and any modeled vector-side work.

- `flops_algo = 65536`
  - Algorithmic GEMM FLOPs only.
  - For this workload: `2 * 32^3 = 65536`.

- `throughput_float_operations_per_cycle = 1136.39`
  - Computed as `flops_micro / cycles`.
  - This is a whole-run throughput metric, so preload, drain, and writeback all stay in the denominator.

- `mac_utilization = 0.5030`
  - Computed as:
    - `active_pe_sum / (valid_mac_cycles * size * size)`
  - This is now a valid-cycle metric.
  - Interpretation: when the MAC pipeline is actually doing valid work, about `50.3%` of the `32 x 32` array is active on average.

- `avg_active_pes_when_active = 515.10`
  - Computed as `active_pe_sum / valid_mac_cycles`.
  - This is the average active PE count conditioned on valid MAC cycles only.

- `avg_active_pes_during_compute_window = 506.92`
  - Computed over all `start == True` cycles, including lower-occupancy compute-window cycles.

- `max_active_pes_in_any_cycle = 896`
  - Peak active PE count observed in a single cycle.

## Bandwidth and Intensity

- `bytes_transmitted = 6144`
  - External data moved across the VLS-visible boundary.
  - One activation tile + one weight tile + one output tile.

- `bytes_algo = 6144`
  - Classical GEMM bytes for A, B, and C.
  - For this workload: `3 * 32 * 32 * 2 = 6144`.

- `arithmetic_intensity_algo = 10.67`
  - `flops_algo / bytes_algo`
  - This is the algorithm-level arithmetic intensity.

- `bytes_internal = 154760`
  - Valid internal array traffic only.
  - This comes from `internal_bytes_valid_total()`, so it excludes zero/invalid movement.

- `arithmetic_intensity_internal = 4.2148`
  - `flops_micro / bytes_internal`
  - A datapath-density metric for useful internal work per useful internal byte moved.

- `external_bandwidth_avg_bytes_per_cycle = 10.70`
  - `bytes_transmitted / cycles`
  - Average over the full run.

- `external_bandwidth_active_bytes_per_cycle = 65.36`
  - `bytes_transmitted / vls_active_cycles`
  - Average only over cycles where the VLS bridge saw activity.

- `internal_bandwidth_bytes_per_cycle = 269.62`
  - `bytes_internal / cycles`
  - Average useful internal traffic over the full run.

## Reuse

- `reuse_act_internal_over_external = 31.3125`
  - `internal_bytes_valid["act_shift"] / bytes_load_act`
  - This is close to the expected `32x` activation reuse intuition for a 32-wide systolic flow.

- `reuse_psum_internal_over_external = 43.1699`
  - `internal_bytes_valid["psum_shift"] / bytes_store_out`
  - Reflects substantial vertical partial-sum movement before output leaves the array.

- `reuse_weight_internal_over_external = 0.0`
  - This is a metric-definition artifact.
  - Weight preload happens before valid MAC cycles, while `internal_bytes_valid["weight_shift"]` only counts valid movement during active compute.

## Queue and Scheduler Analysis

These metrics are sampled once per simulated cycle in the test harness and summarize queue depth over the whole run.

### GSAU / Writeback Side

- `gsau_to_systolic: max 0, avg 0.0`
  - No sustained ingress backlog into TSSA.

- `gsau_from_systolic: max 1, avg 0.0557`
  - Response accumulation is very small.

- `gsau_rd_queue: max 29, avg 3.066`
  - One of the more active queues in the flow.
  - This tracks outstanding result metadata while rows are being retired.

- `gsau_writebacks: max 3, avg 0.1516`
  - Completed GSAU outputs briefly accumulate before the shared VC writeback path drains them.

- `wb_buffer: max 1, avg 0.0052`
  - Shared writeback is not a dominant bottleneck in this run.

### Packet Scheduler

- `scheduler_packets: max 2, avg 0.176`
  - Pending packet backlog is modest.

- `scheduler_build_gsau: max 1, avg 0.1098`
  - Small in-progress packet pressure on the GSAU slot.

- `scheduler_build_vlsu: max 1, avg 0.1533`
  - Small in-progress packet pressure on the VLSU slot.

- `scheduler_build_datapath: max 0, avg 0.0`
  - No vector datapath work in this GEMM flow.

- `scheduler_packet_gsau: max 1, avg 0.0017`
  - Very small queued GSAU occupancy once instructions are packetized.

- `scheduler_packet_vlsu: max 4, avg 0.0174`
  - Some burstiness on the VLSU side, but much lower than older direct-load/store versions of the test.

- `scheduler_packet_datapath: max 0, avg 0.0`
  - Confirms no datapath-issued vector ALU work here.

### VLSU Side

- `vlsu_issue_q: max 3, avg 0.0314`
  - Minor burst absorption only.

- `vlsu_req_q: max 1, avg 0.0070`
  - Requests move through quickly.

- `vlsu_rsp_q: max 1, avg 0.1115`
  - Small response-side buffering only.

- `vlsu_wb_q: max 0, avg 0.0`
  - No meaningful VLSU writeback congestion.

- `vlsu_dst_fifo: max 5, avg 1.5261`
  - A handful of outstanding loads in flight, which is expected.

## Small Architecture Readout

- This harness now models two distinct memory regimes:
  - backend-driven bulk movement for DRAM <-> scratchpad traffic
  - VLSU-driven row traffic for scratchpad <-> VRF exchanges during compute
- The systolic array is not starved, but it is also not running at full-mesh occupancy throughout the valid compute window.
  - `avg_active_pes_when_active = 515.10`
  - `max_active_pes_in_any_cycle = 896`
  - Taken together, these say the `32 x 32` mesh is meaningfully loaded, but the row-by-row feed/retire structure leaves substantial headroom below 1024 PEs.
- Scheduler pressure is present but modest.
  - Packet counts are low.
  - VLSU-side issue buffering is shallow.
  - There is no sign of a scheduler-wide traffic jam.
- The more characteristic pressure is on the result-retirement side of the systolic path.
  - `gsau_rd_queue` is the most consistently elevated queue in the run.
  - `gsau_writebacks` and `vlsu_dst_fifo` show small but persistent in-flight state around result movement.
- The whole-system latency now includes backend preload and backend storeback phases, so `cycles` should be read as a full pipeline number, not just a pure compute-window number.

## Current Bottleneck Hypothesis

The current bottleneck is best described as throughput mismatch across the row pipeline, not scheduler collapse.

- On the front side, the backend preload path adds fixed latency before compute can begin, but it does not appear to build pathological queue pressure once the run is underway.
- During compute, the VLSU and GSAU keep the array fed well enough to maintain steady activity, but not well enough to sustain near-full occupancy across the entire valid window.
- On the back side, output retirement is the clearest pacing point.
  - GSAU must track and drain returning rows in order.
  - Results pass through shared architectural writeback.
  - Output rows are then stored to scratchpad and finally written back to DRAM through the backend.
- That combination makes the post-compute / retire path more plausible as the limiting factor than packet formation or scheduler slot availability.

The machine looks retirement-limited more than issue-limited. The queue data supports a story where compute proceeds steadily. But overall progress is bounded by how quickly rows can be observed, committed, and pushed through the scratchpad-to-DRAM tail.
