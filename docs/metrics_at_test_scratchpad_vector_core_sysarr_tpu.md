# TPU GEMM Stats Report

This note explains the metrics emitted by `stats.log` for
`tests/atalla/test_scratchpad_vector_core_sysarr_tpu.py`.

Current modeled path:

`dram -> backend -> scratchpad -> vlsu -> veggie -> gsau -> tpu -> gsau -> veggie -> vlsu -> scratchpad -> backend -> dram`

The numbers below match the current backend-wired run in
`logs/sysarr_gemm_tpu/stats.log`.

## Summary

- End-to-end latency is `1092` cycles.
- Algorithmic GEMM arithmetic intensity is still `10.67` for a `32 x 32` FP16 GEMM.
- Current correctness is clean in this run:
  - `fp16_saturation_count = 0`
  - `fp16_overflow_count = 0`
  - `max_abs_error = 0.0`
  - `mean_abs_error = 0.0`
- The TPU reaches high but not full instantaneous occupancy in this run:
  - `max_active_pes_in_any_cycle = 896`

## Compute Metrics

- `flops_micro = 415744`
  - Modeled floating-point work performed inside the simulated machine.
  - Includes TPU-side arithmetic and any modeled vector-side work.

- `flops_algo = 65536`
  - Algorithmic GEMM FLOPs only.
  - For this workload: `2 * 32^3 = 65536`.

- `throughput_float_operations_per_cycle = 380.72`
  - Computed as `flops_micro / cycles`.
  - This is a whole-run throughput metric, so backend preload and storeback stay in the denominator.

- `mac_utilization = 0.4384`
  - Computed as:
    - `active_pe_sum / (valid_mac_cycles * size * size)`
  - Interpretation: when the MAC pipeline is doing valid work, about `43.8%` of the `32 x 32` array is active on average.

- `avg_active_pes_when_active = 448.88`
  - Computed as `active_pe_sum / valid_mac_cycles`.
  - This is the average active PE count conditioned on valid MAC cycles only.

- `avg_active_pes_during_compute_window = 431.16`
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

- `bytes_internal = 129676`
  - Valid internal array traffic only.
  - This comes from `internal_bytes_valid_total()`, so it excludes zero/invalid movement.

- `arithmetic_intensity_internal = 3.2060`
  - `flops_micro / bytes_internal`
  - A datapath-density metric for useful internal work per useful internal byte moved.

- `external_bandwidth_avg_bytes_per_cycle = 5.63`
  - `bytes_transmitted / cycles`
  - Average over the full run.

- `external_bandwidth_active_bytes_per_cycle = 76.8`
  - `bytes_transmitted / vls_active_cycles`
  - Average only over cycles where the VLS bridge saw activity.

- `internal_bandwidth_bytes_per_cycle = 118.75`
  - `bytes_internal / cycles`
  - Average useful internal traffic over the full run.

## Reuse

- `reuse_weight_internal_over_external = 16.5`
  - Weight movement is now being counted in the valid internal-byte metric.
  - This says weight data is reused substantially once it is inside the TPU fabric.

- `reuse_act_internal_over_external = 32.0`
  - `internal_bytes_valid["act_shift"] / bytes_load_act`
  - This still matches the expected `32x` activation reuse intuition for a 32-wide systolic flow.

- `reuse_psum_internal_over_external = 13.82`
  - `internal_bytes_valid["psum_shift"] / bytes_store_out`
  - Reflects vertical partial-sum movement before output leaves the array.

## Queue and Scheduler Analysis

These metrics are sampled once per simulated cycle in the test harness and summarize queue depth over the whole run.

### GSAU / Writeback Side

- `gsau_to_systolic: max 0, avg 0.0`
  - No sustained ingress backlog into the TPU array.

- `gsau_from_systolic: max 1, avg 0.0293`
  - Response accumulation is very small.

- `gsau_rd_queue: max 15, avg 1.3187`
  - Still the most consistently elevated queue on the TPU result side.
  - This tracks outstanding result metadata while rows are being retired.

- `gsau_writebacks: max 0, avg 0.0`
  - There is no meaningful queueing at the GSAU writeback staging point in this run.

- `wb_buffer: max 0, avg 0.0`
  - Shared architectural writeback is not the active bottleneck here.

### Packet Scheduler

- `scheduler_packets: max 2, avg 0.0925`
  - Pending packet backlog is modest.

- `scheduler_build_gsau: max 1, avg 0.0577`
  - Small in-progress packet pressure on the GSAU slot.

- `scheduler_build_vlsu: max 1, avg 0.0806`
  - Small in-progress packet pressure on the VLSU slot.

- `scheduler_build_datapath: max 0, avg 0.0`
  - No vector datapath work in this GEMM flow.

- `scheduler_packet_gsau: max 1, avg 0.0009`
  - Very small queued GSAU occupancy once instructions are packetized.

- `scheduler_packet_vlsu: max 4, avg 0.0092`
  - Some burstiness on the VLSU side, but still shallow.

- `scheduler_packet_datapath: max 0, avg 0.0`
  - Confirms no datapath-issued vector ALU work here.

### VLSU Side

- `vlsu_issue_q: max 3, avg 0.0165`
  - Minor burst absorption only.

- `vlsu_req_q: max 1, avg 0.0037`
  - Requests move through quickly.

- `vlsu_rsp_q: max 1, avg 0.0586`
  - Small response-side buffering only.

- `vlsu_wb_q: max 0, avg 0.0`
  - No meaningful VLSU writeback congestion.

- `vlsu_dst_fifo: max 5, avg 0.8022`
  - A handful of outstanding loads in flight, which is expected.

## Phase Breakdown

From `logs/sysarr_gemm_tpu/pipeline_phases.log`:

- `weight_preload = 909` cycles
- `activation_stream = 102` cycles
- `steady_compute = 46` cycles
- `store_tail = 9` cycles
- `drain = 26` cycles

This is the clearest reason the whole-run throughput and bandwidth numbers are lower than the algorithm-only intuition: most of the runtime is still front-loaded into weight preload.

## Small Architecture Readout

- This harness still models two distinct memory regimes:
  - backend-driven bulk movement for DRAM <-> scratchpad traffic
  - VLSU-driven row traffic for scratchpad <-> VRF exchanges during compute
- The TPU is not scheduler-starved.
  - Packet counts are low.
  - VLSU-side issue buffering is shallow.
  - There is no sign of scheduler collapse.
- The TPU is also not running at full-mesh occupancy throughout the valid compute window.
  - `avg_active_pes_when_active = 448.88`
  - `max_active_pes_in_any_cycle = 896`
  - This still says the `32 x 32` mesh is meaningfully loaded, but the row-by-row feed/retire structure leaves substantial headroom below 1024 PEs.
- The most visible performance cost in this run is still outside the packet scheduler.
  - Front-end preload dominates the cycle count.
  - `gsau_rd_queue` remains the most consistently elevated result-side queue.
  - The overall machine still looks paced by row movement and retirement rather than by scheduler slot pressure.

## Current Performance Conclusion

The performance conclusion is the same as before: the machine looks throughput-mismatch / pipeline-tail limited more than scheduler-limited.

- On the front side, backend preload is the dominant fixed-cost phase in the current run.
- During compute, the VLSU and GSAU keep the TPU fed well enough to maintain steady activity, but not well enough to sustain near-full occupancy across the valid window.
- On the back side, result metadata retirement still shows up more clearly than scheduler pressure.
- Queue data continues to support a story where the scheduler is not the collapsing point. Overall progress is still bounded more by data movement and retire behavior than by packet formation.