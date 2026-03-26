# TSSA GEMM Stats Report

This report explains the current metrics in `stats.log` for the end-to-end path:

`dram -> scratchpad -> vlsu -> veggie -> gsau -> tssa -> gsau -> veggie -> vlsu -> scratchpad -> dram`

## Summary

- End-to-end latency is `359` cycles.
- Algorithmic GEMM arithmetic intensity is `10.67`, which matches a `32x32 @ 32x32` FP16 GEMM:
  - `flops_algo = 2 * 32^3 = 65536`
  - `bytes_algo = 3 * 32 * 32 * 2 = 6144`
- The array now reaches full instantaneous occupancy:
  - `max_active_pes_in_any_cycle = 1024`
- Correctness is clean:
  - `fp16_saturation_count = 0`
  - `fp16_overflow_count = 0`
  - `max_abs_error = 0.0`
  - `mean_abs_error = 0.0`

## Compute Metrics

- `flops_micro = 432128`
  - Internal floating-point work counted in the modeled hardware.
  - Includes PE arithmetic, not just algorithmic GEMM FLOPs.

- `throughput_float_operations_per_cycle = 1203.70`
  - `flops_micro / cycles`
  - Useful for comparing schedules or implementations.
  - This is not the same as strict application-level FLOPs/cycle.

- `mac_utilization = 0.1532`
  - Fraction of total end-to-end cycles where at least one PE had a valid MAC.
  - This includes non-compute phases in the denominator, so it is a whole-system utilization metric.

- `avg_active_pes_when_active = 574.84`
  - Average number of active PEs conditioned on at least one PE being active.
  - This says the array is reasonably well utilized during active compute windows.

- `avg_active_pes_during_compute_window = 501.84`
  - Average active PE count across all `start == True` cycles.
  - Slightly lower than the conditional metric because it includes compute-window cycles with lower occupancy.

- `max_active_pes_in_any_cycle = 1024`
  - Peak observed active PE count.
  - This confirms the `32x32` array does fully fill.

## Bandwidth and Intensity

- `bytes_transmitted = 6144`
  - External data moved across the VLS-visible boundary.
  - One activation tile, one weight tile, one output tile.

- `external_bandwidth_avg_bytes_per_cycle = 17.11`
  - Average external bandwidth across the full end-to-end execution.
  - Includes compute, warmup, drain, and store completion cycles in the denominator.

- `external_bandwidth_active_bytes_per_cycle = 64.0`
  - Average external bandwidth during actual load/store transfer cycles.
  - `64 B/cycle` corresponds to one full `32`-element FP16 row per active transfer cycle.

- `bytes_internal = 138608`
  - Valid, non-zero internal traffic during active MAC phases.

- `internal_bandwidth_bytes_per_cycle = 386.09`
  - Internal array traffic per end-to-end cycle.
  - Much larger than external bandwidth, which is expected in a systolic architecture because data is reused by shifting internally.

- `arithmetic_intensity_internal = 3.118`
  - “Useful internal work per useful internal byte moved.”
  - This is a datapath-density metric, not a roofline metric.

- `arithmetic_intensity_algo = 10.67`
  - Classical GEMM arithmetic intensity.
  - Useful for algorithm-level comparison.

## Reuse

- `reuse_act_internal_over_external = 31.125`
  - Activation bytes are reused internally about `31x` relative to external activation traffic.
  - This is very close to the expected `32`-wide systolic reuse intuition.

- `reuse_psum_internal_over_external = 35.32`
  - Partial sums move internally many times before exiting.
  - This reflects substantial vertical psum traffic, which is normal for systolic accumulation.

- `reuse_weight_internal_over_external = 0.0`
  - This is a metric-definition artifact, not a hardware failure.
  - Weight preload happens before `start`, while the internal-valid-byte counter only counts traffic during active MAC phases.

## Queue and Scheduler Analysis

The system is no longer dominated by the old single scheduler backlog. The current scheduler is packet-based, so the queue metrics should be read as packet occupancy plus per-FU packet contents.

### GSAU / Writeback Side

- `gsau_to_systolic: max 0, avg 0.0`
  - No backlog at the TSSA request ingress.
  - The bridge is consuming GSAU requests immediately.

- `gsau_from_systolic: max 1, avg 0.089`
  - Very small response accumulation.

- `gsau_rd_queue: max 27, avg 2.41`
  - This is now one of the more active queues.
  - It tracks destination tags for in-flight systolic results.
  - Interpretation: the array can generate bursts faster than the result-retirement path fully drains them.

- `gsau_writebacks: max 1, avg 0.086`
  - GSAU-completed writebacks are drained immediately into the shared WB path.

- `wb_buffer: max 1, avg 0.003`
  - Shared writeback is not currently a major bottleneck.

### Packet Scheduler

- `scheduler_packets: max 8, avg 0.64`
  - Number of queued VLIW packets waiting behind the currently building packet.

- `scheduler_build_gsau: max 1, avg 0.175`
  - GSAU entries sitting in the in-progress packet builder.
  - Since the packet has only one GSAU slot, this is expected to stay small.

- `scheduler_build_vlsu: max 4, avg 0.187`
  - VLSU entries sitting in the in-progress packet builder.
  - Hitting `4` means the VLSU side fully occupies its packet slot budget in bursts.

- `scheduler_build_datapath: max 0, avg 0.0`
  - No datapath work is being issued in this workload.

- `scheduler_packet_gsau: max 3, avg 0.27`
  - Number of queued GSAU instructions already packed into pending VLIW packets.

- `scheduler_packet_vlsu: max 28, avg 0.40`
  - Number of queued VLSU instructions already packed into pending VLIW packets.
  - This is the dominant packetized scheduler pressure in this workload.

- `scheduler_packet_datapath: max 0, avg 0.0`
  - Again confirms no vector ALU pressure for this test.

### VLSU Side

- `vlsu_issue_q: max 24, avg 1.07`
  - This is a clear burst absorber now.
  - The scheduler and packetization can produce VLSU work faster than the VLSU consumes in short bursts.

- `vlsu_req_q: max 0, avg 0.0`
  - Once a VLSU op is issued, it is forwarded promptly.

- `vlsu_rsp_q: max 1, avg 0.178`
  - Small response-side buffering only.

- `vlsu_wb_q: max 0, avg 0.0`
  - No meaningful writeback congestion inside the VLSU.

- `vlsu_dst_fifo: max 4, avg 0.713`
  - A few outstanding load destinations in flight, but not excessive.

## Small Architecture Readout

- The `32x32` TSSA is now reaching full occupancy, so the array itself is not the obvious bottleneck.
- External streaming is clean at one full FP16 row per active transfer cycle.
- The dominant remaining pressures are:
  - VLSU issue-side burst handling
  - GSAU destination tracking / result retirement
- The packet scheduler appears to be doing its job:
  - packet backlog is moderate
  - per-FU packet occupancy shows the workload is mainly VLSU + GSAU driven
  - datapath slots are unused in this GEMM flow

## Current Bottleneck Hypothesis

The present bottleneck is most likely not packet scheduling. It is more likely the combination of:

- how quickly VLSU-issued memory ops can be consumed in bursts
- how many systolic outputs can be retired through the GSAU rd/writeback path

That fits the queue data better than a scheduler-bound explanation.
