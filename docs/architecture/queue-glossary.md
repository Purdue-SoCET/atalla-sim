# SysArr TPU Queue Glossary

This note defines the queues and FIFO-like buffers in the current
`sysarr_tpu_system` / tiled TPU flow and explains the data direction for each.

Notation used below:

- `producer -> queue -> consumer`
- `metadata` means tags such as destination register, dtype, mask, or row index, not vector payload data.

## Queues Exposed In `queue_max_depths` And `queue_avg_depths`

These are the queue names reported by `QUEUE_NAMES` in
`src/atalla/sysarr_tpu_experiment.py`.

### GSAU Side

- `gsau_to_systolic`
  - Owner: `VectorCore.gsau.to_systolic`
  - Definition: request queue holding GSAU-issued vectors before they are consumed by the systolic-array bridge.
  - Payload: vector data, `is_weight`, `expect_output`, dtype, meta.
  - Direction: scheduler-issued GSAU command -> GSAU request queue -> `GSAUTPUBridge` / systolic array ingress.

- `gsau_from_systolic`
  - Owner: `VectorCore.gsau.from_systolic`
  - Definition: response queue holding completed systolic-array output vectors before GSAU pairs them with destination metadata.
  - Payload: output vector plus meta from the TPU bridge.
  - Direction: `GSAUTPUBridge` / systolic array egress -> GSAU response queue -> GSAU writeback formatter.

- `gsau_rd_queue`
  - Owner: `VectorCore.gsau.rd_queue`
  - Definition: in-order destination FIFO for GSAU operations that expect an output.
  - Payload: destination vreg id, dtype, metadata.
  - Direction: scheduler-issued GSAU op with `expect_output=True` -> destination FIFO -> paired with a returning systolic response.
  - Data direction: logical metadata flow from issue side toward writeback side.

- `gsau_writebacks`
  - Owner: `VectorCore.gsau.writebacks`
  - Definition: GSAU-completed writeback packets waiting to enter the shared vector-core writeback buffer.
  - Payload: destination vreg id, output vector, dtype, mask, meta.
  - Direction: GSAU response matcher -> GSAU writeback queue -> `wb_buffer` -> vector register file.

### Scheduler Side

- `scheduler_packets`
  - Owner: logical view over `VectorCore._build_packet` and `VectorCore.vliw_q`
  - Definition: packet-count view of the scheduler backlog.
  - Payload: this is an aggregate count, not a standalone `SimQueue`.
  - Direction: scheduler enqueue path -> packet assembly / packet queue -> issue stage.

- `scheduler_build_gsau`
  - Owner: `VectorCore._build_packet['gsau']`
  - Definition: not a FIFO object, but the current in-construction packet slot for GSAU instructions.
  - Payload: GSAU instructions not yet flushed into `vliw_q`.
  - Direction: API enqueue -> packet builder -> `vliw_q`.

- `scheduler_build_vlsu`
  - Owner: `VectorCore._build_packet['vlsu']`
  - Definition: current in-construction packet slot for VLSU instructions.
  - Payload: vector memory ops not yet flushed into `vliw_q`.
  - Direction: API enqueue -> packet builder -> `vliw_q`.

- `scheduler_build_datapath`
  - Owner: `VectorCore._build_packet['datapath']`
  - Definition: current in-construction packet slot for datapath instructions.
  - Payload: vector ALU operations not yet flushed into `vliw_q`.
  - Direction: API enqueue -> packet builder -> `vliw_q`.

- `scheduler_packet_gsau`
  - Owner: instructions already stored inside `VectorCore.vliw_q`
  - Definition: aggregate count of queued GSAU instructions inside packetized VLIW entries.
  - Payload: GSAU instructions.
  - Direction: `vliw_q` packet queue -> GSAU issue path.

- `scheduler_packet_vlsu`
  - Owner: instructions already stored inside `VectorCore.vliw_q`
  - Definition: aggregate count of queued VLSU instructions inside packetized VLIW entries.
  - Payload: vector load/store instructions.
  - Direction: `vliw_q` packet queue -> VLSU issue path.

- `scheduler_packet_datapath`
  - Owner: instructions already stored inside `VectorCore.vliw_q`
  - Definition: aggregate count of queued datapath instructions inside packetized VLIW entries.
  - Payload: vector compute instructions.
  - Direction: `vliw_q` packet queue -> datapath issue path.

### Writeback Side

- `wb_buffer`
  - Owner: `VectorCore.wb_buffer.entries`
  - Definition: shared writeback buffer of the vector core.
  - Payload: completed architectural writebacks tagged with destination vreg and data.
  - Direction: vector datapath, GSAU, and VLSU -> writeback buffer -> Veggie / vector register file.

### VLSU Side

- `vlsu_issue_q`
  - Owner: `VLSU.issue_q`
  - Definition: queue of memory instructions accepted from the scheduler but not yet translated into scratchpad requests.
  - Payload: load/store ops with addr, size, scratchpad id, dst/src registers, dtype.
  - Direction: scheduler -> VLSU issue queue -> VLSU request generation.

- `vlsu_req_q`
  - Owner: `VLSU.req_q`
  - Definition: queue of concrete scratchpad requests produced by the VLSU.
  - Payload: load/store requests with scratchpad id, addr, size, and store data when applicable.
  - Direction: VLSU issue stage -> scratchpad request queue -> `VLSFrontendBridge` / scratchpad frontend.

- `vlsu_rsp_q`
  - Owner: `VLSU.rsp_q`
  - Definition: queue of scratchpad load responses waiting to be matched with the destination-tag FIFO.
  - Payload: returned load data plus addr and metadata.
  - Direction: scratchpad frontend / VLS bridge -> VLSU response queue -> destination matching stage.

- `vlsu_wb_q`
  - Owner: `VLSU.wb_q`
  - Definition: load-complete writeback queue after a scratchpad response has been paired with a destination tag.
  - Payload: destination vreg id, load data, dtype, mask, addr, metadata.
  - Direction: VLSU response matcher -> VLSU writeback queue -> `wb_buffer` -> vector register file.

- `vlsu_dst_fifo`
  - Owner: `VLSU.load_dst_fifos[spad]`
  - Definition: per-scratchpad destination FIFO for outstanding loads.
  - Payload: destination register tag and associated metadata.
  - Direction: issued VLSU load -> destination FIFO -> matched against `vlsu_rsp_q` responses.
  - Data direction: metadata flow only.

## Additional Major Queues In The System

These queues are part of the modeled system but are not currently reported in `queue_max_depths` / `queue_avg_depths`.

### Vector Core Internal Queues

- `vliw_q`
  - Owner: `VectorCore.vliw_q`
  - Definition: packet queue of already-built VLIW packets.
  - Direction: scheduler packet builder -> `vliw_q` -> per-unit issue logic.

- `datapath.pending_issue`
  - Owner: `VectorDatapath.pending_issue`
  - Definition: queue of vector ALU instructions waiting to be broadcast to all vector lanes.
  - Direction: scheduler / `VectorCore._issue_datapath` -> datapath pending-issue queue -> vector lanes.

- `collector.completed_vectors`
  - Owner: `ResultCollector.completed_vectors`
  - Definition: queue of completed datapath vector results waiting to be seen as `datapath.result_valid` by the vector core.
  - Direction: vector lanes -> result collector -> vector-core writeback selection -> `wb_buffer`.

- `collector.pending_reductions`
  - Owner: `ResultCollector.pending_reductions`
  - Definition: latency-holding queue for reduction instructions before they become architecturally complete.
  - Direction: lane partial reductions -> reduction latency queue -> `collector.completed_vectors`.

### Transpose Unit Queues

- `transpose.outputs`
  - Owner: `TransposeUnit.outputs`
  - Definition: transposed columns produced by a drain, waiting to be picked up as writebacks. Defaults to one whole tile, so a pop never stalls on its own output; when it is full the FSM parks in `DONE`.
  - Payload: `{"col", "data", "dst"}` per column.
  - Direction: Clos crossbar output -> transpose output queue -> `wb_buffer` -> Veggie.

- `transpose.clos._issue_q`
  - Owner: `TransposeUnit.clos` (a `memory.crossbar.Xbar`)
  - Definition: the Clos network's own issue queue. The FSM keeps one vector in flight, so this only holds a tail the consumer is backpressuring.
  - Payload: one permutation request: shift mask plus a vector of bank values.
  - Direction: transpose FSM -> crossbar pipeline -> row into the banks, or column out.

### Vector Lane / FU Queues

- `laneX.<fu>.entries`
  - Owner: `FunctionalUnitPipeline.entries`
  - Definition: in-flight operations currently executing in one FU pipeline.
  - Direction: lane sequencer -> FU pipeline -> FU completion queue.

- `laneX.<fu>.completed`
  - Owner: `FunctionalUnitPipeline.completed`
  - Definition: FU outputs that finished pipeline latency but have not yet been paired with lane metadata.
  - Direction: FU pipeline -> FU completion queue -> lane metadata pairing.

- `laneX.meta_fifo[fu]`
  - Owner: `VectorLane.meta_fifo[fu]`
  - Definition: metadata FIFO aligned with FU-issued elements so completions can be mapped back to instruction id and vector element index.
  - Payload: inst id, lane element index, destination, reduction info.
  - Direction: lane issue stage -> metadata FIFO -> paired with FU completions.

- `laneX.pending_outputs`
  - Owner: `VectorLane.pending_outputs`
  - Definition: queue of lane results after FU completion and metadata pairing, before delivery to the result collector.
  - Direction: lane completion stage -> pending output queue -> result collector.

### Scratchpad Queues

- `spad.tiles[tid].rq`
  - Owner: `Scratchpad` pad (`memory/scratchpad.py`)
  - Definition: accepted row reads waiting for their pad's read channel; models scpad_cntrl's read FIFO (32 deep by default, `spad_frontend_queue_size`).
  - Direction: `VLSFrontendBridge` load or backend store read -> pad read queue -> bank read enable -> data to the requester 5 cycles later.

- `spad.tiles[tid].wq`
  - Owner: `Scratchpad` pad
  - Definition: accepted row writes waiting for their pad's write channel; models scpad_cntrl's write FIFO. Either queue being full stalls both directions, as w_stall does.
  - Direction: `VLSFrontendBridge` store or backend load write -> pad write queue -> bank write enable.

- `spad.tiles[tid].out`
  - Owner: `Scratchpad` pad
  - Definition: reads that have left the banks and are on their way back (rxbar and tail), keyed by delivery cycle.
  - Direction: bank read -> requester callback.

### Backend Queues

- `backend._tx_queue`
  - Owner: `Backend._tx_queue`
  - Definition: queue of high-level DRAM <-> scratchpad transactions waiting to become active.
  - Payload: load/store transaction descriptors covering many rows and bursts.
  - Direction: driver / test harness -> backend transaction queue -> active backend transaction set.

- `backend._dram_pending`
  - Owner: `Backend._dram_pending`
  - Definition: queue of DRAM burst operations currently in flight.
  - Payload: one DRAM burst read or write operation with remaining latency.
  - Direction: backend burst issuer -> DRAM pending queue -> DRAM completion handler.

- `backend._pending_sram_reads`
  - Owner: `Backend._pending_sram_reads` (Python `deque`, not `SimQueue`)
  - Definition: deferred scratchpad row-read responses for store transactions before they are burst-split into DRAM writes.
  - Direction: scratchpad row read -> deferred read buffer -> backend DRAM write burst builder.

### TPU Boundary FIFOs

- `tpu._weight_boundary[g]`
  - Owner: `SystolicArrayTPU._weight_boundary`
  - Definition: per-group FIFO of incoming weight groups at the array boundary.
  - Direction: GSAU weight stream -> TPU weight boundary FIFO -> cell weight latches.

- `tpu._input_fifo_left[g]`
  - Owner: `SystolicArrayTPU._input_fifo_left`
  - Definition: per-group FIFO of incoming activation groups at the left boundary.
  - Direction: GSAU activation stream -> TPU activation boundary FIFO -> cell activation latches.

- `tpu._input_algo_flags`
  - Owner: `SystolicArrayTPU._input_algo_flags`
  - Definition: FIFO of booleans aligned with activation rows, used for algorithmic accounting and output bookkeeping.
  - Direction: TPU enqueue API -> flag FIFO -> output accounting logic.
  - Data direction: metadata only.

- `tpu._psum_input_fifo_top[j]`
  - Owner: `SystolicArrayTPU._psum_input_fifo_top`
  - Definition: per-column FIFO of incoming partial sums at the top boundary.
  - Direction: psum source / zero seed injection -> psum boundary FIFO -> top-row accumulation inputs.

## Direction Summary By Subsystem

- Scheduler side
  - API enqueue -> build packet -> `vliw_q` -> issue to GSAU / VLSU / datapath.

- Compute side
  - datapath pending issue -> lane FU pipelines -> result collector -> `wb_buffer` -> vector register file.

- Systolic side
  - GSAU request queue -> TPU boundary FIFOs -> TPU compute -> GSAU response queue -> GSAU writebacks -> `wb_buffer`.

- Memory side
  - VLSU issue queue -> VLSU request queue -> scratchpad pad read/write queues -> SRAM bank channels.
  - For loads: SRAM bank / frontend response -> `vlsu_rsp_q` -> `vlsu_wb_q` -> `wb_buffer`.
  - For stores: vector register file / inline data -> VLSU -> scratchpad write path.

- Backend side
  - driver transaction queue -> DRAM burst queue -> scratchpad row handoff or DRAM writeback completion.

## Practical Reading Guide

- `gsau_rd_queue`: metadata FIFO for expected TPU outputs, direction `scheduler/GSAU issue -> GSAU response pairing`
- `vlsu_dst_fifo`: metadata FIFO for outstanding loads, direction `VLSU load issue -> VLSU response matcher`
- `vlsu_req_q`: real memory traffic queue, direction `VLSU -> scratchpad frontend`
- `vlsu_rsp_q`: real load-response queue, direction `scratchpad frontend -> VLSU`
- `wb_buffer`: common architectural sink, direction `vector datapath / GSAU / VLSU -> Veggie`

Those are usually the highest-value queues to interpret first when looking for the active bottleneck.