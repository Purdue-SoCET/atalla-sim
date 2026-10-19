"""A single-tile GEMM experiment on the TPU platform, with detailed metrics.

run_sysarr_tpu_experiment(config) moves weights and activations from DRAM
through one scratchpad pad and VLSU into the systolic array, stores the
outputs back, checks them against a reference, and returns cycle counts per
phase (PHASE_ORDER), queue occupancies and traffic. The sweep scripts in
tools/ call it with different SysArrTPUExperimentConfig settings.
Run one from the command line with `python -m atalla.sysarr_tpu_experiment`.
"""

import argparse
import json
from dataclasses import asdict, dataclass, replace
import os
import sys
from typing import Dict, List, Optional

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from base.clocked_object import Clocked
from memory.dram import DRAM, DRAM_BURST_BYTES
from memory.scratchpad import Scratchpad
from systolic_array.systolic_array_tpu import SystolicArrayTPU
from vector_core.vector_core import VectorCore

from atalla.sysarr_tpu_system import (
    RoundRobinBackendTicker, VLSFrontendBridge, build_tpu_platform, _act_u16,
    _decode_row_u16, _encode_row_u16, _weights_u16)


PHASE_ORDER = [
    "weight_preload",
    "activation_stream",
    "compute_ramp",
    "steady_compute",
    "store_tail",
    "drain",
    "idle",
]


QUEUE_NAMES = [
    "gsau_to_systolic",
    "gsau_from_systolic",
    "gsau_rd_queue",
    "gsau_writebacks",
    "scheduler_packets",
    "scheduler_build_gsau",
    "scheduler_build_vlsu",
    "scheduler_build_datapath",
    "scheduler_packet_gsau",
    "scheduler_packet_vlsu",
    "scheduler_packet_datapath",
    "wb_buffer",
    "vlsu_issue_q",
    "vlsu_req_q",
    "vlsu_rsp_q",
    "vlsu_wb_q",
    "vlsu_dst_fifo",
]


def _read_dram_row_u16(dram: DRAM, addr: int, cols: int):
    row_bytes = int(cols) * 2
    blob = dram.read(int(addr), row_bytes)
    return _decode_row_u16(blob, cols)


def _write_dram_tile_u16(dram: DRAM, base_addr: int, tile: List[List[int]]):
    rows = len(tile)
    if rows == 0:
        return
    cols = len(tile[0])
    row_bytes = cols * 2
    for r in range(rows):
        dram.write(int(base_addr) + r * row_bytes, _encode_row_u16(tile[r]))


def _read_dram_tile_u16(dram: DRAM, base_addr: int, rows: int, cols: int) -> List[List[int]]:
    out = []
    row_bytes = int(cols) * 2
    for r in range(int(rows)):
        blob = dram.read(int(base_addr) + r * row_bytes, row_bytes)
        out.append(_decode_row_u16(blob, cols))
    return out


def _read_slot_vector_u16(spad: Scratchpad, addr: int, vector_len: int, tile_id: int = 0):
    tile = int(tile_id)
    slot = int(addr) % spad.bank_size
    out = []
    for lane in range(vector_len):
        bank = lane
        blob = spad.tiles[tile].banks[bank].mem[slot]
        blob = bytes(blob) if blob is not None else b"\x00\x00"
        if len(blob) < 2:
            blob = blob + (b"\x00" * (2 - len(blob)))
        out.append(int.from_bytes(blob[:2], "little", signed=False))
    return out


def _fp16_from_u16(value: int) -> float:
    return float(np.frombuffer(np.uint16(int(value)).tobytes(), dtype=np.float16)[0])


def _tpu_reference_output(
    act_rows: List[List[int]],
    weight_stream: List[List[int]],
    *,
    size: int,
    dtype: str = "fp16",
) -> List[List[int]]:
    sa = SystolicArrayTPU(size=size, dtype=dtype)
    zero_row = [0.0] * size
    out = []
    out_read_idx = 0
    warmup = sa.warmup_cycles()
    cycle = 0

    for vec in weight_stream:
        assert sa.enqueue_weights([float(x) for x in vec], dtype=dtype)
        sa.set_control(weight_en=True, mac_shift=False, start=False, stall=False)
        sa.tick(cycle)
        cycle += 1

    for vec in act_rows:
        assert sa.enqueue([float(x) for x in vec], dtype=dtype)
        assert sa.enqueue_psums(zero_row, dtype=dtype)
        sa.set_control(weight_en=False, mac_shift=True, start=True, stall=False)
        sa.tick(cycle)
        cycle += 1
        buf = sa.get_buffer()
        while out_read_idx < len(buf):
            if out_read_idx < warmup:
                out_read_idx += 1
                continue
            out.append([int(float(x)) for x in buf[out_read_idx]])
            out_read_idx += 1

    for _ in range(sa.flush_cycles()):
        assert sa.enqueue(zero_row, dtype=dtype, count_algo=False)
        assert sa.enqueue_psums(zero_row, dtype=dtype)
        sa.set_control(weight_en=False, mac_shift=True, start=True, stall=False)
        sa.tick(cycle)
        cycle += 1
        buf = sa.get_buffer()
        while out_read_idx < len(buf):
            if out_read_idx < warmup:
                out_read_idx += 1
                continue
            out.append([int(float(x)) for x in buf[out_read_idx]])
            out_read_idx += 1

    if len(out) >= size:
        return out[-size:]
    return out


def _phase_snapshot(state: Dict, sa: SystolicArrayTPU, tile: int) -> str:
    if state.get("weight_row", 0) < tile:
        return "weight_preload"
    if state.get("act_issue_row", 0) < tile or state.get("act_loads_inflight", 0) > 0:
        return "activation_stream"
    if state.get("next_out_row", 0) < tile and len(state.get("completed_rows", set())) == 0:
        return "compute_ramp"
    if state.get("next_out_row", 0) < tile:
        return "steady_compute"
    if state.get("pending_output_rows") or state.get("store_inflight"):
        return "store_tail"
    if sa.value_ready or len(sa.get_buffer()) > 0:
        return "drain"
    return "idle"


@dataclass
class SysArrTPUExperimentConfig:
    """One experiment's settings. The first four fields label the run in
    sweep output; the rest configure the platform (see build_tpu_platform).
    To add a setting: add a field with a default here, and use it in
    run_sysarr_tpu_experiment; to_dict() picks it up by itself."""
    name: str
    sweep: str
    param_name: str
    param_value: object
    tile: int = 32
    dtype: str = "fp16"
    lane_count: int = 4
    # This experiment drives a single bridge by hand (see clk.objects
    # below), so it uses one VLSU / one pad. Raising this would build
    # bridges that never get ticked.
    vls_count: int = 1
    spad_num_banks: Optional[int] = None      # None: one bank per lane of the tile
    spad_bank_size: Optional[int] = None      # None: max(128, 4 * tile)
    spad_read_latency: int = 2
    spad_write_latency: int = 1
    spad_frontend_queue_size: int = 4
    backend_dram_latency: int = 24
    backend_dram_q_depth: int = 16
    backend_dram_burst_bytes: int = DRAM_BURST_BYTES
    backend_delay_cycles: int = 1
    dram_block_bytes: int = 256
    max_cycles: int = 20000

    def to_dict(self):
        return asdict(self)

    def normalize(self) -> "SysArrTPUExperimentConfig":
        """A copy with the scratchpad defaults filled in, checked."""
        spad_num_banks = self.spad_num_banks if self.spad_num_banks is not None else self.tile
        spad_bank_size = self.spad_bank_size if self.spad_bank_size is not None else max(128, self.tile * 4)
        if spad_num_banks < self.tile:
            raise ValueError("spad_num_banks must be >= tile")
        if spad_bank_size < (self.tile * 3):
            raise ValueError("spad_bank_size must be >= 3 * tile for the chosen address map")
        return replace(self, spad_num_banks=spad_num_banks, spad_bank_size=spad_bank_size)


def _queue_depths(vc: VectorCore) -> Dict[str, int]:
    """How full each queue of the vector core (and its VLSU 0) is right now;
    sampled every cycle for the queue_max_* / queue_avg_* results. To track
    another queue, add it here and to QUEUE_NAMES."""
    vlsu0 = vc.vls_units[0]
    return {
        "gsau_to_systolic": len(vc.gsau.to_systolic),
        "gsau_from_systolic": len(vc.gsau.from_systolic),
        "gsau_rd_queue": len(vc.gsau.rd_queue),
        "gsau_writebacks": len(vc.gsau.writebacks),
        "scheduler_packets": len(vc.scheduler_packets),
        "scheduler_build_gsau": len(vc._build_packet["gsau"]),
        "scheduler_build_vlsu": len(vc._build_packet["vlsu"]),
        "scheduler_build_datapath": len(vc._build_packet["datapath"]),
        "scheduler_packet_gsau": sum(len(pkt["gsau"]) for pkt in vc.vliw_q.items),
        "scheduler_packet_vlsu": sum(len(pkt["vlsu"]) for pkt in vc.vliw_q.items),
        "scheduler_packet_datapath": sum(len(pkt["datapath"]) for pkt in vc.vliw_q.items),
        "wb_buffer": len(vc.wb_buffer.entries),
        "vlsu_issue_q": len(vlsu0.issue_q),
        "vlsu_req_q": len(vlsu0.req_q),
        "vlsu_rsp_q": len(vlsu0.rsp_q),
        "vlsu_wb_q": len(vlsu0.wb_q),
        "vlsu_dst_fifo": len(vlsu0.load_dst_fifos[0]),
    }


def _experiment_result(cfg, state, phase_counts, q_stats, platform, got,
                       expected_ref, expected_cycle) -> Dict[str, object]:
    """Everything one run measured, as a flat dict (a row of the sweep CSVs,
    whose columns follow this order). To report something new, add a key
    here; tools/run_sysarr_tpu_sweeps.py writes whatever keys it finds."""
    tile = cfg.tile
    sa, vc, spad = platform.sa, platform.vc, platform.spad
    backends, vls_bridge = platform.backends, platform.vls_bridge
    bytes_tx = vls_bridge.bytes_load + vls_bridge.bytes_store
    pe_mul = sum(pe.mul_ops for row in sa.array for pe in row)
    pe_add = sum(pe.add_ops for row in sa.array for pe in row)
    pe_mac = sum(pe.mac_ops for row in sa.array for pe in row)
    pe_psum_add = sum(pe.psum_adds for row in sa.array for pe in row)
    vec_total_ops = sum(lane.total_ops for lane in vc.datapath.lanes)
    vec_op_counts: Dict[str, int] = {}
    for lane in vc.datapath.lanes:
        for op, cnt in lane.op_counts.items():
            vec_op_counts[op] = vec_op_counts.get(op, 0) + cnt
    vec_reduce_ops = vc.datapath.collector.reduction_unit.reduce_ops
    flops_micro = pe_mul + pe_add + vec_total_ops + vec_reduce_ops
    bytes_internal = sa.internal_bytes_valid_total()
    arithmetic_intensity_internal = (flops_micro / bytes_internal) if bytes_internal else 0.0
    flops_algo = sa.algo_flops()
    bytes_algo = sa.algo_bytes()
    arithmetic_intensity_algo = sa.algo_arithmetic_intensity()

    mac_utilization = (sa.active_pe_sum / (sa.valid_mac_cycles * tile * tile)) if sa.valid_mac_cycles else 0.0
    avg_active_pes_when_active = (sa.active_pe_sum / sa.valid_mac_cycles) if sa.valid_mac_cycles else 0.0
    avg_active_pes_during_compute_window = (
        sa.compute_window_active_pe_sum / sa.compute_window_cycles
    ) if sa.compute_window_cycles else 0.0
    max_active_pes_in_any_cycle = sa.max_active_pes_in_cycle
    throughput = (flops_micro / state["cycles"]) if state["cycles"] else 0.0
    external_bw = (bytes_tx / state["cycles"]) if state["cycles"] else 0.0
    external_bw_active = (bytes_tx / state["vls_active_cycles"]) if state["vls_active_cycles"] else 0.0
    internal_bw = (bytes_internal / state["cycles"]) if state["cycles"] else 0.0
    reuse_weight = (
        sa.internal_bytes_valid["weight_shift"] / state["bytes_load_wgt"] if state["bytes_load_wgt"] else 0.0
    )
    reuse_act = sa.internal_bytes_valid["act_shift"] / state["bytes_load_act"] if state["bytes_load_act"] else 0.0
    reuse_psum = (
        sa.internal_bytes_valid["psum_shift"] / state["bytes_store_out"] if state["bytes_store_out"] else 0.0
    )

    ref_for_error = expected_ref if len(expected_ref) >= tile else expected_cycle

    max_abs_error = 0.0
    sum_abs_error = 0.0
    count_err = 0
    for i in range(tile):
        for j in range(tile):
            got_f = _fp16_from_u16(got[i][j])
            exp_f = _fp16_from_u16(ref_for_error[i][j])
            err = abs(got_f - exp_f)
            if err > max_abs_error:
                max_abs_error = err
            sum_abs_error += err
            count_err += 1
    mean_abs_error = (sum_abs_error / count_err) if count_err else 0.0

    result: Dict[str, object] = {
        "name": cfg.name,
        "sweep": cfg.sweep,
        "param_name": cfg.param_name,
        "param_value": cfg.param_value,
        "cycles": state["cycles"],
        "pe_mul_ops": pe_mul,
        "pe_add_ops": pe_add,
        "pe_mac_ops": pe_mac,
        "pe_psum_adds": pe_psum_add,
        "vec_total_ops": vec_total_ops,
        "vec_reduce_ops": vec_reduce_ops,
        "flops_micro": flops_micro,
        "bytes_transmitted": bytes_tx,
        "bytes_internal": bytes_internal,
        "arithmetic_intensity_internal": arithmetic_intensity_internal,
        "flops_algo": flops_algo,
        "bytes_algo": bytes_algo,
        "arithmetic_intensity_algo": arithmetic_intensity_algo,
        "mac_utilization": mac_utilization,
        "avg_active_pes_when_active": avg_active_pes_when_active,
        "avg_active_pes_during_compute_window": avg_active_pes_during_compute_window,
        "max_active_pes_in_any_cycle": max_active_pes_in_any_cycle,
        "throughput_float_operations_per_cycle": throughput,
        "external_bandwidth_avg_bytes_per_cycle": external_bw,
        "external_bandwidth_active_bytes_per_cycle": external_bw_active,
        "internal_bandwidth_bytes_per_cycle": internal_bw,
        "reuse_weight_internal_over_external": reuse_weight,
        "reuse_act_internal_over_external": reuse_act,
        "reuse_psum_internal_over_external": reuse_psum,
        "fp16_saturation_count": sa.saturation_count,
        "fp16_overflow_count": sa.overflow_count,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "tile": cfg.tile,
        "dtype": cfg.dtype,
        "lane_count": cfg.lane_count,
        "vls_count": cfg.vls_count,
        "spad_num_banks": cfg.spad_num_banks,
        "spad_bank_size": cfg.spad_bank_size,
        "spad_read_latency": cfg.spad_read_latency,
        "spad_write_latency": cfg.spad_write_latency,
        "spad_frontend_queue_size": cfg.spad_frontend_queue_size,
        "backend_dram_latency": cfg.backend_dram_latency,
        "backend_dram_q_depth": cfg.backend_dram_q_depth,
        "backend_dram_burst_bytes": cfg.backend_dram_burst_bytes,
        "backend_delay_cycles": cfg.backend_delay_cycles,
        "backend_count": len(backends),
        "backend_shared_dram": len({id(backend_obj.dram) for backend_obj in backends}) <= 1,
        "backend_slots_attached": sum(1 for slot in spad.get_stats()["backend_slots"] if slot["attached"]),
        "dram_block_bytes": cfg.dram_block_bytes,
        "valid_mac_cycles": sa.valid_mac_cycles,
        "compute_window_cycles": sa.compute_window_cycles,
        "vls_active_cycles": state["vls_active_cycles"],
        "preload_done_cycle_count": phase_counts["weight_preload"],
        "config": cfg.to_dict(),
        "vec_op_counts": vec_op_counts,
    }

    for phase_name in PHASE_ORDER:
        result[f"phase_{phase_name}"] = phase_counts.get(phase_name, 0)

    if q_stats["samples"]:
        avg_depths = {k: (v / q_stats["samples"]) for k, v in q_stats["sum"].items()}
    else:
        avg_depths = {name: 0.0 for name in QUEUE_NAMES}
    for queue_name in QUEUE_NAMES:
        result[f"queue_max_{queue_name}"] = q_stats["max"].get(queue_name, 0)
        result[f"queue_avg_{queue_name}"] = avg_depths.get(queue_name, 0.0)

    return result


def run_sysarr_tpu_experiment(config: SysArrTPUExperimentConfig) -> Dict[str, object]:
    """Run one experiment and return its results as a dict (see
    _experiment_result for the keys). Raises if the outputs are wrong or the
    run takes more than config.max_cycles.
    """
    cfg = config.normalize()

    tile = cfg.tile
    row_bytes = tile * 2

    platform = build_tpu_platform(
        size=tile,
        dtype=cfg.dtype,
        lane_count=cfg.lane_count,
        vls_count=cfg.vls_count,
        spad_num_banks=cfg.spad_num_banks,
        spad_bank_size=cfg.spad_bank_size,
        spad_read_latency=cfg.spad_read_latency,
        spad_write_latency=cfg.spad_write_latency,
        spad_frontend_queue_size=cfg.spad_frontend_queue_size,
        dram_block_bytes=cfg.dram_block_bytes,
        backend_dram_latency=cfg.backend_dram_latency,
        backend_dram_q_depth=cfg.backend_dram_q_depth,
        backend_dram_burst_bytes=cfg.backend_dram_burst_bytes,
        backend_delay_cycles=cfg.backend_delay_cycles,
        mirror=True,
        vls_bridge_cls=VLSFrontendBridge,
    )
    clk = platform.clk
    sim = platform.sim
    vc = platform.vc
    spad = platform.spad
    sa = platform.sa
    dram = platform.dram
    backend = platform.backend
    backends = platform.backends
    vls_bridge = platform.vls_bridge
    sysarr_bridge = platform.sysarr_bridge
    mirror = platform.mirror
    if backend is None:
        raise ValueError("run_sysarr_tpu_experiment requires an attached backend")

    DRAM_ACT = 0x1000
    DRAM_WGT = 0x2000
    DRAM_OUT = 0x3000
    SPAD_ACT_BASE = 0
    SPAD_WGT_BASE = tile
    SPAD_OUT_BASE = tile * 2

    act = _act_u16(tile)
    wgt = _weights_u16(tile)
    wgt_stream = [[wgt[r][c] for r in range(tile)] for c in range(tile - 1, -1, -1)]
    expected_ref = _tpu_reference_output(act, wgt_stream, size=tile, dtype=cfg.dtype)

    _write_dram_tile_u16(dram, DRAM_ACT, act)
    _write_dram_tile_u16(dram, DRAM_WGT, wgt_stream)

    W_REG = 1
    A_REG = 2
    OUT_REG = 3

    store_commit_timeout = max(1000, cfg.max_cycles)

    state = {
        "cycles": 0,
        "weight_row": 0,
        "weight_issue_row": 0,
        "act_row": 0,
        "act_issue_row": 0,
        "next_out_row": 0,
        "weight_loads_inflight": 0,
        "act_loads_inflight": 0,
        "pending_output_rows": [],
        "store_inflight": {},
        "completed_rows": set(),
        "weights_done": False,
        "bytes_load_wgt": 0,
        "bytes_load_act": 0,
        "bytes_store_out": 0,
        "vls_active_cycles": 0,
        "preload_done": False,
        "preload_wait_drain": False,
        "preload_tx_done": set(),
        "backend_store_rows": set(),
        "backend_store_tx_done": set(),
        "backend_store_txs": {},
        "load_issue_window": spad.tiles[0].depth,
        "store_issue_window": spad.tiles[0].depth,
    }
    q_stats = {"samples": 0, "max": {}, "sum": {}}
    phase_counts = {name: 0 for name in PHASE_ORDER}
    observed_rows = [None for _ in range(tile)]

    def _on_preload_done(name: str):
        state["preload_tx_done"].add(name)
        if len(state["preload_tx_done"]) == 2:
            state["preload_wait_drain"] = True

    def _spad_write_path_idle() -> bool:
        return spad.write_path_idle()

    def _on_backend_store_done(row_idx: int, tx_id: int) -> None:
        del tx_id
        state["backend_store_tx_done"].add(row_idx)
        state["backend_store_txs"].pop(row_idx, None)

    assert (
        backend.driver_to_backend_start_load(
            base_sp_addr=SPAD_ACT_BASE,
            base_dram_addr=DRAM_ACT,
            rows=tile,
            cols=tile,
            callback=lambda _tx: _on_preload_done("act"),
        )
        > 0
    )
    assert (
        backend.driver_to_backend_start_load(
            base_sp_addr=SPAD_WGT_BASE,
            base_dram_addr=DRAM_WGT,
            rows=tile,
            cols=tile,
            callback=lambda _tx: _on_preload_done("wgt"),
        )
        > 0
    )

    def _issue_load(kind: str) -> bool:
        """Ask VLSU 0 for the next "weight" or "act" (activation) row of the
        tile; False once every row has been asked for."""
        if state[kind + "_issue_row"] >= tile:
            return False
        reg, base = (W_REG, SPAD_WGT_BASE) if kind == "weight" else (A_REG, SPAD_ACT_BASE)
        assert vc.enqueue_memory(
            {
                "kind": "load",
                "vls": 0,
                "dst": reg,
                "addr": base + state[kind + "_issue_row"],
                "dtype": cfg.dtype,
            }
        )
        state[kind + "_issue_row"] += 1
        state[kind + "_loads_inflight"] += 1
        if kind == "act" and state["act_issue_row"] >= tile:
            sysarr_bridge.finish_inputs()
        return True

    def _check_preload_drained():
        """Mark the preload done once both DMA loads finished and drained."""
        if not state["preload_done"] and state["preload_wait_drain"] and len(state["preload_tx_done"]) == 2:
            bstats = backend.get_stats()
            if (
                bstats["queued_txs"] == 0
                and bstats["outstanding_txs"] == 0
                and bstats["dram_pending"] == 0
                and _spad_write_path_idle()
            ):
                state["preload_done"] = True

    def _handle_writeback():
        """Route this cycle's vector-core writeback: a weight row goes to the
        array as weights, an activation row as an input, a GSAU result is
        queued to be stored."""
        if vc.wb_valid and vc.last_wb is not None:
            wb = vc.last_wb
            src = wb.get("source")
            dst = wb.get("dst")

            if src == "vlsu" and dst == W_REG and state["weight_loads_inflight"] > 0:
                state["weight_loads_inflight"] -= 1
                wdata = list(wb.get("data", []))
                state["bytes_load_wgt"] += len(wdata) * 2
                assert vc.enqueue_scheduler_instruction(
                    {
                        "unit": "gsau",
                        "vdata": wdata,
                        "is_weight": True,
                        "expect_output": False,
                        "dtype": cfg.dtype,
                    }
                )
                state["weight_row"] += 1
                if state["weight_row"] >= tile:
                    state["weights_done"] = True

            elif src == "vlsu" and dst == A_REG and state["act_loads_inflight"] > 0:
                state["act_loads_inflight"] -= 1
                adata = list(wb.get("data", []))
                state["bytes_load_act"] += len(adata) * 2
                assert vc.enqueue_scheduler_instruction(
                    {
                        "unit": "gsau",
                        "vdata": adata,
                        "dst": OUT_REG,
                        "is_weight": False,
                        "expect_output": True,
                        "dtype": cfg.dtype,
                    }
                )
                state["act_row"] += 1

            elif src == "gsau" and dst == OUT_REG:
                row_idx = state["next_out_row"]
                if row_idx < tile:
                    if observed_rows[row_idx] is None:
                        observed_rows[row_idx] = [int(x) for x in list(wb.get("data", []))[:tile]]
                    state["pending_output_rows"].append({"row": row_idx, "data": list(wb.get("data", []))})
                    state["next_out_row"] += 1

    def _issue_loads():
        """Ask for more weight rows, then (once all weights are in) activation
        rows, up to the scratchpad's issue window."""
        if state["preload_done"]:
            while state["weight_issue_row"] < tile and state["weight_loads_inflight"] < state["load_issue_window"]:
                if not _issue_load("weight"):
                    break

        if state["preload_done"] and state["weights_done"]:
            while state["act_issue_row"] < tile and state["act_loads_inflight"] < state["load_issue_window"]:
                if not _issue_load("act"):
                    break

    def _queue_stores():
        """Send finished output rows to the scratchpad, up to its window."""
        while state["pending_output_rows"] and len(state["store_inflight"]) < state["store_issue_window"]:
            next_item = state["pending_output_rows"].pop(0)
            row_idx = next_item["row"]
            row_data = next_item["data"]
            assert vc.enqueue_memory(
                {
                    "kind": "store",
                    "vls": 0,
                    "data": row_data,
                    "addr": SPAD_OUT_BASE + row_idx,
                    "dtype": cfg.dtype,
                }
            )
            state["store_inflight"][row_idx] = {"data": row_data, "age": 0}

    def _commit_stores():
        """Once a row is in the scratchpad, start its DMA store to DRAM."""
        completed_store_rows = []
        for row_idx, store_meta in list(state["store_inflight"].items()):
            spad_vec = _read_slot_vector_u16(spad, SPAD_OUT_BASE + row_idx, vc.vector_len)
            store_meta["age"] += 1
            if spad_vec == (store_meta["data"] or []):
                if row_idx not in state["backend_store_rows"]:
                    tx_id = backend.driver_to_backend_start_store(
                        base_sp_addr=SPAD_OUT_BASE + row_idx,
                        base_dram_addr=DRAM_OUT + row_idx * row_bytes,
                        rows=1,
                        cols=tile,
                        callback=lambda _tx, row=row_idx: _on_backend_store_done(row, _tx),
                    )
                    assert tx_id > 0
                    state["backend_store_rows"].add(row_idx)
                    state["backend_store_txs"][row_idx] = tx_id
                completed_store_rows.append(row_idx)
            elif store_meta["age"] > store_commit_timeout:
                raise AssertionError("store did not commit to scratchpad")
        for row_idx in completed_store_rows:
            state["store_inflight"].pop(row_idx, None)

    def _verify_dram_rows():
        """Count the rows whose DMA store reached DRAM with the right data."""
        for row_idx in sorted(state["backend_store_tx_done"] - state["completed_rows"]):
            dram_row = _read_dram_row_u16(dram, DRAM_OUT + row_idx * row_bytes, tile)
            expected_row = observed_rows[row_idx] if row_idx < len(observed_rows) else None
            if expected_row is not None and dram_row == expected_row[:tile]:
                state["bytes_store_out"] += len(dram_row) * 2
                state["completed_rows"].add(row_idx)

    def _sample_queues():
        """Record how full every queue is this cycle."""
        q_depths = _queue_depths(vc)
        q_stats["samples"] += 1
        for name, depth in q_depths.items():
            q_stats["sum"][name] = q_stats["sum"].get(name, 0) + depth
            q_stats["max"][name] = max(q_stats["max"].get(name, 0), depth)

    class ExperimentHarness(Clocked):
        def __init__(self):
            super().__init__()
            self.done = False

        def _finish(self) -> None:
            self.done = True
            clk.stop()

        def tick(self, time: float) -> None:
            if self.done:
                return

            _check_preload_drained()
            phase_counts[_phase_snapshot(state, sa, tile)] += 1
            if vls_bridge.activity_this_cycle:
                state["vls_active_cycles"] += 1
            _handle_writeback()
            _issue_loads()
            _queue_stores()
            _commit_stores()
            _verify_dram_rows()
            _sample_queues()
            state["cycles"] += 1
            if len(state["completed_rows"]) >= tile:
                self._finish()
                return
            if state["cycles"] >= cfg.max_cycles:
                self._finish()

    harness = ExperimentHarness()
    backend_ticker = RoundRobinBackendTicker(backends)
    clk.objects = [vc, vls_bridge, sysarr_bridge, backend_ticker, spad, harness]
    clk.schedule_next(0.0)
    sim.run()

    if len(state["completed_rows"]) < tile:
        raise AssertionError("timed out waiting for DRAM->SPAD->VRF->GSAU->TPU->VRF->SPAD->DRAM")

    got = _read_dram_tile_u16(dram, DRAM_OUT, tile, tile)
    expected_cycle = mirror.outputs
    if len(expected_cycle) != tile:
        raise AssertionError("mirror did not produce the expected number of rows")
    if got != expected_cycle:
        raise AssertionError("TPU output mismatch against mirror reference")

    return _experiment_result(cfg, state, phase_counts, q_stats, platform, got,
                              expected_ref, expected_cycle)


def _build_default_cli_config() -> SysArrTPUExperimentConfig:
    return SysArrTPUExperimentConfig(
        name="default_tpu_experiment",
        sweep="manual",
        param_name="none",
        param_value="none",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TPU system-array experiment.")
    parser.add_argument("--tile", type=int, default=32)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--lane-count", type=int, default=4)
    parser.add_argument("--vls-count", type=int, default=1)
    parser.add_argument("--max-cycles", type=int, default=20000)
    parser.add_argument("--backend-dram-latency", type=int, default=24)
    parser.add_argument("--backend-dram-q-depth", type=int, default=16)
    parser.add_argument("--backend-delay-cycles", type=int, default=1)
    args = parser.parse_args()

    cfg = _build_default_cli_config()
    cfg.tile = args.tile
    cfg.dtype = args.dtype
    cfg.lane_count = args.lane_count
    cfg.vls_count = args.vls_count
    cfg.max_cycles = args.max_cycles
    cfg.backend_dram_latency = args.backend_dram_latency
    cfg.backend_dram_q_depth = args.backend_dram_q_depth
    cfg.backend_delay_cycles = args.backend_delay_cycles

    result = run_sysarr_tpu_experiment(cfg)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
