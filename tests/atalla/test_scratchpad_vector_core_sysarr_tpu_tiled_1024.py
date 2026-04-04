import os
import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from atalla.sysarr_tpu_experiment import MetricsVLSFrontendBridge, QUEUE_NAMES
from atalla.sysarr_tpu_system import GSAUTPUBridge
from base.debug import close_debug, configure_debug, dprintf
from memory.backend import Backend
from memory.dram import DRAM
from memory.sc_sram_banks import _xor_bank
from memory.scratchpad import Scratchpad
from systolic_array.systolic_array_tpu import SystolicArrayTPU
from vector_core.vector_core import VectorCore


TILE = 32
MATRIX = 1024
NUM_TILES = MATRIX // TILE


def _act_matrix_u16(size: int) -> np.ndarray:
    rows = np.arange(size, dtype=np.int32)[:, None]
    cols = np.arange(size, dtype=np.int32)[None, :]
    return ((cols * size + rows) % 3 + 1).astype(np.int32)


def _weights_matrix_u16(size: int) -> np.ndarray:
    rows = np.arange(size, dtype=np.int32)[:, None]
    cols = np.arange(size, dtype=np.int32)[None, :]
    return ((rows * size + cols) % 5 + 1).astype(np.int32)


def _tile_weight_stream(tile: np.ndarray) -> List[List[int]]:
    size = int(tile.shape[0])
    return [[int(tile[r, c]) for r in range(size)] for c in range(size - 1, -1, -1)]


def _encode_row_u16(row: List[int]) -> bytes:
    return b"".join(int(v).to_bytes(2, "little", signed=False) for v in row)


def _decode_row_u16(blob: bytes, cols: int) -> List[int]:
    out = []
    for i in range(int(cols)):
        start = i * 2
        out.append(int.from_bytes(blob[start : start + 2], "little", signed=False))
    return out


def _write_dram_tile_u16(dram: DRAM, base_addr: int, tile: List[List[int]]) -> None:
    rows = len(tile)
    cols = len(tile[0]) if rows else 0
    row_bytes = cols * 2
    for r in range(rows):
        dram.write(int(base_addr) + r * row_bytes, _encode_row_u16(tile[r]))


def _read_dram_row_u16(dram: DRAM, addr: int, cols: int) -> List[int]:
    row_bytes = cols * 2
    blob = dram.read(int(addr), row_bytes)
    return _decode_row_u16(blob, cols)


def _fp16_bits(value: float) -> int:
    return int(np.asarray(value, dtype=np.float16).view(np.uint16).item())


def _read_slot_vector_u16(spad: Scratchpad, addr: int, vector_len: int) -> List[int]:
    tile = 0
    slot = int(addr) % spad.bank_size
    out = []
    for lane in range(vector_len):
        bank = _xor_bank(slot, lane, spad.num_banks)
        blob = spad.tiles[tile].banks[bank].mem[slot]
        blob = bytes(blob) if blob is not None else b"\x00\x00"
        if len(blob) < 2:
            blob = blob + (b"\x00" * (2 - len(blob)))
        out.append(int.from_bytes(blob[:2], "little", signed=False))
    return out


def _write_slot_vector_u16(spad: Scratchpad, addr: int, values: List[int]) -> None:
    tile = 0
    slot = int(addr) % spad.bank_size
    padded = list(values)
    for lane in range(len(padded), spad.num_banks):
        padded.append(0)
    for lane, value in enumerate(padded[: spad.num_banks]):
        bank = _xor_bank(slot, lane, spad.num_banks)
        spad.tiles[tile].banks[bank].mem[slot] = int(value).to_bytes(2, "little", signed=False)


def _fp16_from_u16(value: int) -> float:
    return float(np.frombuffer(np.uint16(int(value)).tobytes(), dtype=np.float16)[0])


def _decode_fp16_vector(values: List[int]) -> List[float]:
    return [_fp16_from_u16(value) for value in values]


def _encode_fp16_vector(values: List[float]) -> List[int]:
    return [_fp16_bits(value) for value in values]


def _expected_tiled_fp16_accum(act: np.ndarray, wgt: np.ndarray, tile_size: int) -> np.ndarray:
    matrix_size = int(act.shape[0])
    num_tiles = matrix_size // int(tile_size)
    accum = np.zeros((matrix_size, matrix_size), dtype=np.float16)
    for tk in range(num_tiles):
        k_slice = slice(tk * tile_size, (tk + 1) * tile_size)
        partial = (act[:, k_slice] @ wgt[k_slice, :]).astype(np.float16)
        accum = (accum + partial).astype(np.float16)
    return np.rint(accum).astype(np.int32)


def _summarize_cycle_values(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "total": 0, "min": 0, "max": 0, "avg": 0.0}
    total = int(sum(values))
    return {
        "count": len(values),
        "total": total,
        "min": int(min(values)),
        "max": int(max(values)),
        "avg": float(total / len(values)),
    }


@dataclass
class TileJob:
    ti: int
    tj: int
    tk: int
    slot: int
    act_tile: List[List[int]]
    wgt_stream: List[List[int]]
    preload_tx_done: set = field(default_factory=set)
    preload_ready: bool = False
    compute_started: bool = False
    weight_row: int = 0
    weight_issue_row: int = 0
    weight_loads_inflight: int = 0
    weights_done: bool = False
    act_row: int = 0
    act_issue_row: int = 0
    act_loads_inflight: int = 0
    next_out_row: int = 0
    pending_accum_rows: List[Dict] = field(default_factory=list)
    accum_loads_inflight: int = 0
    accum_load_rows: Dict[int, Dict] = field(default_factory=dict)
    pending_sum_rows: List[Dict] = field(default_factory=list)
    sum_issue_rows: List[int] = field(default_factory=list)
    store_inflight: Dict[int, Dict] = field(default_factory=dict)
    completed_rows: set = field(default_factory=set)
    cycle_start: Optional[int] = None
    cycle_done: Optional[int] = None


class TiledTPUCosim:
    def __init__(self, matrix_size: int = MATRIX, tile_size: int = TILE, dtype: str = "fp16"):
        self.matrix_size = int(matrix_size)
        self.tile_size = int(tile_size)
        self.num_tiles = self.matrix_size // self.tile_size
        assert self.matrix_size % self.tile_size == 0
        self.dtype = str(dtype)

        self.W_REG = 1
        self.A_REG = 2
        self.OUT_REG = 3
        self.ACC_REG = 4
        self.SUM_REG = 5

        self.ACT_SLOT_BASES = [0, self.tile_size]
        self.WGT_SLOT_BASES = [2 * self.tile_size, 3 * self.tile_size]
        self.ACC_SLOT_BASE = 6 * self.tile_size

        self.DRAM_ACT_STAGE = [0x100000, 0x110000]
        self.DRAM_WGT_STAGE = [0x200000, 0x210000]
        self.DRAM_OUT = 0x300000

        self.global_cycle = 0
        self.timeline: List[str] = []
        self.total_tile_pairs = 0
        self.progress_total = 32_768
        self.output_gemms_completed = 0
        self.output_gemms_total = self.matrix_size * self.matrix_size
        self._init_metrics()

        self._init_platform()

    def _init_metrics(self) -> None:
        self.bytes_load_wgt = 0
        self.bytes_load_act = 0
        self.bytes_store_out = 0
        self.vls_active_cycles = 0
        self.gemm_cycle_records: List[Dict[str, int]] = []
        self.sdma_load_cycle_records: List[Dict[str, object]] = []
        self.queue_samples = 0
        self.queue_depth_sums = {name: 0 for name in QUEUE_NAMES}
        self.queue_depth_max = {name: 0 for name in QUEUE_NAMES}
        self.sa_totals = {
            "pe_mul_ops": 0,
            "pe_add_ops": 0,
            "pe_mac_ops": 0,
            "pe_psum_adds": 0,
            "bytes_internal": 0,
            "valid_mac_cycles": 0,
            "active_pe_sum": 0,
            "compute_window_cycles": 0,
            "compute_window_active_pe_sum": 0,
            "max_active_pes_in_any_cycle": 0,
            "fp16_saturation_count": 0,
            "fp16_overflow_count": 0,
            "internal_weight_bytes": 0,
            "internal_act_bytes": 0,
            "internal_psum_bytes": 0,
        }

    def _record_sdma_load_completion(self, job: TileJob, kind: str, launch_cycle: int) -> None:
        job.preload_tx_done.add(kind)
        self.sdma_load_cycle_records.append(
            {
                "ti": job.ti,
                "tj": job.tj,
                "tk": job.tk,
                "slot": job.slot,
                "kind": kind,
                "cycles": int(self.global_cycle - launch_cycle),
            }
        )

    def _sample_queue_depths(self) -> None:
        vlsu0 = self.vc.vls_units[0]
        q_depths = {
            "gsau_to_systolic": len(self.vc.gsau.to_systolic),
            "gsau_from_systolic": len(self.vc.gsau.from_systolic),
            "gsau_rd_queue": len(self.vc.gsau.rd_queue),
            "gsau_writebacks": len(self.vc.gsau.writebacks),
            "scheduler_packets": len(self.vc.scheduler_packets),
            "scheduler_build_gsau": len(self.vc._build_packet["gsau"]),
            "scheduler_build_vlsu": len(self.vc._build_packet["vlsu"]),
            "scheduler_build_datapath": len(self.vc._build_packet["datapath"]),
            "scheduler_packet_gsau": sum(len(pkt["gsau"]) for pkt in self.vc.vliw_q.items),
            "scheduler_packet_vlsu": sum(len(pkt["vlsu"]) for pkt in self.vc.vliw_q.items),
            "scheduler_packet_datapath": sum(len(pkt["datapath"]) for pkt in self.vc.vliw_q.items),
            "wb_buffer": len(self.vc.wb_buffer.entries),
            "vlsu_issue_q": len(vlsu0.issue_q),
            "vlsu_req_q": len(vlsu0.req_q),
            "vlsu_rsp_q": len(vlsu0.rsp_q),
            "vlsu_wb_q": len(vlsu0.wb_q),
            "vlsu_dst_fifo": len(vlsu0.load_dst_fifos[0]),
        }
        self.queue_samples += 1
        for name, depth in q_depths.items():
            self.queue_depth_sums[name] += depth
            self.queue_depth_max[name] = max(self.queue_depth_max[name], depth)

    def _accumulate_sa_metrics(self) -> None:
        self.sa_totals["pe_mul_ops"] += self.sa.metrics["mul_ops"]
        self.sa_totals["pe_add_ops"] += self.sa.metrics["add_ops"]
        self.sa_totals["pe_mac_ops"] += self.sa.metrics["mac_ops"]
        self.sa_totals["pe_psum_adds"] += self.sa.metrics["psum_adds"]
        self.sa_totals["bytes_internal"] += self.sa.internal_bytes_valid_total()
        self.sa_totals["valid_mac_cycles"] += self.sa.valid_mac_cycles
        self.sa_totals["active_pe_sum"] += self.sa.active_pe_sum
        self.sa_totals["compute_window_cycles"] += self.sa.compute_window_cycles
        self.sa_totals["compute_window_active_pe_sum"] += self.sa.compute_window_active_pe_sum
        self.sa_totals["max_active_pes_in_any_cycle"] = max(
            self.sa_totals["max_active_pes_in_any_cycle"],
            self.sa.max_active_pes_in_cycle,
        )
        self.sa_totals["fp16_saturation_count"] += self.sa.saturation_count
        self.sa_totals["fp16_overflow_count"] += self.sa.overflow_count
        self.sa_totals["internal_weight_bytes"] += self.sa.internal_bytes_valid["weight_shift"]
        self.sa_totals["internal_act_bytes"] += self.sa.internal_bytes_valid["act_shift"]
        self.sa_totals["internal_psum_bytes"] += self.sa.internal_bytes_valid["psum_shift"]

    def build_stats(self, got: np.ndarray, expected: Optional[np.ndarray] = None) -> Dict[str, object]:
        vec_total_ops = sum(lane.total_ops for lane in self.vc.datapath.lanes)
        vec_op_counts: Dict[str, int] = {}
        for lane in self.vc.datapath.lanes:
            for op, count in lane.op_counts.items():
                vec_op_counts[op] = vec_op_counts.get(op, 0) + count
        vec_reduce_ops = self.vc.datapath.collector.reduction_unit.reduce_ops
        gemm_total_cycles = [int(record["cycles"]) for record in self.gemm_cycle_records]
        sdma_load_cycles = [int(record["cycles"]) for record in self.sdma_load_cycle_records]
        sa_compute_cycles = [int(record["systolic_array_compute_cycles"]) for record in self.gemm_cycle_records]
        sa_valid_mac_cycles = [int(record["systolic_array_valid_mac_cycles"]) for record in self.gemm_cycle_records]

        pe_mul = self.sa_totals["pe_mul_ops"]
        pe_add = self.sa_totals["pe_add_ops"]
        pe_mac = self.sa_totals["pe_mac_ops"]
        pe_psum_adds = self.sa_totals["pe_psum_adds"]
        bytes_internal = self.sa_totals["bytes_internal"]
        flops_micro = pe_mul + pe_add + vec_total_ops + vec_reduce_ops
        arithmetic_intensity_internal = (flops_micro / bytes_internal) if bytes_internal else 0.0

        valid_mac_cycles = self.sa_totals["valid_mac_cycles"]
        compute_window_cycles = self.sa_totals["compute_window_cycles"]
        mac_utilization = (
            self.sa_totals["active_pe_sum"] / (valid_mac_cycles * self.tile_size * self.tile_size)
        ) if valid_mac_cycles else 0.0
        avg_active_pes_when_active = (
            self.sa_totals["active_pe_sum"] / valid_mac_cycles
        ) if valid_mac_cycles else 0.0
        avg_active_pes_during_compute_window = (
            self.sa_totals["compute_window_active_pe_sum"] / compute_window_cycles
        ) if compute_window_cycles else 0.0

        bytes_transmitted = self.vls_bridge.bytes_load + self.vls_bridge.bytes_store
        throughput = (flops_micro / self.global_cycle) if self.global_cycle else 0.0
        external_bw = (bytes_transmitted / self.global_cycle) if self.global_cycle else 0.0
        external_bw_active = (bytes_transmitted / self.vls_active_cycles) if self.vls_active_cycles else 0.0
        internal_bw = (bytes_internal / self.global_cycle) if self.global_cycle else 0.0
        reuse_weight = (
            self.sa_totals["internal_weight_bytes"] / self.bytes_load_wgt
        ) if self.bytes_load_wgt else 0.0
        reuse_act = (
            self.sa_totals["internal_act_bytes"] / self.bytes_load_act
        ) if self.bytes_load_act else 0.0
        reuse_psum = (
            self.sa_totals["internal_psum_bytes"] / self.bytes_store_out
        ) if self.bytes_store_out else 0.0

        if self.queue_samples:
            queue_avg_depths = {
                name: self.queue_depth_sums[name] / self.queue_samples for name in QUEUE_NAMES
            }
        else:
            queue_avg_depths = {name: 0.0 for name in QUEUE_NAMES}
        queue_max_depths = {name: self.queue_depth_max[name] for name in QUEUE_NAMES}

        if expected is None:
            max_abs_error = 0.0
            mean_abs_error = 0.0
        else:
            abs_err = np.abs(got.astype(np.float64) - expected.astype(np.float64))
            max_abs_error = float(np.max(abs_err)) if abs_err.size else 0.0
            mean_abs_error = float(np.mean(abs_err)) if abs_err.size else 0.0

        return {
            "cycles": self.global_cycle,
            "matrix_total_cycles": self.global_cycle,
            "gemm_total_cycles_summary": _summarize_cycle_values(gemm_total_cycles),
            "sdma_load_cycles_summary": _summarize_cycle_values(sdma_load_cycles),
            "systolic_array_compute_cycles_summary": _summarize_cycle_values(sa_compute_cycles),
            "systolic_array_valid_mac_cycles_summary": _summarize_cycle_values(sa_valid_mac_cycles),
            "pe_mul_ops": pe_mul,
            "pe_add_ops": pe_add,
            "pe_mac_ops": pe_mac,
            "pe_psum_adds": pe_psum_adds,
            "vec_total_ops": vec_total_ops,
            "vec_op_counts": vec_op_counts,
            "vec_reduce_ops": vec_reduce_ops,
            "flops_micro": flops_micro,
            "bytes_transmitted": bytes_transmitted,
            "bytes_internal": bytes_internal,
            "arithmetic_intensity_internal": arithmetic_intensity_internal,
            "mac_utilization": mac_utilization,
            "avg_active_pes_when_active": avg_active_pes_when_active,
            "avg_active_pes_during_compute_window": avg_active_pes_during_compute_window,
            "max_active_pes_in_any_cycle": self.sa_totals["max_active_pes_in_any_cycle"],
            "throughput_float_operations_per_cycle": throughput,
            "external_bandwidth_avg_bytes_per_cycle": external_bw,
            "external_bandwidth_active_bytes_per_cycle": external_bw_active,
            "internal_bandwidth_bytes_per_cycle": internal_bw,
            "reuse_weight_internal_over_external": reuse_weight,
            "reuse_act_internal_over_external": reuse_act,
            "reuse_psum_internal_over_external": reuse_psum,
            "queue_max_depths": queue_max_depths,
            "queue_avg_depths": queue_avg_depths,
            "fp16_saturation_count": self.sa_totals["fp16_saturation_count"],
            "fp16_overflow_count": self.sa_totals["fp16_overflow_count"],
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "matrix_size": self.matrix_size,
            "tile_size": self.tile_size,
            "num_tiles": self.num_tiles,
            "prefetch_slots": 2,
            "total_tile_pairs": self.total_tile_pairs,
            "total_output_tiles": self.num_tiles ** 2,
        }

    def _init_platform(self) -> None:
        self.vc = VectorCore(
            veggie_size=self.tile_size * 16,
            lane_count=4,
            vls_count=1,
            fu_latencies={"alu": 1},
            dtype=self.dtype,
        )
        self.spad = Scratchpad(
            num_banks=32,
            bank_size=512,
            read_latency=2,
            write_latency=2,
            xbar_delay=3,
            elem_bytes=2,
            frontend_queue_size=4,
        )
        self.backend = Backend(dram_latency=24, dram_q_depth=16, dram_burst_bytes=32, elem_bytes=2)
        self.dram = DRAM(block_bytes=256)
        self.spad.attach_backend(self.backend)
        self.backend.attach_dram(self.dram)
        self.vls_bridge = MetricsVLSFrontendBridge(self.vc, self.spad, vls_id=0, frontend_id=0)
        self.load_issue_window = self.spad.frontends[0].readq.max_size + 1
        self.store_issue_window = self.spad.frontends[0].writeq.max_size + 1
        self.tile_row_bytes = self.tile_size * 2
        self._reset_compute_pipeline()

    def _reset_compute_pipeline(self) -> None:
        self.sa = SystolicArrayTPU(size=self.tile_size, dtype=self.dtype)
        self.sysarr_bridge = GSAUTPUBridge(self.vc, self.sa, mirror=None)

    def _slot_ready(self, base_addr: int, expected_rows: List[List[int]]) -> bool:
        for row_idx, expected in enumerate(expected_rows):
            got = _read_slot_vector_u16(self.spad, base_addr + row_idx, self.vc.vector_len)
            if got[: self.tile_size] != list(expected):
                return False
        return True

    def _clear_accumulator_tile(self) -> None:
        zero_vec = [0] * self.vc.vector_len
        for row_idx in range(self.tile_size):
            _write_slot_vector_u16(self.spad, self.ACC_SLOT_BASE + row_idx, zero_vec)
            self.dram.write(
                self.DRAM_OUT + row_idx * self.tile_row_bytes,
                b"\x00" * self.tile_row_bytes,
            )

    def _launch_prefetch(self, job: TileJob) -> None:
        launch_cycle = self.global_cycle
        _write_dram_tile_u16(self.dram, self.DRAM_ACT_STAGE[job.slot], job.act_tile)
        _write_dram_tile_u16(self.dram, self.DRAM_WGT_STAGE[job.slot], job.wgt_stream)
        assert self.backend.driver_to_backend_start_load(
            base_sp_addr=self.ACT_SLOT_BASES[job.slot],
            base_dram_addr=self.DRAM_ACT_STAGE[job.slot],
            rows=self.tile_size,
            cols=self.tile_size,
            callback=lambda _tx, _job=job, _cycle=launch_cycle: self._record_sdma_load_completion(
                _job, "act", _cycle
            ),
        ) > 0
        assert self.backend.driver_to_backend_start_load(
            base_sp_addr=self.WGT_SLOT_BASES[job.slot],
            base_dram_addr=self.DRAM_WGT_STAGE[job.slot],
            rows=self.tile_size,
            cols=self.tile_size,
            callback=lambda _tx, _job=job, _cycle=launch_cycle: self._record_sdma_load_completion(
                _job, "wgt", _cycle
            ),
        ) > 0
        self.timeline.append(
            f"cycle {self.global_cycle}: prefetch launch ti={job.ti} tj={job.tj} tk={job.tk} slot={job.slot}"
        )

    def _issue_weight_load(self, job: TileJob) -> bool:
        if job.weight_issue_row >= self.tile_size:
            return False
        assert self.vc.enqueue_memory(
            {
                "kind": "load",
                "vls": 0,
                "dst": self.W_REG,
                "addr": self.WGT_SLOT_BASES[job.slot] + job.weight_issue_row,
                "dtype": self.dtype,
            }
        )
        job.weight_issue_row += 1
        job.weight_loads_inflight += 1
        return True

    def _issue_act_load(self, job: TileJob) -> bool:
        if job.act_issue_row >= self.tile_size:
            return False
        assert self.vc.enqueue_memory(
            {
                "kind": "load",
                "vls": 0,
                "dst": self.A_REG,
                "addr": self.ACT_SLOT_BASES[job.slot] + job.act_issue_row,
                "dtype": self.dtype,
            }
        )
        job.act_issue_row += 1
        job.act_loads_inflight += 1
        if job.act_issue_row >= self.tile_size:
            self.sysarr_bridge.finish_inputs()
        return True

    def _step(self) -> Optional[Dict]:
        self.spad.now = self.global_cycle
        self.vls_bridge.start_cycle()
        self.vc.tick()
        self.vls_bridge.tick()
        self.sysarr_bridge.tick()
        self.backend.tick(self.global_cycle)
        self.spad.tick(self.global_cycle)
        if self.vls_bridge.activity_this_cycle:
            self.vls_active_cycles += 1
        self._sample_queue_depths()
        wb = self.vc.last_wb if self.vc.wb_valid else None
        self.global_cycle += 1
        return wb

    def _datapath_idle(self) -> bool:
        if len(self.vc.datapath.pending_issue) > 0:
            return False
        if self.vc.datapath.result_valid or self.vc._datapath_wb_hold is not None:
            return False
        if self.vc.datapath.collector.inflight:
            return False
        if len(self.vc.datapath.collector.completed_vectors) > 0:
            return False
        if len(self.vc.datapath.collector.pending_reductions) > 0:
            return False
        for lane in self.vc.datapath.lanes:
            if len(lane.pending_outputs) > 0:
                return False
            if any(ctx is not None for ctx in lane.fu_ctx.values()):
                return False
            if any(len(meta_fifo) > 0 for meta_fifo in lane.meta_fifo.values()):
                return False
            if any(len(fu.entries) > 0 or len(fu.completed) > 0 for fu in lane.fus.values()):
                return False
        return True

    def _compute_path_idle(self) -> bool:
        if len(self.vc.vliw_q) > 0:
            return False
        if any(self.vc._build_packet[key] for key in ("gsau", "vlsu", "datapath")):
            return False
        if not self._datapath_idle():
            return False
        if len(self.vc.gsau.to_systolic) > 0 or len(self.vc.gsau.from_systolic) > 0:
            return False
        if len(self.vc.gsau.rd_queue) > 0 or len(self.vc.gsau.writebacks) > 0:
            return False
        if len(self.vc.wb_buffer.entries) > 0:
            return False
        for vls in self.vc.vls_units:
            if len(vls.issue_q) > 0 or len(vls.req_q) > 0 or len(vls.rsp_q) > 0 or len(vls.wb_q) > 0:
                return False
            if any(len(fifo) > 0 for fifo in vls.load_dst_fifos):
                return False
        if self.sa._algo_out_pending > 0:
            return False
        if self.sa.value_ready or self.sa.start or self.sa.weight_en or self.sa.mac_shift:
            return False
        if self.sysarr_bridge._pending_meta:
            return False
        if self.sysarr_bridge._flush_pending > 0:
            return False
        if self.sysarr_bridge._out_read_idx < len(self.sa.get_buffer()):
            return False
        return True

    def _drain_compute_path(self, limit: int = 2000) -> None:
        waited = 0
        while not self._compute_path_idle():
            self._step()
            waited += 1
            if waited > limit:
                raise AssertionError("compute path failed to drain")

    def _drain_accumulator_tile_to_dram(self) -> np.ndarray:
        tile_out = np.zeros((self.tile_size, self.tile_size), dtype=np.int32)
        completed_rows = set()
        next_row_to_store = 0

        while len(completed_rows) < self.tile_size:
            while next_row_to_store < self.tile_size:
                row_idx = next_row_to_store
                tx_id = self.backend.driver_to_backend_start_store(
                    base_sp_addr=self.ACC_SLOT_BASE + row_idx,
                    base_dram_addr=self.DRAM_OUT + row_idx * self.tile_row_bytes,
                    rows=1,
                    cols=self.tile_size,
                    callback=lambda _tx, _row=row_idx: completed_rows.add(_row),
                )
                if tx_id <= 0:
                    break
                next_row_to_store += 1
            self._step()

        self.bytes_store_out += self.tile_size * self.tile_row_bytes

        for row_idx in range(self.tile_size):
            dram_row = _read_dram_row_u16(
                self.dram,
                self.DRAM_OUT + row_idx * self.tile_row_bytes,
                self.tile_size,
            )
            tile_out[row_idx, :] = np.rint(
                np.asarray([_fp16_from_u16(v) for v in dram_row], dtype=np.float64)
            ).astype(np.int32)
        return tile_out

    def _run_output_tile(self, ti: int, tj: int, act: np.ndarray, wgt: np.ndarray) -> np.ndarray:
        self._reset_compute_pipeline()
        self._clear_accumulator_tile()
        row_slice = slice(ti * self.tile_size, (ti + 1) * self.tile_size)
        col_slice = slice(tj * self.tile_size, (tj + 1) * self.tile_size)

        jobs_by_slot: Dict[int, TileJob] = {}
        current_job: Optional[TileJob] = None
        next_k_to_launch = 0
        completed_k = 0
        tile_start_cycle = self.global_cycle

        while completed_k < self.num_tiles:
            if self.global_cycle - tile_start_cycle > 1_000_000:
                raise AssertionError(f"output tile ti={ti} tj={tj} exceeded cycle budget")

            free_slots = [slot for slot in (0, 1) if slot not in jobs_by_slot]
            while free_slots and next_k_to_launch < self.num_tiles:
                slot = free_slots.pop(0)
                k_slice = slice(next_k_to_launch * self.tile_size, (next_k_to_launch + 1) * self.tile_size)
                job = TileJob(
                    ti=ti,
                    tj=tj,
                    tk=next_k_to_launch,
                    slot=slot,
                    act_tile=act[row_slice, k_slice].astype(np.int32).tolist(),
                    wgt_stream=_tile_weight_stream(wgt[k_slice, col_slice]),
                )
                jobs_by_slot[slot] = job
                self._launch_prefetch(job)
                next_k_to_launch += 1

            for job in list(jobs_by_slot.values()):
                if not job.preload_ready and job.preload_tx_done == {"act", "wgt"}:
                    if self._slot_ready(self.ACT_SLOT_BASES[job.slot], job.act_tile) and self._slot_ready(
                        self.WGT_SLOT_BASES[job.slot], job.wgt_stream
                    ):
                        job.preload_ready = True
                        self.timeline.append(
                            f"cycle {self.global_cycle}: preload ready ti={job.ti} tj={job.tj} tk={job.tk} slot={job.slot}"
                        )

            if current_job is None:
                ready_jobs = sorted(
                    (job for job in jobs_by_slot.values() if job.preload_ready and not job.compute_started),
                    key=lambda job: job.tk,
                )
                if ready_jobs:
                    current_job = ready_jobs[0]
                    current_job.compute_started = True
                    current_job.cycle_start = self.global_cycle
                    self.timeline.append(
                        f"cycle {self.global_cycle}: compute start ti={ti} tj={tj} tk={current_job.tk} slot={current_job.slot}"
                    )

            wb = self._step()

            if current_job is None:
                continue

            if wb is not None:
                src = wb.get("source")
                dst = wb.get("dst")

                if src == "vlsu" and dst == self.W_REG and current_job.weight_loads_inflight > 0:
                    current_job.weight_loads_inflight -= 1
                    self.bytes_load_wgt += len(list(wb.get("data", []))) * 2
                    assert self.vc.enqueue_scheduler_instruction(
                        {
                            "unit": "gsau",
                            "vdata": list(wb.get("data", [])),
                            "is_weight": True,
                            "expect_output": False,
                            "dtype": self.dtype,
                        }
                    )
                    current_job.weight_row += 1
                    if current_job.weight_row >= self.tile_size:
                        current_job.weights_done = True

                elif src == "vlsu" and dst == self.A_REG and current_job.act_loads_inflight > 0:
                    current_job.act_loads_inflight -= 1
                    self.bytes_load_act += len(list(wb.get("data", []))) * 2
                    assert self.vc.enqueue_scheduler_instruction(
                        {
                            "unit": "gsau",
                            "vdata": list(wb.get("data", [])),
                            "dst": self.OUT_REG,
                            "is_weight": False,
                            "expect_output": True,
                            "dtype": self.dtype,
                        }
                    )
                    current_job.act_row += 1

                elif src == "vlsu" and dst == self.ACC_REG and current_job.accum_loads_inflight > 0:
                    current_job.accum_loads_inflight -= 1
                    load_addr = int(wb.get("meta", {}).get("addr", -1))
                    row_idx = load_addr - self.ACC_SLOT_BASE
                    if row_idx not in current_job.accum_load_rows:
                        raise AssertionError("unexpected accumulator row load")
                    partial_bits = current_job.accum_load_rows.pop(row_idx)["data"]
                    assert self.vc.enqueue_compute(
                        op="add",
                        dst=self.SUM_REG,
                        src0=_decode_fp16_vector(partial_bits),
                        src1=_decode_fp16_vector(list(wb.get("data", []))),
                    )
                    current_job.sum_issue_rows.append(row_idx)

                elif src == "gsau" and dst == self.OUT_REG:
                    row_idx = current_job.next_out_row
                    if row_idx < self.tile_size:
                        current_job.pending_accum_rows.append({"row": row_idx, "data": list(wb.get("data", []))})
                        current_job.next_out_row += 1

                elif src == "datapath" and dst == self.SUM_REG:
                    if not current_job.sum_issue_rows:
                        raise AssertionError("unexpected accumulator datapath writeback")
                    row_idx = current_job.sum_issue_rows.pop(0)
                    current_job.pending_sum_rows.append(
                        {"row": row_idx, "data": _encode_fp16_vector(list(wb.get("data", [])))}
                    )

            while current_job.weight_issue_row < self.tile_size and current_job.weight_loads_inflight < self.load_issue_window:
                if not self._issue_weight_load(current_job):
                    break

            if current_job.weights_done:
                while current_job.act_issue_row < self.tile_size and current_job.act_loads_inflight < self.load_issue_window:
                    if not self._issue_act_load(current_job):
                        break

            while current_job.pending_accum_rows and current_job.accum_loads_inflight < self.load_issue_window:
                next_item = current_job.pending_accum_rows.pop(0)
                row_idx = next_item["row"]
                assert self.vc.enqueue_memory(
                    {
                        "kind": "load",
                        "vls": 0,
                        "dst": self.ACC_REG,
                        "addr": self.ACC_SLOT_BASE + row_idx,
                        "dtype": self.dtype,
                    }
                )
                current_job.accum_load_rows[row_idx] = {"data": list(next_item["data"]), "age": 0}
                current_job.accum_loads_inflight += 1

            while current_job.pending_sum_rows and len(current_job.store_inflight) < self.store_issue_window:
                next_item = current_job.pending_sum_rows.pop(0)
                row_idx = next_item["row"]
                row_data = next_item["data"]
                assert self.vc.enqueue_memory(
                    {
                        "kind": "store",
                        "vls": 0,
                        "data": row_data,
                        "addr": self.ACC_SLOT_BASE + row_idx,
                        "dtype": self.dtype,
                    }
                )
                current_job.store_inflight[row_idx] = {"data": row_data, "age": 0}

            for row_idx, store_meta in list(current_job.store_inflight.items()):
                spad_vec = _read_slot_vector_u16(
                    self.spad,
                    self.ACC_SLOT_BASE + row_idx,
                    self.vc.vector_len,
                )
                store_meta["age"] += 1
                if spad_vec == list(store_meta["data"]):
                    current_job.completed_rows.add(row_idx)
                    current_job.store_inflight.pop(row_idx, None)
                elif store_meta["age"] > 4000:
                    raise AssertionError("store did not commit to scratchpad")

            if len(current_job.completed_rows) >= self.tile_size:
                self.timeline.append(
                    f"cycle {self.global_cycle}: compute done ti={ti} tj={tj} tk={current_job.tk} slot={current_job.slot}"
                )
                self.total_tile_pairs += 1
                completed_k += 1
                jobs_by_slot.pop(current_job.slot, None)
                self._drain_compute_path()
                gemm_end_cycle = self.global_cycle
                current_job.cycle_done = gemm_end_cycle
                self.gemm_cycle_records.append(
                    {
                        "ti": current_job.ti,
                        "tj": current_job.tj,
                        "tk": current_job.tk,
                        "slot": current_job.slot,
                        "cycles": int(gemm_end_cycle - (current_job.cycle_start or gemm_end_cycle)),
                        "systolic_array_compute_cycles": int(self.sa.compute_window_cycles),
                        "systolic_array_valid_mac_cycles": int(self.sa.valid_mac_cycles),
                    }
                )
                self._accumulate_sa_metrics()
                current_job = None
                if completed_k < self.num_tiles:
                    self._reset_compute_pipeline()

        self._drain_compute_path()
        return self._drain_accumulator_tile_to_dram()

    def run(self, act: np.ndarray, wgt: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        got = np.zeros((self.matrix_size, self.matrix_size), dtype=np.int32)

        for ti in range(self.num_tiles):
            for tj in range(self.num_tiles):
                tile_out = self._run_output_tile(ti, tj, act, wgt)
                row_slice = slice(ti * self.tile_size, (ti + 1) * self.tile_size)
                col_slice = slice(tj * self.tile_size, (tj + 1) * self.tile_size)
                got[row_slice, col_slice] = tile_out
                self.output_gemms_completed += self.tile_size * self.tile_size
                print(
                    (
                        f"GEMM outputs completed: {self.output_gemms_completed:,}/"
                        f"{self.output_gemms_total:,} "
                        f"(tile ti={ti} tj={tj})"
                    ),
                    flush=True,
                )
                self.timeline.append(
                    f"cycle {self.global_cycle}: output tile complete ti={ti} tj={tj}"
                )

        stats = {
            "matrix_size": self.matrix_size,
            "tile_size": self.tile_size,
            "num_tiles": self.num_tiles,
            "prefetch_slots": 2,
            "total_tile_pairs": self.total_tile_pairs,
            "total_output_tiles": self.num_tiles ** 2,
            "cycles": self.global_cycle,
        }
        return got, stats


def run_tiled_tpu_cosim(matrix_size: int = MATRIX, tile_size: int = TILE) -> Tuple[np.ndarray, Dict[str, int], TiledTPUCosim]:
    act = _act_matrix_u16(matrix_size)
    wgt = _weights_matrix_u16(matrix_size)
    runner = TiledTPUCosim(matrix_size=matrix_size, tile_size=tile_size, dtype="fp16")
    expected = _expected_tiled_fp16_accum(act, wgt, tile_size)
    got, _ = runner.run(act, wgt)
    stats = runner.build_stats(got, expected)
    return got, stats, runner


def _write_logs(runner: TiledTPUCosim, stats: Dict[str, int], log_dir: Path) -> None:
    configure_debug(
        flags=["stats", "schedule", "gemm_cycles", "sdma_load_cycles"],
        log_dir=str(log_dir),
    )
    try:
        for key, value in stats.items():
            dprintf("stats", f"{key} {value}")
        for line in runner.timeline:
            dprintf("schedule", line)
        dprintf(
            "gemm_cycles",
            "ti tj tk slot total_cycles systolic_array_compute_cycles systolic_array_valid_mac_cycles",
        )
        for record in runner.gemm_cycle_records:
            dprintf(
                "gemm_cycles",
                (
                    f"{record['ti']} {record['tj']} {record['tk']} {record['slot']} "
                    f"{record['cycles']} {record['systolic_array_compute_cycles']} "
                    f"{record['systolic_array_valid_mac_cycles']}"
                ),
            )
        dprintf("sdma_load_cycles", "ti tj tk slot kind total_cycles")
        for record in runner.sdma_load_cycle_records:
            dprintf(
                "sdma_load_cycles",
                (
                    f"{record['ti']} {record['tj']} {record['tk']} {record['slot']} "
                    f"{record['kind']} {record['cycles']}"
                ),
            )
    finally:
        close_debug()


def test_scratchpad_vector_core_sysarr_tpu_tiled_1024x1024() -> None:
    act = _act_matrix_u16(MATRIX)
    wgt = _weights_matrix_u16(MATRIX)
    expected = _expected_tiled_fp16_accum(act, wgt, TILE)

    runner = TiledTPUCosim(matrix_size=MATRIX, tile_size=TILE, dtype="fp16")
    got, _ = runner.run(act, wgt)
    stats = runner.build_stats(got, expected)

    assert np.array_equal(got, expected)

    log_dir = Path(__file__).resolve().parents[2] / "logs" / "sysarr_gemm_tpu_tiled_1024"
    _write_logs(runner, stats, log_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cycle-accurate tiled TPU co-sim")
    parser.add_argument("--matrix-size", type=int, default=MATRIX)
    parser.add_argument("--tile-size", type=int, default=TILE)
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "logs" / "sysarr_gemm_tpu_tiled_1024"),
    )
    args = parser.parse_args()

    act = _act_matrix_u16(args.matrix_size)
    wgt = _weights_matrix_u16(args.matrix_size)
    expected = _expected_tiled_fp16_accum(act, wgt, args.tile_size)

    runner = TiledTPUCosim(matrix_size=args.matrix_size, tile_size=args.tile_size, dtype="fp16")
    got, _ = runner.run(act, wgt)
    stats = runner.build_stats(got, expected)
    if not np.array_equal(got, expected):
        raise AssertionError("tiled TPU co-sim result mismatch")
    _write_logs(runner, stats, Path(args.log_dir))
    print(stats)


if __name__ == "__main__":
    main()
