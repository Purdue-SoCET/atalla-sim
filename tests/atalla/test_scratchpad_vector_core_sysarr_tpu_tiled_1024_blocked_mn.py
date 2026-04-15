import argparse
import importlib.util
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from base.debug import close_debug, configure_debug, dprintf


def _load_legacy_module():
    legacy_path = Path(__file__).with_name("test_scratchpad_vector_core_sysarr_tpu_tiled_1024.py")
    spec = importlib.util.spec_from_file_location("_legacy_tiled_1024", legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load legacy tiled harness from {legacy_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


legacy = _load_legacy_module()

MATRIX = legacy.MATRIX
TILE = legacy.TILE
TiledTPUCosim = legacy.TiledTPUCosim
_act_matrix_u16 = legacy._act_matrix_u16
_weights_matrix_u16 = legacy._weights_matrix_u16
_expected_tiled_fp16_accum = legacy._expected_tiled_fp16_accum
_tile_weight_stream = legacy._tile_weight_stream
_write_dram_tile_u16 = legacy._write_dram_tile_u16
_read_dram_row_u16 = legacy._read_dram_row_u16
_read_slot_vector_u16 = legacy._read_slot_vector_u16
_write_slot_vector_u16 = legacy._write_slot_vector_u16
_fp16_from_u16 = legacy._fp16_from_u16
_decode_fp16_vector = legacy._decode_fp16_vector
_encode_fp16_vector = legacy._encode_fp16_vector


@dataclass(frozen=True)
class ScratchpadTileBuffer:
    kind: str
    slot_id: int
    tile_id: int
    base_addr: int
    dram_addr: int


@dataclass
class ReuseBlockLayout:
    block_rows: int
    block_cols: int
    act_slots: List[ScratchpadTileBuffer]
    wgt_slots: List[ScratchpadTileBuffer]
    psum_slots: Dict[Tuple[int, int], ScratchpadTileBuffer]
    rows_used_per_tile: List[int]


@dataclass
class BlockedTileJob:
    ti: int
    tj: int
    tk: int
    slot: int
    tag: str
    act_tile: List[List[int]]
    wgt_stream: List[List[int]]
    preload_tx_done: set = field(default_factory=set)
    preload_launch_cycle: Optional[int] = None
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
    act_buffer: Optional[ScratchpadTileBuffer] = None
    wgt_buffer: Optional[ScratchpadTileBuffer] = None
    psum_buffer: Optional[ScratchpadTileBuffer] = None


class ScratchpadLayoutAllocator:
    def __init__(self, *, bank_size: int, tile_size: int, tile_count: int = 2):
        self.bank_size = int(bank_size)
        self.tile_size = int(tile_size)
        self.tile_count = int(tile_count)
        self._next_row = [0 for _ in range(self.tile_count)]

    def allocate(
        self,
        *,
        kind: str,
        count: int,
        preferred_tiles: Tuple[int, ...],
        dram_base: int,
        tile_bytes: int,
    ) -> List[ScratchpadTileBuffer]:
        buffers: List[ScratchpadTileBuffer] = []
        for slot_id in range(int(count)):
            chosen_tile = None
            for tile_id in preferred_tiles:
                next_row = self._next_row[tile_id]
                if next_row + self.tile_size <= self.bank_size:
                    chosen_tile = tile_id
                    break
            if chosen_tile is None:
                raise ValueError(
                    f"scratchpad capacity exceeded while allocating {kind} buffer {slot_id}"
                )
            base_addr = self._next_row[chosen_tile]
            self._next_row[chosen_tile] += self.tile_size
            buffers.append(
                ScratchpadTileBuffer(
                    kind=str(kind),
                    slot_id=int(slot_id),
                    tile_id=int(chosen_tile),
                    base_addr=int(base_addr),
                    dram_addr=int(dram_base) + int(slot_id) * int(tile_bytes),
                )
            )
        return buffers

    @property
    def rows_used_per_tile(self) -> List[int]:
        return list(self._next_row)


class MNReuseBlockedTPUCosim(TiledTPUCosim):
    """
    Blocked tiled harness driven by notebook-style `M`/`N` reuse groups.

    `weight_reuse_m` is the number of resident weight tiles in a block. Each
    resident weight tile is reused across `activation_reuse_n` streamed
    activation tiles before the TPU weight state is drained and replaced.
    Each `(ti_block, tj_block, tk)` step therefore keeps `N` activation tiles,
    `M` weight tiles, and `M*N` psum tiles resident.
    """

    DRAM_ACT_STAGE_BASE = 0x10000000
    DRAM_WGT_STAGE_BASE = 0x20000000
    DRAM_PSUM_STAGE_BASE = 0x30000000

    def __init__(
        self,
        matrix_size: int = MATRIX,
        tile_size: int = TILE,
        weight_reuse_m: int = 1,
        activation_reuse_n: int = 1,
        dtype: str = "fp16",
        spad_frontend_queue_size: int = 4,
    ):
        self.weight_reuse_m = max(1, int(weight_reuse_m))
        self.activation_reuse_n = max(1, int(activation_reuse_n))
        super().__init__(
            matrix_size=matrix_size,
            tile_size=tile_size,
            dtype=dtype,
            spad_frontend_queue_size=spad_frontend_queue_size,
        )
        self.weight_reuse_m = min(self.weight_reuse_m, self.num_tiles)
        self.activation_reuse_n = min(self.activation_reuse_n, self.num_tiles)

    def build_stats(self, got: np.ndarray, expected: Optional[np.ndarray] = None) -> Dict[str, object]:
        stats = super().build_stats(got, expected)
        stats.update(
            {
                "weight_reuse_m": self.weight_reuse_m,
                "activation_reuse_n": self.activation_reuse_n,
                "execution_policy": (
                    "weight-stationary per tk: load one resident weight tile into the TPU, "
                    "stream activation tiles back-to-back, then flush before the next weight tile"
                ),
                "slot_policy": (
                    "weights prefer tile0, activations prefer tile1, psums prefer tile1 then spill to tile0; "
                    "every logical buffer carries its own tile/frontend id"
                ),
            }
        )
        return stats

    def _build_reuse_layout(self, block_rows: int, block_cols: int) -> ReuseBlockLayout:
        tile_bytes = self.tile_size * self.tile_row_bytes
        allocator = ScratchpadLayoutAllocator(
            bank_size=self.spad.bank_size,
            tile_size=self.tile_size,
            tile_count=len(self.spad.tiles),
        )
        act_slots = allocator.allocate(
            kind="act",
            count=block_rows,
            preferred_tiles=(1, 0),
            dram_base=self.DRAM_ACT_STAGE_BASE,
            tile_bytes=tile_bytes,
        )
        wgt_slots = allocator.allocate(
            kind="wgt",
            count=block_cols,
            preferred_tiles=(0, 1),
            dram_base=self.DRAM_WGT_STAGE_BASE,
            tile_bytes=tile_bytes,
        )
        psum_flat = allocator.allocate(
            kind="psum",
            count=block_rows * block_cols,
            preferred_tiles=(1, 0),
            dram_base=self.DRAM_PSUM_STAGE_BASE,
            tile_bytes=tile_bytes,
        )
        psum_slots = {
            (row_idx, col_idx): psum_flat[row_idx * block_cols + col_idx]
            for row_idx in range(block_rows)
            for col_idx in range(block_cols)
        }
        return ReuseBlockLayout(
            block_rows=int(block_rows),
            block_cols=int(block_cols),
            act_slots=act_slots,
            wgt_slots=wgt_slots,
            psum_slots=psum_slots,
            rows_used_per_tile=allocator.rows_used_per_tile,
        )

    def _clear_block_psums(self, layout: ReuseBlockLayout) -> None:
        zero_vec = [0] * self.vc.vector_len
        for psum_slot in layout.psum_slots.values():
            for row_idx in range(self.tile_size):
                _write_slot_vector_u16(
                    self.spad,
                    psum_slot.base_addr + row_idx,
                    zero_vec,
                    tile_id=psum_slot.tile_id,
                )
            self.dram.write(psum_slot.dram_addr, b"\x00" * (self.tile_size * self.tile_row_bytes))

    def _buffer_ready(self, buffer_slot: ScratchpadTileBuffer, expected_rows: List[List[int]]) -> bool:
        return self._slot_ready(
            buffer_slot.base_addr,
            expected_rows,
            tile_id=buffer_slot.tile_id,
        )

    def _ensure_kernel_tag_info(self, *, ti: int, tj: int, tk: int, slot: int) -> str:
        tag = self._make_kernel_tag(ti, tj, tk)
        if tag not in self.kernel_tag_info:
            self.kernel_tag_info[tag] = {
                "ti": int(ti),
                "tj": int(tj),
                "tk": int(tk),
                "slot": int(slot),
            }
        return tag

    def _prefetch_block_operands(
        self,
        *,
        layout: ReuseBlockLayout,
        ti0: int,
        tj0: int,
        tk: int,
        act: np.ndarray,
        wgt: np.ndarray,
    ) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
        k_slice = slice(tk * self.tile_size, (tk + 1) * self.tile_size)
        act_tiles: List[List[List[int]]] = []
        wgt_tiles: List[List[List[int]]] = []

        for row_idx, act_slot in enumerate(layout.act_slots):
            tile_row = ti0 + row_idx
            row_slice = slice(tile_row * self.tile_size, (tile_row + 1) * self.tile_size)
            act_tile = act[row_slice, k_slice].astype(np.int32).tolist()
            act_tiles.append(act_tile)
            _write_dram_tile_u16(self.dram, act_slot.dram_addr, act_tile)

        for col_idx, wgt_slot in enumerate(layout.wgt_slots):
            tile_col = tj0 + col_idx
            col_slice = slice(tile_col * self.tile_size, (tile_col + 1) * self.tile_size)
            wgt_tile = _tile_weight_stream(wgt[k_slice, col_slice])
            wgt_tiles.append(wgt_tile)
            _write_dram_tile_u16(self.dram, wgt_slot.dram_addr, wgt_tile)

        pending_by_tile = {tile_id: [] for tile_id in range(len(self.backends))}
        completed = set()
        launch_cycle: Dict[Tuple[str, int], int] = {}

        def _make_load_done(
            kind: str,
            slot_id: int,
            tile_row: int,
            tile_col: int,
            tag_targets: List[Tuple[str, int, int, int]],
        ):
            def _cb(_tx_id: int) -> None:
                completed.add((kind, slot_id))
                duration = int(self.global_cycle - launch_cycle[(kind, slot_id)])
                for tag, ti, tj, slot in tag_targets:
                    if tag not in self.kernel_tag_info:
                        self.kernel_tag_info[tag] = {
                            "ti": int(ti),
                            "tj": int(tj),
                            "tk": int(tk),
                            "slot": int(slot),
                        }
                    self._cover_kernel_path(tag, f"sdma_{kind}", int(launch_cycle[(kind, slot_id)]), int(self.global_cycle + 1))
                self.sdma_load_cycle_records.append(
                    {
                        "ti": tile_row,
                        "tj": tile_col,
                        "tk": tk,
                        "slot": slot_id,
                        "kind": kind,
                        "cycles": duration,
                    }
                )
            return _cb

        for row_idx, act_slot in enumerate(layout.act_slots):
            tag_targets = [
                (
                    self._ensure_kernel_tag_info(
                        ti=ti0 + row_idx,
                        tj=tj0 + col_idx,
                        tk=tk,
                        slot=layout.psum_slots[(row_idx, col_idx)].slot_id,
                    ),
                    ti0 + row_idx,
                    tj0 + col_idx,
                    layout.psum_slots[(row_idx, col_idx)].slot_id,
                )
                for col_idx in range(layout.block_cols)
            ]
            pending_by_tile[act_slot.tile_id].append(
                {
                    "kind": "act",
                    "slot": act_slot,
                    "tile_row": ti0 + row_idx,
                    "tile_col": tj0,
                    "callback": _make_load_done(
                        "act",
                        act_slot.slot_id,
                        ti0 + row_idx,
                        tj0,
                        tag_targets,
                    ),
                }
            )
        for col_idx, wgt_slot in enumerate(layout.wgt_slots):
            tag_targets = [
                (
                    self._ensure_kernel_tag_info(
                        ti=ti0 + row_idx,
                        tj=tj0 + col_idx,
                        tk=tk,
                        slot=layout.psum_slots[(row_idx, col_idx)].slot_id,
                    ),
                    ti0 + row_idx,
                    tj0 + col_idx,
                    layout.psum_slots[(row_idx, col_idx)].slot_id,
                )
                for row_idx in range(layout.block_rows)
            ]
            pending_by_tile[wgt_slot.tile_id].append(
                {
                    "kind": "wgt",
                    "slot": wgt_slot,
                    "tile_row": ti0,
                    "tile_col": tj0 + col_idx,
                    "callback": _make_load_done(
                        "wgt",
                        wgt_slot.slot_id,
                        ti0,
                        tj0 + col_idx,
                        tag_targets,
                    ),
                }
            )

        wait_cycles = 0
        total_loads = len(layout.act_slots) + len(layout.wgt_slots)
        while any(pending_by_tile[tile_id] for tile_id in pending_by_tile) or len(completed) < total_loads:
            for tile_id, queue in pending_by_tile.items():
                if not queue:
                    continue
                request = queue[0]
                slot = request["slot"]
                tx_id = self.backends[tile_id].driver_to_backend_start_load(
                    base_sp_addr=slot.base_addr,
                    base_dram_addr=slot.dram_addr,
                    rows=self.tile_size,
                    cols=self.tile_size,
                    callback=request["callback"],
                )
                if tx_id > 0:
                    launch_cycle[(request["kind"], slot.slot_id)] = self.global_cycle
                    queue.pop(0)
            self._step()
            wait_cycles += 1
            if wait_cycles > 200_000:
                raise AssertionError("blocked operand preload exceeded cycle budget")

        ready_wait = 0
        while True:
            act_ready = all(self._buffer_ready(slot, expected) for slot, expected in zip(layout.act_slots, act_tiles))
            wgt_ready = all(self._buffer_ready(slot, expected) for slot, expected in zip(layout.wgt_slots, wgt_tiles))
            if act_ready and wgt_ready:
                break
            self._step()
            ready_wait += 1
            if ready_wait > 50_000:
                raise AssertionError("preloaded block operands did not become visible in scratchpad")

        return act_tiles, wgt_tiles

    def _issue_weight_load_from_buffer(self, job: BlockedTileJob) -> bool:
        if job.weight_issue_row >= self.tile_size or job.wgt_buffer is None:
            return False
        assert self.vc.enqueue_memory(
            {
                "kind": "load",
                "vls": job.wgt_buffer.tile_id,
                "dst": self.W_REG,
                "addr": job.wgt_buffer.base_addr + job.weight_issue_row,
                "dtype": self.dtype,
                "meta": self._job_meta(job, "vls_wgt", row=job.weight_issue_row),
            }
        )
        job.weight_issue_row += 1
        job.weight_loads_inflight += 1
        return True

    def _issue_act_load_from_buffer(self, job: BlockedTileJob) -> bool:
        if job.act_issue_row >= self.tile_size or job.act_buffer is None:
            return False
        assert self.vc.enqueue_memory(
            {
                "kind": "load",
                "vls": job.act_buffer.tile_id,
                "dst": self.A_REG,
                "addr": job.act_buffer.base_addr + job.act_issue_row,
                "dtype": self.dtype,
                "meta": self._job_meta(job, "vls_act", row=job.act_issue_row),
            }
        )
        job.act_issue_row += 1
        job.act_loads_inflight += 1
        return True

    def _run_resident_kernel(self, job: BlockedTileJob) -> None:
        if job.psum_buffer is None:
            raise ValueError("kernel job is missing psum_buffer")

        self._reset_compute_pipeline()
        self._register_job_tag(job)
        job.compute_started = True
        job.cycle_start = self.global_cycle
        self._mark_kernel_path_start(job.tag, "compute_window", self.global_cycle)
        self._mark_kernel_path_start(job.tag, "kernel_total", self.global_cycle)
        finish_inputs_issued = False

        while len(job.completed_rows) < self.tile_size:
            wb = self._step()

            if wb is not None:
                src = wb.get("source")
                dst = wb.get("dst")
                wb_meta = dict(wb.get("meta", {}) or {})

                if src == "vlsu" and dst == self.W_REG and job.weight_loads_inflight > 0:
                    job.weight_loads_inflight -= 1
                    self.bytes_load_wgt += len(list(wb.get("data", []))) * 2
                    assert self.vc.enqueue_scheduler_instruction(
                        {
                            "unit": "gsau",
                            "vdata": list(wb.get("data", [])),
                            "is_weight": True,
                            "expect_output": False,
                            "dtype": self.dtype,
                            "meta": self._job_meta(job, "gsau_wgt", row=job.weight_row),
                        }
                    )
                    job.weight_row += 1
                    if job.weight_row >= self.tile_size:
                        job.weights_done = True

                elif src == "vlsu" and dst == self.A_REG and job.act_loads_inflight > 0:
                    job.act_loads_inflight -= 1
                    self.bytes_load_act += len(list(wb.get("data", []))) * 2
                    assert self.vc.enqueue_scheduler_instruction(
                        {
                            "unit": "gsau",
                            "vdata": list(wb.get("data", [])),
                            "dst": self.OUT_REG,
                            "is_weight": False,
                            "expect_output": True,
                            "dtype": self.dtype,
                            "meta": self._job_meta(job, "gsau_act", row=job.act_row),
                        }
                    )
                    job.act_row += 1

                elif src == "vlsu" and dst == self.ACC_REG and job.accum_loads_inflight > 0:
                    job.accum_loads_inflight -= 1
                    load_addr = int(wb_meta.get("addr", -1))
                    row_idx = load_addr - job.psum_buffer.base_addr
                    if row_idx not in job.accum_load_rows:
                        raise AssertionError("unexpected accumulator row load")
                    partial_bits = job.accum_load_rows.pop(row_idx)["data"]
                    self._mark_kernel_path_start(job.tag, "datapath_add", self.global_cycle)
                    assert self.vc.enqueue_compute(
                        op="add",
                        dst=self.SUM_REG,
                        src0=_decode_fp16_vector(partial_bits),
                        src1=_decode_fp16_vector(list(wb.get("data", []))),
                    )
                    job.sum_issue_rows.append(row_idx)

                elif src == "gsau" and dst == self.OUT_REG:
                    tag = str(wb_meta.get("tag", job.tag))
                    self._mark_kernel_path_end(tag, "gsau_rsp", self.global_cycle)
                    row_idx = job.next_out_row
                    if row_idx < self.tile_size:
                        job.pending_accum_rows.append({"row": row_idx, "data": list(wb.get("data", []))})
                        job.next_out_row += 1

                elif src == "datapath" and dst == self.SUM_REG:
                    self._mark_kernel_path_end(job.tag, "datapath_add", self.global_cycle)
                    if not job.sum_issue_rows:
                        raise AssertionError("unexpected accumulator datapath writeback")
                    row_idx = job.sum_issue_rows.pop(0)
                    job.pending_sum_rows.append(
                        {"row": row_idx, "data": _encode_fp16_vector(list(wb.get("data", [])))}
                    )

            while job.weight_issue_row < self.tile_size and job.weight_loads_inflight < self.load_issue_window:
                if not self._issue_weight_load_from_buffer(job):
                    break

            if job.weights_done:
                while job.act_issue_row < self.tile_size and job.act_loads_inflight < self.load_issue_window:
                    if not self._issue_act_load_from_buffer(job):
                        break
                    if job.act_issue_row >= self.tile_size and not finish_inputs_issued:
                        self.sysarr_bridge.finish_inputs()
                        finish_inputs_issued = True

            while job.pending_accum_rows and job.accum_loads_inflight < self.load_issue_window:
                next_item = job.pending_accum_rows.pop(0)
                row_idx = next_item["row"]
                assert self.vc.enqueue_memory(
                    {
                        "kind": "load",
                        "vls": job.psum_buffer.tile_id,
                        "dst": self.ACC_REG,
                        "addr": job.psum_buffer.base_addr + row_idx,
                        "dtype": self.dtype,
                        "meta": self._job_meta(job, "vls_psum_load", row=row_idx),
                    }
                )
                job.accum_load_rows[row_idx] = {"data": list(next_item["data"]), "age": 0}
                job.accum_loads_inflight += 1

            psum_store_window = self.spad.frontends[job.psum_buffer.tile_id].writeq.max_size + 1
            while job.pending_sum_rows and len(job.store_inflight) < psum_store_window:
                next_item = job.pending_sum_rows.pop(0)
                row_idx = next_item["row"]
                row_data = next_item["data"]
                assert self.vc.enqueue_memory(
                    {
                        "kind": "store",
                        "vls": job.psum_buffer.tile_id,
                        "data": row_data,
                        "addr": job.psum_buffer.base_addr + row_idx,
                        "dtype": self.dtype,
                        "meta": self._job_meta(job, "vls_psum_store", row=row_idx),
                    }
                )
                job.store_inflight[row_idx] = {"data": row_data, "age": 0}

            for row_idx, store_meta in list(job.store_inflight.items()):
                spad_vec = _read_slot_vector_u16(
                    self.spad,
                    job.psum_buffer.base_addr + row_idx,
                    self.vc.vector_len,
                    tile_id=job.psum_buffer.tile_id,
                )
                store_meta["age"] += 1
                if spad_vec == list(store_meta["data"]):
                    job.completed_rows.add(row_idx)
                    self._mark_kernel_path_end(job.tag, "vls_psum_store", self.global_cycle)
                    job.store_inflight.pop(row_idx, None)
                elif store_meta["age"] > 4000:
                    raise AssertionError("store did not commit to scratchpad")

        self.total_tile_pairs += 1
        self._drain_compute_path()
        gemm_end_cycle = self.global_cycle
        job.cycle_done = gemm_end_cycle
        self._mark_kernel_path_end(job.tag, "compute_window", gemm_end_cycle)
        self._mark_kernel_path_end(job.tag, "kernel_total", gemm_end_cycle)
        self.gemm_cycle_records.append(
            {
                "ti": job.ti,
                "tj": job.tj,
                "tk": job.tk,
                "slot": job.slot,
                "cycles": int(gemm_end_cycle - (job.cycle_start or gemm_end_cycle)),
                "systolic_array_compute_cycles": int(self.sa.compute_window_cycles),
                "systolic_array_valid_mac_cycles": int(self.sa.valid_mac_cycles),
            }
        )
        self._accumulate_sa_metrics()
        print(
            (
                f"TK produced: ti={job.ti} tj={job.tj} tk={job.tk} "
                f"slot={job.slot} cycles={int(gemm_end_cycle - (job.cycle_start or gemm_end_cycle))}"
            ),
            flush=True,
        )

    def _run_weight_stationary_batch(self, jobs: List[BlockedTileJob]) -> None:
        if not jobs:
            return

        shared_weight = jobs[0].wgt_buffer
        if shared_weight is None:
            raise ValueError("weight-stationary batch is missing a shared weight buffer")

        self._reset_compute_pipeline()

        jobs_by_tag: Dict[str, BlockedTileJob] = {}
        for job in jobs:
            if job.act_buffer is None or job.wgt_buffer is None or job.psum_buffer is None:
                raise ValueError("weight-stationary batch job is missing required scratchpad buffers")
            self._register_job_tag(job)
            jobs_by_tag[job.tag] = job

        phase_start_cycle = self.global_cycle
        weight_issue_row = 0
        weight_row = 0
        weight_loads_inflight = 0
        weights_done = False
        finish_inputs_issued = False
        next_stream_job = 0
        completed_jobs: List[BlockedTileJob] = []
        pending_datapath_rows: List[Tuple[BlockedTileJob, int, List[int], List[int]]] = []
        datapath_issue_fifo: List[Tuple[BlockedTileJob, int]] = []
        weight_vls_start: Optional[int] = None
        weight_vls_end: Optional[int] = None
        weight_gsau_start: Optional[int] = None
        weight_gsau_end: Optional[int] = None
        datapath_backlog_limit = 96

        def _read_loads_inflight_for_tile(tile_id: int) -> int:
            total = 0
            if shared_weight.tile_id == tile_id:
                total += weight_loads_inflight
            total += sum(
                job.act_loads_inflight
                for job in jobs
                if job.act_buffer is not None and job.act_buffer.tile_id == tile_id
            )
            total += sum(
                job.accum_loads_inflight
                for job in jobs
                if job.psum_buffer is not None and job.psum_buffer.tile_id == tile_id
            )
            return total

        def _store_inflight_for_tile(tile_id: int) -> int:
            return sum(
                len(job.store_inflight)
                for job in jobs
                if job.psum_buffer is not None and job.psum_buffer.tile_id == tile_id
            )

        def _scheduler_datapath_backlog() -> int:
            return (
                len(self.vc.datapath.pending_issue)
                + len(self.vc._build_packet["datapath"])
                + sum(len(pkt["datapath"]) for pkt in self.vc.vliw_q.items)
                + len(pending_datapath_rows)
                + len(datapath_issue_fifo)
            )

        while len(completed_jobs) < len(jobs):
            if self.global_cycle - phase_start_cycle > 400_000:
                raise AssertionError("weight-stationary batch exceeded cycle budget")

            wb = self._step()

            if wb is not None:
                src = wb.get("source")
                dst = wb.get("dst")
                wb_meta = dict(wb.get("meta", {}) or {})

                if src == "vlsu" and dst == self.W_REG and weight_loads_inflight > 0:
                    weight_loads_inflight -= 1
                    weight_vls_end = self.global_cycle
                    self.bytes_load_wgt += len(list(wb.get("data", []))) * 2
                    if weight_gsau_start is None:
                        weight_gsau_start = self.global_cycle
                    assert self.vc.enqueue_scheduler_instruction(
                        {
                            "unit": "gsau",
                            "vdata": list(wb.get("data", [])),
                            "is_weight": True,
                            "expect_output": False,
                            "dtype": self.dtype,
                            "meta": {
                                "tag": jobs[0].tag,
                                "path": "gsau_wgt_shared",
                                "tj": int(jobs[0].tj),
                                "tk": int(jobs[0].tk),
                                "row": int(weight_row),
                            },
                        }
                    )
                    weight_row += 1
                    weight_gsau_end = self.global_cycle + 1
                    if weight_row >= self.tile_size:
                        weights_done = True

                elif src == "vlsu" and dst == self.A_REG:
                    tag = str(wb_meta.get("tag", ""))
                    if tag not in jobs_by_tag:
                        raise AssertionError("unexpected activation load response")
                    job = jobs_by_tag[tag]
                    if job.act_loads_inflight <= 0:
                        raise AssertionError("activation load response without inflight tracker")
                    job.act_loads_inflight -= 1
                    self.bytes_load_act += len(list(wb.get("data", []))) * 2
                    row_idx = int(wb_meta.get("row", job.act_row))
                    assert self.vc.enqueue_scheduler_instruction(
                        {
                            "unit": "gsau",
                            "vdata": list(wb.get("data", [])),
                            "dst": self.OUT_REG,
                            "is_weight": False,
                            "expect_output": True,
                            "dtype": self.dtype,
                            "meta": self._job_meta(job, "gsau_act", row=row_idx),
                        }
                    )
                    job.act_row = max(job.act_row, row_idx + 1)

                elif src == "vlsu" and dst == self.ACC_REG:
                    tag = str(wb_meta.get("tag", ""))
                    if tag not in jobs_by_tag:
                        raise AssertionError("unexpected accumulator load response")
                    job = jobs_by_tag[tag]
                    if job.accum_loads_inflight <= 0:
                        raise AssertionError("accumulator load response without inflight tracker")
                    job.accum_loads_inflight -= 1
                    row_idx = int(wb_meta.get("row", -1))
                    if row_idx not in job.accum_load_rows:
                        raise AssertionError("unexpected accumulator row load")
                    partial_bits = job.accum_load_rows.pop(row_idx)["data"]
                    pending_datapath_rows.append(
                        (job, row_idx, list(partial_bits), list(wb.get("data", [])))
                    )

                elif src == "gsau" and dst == self.OUT_REG:
                    tag = str(wb_meta.get("tag", ""))
                    if tag not in jobs_by_tag:
                        raise AssertionError("unexpected systolic response tag")
                    job = jobs_by_tag[tag]
                    self._mark_kernel_path_end(tag, "gsau_rsp", self.global_cycle)
                    row_idx = int(wb_meta.get("row", job.next_out_row))
                    job.pending_accum_rows.append({"row": row_idx, "data": list(wb.get("data", []))})
                    job.next_out_row = max(job.next_out_row, row_idx + 1)

                elif src == "datapath" and dst == self.SUM_REG:
                    if not datapath_issue_fifo:
                        raise AssertionError("unexpected accumulator datapath writeback")
                    job, row_idx = datapath_issue_fifo.pop(0)
                    self._mark_kernel_path_end(job.tag, "datapath_add", self.global_cycle)
                    job.pending_sum_rows.append(
                        {"row": row_idx, "data": _encode_fp16_vector(list(wb.get("data", [])))}
                    )

            while weight_issue_row < self.tile_size and _read_loads_inflight_for_tile(shared_weight.tile_id) < self.load_issue_window:
                if weight_vls_start is None:
                    weight_vls_start = self.global_cycle
                assert self.vc.enqueue_memory(
                    {
                        "kind": "load",
                        "vls": shared_weight.tile_id,
                        "dst": self.W_REG,
                        "addr": shared_weight.base_addr + weight_issue_row,
                        "dtype": self.dtype,
                        "meta": {
                            "tag": jobs[0].tag,
                            "path": "vls_wgt_shared",
                            "tj": int(jobs[0].tj),
                            "tk": int(jobs[0].tk),
                            "row": int(weight_issue_row),
                        },
                    }
                )
                weight_issue_row += 1
                weight_loads_inflight += 1

            if weights_done and next_stream_job < len(jobs):
                job = jobs[next_stream_job]
                if not job.compute_started:
                    job.compute_started = True
                    job.cycle_start = self.global_cycle
                    self._mark_kernel_path_start(job.tag, "compute_window", self.global_cycle)
                    self._mark_kernel_path_start(job.tag, "kernel_total", self.global_cycle)
                    if weight_vls_start is not None and weight_vls_end is not None:
                        self._cover_kernel_path(job.tag, "vls_wgt", weight_vls_start, weight_vls_end)
                    if weight_gsau_start is not None and weight_gsau_end is not None:
                        self._cover_kernel_path(job.tag, "gsau_wgt", weight_gsau_start, weight_gsau_end)

                while (
                    job.act_issue_row < self.tile_size
                    and _read_loads_inflight_for_tile(job.act_buffer.tile_id) < self.load_issue_window
                ):
                    if not self._issue_act_load_from_buffer(job):
                        break

                if job.act_issue_row >= self.tile_size:
                    next_stream_job += 1
                    if next_stream_job >= len(jobs) and not finish_inputs_issued:
                        self.sysarr_bridge.finish_inputs()
                        finish_inputs_issued = True

            while pending_datapath_rows and _scheduler_datapath_backlog() < datapath_backlog_limit:
                job, row_idx, partial_bits, accum_bits = pending_datapath_rows.pop(0)
                self._mark_kernel_path_start(job.tag, "datapath_add", self.global_cycle)
                assert self.vc.enqueue_compute(
                    op="add",
                    dst=self.SUM_REG,
                    src0=_decode_fp16_vector(partial_bits),
                    src1=_decode_fp16_vector(accum_bits),
                )
                datapath_issue_fifo.append((job, row_idx))

            for job in jobs:
                while (
                    job.pending_accum_rows
                    and job.psum_buffer is not None
                    and _read_loads_inflight_for_tile(job.psum_buffer.tile_id) < self.load_issue_window
                    and _scheduler_datapath_backlog() < datapath_backlog_limit
                ):
                    next_item = job.pending_accum_rows.pop(0)
                    row_idx = int(next_item["row"])
                    assert self.vc.enqueue_memory(
                        {
                            "kind": "load",
                            "vls": job.psum_buffer.tile_id,
                            "dst": self.ACC_REG,
                            "addr": job.psum_buffer.base_addr + row_idx,
                            "dtype": self.dtype,
                            "meta": self._job_meta(job, "vls_psum_load", row=row_idx),
                        }
                    )
                    job.accum_load_rows[row_idx] = {"data": list(next_item["data"]), "age": 0}
                    job.accum_loads_inflight += 1

                psum_store_window = self.spad.frontends[job.psum_buffer.tile_id].writeq.max_size + 1
                while (
                    job.pending_sum_rows
                    and job.psum_buffer is not None
                    and _store_inflight_for_tile(job.psum_buffer.tile_id) < psum_store_window
                ):
                    next_item = job.pending_sum_rows.pop(0)
                    row_idx = int(next_item["row"])
                    row_data = list(next_item["data"])
                    assert self.vc.enqueue_memory(
                        {
                            "kind": "store",
                            "vls": job.psum_buffer.tile_id,
                            "data": row_data,
                            "addr": job.psum_buffer.base_addr + row_idx,
                            "dtype": self.dtype,
                            "meta": self._job_meta(job, "vls_psum_store", row=row_idx),
                        }
                    )
                    job.store_inflight[row_idx] = {"data": row_data, "age": 0}

                for row_idx, store_meta in list(job.store_inflight.items()):
                    spad_vec = _read_slot_vector_u16(
                        self.spad,
                        job.psum_buffer.base_addr + row_idx,
                        self.vc.vector_len,
                        tile_id=job.psum_buffer.tile_id,
                    )
                    store_meta["age"] += 1
                    if spad_vec == list(store_meta["data"]):
                        job.completed_rows.add(row_idx)
                        self._mark_kernel_path_end(job.tag, "vls_psum_store", self.global_cycle)
                        job.store_inflight.pop(row_idx, None)
                    elif store_meta["age"] > 4000:
                        raise AssertionError("store did not commit to scratchpad")

                if job.cycle_done is None and len(job.completed_rows) >= self.tile_size:
                    job.cycle_done = self.global_cycle
                    self.total_tile_pairs += 1
                    self._mark_kernel_path_end(job.tag, "compute_window", self.global_cycle)
                    self._mark_kernel_path_end(job.tag, "kernel_total", self.global_cycle)
                    completed_jobs.append(job)
                    print(
                        (
                            f"TK produced: ti={job.ti} tj={job.tj} tk={job.tk} "
                            f"slot={job.slot} cycles={int(self.global_cycle - (job.cycle_start or self.global_cycle))}"
                        ),
                        flush=True,
                    )

        self._drain_compute_path(limit=8000)

        phase_compute_total = int(self.sa.compute_window_cycles)
        phase_valid_total = int(self.sa.valid_mac_cycles)
        job_count = len(completed_jobs)
        compute_share, compute_rem = divmod(phase_compute_total, job_count)
        valid_share, valid_rem = divmod(phase_valid_total, job_count)
        for idx, job in enumerate(completed_jobs):
            self.gemm_cycle_records.append(
                {
                    "ti": job.ti,
                    "tj": job.tj,
                    "tk": job.tk,
                    "slot": job.slot,
                    "cycles": int((job.cycle_done or self.global_cycle) - (job.cycle_start or self.global_cycle)),
                    "systolic_array_compute_cycles": int(compute_share + (1 if idx < compute_rem else 0)),
                    "systolic_array_valid_mac_cycles": int(valid_share + (1 if idx < valid_rem else 0)),
                }
            )

        self._accumulate_sa_metrics()

    def _drain_psum_buffer(self, psum_slot: ScratchpadTileBuffer) -> np.ndarray:
        tile_out = np.zeros((self.tile_size, self.tile_size), dtype=np.int32)
        completed_rows = set()
        next_row_to_store = 0
        backend = self.backends[psum_slot.tile_id]

        while len(completed_rows) < self.tile_size:
            while next_row_to_store < self.tile_size:
                row_idx = next_row_to_store
                tx_id = backend.driver_to_backend_start_store(
                    base_sp_addr=psum_slot.base_addr + row_idx,
                    base_dram_addr=psum_slot.dram_addr + row_idx * self.tile_row_bytes,
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
                psum_slot.dram_addr + row_idx * self.tile_row_bytes,
                self.tile_size,
            )
            tile_out[row_idx, :] = np.rint(
                np.asarray([_fp16_from_u16(v) for v in dram_row], dtype=np.float64)
            ).astype(np.int32)
        return tile_out

    def _run_output_block(self, ti0: int, tj0: int, act: np.ndarray, wgt: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
        activation_tiles = min(self.activation_reuse_n, self.num_tiles - ti0)
        weight_tiles = min(self.weight_reuse_m, self.num_tiles - tj0)
        layout = self._build_reuse_layout(activation_tiles, weight_tiles)
        self._clear_block_psums(layout)
        self.timeline.append(
            (
                f"cycle {self.global_cycle}: block start ti0={ti0} tj0={tj0} "
                f"activation_tiles={activation_tiles} weight_tiles={weight_tiles} rows_used={layout.rows_used_per_tile}"
            )
        )

        for tk in range(self.num_tiles):
            act_tiles, wgt_tiles = self._prefetch_block_operands(
                layout=layout,
                ti0=ti0,
                tj0=tj0,
                tk=tk,
                act=act,
                wgt=wgt,
            )
            for col_idx, wgt_slot in enumerate(layout.wgt_slots):
                weight_jobs: List[BlockedTileJob] = []
                for row_idx, act_slot in enumerate(layout.act_slots):
                    weight_jobs.append(
                        BlockedTileJob(
                            ti=ti0 + row_idx,
                            tj=tj0 + col_idx,
                            tk=tk,
                            slot=layout.psum_slots[(row_idx, col_idx)].slot_id,
                            tag=self._make_kernel_tag(ti0 + row_idx, tj0 + col_idx, tk),
                            act_tile=act_tiles[row_idx],
                            wgt_stream=wgt_tiles[col_idx],
                            act_buffer=act_slot,
                            wgt_buffer=wgt_slot,
                            psum_buffer=layout.psum_slots[(row_idx, col_idx)],
                        )
                    )
                self._run_weight_stationary_batch(weight_jobs)

        outputs: Dict[Tuple[int, int], np.ndarray] = {}
        for row_idx in range(activation_tiles):
            for col_idx in range(weight_tiles):
                outputs[(ti0 + row_idx, tj0 + col_idx)] = self._drain_psum_buffer(
                    layout.psum_slots[(row_idx, col_idx)]
                )

        self.timeline.append(
            (
                f"cycle {self.global_cycle}: block complete ti0={ti0} tj0={tj0} "
                f"activation_tiles={activation_tiles} weight_tiles={weight_tiles}"
            )
        )
        return outputs

    def run(self, act: np.ndarray, wgt: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        got = np.zeros((self.matrix_size, self.matrix_size), dtype=np.int32)

        for ti0 in range(0, self.num_tiles, self.activation_reuse_n):
            for tj0 in range(0, self.num_tiles, self.weight_reuse_m):
                block_outputs = self._run_output_block(ti0, tj0, act, wgt)
                for (ti, tj), tile_out in block_outputs.items():
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

        stats = {
            "matrix_size": self.matrix_size,
            "tile_size": self.tile_size,
            "num_tiles": self.num_tiles,
            "weight_reuse_m": self.weight_reuse_m,
            "activation_reuse_n": self.activation_reuse_n,
            "total_tile_pairs": self.total_tile_pairs,
            "total_output_tiles": self.num_tiles ** 2,
            "cycles": self.global_cycle,
        }
        return got, stats


def run_blocked_mn_tpu_cosim(
    matrix_size: int = MATRIX,
    tile_size: int = TILE,
    weight_reuse_m: int = 1,
    activation_reuse_n: int = 1,
    spad_frontend_queue_size: int = 4,
) -> Tuple[np.ndarray, Dict[str, int], MNReuseBlockedTPUCosim]:
    act = _act_matrix_u16(matrix_size)
    wgt = _weights_matrix_u16(matrix_size)
    runner = MNReuseBlockedTPUCosim(
        matrix_size=matrix_size,
        tile_size=tile_size,
        weight_reuse_m=weight_reuse_m,
        activation_reuse_n=activation_reuse_n,
        dtype="fp16",
        spad_frontend_queue_size=spad_frontend_queue_size,
    )
    expected = _expected_tiled_fp16_accum(act, wgt, tile_size)
    got, _ = runner.run(act, wgt)
    stats = runner.build_stats(got, expected)
    return got, stats, runner


def _write_tk_gantt_plots(runner: MNReuseBlockedTPUCosim, log_dir: Path) -> None:
    try:
        from tools.plot_tiled_sysarr_tpu_gantt import _filter_rows, plot_gantt
    except BaseException as exc:
        print(f"Skipping tk Gantt plot generation: {exc}", flush=True)
        return

    rows = runner.kernel_gantt_rows()
    if not rows:
        return

    tile_keys = sorted({(int(row["ti"]), int(row["tj"])) for row in rows})
    if not tile_keys:
        return

    ti, tj = tile_keys[0]
    filtered = _filter_rows(
        rows,
        ti=ti,
        tj=tj,
        tk=None,
        tags=[],
        max_tags=None,
        include_envelopes=True,
    )
    if not filtered:
        return

    stem = f"kernel_gantt_ti{ti:02d}_tj{tj:02d}"
    plot_gantt(
        filtered,
        log_dir / f"{stem}.png",
        f"Blocked TPU tk-calculation Gantt (ti={ti}, tj={tj})",
        row_mode="tag",
    )
    plot_gantt(
        filtered,
        log_dir / f"{stem}_slots.png",
        f"Blocked TPU tk-calculation Gantt, slot-collapsed (ti={ti}, tj={tj})",
        row_mode="slot",
    )


def _write_reuse_gantt_plots(runner: MNReuseBlockedTPUCosim, log_dir: Path) -> None:
    try:
        from tools.plot_tiled_sysarr_tpu_gantt import _filter_rows, plot_gantt
    except BaseException as exc:
        print(f"Skipping reuse Gantt plot generation: {exc}", flush=True)
        return

    rows = runner.kernel_gantt_rows()
    if not rows:
        return

    first_row = min(rows, key=lambda row: (int(row["tk"]), int(row["ti"]), int(row["tj"]), int(row["slot"])))
    ti = int(first_row["ti"])
    tj = int(first_row["tj"])
    tk = int(first_row["tk"])

    weight_reuse_rows = _filter_rows(
        rows,
        ti=None,
        tj=tj,
        tk=tk,
        tags=[],
        max_tags=None,
        include_envelopes=True,
    )
    if weight_reuse_rows:
        stem = f"kernel_gantt_weight_reuse_tj{tj:02d}_tk{tk:02d}"
        plot_gantt(
            weight_reuse_rows,
            log_dir / f"{stem}.png",
            f"Blocked TPU weight-reuse Gantt (tj={tj}, tk={tk})",
            row_mode="tag",
        )
        plot_gantt(
            weight_reuse_rows,
            log_dir / f"{stem}_slots.png",
            f"Blocked TPU weight-reuse Gantt, slot-collapsed (tj={tj}, tk={tk})",
            row_mode="slot",
        )

    activation_reuse_rows = _filter_rows(
        rows,
        ti=ti,
        tj=None,
        tk=tk,
        tags=[],
        max_tags=None,
        include_envelopes=True,
    )
    if activation_reuse_rows:
        stem = f"kernel_gantt_activation_reuse_ti{ti:02d}_tk{tk:02d}"
        plot_gantt(
            activation_reuse_rows,
            log_dir / f"{stem}.png",
            f"Blocked TPU activation-reuse Gantt (ti={ti}, tk={tk})",
            row_mode="tag",
        )
        plot_gantt(
            activation_reuse_rows,
            log_dir / f"{stem}_slots.png",
            f"Blocked TPU activation-reuse Gantt, slot-collapsed (ti={ti}, tk={tk})",
            row_mode="slot",
        )


def _write_presentation_plots(runner: MNReuseBlockedTPUCosim, stats: Dict[str, object], log_dir: Path) -> None:
    try:
        from tools.plot_tiled_sysarr_tpu_gantt import write_presentation_plot_set
    except BaseException as exc:
        print(f"Skipping presentation plot generation: {exc}", flush=True)
        return

    rows = runner.kernel_gantt_rows()
    if not rows:
        return

    first_row = min(rows, key=lambda row: (int(row["tk"]), int(row["ti"]), int(row["tj"]), int(row["slot"])))
    write_presentation_plot_set(
        rows,
        log_dir,
        stats=stats,
        tj=int(first_row["tj"]),
        tk=int(first_row["tk"]),
    )


def _write_logs(runner: MNReuseBlockedTPUCosim, stats: Dict[str, object], log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_debug(
        flags=["stats", "schedule", "gemm_cycles", "sdma_load_cycles", "gantt", "harness"],
        log_dir=str(log_dir),
    )
    try:
        dprintf("harness", "TPU tiled blocked M/N harness overview")
        dprintf(
            "harness",
            (
                "This harness groups output tiles into notebook-style reuse blocks. "
                "For each (ti_block, tj_block, tk), it preloads N activation tiles and M weight tiles, "
                "loads one resident weight tile into the TPU, then streams activation tiles back-to-back before "
                "flushing for the next resident weight tile."
            ),
        )
        dprintf(
            "harness",
            (
                "Logical scratchpad buffers are decoupled from the two physical scratchpad tiles. "
                "Weights prefer tile 0, activations prefer tile 1, and psums prefer tile 1 before spilling to tile 0."
            ),
        )
        dprintf(
            "harness",
            (
                f"Config matrix_size={runner.matrix_size} tile_size={runner.tile_size} num_tiles={runner.num_tiles} "
                f"weight_reuse_m={runner.weight_reuse_m} activation_reuse_n={runner.activation_reuse_n}"
            ),
        )
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
        dprintf("gantt", "tag ti tj tk slot path start_cycle end_cycle duration_cycles touches")
        for row in runner.kernel_gantt_rows():
            dprintf(
                "gantt",
                (
                    f"{row['tag']} {row['ti']} {row['tj']} {row['tk']} {row['slot']} "
                    f"{row['path']} {row['start_cycle']} {row['end_cycle']} "
                    f"{row['duration_cycles']} {row['touches']}"
                ),
            )
    finally:
        close_debug()
    _write_tk_gantt_plots(runner, log_dir)
    _write_reuse_gantt_plots(runner, log_dir)
    _write_presentation_plots(runner, stats, log_dir)


def test_blocked_mn_tpu_cosim_64x64_two_by_two_reuse() -> None:
    matrix_size = 64
    tile_size = 32
    act = _act_matrix_u16(matrix_size)
    wgt = _weights_matrix_u16(matrix_size)
    expected = _expected_tiled_fp16_accum(act, wgt, tile_size)

    runner = MNReuseBlockedTPUCosim(
        matrix_size=matrix_size,
        tile_size=tile_size,
        weight_reuse_m=2,
        activation_reuse_n=2,
        dtype="fp16",
    )
    got, _ = runner.run(act, wgt)
    stats = runner.build_stats(got, expected)

    assert np.array_equal(got, expected)
    assert stats["weight_reuse_m"] == 2
    assert stats["activation_reuse_n"] == 2
    assert stats["total_output_tiles"] == 4


def test_blocked_mn_weight_stationary_pipelines_activation_tiles() -> None:
    matrix_size = 64
    tile_size = 32
    act = _act_matrix_u16(matrix_size)
    wgt = _weights_matrix_u16(matrix_size)
    expected = _expected_tiled_fp16_accum(act, wgt, tile_size)

    runner = MNReuseBlockedTPUCosim(
        matrix_size=matrix_size,
        tile_size=tile_size,
        weight_reuse_m=1,
        activation_reuse_n=2,
        dtype="fp16",
    )
    got, _ = runner.run(act, wgt)

    assert np.array_equal(got, expected)

    spans = {(row["tag"], row["path"]): row for row in runner.kernel_gantt_rows()}
    first_kernel = spans[("ti00_tj00_tk00", "compute_window")]
    second_kernel = spans[("ti01_tj00_tk00", "compute_window")]

    assert int(first_kernel["start_cycle"]) < int(second_kernel["start_cycle"])
    assert int(second_kernel["start_cycle"]) < int(first_kernel["end_cycle"])


def test_blocked_mn_gantt_contains_sdma_spans() -> None:
    runner = MNReuseBlockedTPUCosim(
        matrix_size=64,
        tile_size=32,
        weight_reuse_m=1,
        activation_reuse_n=2,
        dtype="fp16",
    )
    act = _act_matrix_u16(64)
    wgt = _weights_matrix_u16(64)
    got, _ = runner.run(act, wgt)
    expected = _expected_tiled_fp16_accum(act, wgt, 32)

    assert np.array_equal(got, expected)

    paths = {row["path"] for row in runner.kernel_gantt_rows()}
    assert "sdma_act" in paths
    assert "sdma_wgt" in paths


def test_blocked_mn_medium_reuse_batch_avoids_datapath_overflow() -> None:
    matrix_size = 128
    tile_size = 32
    act = _act_matrix_u16(matrix_size)
    wgt = _weights_matrix_u16(matrix_size)
    expected = _expected_tiled_fp16_accum(act, wgt, tile_size)

    runner = MNReuseBlockedTPUCosim(
        matrix_size=matrix_size,
        tile_size=tile_size,
        weight_reuse_m=4,
        activation_reuse_n=4,
        dtype="fp16",
    )
    got, _ = runner.run(act, wgt)

    assert np.array_equal(got, expected)
    assert len(runner.gemm_cycle_records) == 64


def test_blocked_mn_write_logs_emits_tk_gantt_plots() -> None:
    matrix_size = 64
    tile_size = 32
    act = _act_matrix_u16(matrix_size)
    wgt = _weights_matrix_u16(matrix_size)
    expected = _expected_tiled_fp16_accum(act, wgt, tile_size)

    runner = MNReuseBlockedTPUCosim(
        matrix_size=matrix_size,
        tile_size=tile_size,
        weight_reuse_m=1,
        activation_reuse_n=2,
        dtype="fp16",
    )
    got, _ = runner.run(act, wgt)
    stats = runner.build_stats(got, expected)

    assert np.array_equal(got, expected)

    with tempfile.TemporaryDirectory() as tmp_dir:
        log_dir = Path(tmp_dir)
        _write_logs(runner, stats, log_dir)
        tk_gantt = log_dir / "kernel_gantt_ti00_tj00.png"
        tk_gantt_slots = log_dir / "kernel_gantt_ti00_tj00_slots.png"
        assert tk_gantt.exists()
        assert tk_gantt_slots.exists()
        assert tk_gantt.stat().st_size > 0
        assert tk_gantt_slots.stat().st_size > 0

        weight_reuse = log_dir / "kernel_gantt_weight_reuse_tj00_tk00.png"
        weight_reuse_slots = log_dir / "kernel_gantt_weight_reuse_tj00_tk00_slots.png"
        activation_reuse = log_dir / "kernel_gantt_activation_reuse_ti00_tk00.png"
        activation_reuse_slots = log_dir / "kernel_gantt_activation_reuse_ti00_tk00_slots.png"
        assert weight_reuse.exists()
        assert weight_reuse_slots.exists()
        assert activation_reuse.exists()
        assert activation_reuse_slots.exists()
        assert weight_reuse.stat().st_size > 0
        assert weight_reuse_slots.stat().st_size > 0
        assert activation_reuse.stat().st_size > 0
        assert activation_reuse_slots.stat().st_size > 0

        presentation_block = log_dir / "presentation_block_overview_tj00_tk00.png"
        presentation_weight_flow = log_dir / "presentation_weight_flow_tj00_tk00.png"
        presentation_weight_compute = log_dir / "presentation_weight_compute_tj00_tk00.png"
        presentation_reuse_balance = log_dir / "presentation_reuse_balance_tk00.png"
        assert presentation_block.exists()
        assert presentation_weight_flow.exists()
        assert presentation_weight_compute.exists()
        assert presentation_reuse_balance.exists()
        assert presentation_block.stat().st_size > 0
        assert presentation_weight_flow.stat().st_size > 0
        assert presentation_weight_compute.stat().st_size > 0
        assert presentation_reuse_balance.stat().st_size > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Cycle-accurate blocked tiled TPU co-sim")
    parser.add_argument("--matrix-size", type=int, default=MATRIX)
    parser.add_argument("--tile-size", type=int, default=TILE)
    parser.add_argument("--weight-reuse-m", type=int, default=1)
    parser.add_argument("--activation-reuse-n", type=int, default=1)
    parser.add_argument("--spad-frontend-queue-size", type=int, default=4)
    parser.add_argument(
        "--log-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "logs" / "sysarr_gemm_tpu_tiled_1024_blocked_mn"),
    )
    args = parser.parse_args()

    act = _act_matrix_u16(args.matrix_size)
    wgt = _weights_matrix_u16(args.matrix_size)
    expected = _expected_tiled_fp16_accum(act, wgt, args.tile_size)

    runner = MNReuseBlockedTPUCosim(
        matrix_size=args.matrix_size,
        tile_size=args.tile_size,
        weight_reuse_m=args.weight_reuse_m,
        activation_reuse_n=args.activation_reuse_n,
        dtype="fp16",
        spad_frontend_queue_size=args.spad_frontend_queue_size,
    )
    got, _ = runner.run(act, wgt)
    stats = runner.build_stats(got, expected)
    if not np.array_equal(got, expected):
        raise AssertionError("blocked tiled TPU co-sim result mismatch")
    _write_logs(runner, stats, Path(args.log_dir))
    print(stats)



if __name__ == "__main__":
    main()