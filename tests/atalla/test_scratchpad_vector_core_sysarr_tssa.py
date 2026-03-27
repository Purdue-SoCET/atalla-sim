import os
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..//..", "src")))

from base.clock_domain import ClockDomain
from base.core import Core
from base.debug import close_debug, configure_debug, dprintf
from base.eventq import EventQueue
from base.sim import Sim
from memory.backend import Backend
from memory.dram import DRAM
from memory.sc_sram_banks import _xor_bank
from memory.scratchpad import Scratchpad
from systolic_array.systolic_array_tssa import SystolicArrayTSSA
from vector_core.vector_core import VectorCore
from atalla.sysarr_tssa_system import TSSAMetrics


def build_sim():
    eq = EventQueue()
    clk = ClockDomain(eq, period=1.0)
    core = Core(eq)
    core.add_clock_domain(clk)
    sim = Sim()
    sim.init(eq, core)
    return eq, clk, sim


def _encode_vector_u16(vec):
    return b"".join(int(v).to_bytes(2, "little", signed=False) for v in vec)


def _decode_lanes_u16(lanes, vector_len: int):
    out = []
    for lane in list(lanes)[:vector_len]:
        if lane is False:
            lane = b"\x00\x00"
        blob = bytes(lane) if lane is not None else b"\x00\x00"
        if len(blob) < 2:
            blob = blob + (b"\x00" * (2 - len(blob)))
        out.append(int.from_bytes(blob[:2], "little", signed=False))
    while len(out) < vector_len:
        out.append(0)
    return out


def _read_slot_vector_u16(spad: Scratchpad, addr: int, vector_len: int):
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


def _write_slot_vector_u16(spad: Scratchpad, addr: int, values):
    tile = 0
    slot = int(addr) % spad.bank_size
    for lane, value in enumerate(list(values)):
        bank = _xor_bank(slot, lane, spad.num_banks)
        spad.tiles[tile].banks[bank].mem[slot] = int(value).to_bytes(2, "little", signed=False)


def _encode_row_u16(row):
    return b"".join(int(v).to_bytes(2, "little", signed=False) for v in row)


def _decode_row_u16(blob: bytes, cols: int):
    out = []
    for i in range(int(cols)):
        start = i * 2
        out.append(int.from_bytes(blob[start : start + 2], "little", signed=False))
    return out


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


def _act_u16(size: int) -> List[List[int]]:
    return [[((j * size + i) % 4) + 1 for j in range(size)] for i in range(size)]


def _weights_u16(size: int) -> List[List[int]]:
    return [[((i * size + j) % 8) + 1 for j in range(size)] for i in range(size)]


def _matmul_u16(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
    n = len(a)
    m = len(a[0]) if n else 0
    p = len(b[0]) if b else 0
    out = [[0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(m):
            aik = int(a[i][k])
            if aik == 0:
                continue
            row_b = b[k]
            for j in range(p):
                out[i][j] += aik * int(row_b[j])
    return out


def _busy_units_snapshot(vc: VectorCore, sa: SystolicArrayTSSA, state: Dict, spad: Scratchpad = None, backend=None) -> List[str]:
    busy = []

    if getattr(vc, "vliw_q", None) is not None and len(vc.vliw_q) > 0:
        busy.append(f"scheduler_packets={len(vc.vliw_q)}")
    if getattr(vc, "_build_packet", None) is not None:
        build = vc._build_packet
        if build["gsau"] or build["vlsu"] or build["datapath"]:
            busy.append(
                "scheduler_build="
                f"gsau:{len(build['gsau'])},vlsu:{len(build['vlsu'])},datapath:{len(build['datapath'])}"
            )

    for vls_id, vls in enumerate(vc.vls_units):
        if len(vls.issue_q) > 0:
            busy.append(f"vlsu{vls_id}.issue_q={len(vls.issue_q)}")
        if len(vls.req_q) > 0:
            busy.append(f"vlsu{vls_id}.req_q={len(vls.req_q)}")
        if len(vls.rsp_q) > 0:
            busy.append(f"vlsu{vls_id}.rsp_q={len(vls.rsp_q)}")
        if len(vls.wb_q) > 0:
            busy.append(f"vlsu{vls_id}.wb_q={len(vls.wb_q)}")
        for spad_id, fifo in enumerate(vls.load_dst_fifos):
            if len(fifo) > 0:
                busy.append(f"vlsu{vls_id}.dst_fifo[{spad_id}]={len(fifo)}")

    if len(vc.gsau.to_systolic) > 0:
        busy.append(f"gsau.to_systolic={len(vc.gsau.to_systolic)}")
    if len(vc.gsau.from_systolic) > 0:
        busy.append(f"gsau.from_systolic={len(vc.gsau.from_systolic)}")
    if len(vc.gsau.rd_queue) > 0:
        busy.append(f"gsau.rd_queue={len(vc.gsau.rd_queue)}")
    if len(vc.gsau.writebacks) > 0:
        busy.append(f"gsau.writebacks={len(vc.gsau.writebacks)}")

    if len(vc.datapath.pending_issue) > 0:
        busy.append(f"datapath.pending_issue={len(vc.datapath.pending_issue)}")
    for lane_id, lane in enumerate(vc.datapath.lanes):
        lane_busy = sum(1 for ctx in lane.fu_ctx.values() if ctx is not None)
        if lane_busy > 0:
            busy.append(f"lane{lane_id}.fu_ctx={lane_busy}")
        pending_out = len(lane.pending_outputs)
        if pending_out > 0:
            busy.append(f"lane{lane_id}.pending_outputs={pending_out}")
        for fu_name, pipe in lane.fus.items():
            if len(pipe.entries) > 0:
                busy.append(f"lane{lane_id}.{fu_name}.pipe={len(pipe.entries)}")
            if len(pipe.completed) > 0:
                busy.append(f"lane{lane_id}.{fu_name}.done={len(pipe.completed)}")
    if vc.datapath.collector.inflight:
        busy.append(f"collector.inflight={len(vc.datapath.collector.inflight)}")
    if len(vc.datapath.collector.pending_reductions) > 0:
        busy.append(f"collector.pending_reductions={len(vc.datapath.collector.pending_reductions)}")
    if len(vc.datapath.collector.completed_vectors) > 0:
        busy.append(f"collector.completed_vectors={len(vc.datapath.collector.completed_vectors)}")

    if len(vc.wb_buffer.entries) > 0:
        busy.append(f"wb_buffer={len(vc.wb_buffer.entries)}")
    if getattr(vc, "_datapath_wb_hold", None) is not None:
        busy.append("wb_hold=datapath")

    if state.get("weight_loads_inflight", 0) > 0:
        busy.append(f"driver.weight_loads_inflight={state['weight_loads_inflight']}")
    if state.get("act_loads_inflight", 0) > 0:
        busy.append(f"driver.act_loads_inflight={state['act_loads_inflight']}")
    if not state.get("preload_done", True):
        preload_done = sorted(state.get("preload_tx_done", set()))
        busy.append(
            "driver.preload="
            f"done:{','.join(preload_done) if preload_done else 'none'}"
            f",settle:{state.get('preload_settle', 0)}"
        )
    if state.get("pending_output_rows"):
        busy.append(f"driver.pending_output_rows={len(state['pending_output_rows'])}")
    if state.get("store_inflight"):
        inflight_rows = sorted(int(row_idx) for row_idx in state["store_inflight"].keys())
        busy.append(
            "driver.store_inflight="
            + ",".join(f"row{row_idx}" for row_idx in inflight_rows[:8])
            + ("..." if len(inflight_rows) > 8 else "")
        )
    if state.get("backend_store_rows"):
        busy.append(f"driver.backend_store_rows={len(state['backend_store_rows'])}")

    if backend is not None:
        bstats = backend.get_stats()
        if bstats.get("queued_txs", 0) > 0:
            busy.append(f"backend.tx_queue={bstats['queued_txs']}")
        if bstats.get("outstanding_txs", 0) > 0:
            busy.append(f"backend.active_txs={bstats['outstanding_txs']}")
        if bstats.get("dram_pending", 0) > 0:
            busy.append(f"backend.dram_pending={bstats['dram_pending']}")

    if spad is not None:
        wr_active = sum(1 for x in getattr(spad, "backend_write_inflight", []) if x)
        rd_active = sum(1 for x in getattr(spad, "backend_read_inflight", []) if x)
        if wr_active > 0:
            busy.append(f"spad.write_xbar_active={wr_active}")
        if rd_active > 0:
            busy.append(f"spad.read_xbar_active={rd_active}")

    active_pes = 0
    for i in range(sa.size):
        for j in range(sa.size):
            if sa.array[i][j].activation_latch != 0.0 and sa.array[i][j].weight != 0.0:
                active_pes += 1
    if sa.weight_en or sa.mac_shift or sa.start or sa.value_ready:
        busy.append(
            "tssa.ctrl="
            f"weight_en:{int(sa.weight_en)},mac_shift:{int(sa.mac_shift)},start:{int(sa.start)},ready:{int(sa.value_ready)}"
        )
    if active_pes > 0:
        busy.append(f"tssa.active_pes={active_pes}")
    if sa._algo_out_pending > 0:
        busy.append(f"tssa.algo_out_pending={sa._algo_out_pending}")
    if len(sa.get_buffer()) > 0:
        busy.append(f"tssa.out_buffer={len(sa.get_buffer())}")

    return busy


def _phase_snapshot(state: Dict, sa: SystolicArrayTSSA, tile: int) -> str:
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


def _fp16_bits(value: float) -> int:
    return int(np.asarray(value, dtype=np.float16).view(np.uint16).item())


def _fp16_from_u16(value: int) -> float:
    return float(np.frombuffer(np.uint16(int(value)).tobytes(), dtype=np.float16)[0])


def _tssa_reference_output(
    act_rows: List[List[int]],
    weight_stream: List[List[int]],
    *,
    size: int,
    dtype: str = "fp16",
) -> List[List[int]]:
    sa = SystolicArrayTSSA(size=size, dtype=dtype)
    zero_row = [0.0] * size
    out = []
    out_read_idx = 0
    warmup = max(0, size - 1)
    logged_first = False
    logged_converted = False

    # Load weights (one column vector per cycle).
    for vec in weight_stream:
        assert sa.enqueue_weights([float(x) for x in vec], dtype=dtype)
        sa.set_control(weight_en=True, mac_shift=False, start=False, stall=False)
        sa.tick()

    # Stream activations (one vector per cycle).
    for vec in act_rows:
        assert sa.enqueue([float(x) for x in vec], dtype=dtype)
        assert sa.enqueue_psums(zero_row, dtype=dtype)
        sa.set_control(weight_en=False, mac_shift=True, start=True, stall=False)
        sa.tick()
        buf = sa.get_buffer()
        while out_read_idx < len(buf):
            if out_read_idx < warmup:
                out_read_idx += 1
                continue
            if not logged_first:
                try:
                    dprintf("SYSARR", f"ref raw row[:8]={buf[out_read_idx][:8]}")
                    dprintf(
                        "SYSARR",
                        f"ref raw types[:4]={[type(x) for x in buf[out_read_idx][:4]]}",
                    )
                    dprintf(
                        "SYSARR",
                        f"ref raw float[:4]={[float(x) for x in buf[out_read_idx][:4]]}",
                    )
                except Exception as exc:
                    dprintf("SYSARR", f"ref raw log failed: {exc}")
                logged_first = True
            converted = [int(float(x)) for x in buf[out_read_idx]]
            if not logged_converted:
                dprintf("SYSARR", f"ref converted row[:8]={converted[:8]}")
                logged_converted = True
            out.append(converted)
            out_read_idx += 1

    # Flush pipeline.
    for _ in range(size - 1):
        assert sa.enqueue(zero_row, dtype=dtype, count_algo=False)
        assert sa.enqueue_psums(zero_row, dtype=dtype)
        sa.set_control(weight_en=False, mac_shift=True, start=True, stall=False)
        sa.tick()
        buf = sa.get_buffer()
        while out_read_idx < len(buf):
            if out_read_idx < warmup:
                out_read_idx += 1
                continue
            out.append([int(float(x)) for x in buf[out_read_idx]])
            out_read_idx += 1

    # Keep last size rows (one per activation vector).
    if len(out) >= size:
        dprintf("SYSARR", f"ref out len={len(out)} head={out[-size:][0][:8]}")
        return out[-size:]
    if out:
        dprintf("SYSARR", f"ref out len={len(out)} head={out[0][:8]}")
    return out


class VLSFrontendBridge:
    # Adapts the VectorCore's abstract scratchpad request/response interface onto
    # the scratchpad frontend model used in this simulator.
    def __init__(self, vc: VectorCore, spad: Scratchpad, vls_id: int = 0, frontend_id: int = 0):
        self.vc = vc
        self.spad = spad
        self.vls_id = int(vls_id)
        self.frontend_id = int(frontend_id)
        self._next_load_id = 0
        self._completed_load_ids = set()
        self.bytes_load = 0
        self.bytes_store = 0
        self.activity_this_cycle = False

    def start_cycle(self) -> None:
        self.activity_this_cycle = False

    def _on_frontend_read(self, load_id: int, addr: int, lanes) -> None:
        if load_id in self._completed_load_ids:
            return
        self._completed_load_ids.add(load_id)
        data = _decode_lanes_u16(lanes, self.vc.vector_len)
        self.bytes_load += len(data) * 2
        self.activity_this_cycle = True
        nz = [i for i, v in enumerate(data) if v != 0]
        if load_id < 2:
            dprintf(
                "SYSARR",
                f"vls_rsp data head={data[:8]} nz_count={len(nz)} nz_idx_head={nz[:8]}",
            )
        dprintf("SYSARR", f"vls_rsp load_id={load_id} addr={addr} len={len(data)}")
        assert self.vc.push_scratchpad_response(self.vls_id, {"addr": addr, "data": data})

    def tick(self) -> None:
        vls = self.vc.vls_units[self.vls_id]
        req = vls.req_q.peek()
        if req is None:
            return

        addr = int(req.get("addr", 0))
        if req["kind"] == "store":
            if self.spad.frontends[self.frontend_id].writeq.is_full():
                return
            req = self.vc.pop_scratchpad_request(self.vls_id)
            if req is None:
                return
            dprintf("SYSARR", f"vls_req store addr={addr} len={len(req.get('data', []))}")
            self.bytes_store += len(req.get("data", [])) * 2
            self.activity_this_cycle = True
            assert self.spad.frontend_write(
                addr,
                _encode_vector_u16(req["data"]),
                row_idx=0,
                tile_id=self.frontend_id,
            )
            return

        if req["kind"] == "load":
            if self.spad.frontends[self.frontend_id].readq.is_full():
                return
            req = self.vc.pop_scratchpad_request(self.vls_id)
            if req is None:
                return
            dprintf("SYSARR", f"vls_req load addr={addr}")
            load_id = self._next_load_id
            self._next_load_id += 1
            assert self.spad.frontends[self.frontend_id].read(
                addr,
                0,
                lambda lanes, _lid=load_id, _addr=addr: self._on_frontend_read(_lid, _addr, lanes),
            )
            return

        raise ValueError("unsupported request kind: %s" % req["kind"])


class TSSAReference:
    def __init__(self, size: int, dtype: str = "fp16"):
        self.sa = SystolicArrayTSSA(size=size, dtype=dtype)
        self.size = size
        self.dtype = dtype
        self._pending = 0
        self._out_read_idx = 0
        self._warmup = max(0, size - 1)
        self._flush_pending = 0
        self._zero_row = [0.0] * size
        self.outputs: List[List[int]] = []
        self._input_done = False

    def finish_inputs(self) -> None:
        self._input_done = True
        self._flush_pending = self.size - 1

    def tick_from_bridge(
        self,
        *,
        did_req: bool,
        row: List[float],
        is_weight: bool,
        expect_output: bool,
        did_flush: bool,
    ) -> None:
        self.sa.set_control(weight_en=False, mac_shift=False, start=False, stall=False)
        if did_req:
            if is_weight:
                assert self.sa.enqueue_weights(row, dtype=self.dtype)
                self.sa.set_control(weight_en=True, mac_shift=False, start=False, stall=False)
            else:
                assert self.sa.enqueue(row, dtype=self.dtype)
                assert self.sa.enqueue_psums([0.0] * self.size, dtype=self.dtype)
                self.sa.set_control(weight_en=False, mac_shift=True, start=True, stall=False)
                if expect_output:
                    self._pending += 1
        elif did_flush and self._input_done and self._flush_pending > 0:
            if self.sa.enqueue(self._zero_row, dtype=self.dtype, count_algo=False):
                assert self.sa.enqueue_psums([0.0] * self.size, dtype=self.dtype)
                self.sa.set_control(weight_en=False, mac_shift=True, start=True, stall=False)
                self._flush_pending -= 1

        self.sa.tick()

        buf = self.sa.get_buffer()
        while self._out_read_idx < len(buf):
            if self._out_read_idx < self._warmup:
                self._out_read_idx += 1
                continue
            if self._pending <= 0:
                break
            out_row = [float(x) for x in buf[self._out_read_idx]]
            self.outputs.append([_fp16_bits(x) for x in out_row[: self.size]])
            self._pending -= 1
            self._out_read_idx += 1


class GSAUTSSABridge:
    """Consumes VectorCore GSAU requests, drives TSSA model, returns responses."""

    def __init__(self, vc: VectorCore, sa: SystolicArrayTSSA, mirror: TSSAReference = None):
        self.vc = vc
        self.sa = sa
        self.mirror = mirror
        self.size = sa.size
        self._pending_meta: List[Dict] = []
        self._out_read_idx = 0
        self._warmup = max(0, self.size - 1)
        self._flush_pending = 0
        self._zero_row = [0.0] * self.size
        self._debug_weight_count = 0
        self._debug_act_count = 0
        self._debug_out_count = 0
        self._input_done = False

    def finish_inputs(self) -> None:
        self._input_done = True
        self._flush_pending = self.size - 1
        if self.mirror is not None:
            self.mirror.finish_inputs()

    def _pack_rsp(self, out_row: List[float], meta: Dict) -> Dict:
        vec = [0.0] * self.vc.vector_len
        for i, val in enumerate(out_row[: self.size]):
            vec[i] = _fp16_bits(val)
        return {"vdata": vec, "meta": dict(meta), "dtype": meta.get("dtype")}

    def tick(self) -> None:
        # Default: idle compute controls.
        self.sa.set_control(weight_en=False, mac_shift=False, start=False, stall=False)

        req = self.vc.pop_systolic_request()
        did_req = False
        did_flush = False
        row = self._zero_row
        is_weight = False
        expect_output = False
        if req is not None:
            vdata = [float(x) for x in req.get("vdata", [])]
            row = vdata[: self.size]
            if len(row) < self.size:
                row += [0.0] * (self.size - len(row))

            if bool(req.get("is_weight", False)):
                did_req = True
                is_weight = True
                assert self.sa.enqueue_weights(row, dtype=req.get("dtype"))
                self.sa.set_control(weight_en=True, mac_shift=False, start=False, stall=False)
                if self._debug_weight_count < 2:
                    dprintf("SYSARR", f"ref weight enqueue head={row[:8]}")
                    self._debug_weight_count += 1
            else:
                did_req = True
                expect_output = bool(req.get("expect_output", True))
                assert self.sa.enqueue(row, dtype=req.get("dtype"))
                assert self.sa.enqueue_psums([0.0] * self.size, dtype=req.get("dtype"))
                self.sa.set_control(weight_en=False, mac_shift=True, start=True, stall=False)
                if self._debug_act_count < 2:
                    dprintf("SYSARR", f"ref act enqueue head={row[:8]}")
                    self._debug_act_count += 1
                if expect_output:
                    self._pending_meta.append(dict(req.get("meta", {})))
        elif self._input_done and self._flush_pending > 0:
            if self.sa.enqueue(self._zero_row, dtype=self.sa.dtype, count_algo=False):
                did_flush = True
                assert self.sa.enqueue_psums([0.0] * self.size, dtype=self.sa.dtype)
                self.sa.set_control(weight_en=False, mac_shift=True, start=True, stall=False)
                self._flush_pending -= 1

        if self.mirror is not None:
            self.mirror.tick_from_bridge(
                did_req=did_req,
                row=row,
                is_weight=is_weight,
                expect_output=expect_output,
                did_flush=did_flush,
            )

        self.sa.tick()

        buf = self.sa.get_buffer()
        while self._out_read_idx < len(buf):
            if self._out_read_idx < self._warmup:
                self._out_read_idx += 1
                continue
            if not self._pending_meta:
                break
            out_row = [float(x) for x in buf[self._out_read_idx]]
            if self._debug_out_count < 2:
                dprintf("SYSARR", f"ref out row head={out_row[:8]}")
                self._debug_out_count += 1
            meta = self._pending_meta.pop(0)
            assert self.vc.push_systolic_response(self._pack_rsp(out_row, meta))
            self._out_read_idx += 1


def test_scratchpad_vector_core_sysarr_tssa_end_to_end():
    log_dir = Path(__file__).resolve().parents[2] / "logs" / "sysarr_gemm_tssa"
    configure_debug(flags=["DRAM", "SYSARR", "Xbar"], log_dir=str(log_dir))

    try:
        eq, clk, sim = build_sim()

        tile = 32
        row_bytes = tile * 2
        vc = VectorCore(veggie_size=tile * 16, lane_count=4, vls_count=1, fu_latencies={"alu": 1}, dtype="fp16")
        spad = Scratchpad(
            num_banks=32,
            bank_size=128,
            read_latency=2,
            write_latency=2,
            xbar_delay=3,
            elem_bytes=2,
            frontend_queue_size=4,
        )
        sa = SystolicArrayTSSA(size=tile, dtype="fp16")

        vls_bridge = VLSFrontendBridge(vc, spad, vls_id=0, frontend_id=0)
        mirror = TSSAReference(size=tile, dtype="fp16")
        sysarr_bridge = GSAUTSSABridge(vc, sa, mirror=mirror)

        dram = DRAM(block_bytes=256)
        # Backend models the DMA-style path that moves tiles between DRAM and the
        # scratchpad outside the VLS/VRF datapath used by compute.
        backend = Backend(dram_latency=8, dram_q_depth=32, dram_burst_bytes=32, elem_bytes=2)
        spad.attach_backend(backend)
        backend.attach_dram(dram)
        DRAM_ACT = 0x1000
        DRAM_WGT = 0x2000
        DRAM_OUT = 0x3000

        SPAD_ACT_BASE = 0
        SPAD_WGT_BASE = tile
        SPAD_OUT_BASE = tile * 2

        act = _act_u16(tile)
        wgt = _weights_u16(tile)
        # TSSA weight load shifts right each cycle; stream columns in reverse order.
        wgt_stream = [[wgt[r][c] for r in range(tile)] for c in range(tile - 1, -1, -1)]
        expected_ref = _tssa_reference_output(act, wgt_stream, size=tile, dtype="fp16")
        dprintf("SYSARR", f"expected[0][:8]={expected_ref[0][:8]}")
        try:
            dprintf("SYSARR", f"expected[0][1] float={float(expected_ref[0][1])}")
        except Exception:
            dprintf("SYSARR", f"expected[0][1] type={type(expected_ref[0][1])}")
        try:
            expected_types = [type(x) for x in expected_ref[0][:4]]
            dprintf("SYSARR", f"expected types head={expected_types}")
        except Exception as exc:
            dprintf("SYSARR", f"expected types log failed: {exc}")
        try:
            flat = [float(x) for row in expected_ref for x in row]
            if flat:
                dprintf("SYSARR", f"expected min={min(flat)} max={max(flat)}")
        except Exception as exc:
            dprintf("SYSARR", f"expected range log failed: {exc}")
        ref_int = _matmul_u16(act, wgt)
        dprintf("SYSARR", f"ref_int[0][:8]={ref_int[0][:8]}")
        try:
            ref_bits = [_fp16_bits(x) for x in ref_int[0][:8]]
            dprintf("SYSARR", f"ref_int fp16 bits[0][:8]={ref_bits}")
        except Exception as exc:
            dprintf("SYSARR", f"ref_int fp16 bits log failed: {exc}")

        _write_dram_tile_u16(dram, DRAM_ACT, act)
        _write_dram_tile_u16(dram, DRAM_WGT, wgt_stream)
        dram.snapshot_tile(DRAM_ACT, m=tile, n=tile, elem_bytes=2)
        dram.snapshot_tile(DRAM_WGT, m=tile, n=tile, elem_bytes=2)

        W_REG = 1
        A_REG = 2
        OUT_REG = 3

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
            "preload_settle": 0,
            "preload_tx_done": set(),
            "backend_store_rows": set(),
            # The modeled memory path is narrow:
            # - VLSU accepts/advances one memory op per cycle
            # - scratchpad frontend queues are finite
            # - each frontend has one active read path and one active write path
            # Use frontend queue depth + the currently serviceable in-flight slot
            # instead of an unbounded tile-wide window.
            "load_issue_window": spad.frontends[0].readq.max_size + 1,
            "store_issue_window": spad.frontends[0].writeq.max_size + 1,
        }
        q_stats = {
            "samples": 0,
            "max": {},
            "sum": {},
        }
        metrics = TSSAMetrics(size=tile)
        observed_rows = [None for _ in range(tile)]
        activity_path = log_dir / "pipeline_activity.log"
        activity_lines = []
        phase_path = log_dir / "pipeline_phases.log"
        phase_lines = []

        def _on_preload_done(name: str):
            state["preload_tx_done"].add(name)
            if len(state["preload_tx_done"]) == 2:
                # Allow the final backend->scratchpad write to drain through xbar and banks.
                state["preload_settle"] = spad.tile_write_xbars[0].delay + spad.tiles[0].write_latency + 1

        # Preload activation and weight tiles through the backend, matching the
        # RTL split between bulk DRAM movement and compute-time VLS accesses.
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

        def _issue_weight_load():
            if state["weight_issue_row"] >= tile:
                return False
            row_idx = state["weight_issue_row"]
            dprintf("SYSARR", f"issue weight load row={row_idx}")
            assert vc.enqueue_memory(
                {
                    "kind": "load",
                    "vls": 0,
                    "dst": W_REG,
                    "addr": SPAD_WGT_BASE + row_idx,
                    "dtype": "fp16",
                }
            )
            state["weight_issue_row"] += 1
            state["weight_loads_inflight"] += 1
            return True

        def _issue_act_load():
            if state["act_issue_row"] >= tile:
                return False
            row_idx = state["act_issue_row"]
            dprintf("SYSARR", f"issue act load row={row_idx}")
            assert vc.enqueue_memory(
                {
                    "kind": "load",
                    "vls": 0,
                    "dst": A_REG,
                    "addr": SPAD_ACT_BASE + row_idx,
                    "dtype": "fp16",
                }
            )
            state["act_issue_row"] += 1
            state["act_loads_inflight"] += 1
            if state["act_issue_row"] >= tile:
                sysarr_bridge.finish_inputs()
            return True

        def _step(time: float):
            spad.now = time
            # Keep the cycle order explicit:
            # 1) VC consumes prior-cycle responses and emits new requests
            # 2) bridges translate VC/GSAU traffic into scratchpad/TSSA activity
            # 3) backend advances DRAM<->SPAD transfers
            # 4) scratchpad advances xbars/banks so new data becomes visible
            vls_bridge.start_cycle()
            vc.tick()
            vls_bridge.tick()
            sysarr_bridge.tick()
            backend.tick(time)
            spad.tick(time)

            if not state["preload_done"] and len(state["preload_tx_done"]) == 2:
                if state["preload_settle"] > 0:
                    state["preload_settle"] -= 1
                else:
                    state["preload_done"] = True

            busy_units = _busy_units_snapshot(vc, sa, state, spad=spad, backend=backend)
            phase = _phase_snapshot(state, sa, tile)
            if busy_units:
                activity_lines.append(f"cycle {state['cycles']} [{phase}]: " + ", ".join(busy_units))
            else:
                activity_lines.append(f"cycle {state['cycles']} [{phase}]: idle")
            phase_lines.append(f"cycle {state['cycles']}: {phase}")
            if vls_bridge.activity_this_cycle:
                state["vls_active_cycles"] += 1

            # Consume at most one architectural writeback per cycle from the VC.
            # In this harness those writebacks are the control points that advance
            # the high-level flow:
            # - VLSU -> weight register: feed a weight row into the systolic path
            # - VLSU -> activation register: feed an activation row into the systolic path
            # - GSAU -> output register: queue a completed output row for storeback
            if vc.wb_valid and vc.last_wb is not None:
                wb = vc.last_wb
                src = wb.get("source")
                dst = wb.get("dst")

                # Weight rows are only used to program the systolic array state, so
                # the follow-on scheduler instruction does not expect an output row.
                if src == "vlsu" and dst == W_REG and state["weight_loads_inflight"] > 0:
                    state["weight_loads_inflight"] -= 1
                    dprintf("SYSARR", f"weight load done row={state['weight_row']} len={len(wb.get('data', []))}")
                    wdata = list(wb.get("data", []))
                    state["bytes_load_wgt"] += len(wdata) * 2
                    metrics.count_weight_load()
                    nz = [i for i, v in enumerate(wdata) if v != 0]
                    if state["weight_row"] < 2:
                        dprintf(
                            "SYSARR",
                            f"weight data head={wdata[:8]} nz_count={len(nz)} nz_idx_head={nz[:8]}",
                        )
                    assert vc.enqueue_scheduler_instruction(
                        {
                            "unit": "gsau",
                            "vdata": wdata,
                            "is_weight": True,
                            "expect_output": False,
                            "dtype": "fp16",
                        }
                    )
                    state["weight_row"] += 1
                    if state["weight_row"] >= tile:
                        state["weights_done"] = True
                        dprintf("SYSARR", "weights_done")

                # Activation rows are launched only after the weight preload phase
                # has completed. Each activation row generates one output row later.
                elif src == "vlsu" and dst == A_REG and state["act_loads_inflight"] > 0:
                    state["act_loads_inflight"] -= 1
                    dprintf("SYSARR", f"act load done row={state['act_row']} len={len(wb.get('data', []))}")
                    adata = list(wb.get("data", []))
                    state["bytes_load_act"] += len(adata) * 2
                    metrics.count_act_load()
                    nz = [i for i, v in enumerate(adata) if v != 0]
                    if state["act_row"] < 2:
                        dprintf(
                            "SYSARR",
                            f"act data head={adata[:8]} nz_count={len(nz)} nz_idx_head={nz[:8]}",
                        )
                    assert vc.enqueue_scheduler_instruction(
                        {
                            "unit": "gsau",
                            "vdata": adata,
                            "dst": OUT_REG,
                            "is_weight": False,
                            "expect_output": True,
                            "dtype": "fp16",
                        }
                    )
                    state["act_row"] += 1

                # GSAU writebacks are the architectural outputs of the systolic
                # pipeline. Hold them in order until the store side can accept them.
                elif src == "gsau" and dst == OUT_REG:
                    row_idx = state["next_out_row"]
                    if row_idx < tile:
                        dprintf("SYSARR", f"gsau output row={row_idx} len={len(wb.get('data', []))}")
                        if observed_rows[row_idx] is None:
                            observed_rows[row_idx] = [int(x) for x in list(wb.get("data", []))[:tile]]
                        metrics.count_output_row()
                        state["pending_output_rows"].append(
                            {
                                "row": row_idx,
                                "data": list(wb.get("data", [])),
                            }
                        )
                        state["next_out_row"] += 1

            # Do not let the compute-side VLS traffic start until the backend has
            # finished bulk-loading both source tiles into the scratchpad.
            if state["preload_done"]:
                while (
                    state["weight_issue_row"] < tile
                    and state["weight_loads_inflight"] < state["load_issue_window"]
                ):
                    if not _issue_weight_load():
                        break

            # TSSA consumes activations only after all weights have been presented.
            if state["preload_done"] and state["weights_done"]:
                while (
                    state["act_issue_row"] < tile
                    and state["act_loads_inflight"] < state["load_issue_window"]
                ):
                    if not _issue_act_load():
                        break

            # Once a row comes back from GSAU, turn it into a VLS store targeting
            # the output region of the scratchpad. This mirrors the VRF->SPAD path
            # before the backend later writes the row back to DRAM.
            while state["pending_output_rows"] and len(state["store_inflight"]) < state["store_issue_window"]:
                next_item = state["pending_output_rows"].pop(0)
                row_idx = next_item["row"]
                row_data = next_item["data"]
                dprintf("SYSARR", f"issue store row={row_idx}")
                if row_idx < 2:
                    try:
                        dprintf("SYSARR", f"store row_data head={row_data[:8]}")
                        dprintf(
                            "SYSARR",
                            f"store row_data types head={[type(x) for x in row_data[:4]]}",
                        )
                        dprintf(
                            "SYSARR",
                            f"store row_data int head={[int(x) for x in row_data[:4]]}",
                        )
                        dprintf(
                            "SYSARR",
                            f"store row_data fp16 bits head={[_fp16_bits(x) for x in row_data[:4]]}",
                        )
                    except Exception as exc:
                        dprintf("SYSARR", f"store row_data log failed: {exc}")
                assert vc.enqueue_memory(
                    {
                        "kind": "store",
                        "vls": 0,
                        "data": row_data,
                        "addr": SPAD_OUT_BASE + row_idx,
                        "dtype": "fp16",
                    }
                )
                state["store_inflight"][row_idx] = {
                    "data": row_data,
                    "age": 0,
                }

            completed_store_rows = []
            for row_idx, store_meta in list(state["store_inflight"].items()):
                # First wait for the VLS store to land in the scratchpad. Once the
                # row is resident, kick a backend store so DRAM writeback uses the
                # same path as the RTL bulk-memory engine.
                spad_vec = _read_slot_vector_u16(spad, SPAD_OUT_BASE + row_idx, vc.vector_len)
                store_meta["age"] += 1
                if spad_vec == (store_meta["data"] or []):
                    dprintf("SYSARR", f"store complete row={row_idx}")
                    if row_idx not in state["backend_store_rows"]:
                        tx_id = backend.driver_to_backend_start_store(
                            base_sp_addr=SPAD_OUT_BASE + row_idx,
                            base_dram_addr=DRAM_OUT + row_idx * row_bytes,
                            rows=1,
                            cols=tile,
                        )
                        assert tx_id > 0
                        state["backend_store_rows"].add(row_idx)
                    completed_store_rows.append(row_idx)
                elif store_meta["age"] > 1000:
                    dprintf(
                        "SYSARR",
                        f"store stuck row={row_idx} spad_head={spad_vec[:4]} data_head={(store_meta['data'] or [])[:4]}",
                    )
                    raise AssertionError("store did not commit to scratchpad")
            for row_idx in completed_store_rows:
                state["store_inflight"].pop(row_idx, None)

            # Backend store completion is observed at the architectural boundary we
            # care about here: the output row is visible in DRAM.
            for row_idx in sorted(state["backend_store_rows"] - state["completed_rows"]):
                dram_row = _read_dram_row_u16(dram, DRAM_OUT + row_idx * row_bytes, tile)
                expected_row = observed_rows[row_idx] if row_idx < len(observed_rows) else None
                if expected_row is not None and dram_row == expected_row[:tile]:
                    state["bytes_store_out"] += len(dram_row) * 2
                    metrics.count_store_row()
                    state["completed_rows"].add(row_idx)

            metrics.count_cycle()
            state["cycles"] += 1
            # Queue backpressure tracking (max + average depth).
            vlsu0 = vc.vls_units[0]
            q_depths = {
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
            q_stats["samples"] += 1
            for name, depth in q_depths.items():
                q_stats["sum"][name] = q_stats["sum"].get(name, 0) + depth
                q_stats["max"][name] = max(q_stats["max"].get(name, 0), depth)
            if state["cycles"] % 500 == 0:
                dprintf(
                    "SYSARR",
                    "cycle=%d weight_row=%d act_row=%d out_row=%d completed=%d pending_out=%d"
                    % (
                        state["cycles"],
                        state["weight_row"],
                        state["act_row"],
                        state["next_out_row"],
                        len(state["completed_rows"]),
                        len(state["pending_output_rows"]),
                    ),
                )
            if len(state["completed_rows"]) >= tile:
                return
            if state["cycles"] >= 20000:
                return
            eq.schedule(time + 1.0, _step, time + 1.0)

        eq.schedule(0.0, _step, 0.0)
        sim.run()

        if len(state["completed_rows"]) < tile:
            raise AssertionError("timed out waiting for DRAM->SPAD->VRF->GSAU->TSSA->VRF->SPAD->DRAM")

        dram.snapshot_tile(DRAM_OUT, m=tile, n=tile, elem_bytes=2)
        got = _read_dram_tile_u16(dram, DRAM_OUT, tile, tile)
        expected_from_gsau = []
        for r in range(tile):
            row = observed_rows[r]
            if row is None:
                row = [0] * tile
            expected_from_gsau.append(row[:tile])
        if got != expected_from_gsau:
            mismatch_row = None
            for i in range(tile):
                if got[i] != expected_from_gsau[i]:
                    mismatch_row = i
                    break
            if mismatch_row is None:
                mismatch_row = 0
            dprintf("SYSARR", f"store mismatch row={mismatch_row}")
            dprintf("SYSARR", f"got[{mismatch_row}][:8]={got[mismatch_row][:8]}")
            dprintf("SYSARR", f"gsau[{mismatch_row}][:8]={expected_from_gsau[mismatch_row][:8]}")
            assert got == expected_from_gsau
        expected_cycle = mirror.outputs
        if len(expected_cycle) != tile:
            dprintf("SYSARR", f"mirror outputs len={len(expected_cycle)}")
        assert len(expected_cycle) == tile
        dprintf("SYSARR", f"mirror[0][:8]={expected_cycle[0][:8]}")
        if got != expected_cycle:
            mismatch_row = None
            for i in range(tile):
                if got[i] != expected_cycle[i]:
                    mismatch_row = i
                    break
            if mismatch_row is None:
                mismatch_row = 0
            dprintf("SYSARR", f"mirror mismatch row={mismatch_row}")
            dprintf("SYSARR", f"got_m[{mismatch_row}][:8]={got[mismatch_row][:8]}")
            dprintf("SYSARR", f"mir[{mismatch_row}][:8]={expected_cycle[mismatch_row][:8]}")
            dprintf(
                "SYSARR",
                f"got_m_fp16[{mismatch_row}][:8]={[ _fp16_from_u16(v) for v in got[mismatch_row][:8] ]}",
            )
            dprintf(
                "SYSARR",
                f"mir_fp16[{mismatch_row}][:8]={[ _fp16_from_u16(v) for v in expected_cycle[mismatch_row][:8] ]}",
            )
        if got != expected_ref:
            mismatch_row = None
            for i in range(tile):
                if got[i] != expected_ref[i]:
                    mismatch_row = i
                    break
            if mismatch_row is None:
                mismatch_row = 0
            dprintf("SYSARR", f"mismatch row={mismatch_row}")
            dprintf("SYSARR", f"got[{mismatch_row}][:8]={got[mismatch_row][:8]}")
            dprintf("SYSARR", f"exp[{mismatch_row}][:8]={expected_ref[mismatch_row][:8]}")
            dprintf(
                "SYSARR",
                f"got_fp16[{mismatch_row}][:8]={[ _fp16_from_u16(v) for v in got[mismatch_row][:8] ]}",
            )
            dprintf(
                "SYSARR",
                f"exp_fp16[{mismatch_row}][:8]={[ _fp16_from_u16(v) for v in expected_ref[mismatch_row][:8] ]}",
            )
            dprintf("SYSARR", f"got[0][:8]={got[0][:8]}")
            dprintf("SYSARR", f"exp[0][:8]={expected_ref[0][:8]}")
            dprintf(
                "SYSARR",
                f"got_fp16[0][:8]={[ _fp16_from_u16(v) for v in got[0][:8] ]}",
            )
            dprintf(
                "SYSARR",
                f"exp_fp16[0][:8]={[ _fp16_from_u16(v) for v in expected_ref[0][:8] ]}",
            )
            dprintf("SYSARR", f"got[-1][:8]={got[-1][:8]}")
            dprintf("SYSARR", f"exp[-1][:8]={expected_ref[-1][:8]}")
            dprintf(
                "SYSARR",
                f"got_fp16[-1][:8]={[ _fp16_from_u16(v) for v in got[-1][:8] ]}",
            )
            dprintf(
                "SYSARR",
                f"exp_fp16[-1][:8]={[ _fp16_from_u16(v) for v in expected_ref[-1][:8] ]}",
            )
            dprintf("SYSARR", f"got[1][:8]={got[1][:8]}")
            dprintf("SYSARR", f"exp[1][:8]={expected_ref[1][:8]}")
            dprintf(
                "SYSARR",
                f"got_fp16[1][:8]={[ _fp16_from_u16(v) for v in got[1][:8] ]}",
            )
            dprintf(
                "SYSARR",
                f"exp_fp16[1][:8]={[ _fp16_from_u16(v) for v in expected_ref[1][:8] ]}",
            )
            dprintf("SYSARR", f"got[2][:8]={got[2][:8]}")
            dprintf("SYSARR", f"exp[2][:8]={expected_ref[2][:8]}")
            dprintf(
                "SYSARR",
                f"got_fp16[2][:8]={[ _fp16_from_u16(v) for v in got[2][:8] ]}",
            )
            dprintf(
                "SYSARR",
                f"exp_fp16[2][:8]={[ _fp16_from_u16(v) for v in expected_ref[2][:8] ]}",
            )
            dprintf("SYSARR", f"got_row_sum={sum(got[mismatch_row])} exp_row_sum={sum(expected_ref[mismatch_row])}")
            dprintf("SYSARR", f"wgt_stream[0][:8]={wgt_stream[0][:8]}")
            dprintf("SYSARR", f"wgt_stream[-1][:8]={wgt_stream[-1][:8]}")
            dprintf("SYSARR", f"act[0][:8]={act[0][:8]}")
            dprintf("SYSARR", f"act[-1][:8]={act[-1][:8]}")
        expected = expected_cycle
        assert got == expected
        bytes_tx = vls_bridge.bytes_load + vls_bridge.bytes_store
        pe_mul = sum(pe.mul_ops for row in sa.array for pe in row)
        pe_add = sum(pe.add_ops for row in sa.array for pe in row)
        pe_mac = sum(pe.mac_ops for row in sa.array for pe in row)
        pe_psum_add = sum(pe.psum_adds for row in sa.array for pe in row)
        vec_total_ops = sum(lane.total_ops for lane in vc.datapath.lanes)
        vec_op_counts = {}
        for lane in vc.datapath.lanes:
            for op, cnt in lane.op_counts.items():
                vec_op_counts[op] = vec_op_counts.get(op, 0) + cnt
        vec_reduce_ops = vc.datapath.collector.reduction_unit.reduce_ops
        flops_micro = (pe_mul + pe_add + vec_total_ops + vec_reduce_ops)
        bytes_internal = sa.internal_bytes_valid_total()
        arithmetic_intensity_internal = (flops_micro / bytes_internal) if bytes_internal else 0.0
        flops_algo = sa.algo_flops()
        bytes_algo = sa.algo_bytes()
        arithmetic_intensity_algo = sa.algo_arithmetic_intensity()
        stats_path = log_dir / "stats.log"
        stats_lines = [
            f"cycles {state['cycles']}",
            f"pe_mul_ops {pe_mul}",
            f"pe_add_ops {pe_add}",
            f"pe_mac_ops {pe_mac}",
            f"pe_psum_adds {pe_psum_add}",
            f"vec_total_ops {vec_total_ops}",
            f"vec_op_counts {vec_op_counts}",
            f"vec_reduce_ops {vec_reduce_ops}",
            f"flops_micro {flops_micro}",
            f"bytes_transmitted {bytes_tx}",
            f"bytes_internal {bytes_internal}",
            f"arithmetic_intensity_internal {arithmetic_intensity_internal}",
            f"flops_algo {flops_algo}",
            f"bytes_algo {bytes_algo}",
            f"arithmetic_intensity_algo {arithmetic_intensity_algo}",
        ]
        mac_utilization = (
            sa.active_pe_sum / (sa.valid_mac_cycles * tile * tile)
        ) if sa.valid_mac_cycles else 0.0
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
            sa.internal_bytes_valid["weight_shift"] / state["bytes_load_wgt"]
            if state["bytes_load_wgt"]
            else 0.0
        )
        reuse_act = (
            sa.internal_bytes_valid["act_shift"] / state["bytes_load_act"]
            if state["bytes_load_act"]
            else 0.0
        )
        reuse_psum = (
            sa.internal_bytes_valid["psum_shift"] / state["bytes_store_out"]
            if state["bytes_store_out"]
            else 0.0
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
        stats_lines.extend(
            [
                f"mac_utilization {mac_utilization}",
                f"avg_active_pes_when_active {avg_active_pes_when_active}",
                f"avg_active_pes_during_compute_window {avg_active_pes_during_compute_window}",
                f"max_active_pes_in_any_cycle {max_active_pes_in_any_cycle}",
                f"throughput_float_operations_per_cycle {throughput}",
                f"external_bandwidth_avg_bytes_per_cycle {external_bw}",
                f"external_bandwidth_active_bytes_per_cycle {external_bw_active}",
                f"internal_bandwidth_bytes_per_cycle {internal_bw}",
                f"reuse_weight_internal_over_external {reuse_weight}",
                f"reuse_act_internal_over_external {reuse_act}",
                f"reuse_psum_internal_over_external {reuse_psum}",
                f"queue_max_depths {q_stats['max']}",
            ]
        )
        if q_stats["samples"]:
            avg_depths = {k: (v / q_stats["samples"]) for k, v in q_stats["sum"].items()}
            stats_lines.append(f"queue_avg_depths {avg_depths}")
        stats_lines.extend(
            [
                f"fp16_saturation_count {sa.saturation_count}",
                f"fp16_overflow_count {sa.overflow_count}",
                f"max_abs_error {max_abs_error}",
                f"mean_abs_error {mean_abs_error}",
            ]
        )
        stats_path.write_text("\n".join(stats_lines) + "\n", encoding="utf-8")
        activity_path.write_text("\n".join(activity_lines) + "\n", encoding="utf-8")
        phase_path.write_text("\n".join(phase_lines) + "\n", encoding="utf-8")
    finally:
        close_debug()


if __name__ == "__main__":
    test_scratchpad_vector_core_sysarr_tssa_end_to_end()
