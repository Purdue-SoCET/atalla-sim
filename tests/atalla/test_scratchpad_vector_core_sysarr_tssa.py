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
        while True:
            req = self.vc.pop_scratchpad_request(self.vls_id)
            if req is None:
                break

            addr = int(req.get("addr", 0))
            if req["kind"] == "store":
                dprintf("SYSARR", f"vls_req store addr={addr} len={len(req.get('data', []))}")
                self.bytes_store += len(req.get("data", [])) * 2
                self.activity_this_cycle = True
                assert self.spad.frontend_write(
                    addr,
                    _encode_vector_u16(req["data"]),
                    row_idx=0,
                    tile_id=self.frontend_id,
                )
                continue

            if req["kind"] == "load":
                dprintf("SYSARR", f"vls_req load addr={addr}")
                load_id = self._next_load_id
                self._next_load_id += 1
                assert self.spad.frontends[self.frontend_id].read(
                    addr,
                    0,
                    lambda lanes, _lid=load_id, _addr=addr: self._on_frontend_read(_lid, _addr, lanes),
                )
                continue

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
                self._flush_pending = self.size - 1
        elif did_flush and self._flush_pending > 0:
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
                self._flush_pending = self.size - 1
        elif self._flush_pending > 0:
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
            read_latency=1,
            write_latency=1,
            xbar_delay=1,
            elem_bytes=2,
            frontend_queue_size=4,
        )
        sa = SystolicArrayTSSA(size=tile, dtype="fp16")

        vls_bridge = VLSFrontendBridge(vc, spad, vls_id=0, frontend_id=0)
        mirror = TSSAReference(size=tile, dtype="fp16")
        sysarr_bridge = GSAUTSSABridge(vc, sa, mirror=mirror)

        dram = DRAM(block_bytes=256)
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

        for r in range(tile):
            act_row = _read_dram_row_u16(dram, DRAM_ACT + r * row_bytes, tile)
            wgt_row = _read_dram_row_u16(dram, DRAM_WGT + r * row_bytes, tile)
            _write_slot_vector_u16(spad, SPAD_ACT_BASE + r, act_row)
            _write_slot_vector_u16(spad, SPAD_WGT_BASE + r, wgt_row)

        W_REG = 1
        A_REG = 2
        OUT_REG = 3

        state = {
            "cycles": 0,
            "weight_row": 0,
            "act_row": 0,
            "next_out_row": 0,
            "pending_weight_load": False,
            "pending_act_load": False,
            "pending_output_rows": [],
            "store_inflight": False,
            "store_row_idx": None,
            "store_row_data": None,
            "store_age": 0,
            "completed_rows": set(),
            "weights_done": False,
            "bytes_load_wgt": 0,
            "bytes_load_act": 0,
            "bytes_store_out": 0,
            "vls_active_cycles": 0,
        }
        q_stats = {
            "samples": 0,
            "max": {},
            "sum": {},
        }
        metrics = TSSAMetrics(size=tile)
        observed_rows = [None for _ in range(tile)]

        def _issue_weight_load():
            if state["pending_weight_load"] or state["weight_row"] >= tile:
                return
            dprintf("SYSARR", f"issue weight load row={state['weight_row']}")
            assert vc.enqueue_memory(
                {
                    "kind": "load",
                    "vls": 0,
                    "dst": W_REG,
                    "addr": SPAD_WGT_BASE + state["weight_row"],
                    "dtype": "fp16",
                }
            )
            state["pending_weight_load"] = True

        def _issue_act_load():
            if state["pending_act_load"] or state["act_row"] >= tile:
                return
            dprintf("SYSARR", f"issue act load row={state['act_row']}")
            assert vc.enqueue_memory(
                {
                    "kind": "load",
                    "vls": 0,
                    "dst": A_REG,
                    "addr": SPAD_ACT_BASE + state["act_row"],
                    "dtype": "fp16",
                }
            )
            state["pending_act_load"] = True

        def _step(time: float):
            spad.now = time
            vls_bridge.start_cycle()
            vc.tick()
            vls_bridge.tick()
            sysarr_bridge.tick()
            spad.tick(time)
            if vls_bridge.activity_this_cycle:
                state["vls_active_cycles"] += 1

            if vc.wb_valid and vc.last_wb is not None:
                wb = vc.last_wb
                src = wb.get("source")
                dst = wb.get("dst")

                if src == "vlsu" and dst == W_REG and state["pending_weight_load"]:
                    state["pending_weight_load"] = False
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
                    if state["weight_row"] < tile:
                        _issue_weight_load()
                    else:
                        state["weights_done"] = True
                        dprintf("SYSARR", "weights_done")

                elif src == "vlsu" and dst == A_REG and state["pending_act_load"]:
                    state["pending_act_load"] = False
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
                    if state["act_row"] < tile:
                        _issue_act_load()

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

            if state["weights_done"] and (not state["pending_act_load"]) and state["act_row"] < tile:
                _issue_act_load()

            if (not state["store_inflight"]) and state["pending_output_rows"]:
                next_item = state["pending_output_rows"][0]
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
                state["store_inflight"] = True
                state["store_row_idx"] = row_idx
                state["store_row_data"] = row_data
                state["store_age"] = 0

            if state["store_inflight"] and state["store_row_idx"] is not None:
                row_idx = state["store_row_idx"]
                spad_vec = _read_slot_vector_u16(spad, SPAD_OUT_BASE + row_idx, vc.vector_len)
                state["store_age"] += 1
                if spad_vec == (state["store_row_data"] or []):
                    dprintf("SYSARR", f"store complete row={row_idx}")
                    dram.write(DRAM_OUT + row_idx * row_bytes, _encode_row_u16(spad_vec))
                    state["bytes_store_out"] += len(spad_vec) * 2
                    metrics.count_store_row()
                    state["completed_rows"].add(row_idx)
                    state["store_inflight"] = False
                    state["store_row_idx"] = None
                    state["store_row_data"] = None
                    state["store_age"] = 0
                    if state["pending_output_rows"] and state["pending_output_rows"][0]["row"] == row_idx:
                        state["pending_output_rows"].pop(0)
                elif state["store_age"] > 1000:
                    dprintf(
                        "SYSARR",
                        f"store stuck row={row_idx} spad_head={spad_vec[:4]} data_head={(state['store_row_data'] or [])[:4]}",
                    )
                    raise AssertionError("store did not commit to scratchpad")

            metrics.count_cycle()
            state["cycles"] += 1
            # Queue backpressure tracking (max + average depth).
            vlsu0 = vc.vls_units[0]
            q_depths = {
                "gsau_to_systolic": len(vc.gsau.to_systolic),
                "gsau_from_systolic": len(vc.gsau.from_systolic),
                "gsau_rd_queue": len(vc.gsau.rd_queue),
                "gsau_writebacks": len(vc.gsau.writebacks),
                "scheduler_q": len(vc.scheduler_q),
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

        _issue_weight_load()
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
        mac_utilization = (sa.valid_mac_cycles / state["cycles"]) if state["cycles"] else 0.0
        avg_active_pes_when_active = (sa.active_pe_sum / sa.valid_mac_cycles) if sa.valid_mac_cycles else 0.0
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
    finally:
        close_debug()


if __name__ == "__main__":
    test_scratchpad_vector_core_sysarr_tssa_end_to_end()
