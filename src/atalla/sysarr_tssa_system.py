from typing import List, Dict, Optional, Tuple

import numpy as np

from base.clock_domain import ClockDomain
from base.core import Core
from base.eventq import EventQueue
from base.sim import Sim
from memory.dram import DRAM
from memory.sc_sram_banks import _xor_bank
from memory.scratchpad import Scratchpad
from systolic_array.systolic_array_tssa import SystolicArrayTSSA
from vector_core.vector_core import VectorCore


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


def _encode_row_u16(row):
    return b"".join(int(v).to_bytes(2, "little", signed=False) for v in row)


def _decode_row_u16(blob: bytes, cols: int):
    out = []
    for i in range(int(cols)):
        start = i * 2
        out.append(int.from_bytes(blob[start : start + 2], "little", signed=False))
    return out


def _fp16_bits(value: float) -> int:
    return int(np.asarray(value, dtype=np.float16).view(np.uint16).item())


class VLSFrontendBridge:
    def __init__(self, vc: VectorCore, spad: Scratchpad, vls_id: int = 0, frontend_id: int = 0):
        self.vc = vc
        self.spad = spad
        self.vls_id = int(vls_id)
        self.frontend_id = int(frontend_id)
        self._next_load_id = 0
        self._completed_load_ids = set()

    def _on_frontend_read(self, load_id: int, addr: int, lanes) -> None:
        if load_id in self._completed_load_ids:
            return
        self._completed_load_ids.add(load_id)
        data = _decode_lanes_u16(lanes, self.vc.vector_len)
        assert self.vc.push_scratchpad_response(self.vls_id, {"addr": addr, "data": data})

    def tick(self) -> None:
        while True:
            req = self.vc.pop_scratchpad_request(self.vls_id)
            if req is None:
                break

            addr = int(req.get("addr", 0))
            if req["kind"] == "store":
                assert self.spad.frontend_write(
                    addr,
                    _encode_vector_u16(req["data"]),
                    row_idx=0,
                    tile_id=self.frontend_id,
                )
                continue

            if req["kind"] == "load":
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
            if self.sa.enqueue(self._zero_row, dtype=self.dtype):
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
    def __init__(self, vc: VectorCore, sa: SystolicArrayTSSA, mirror: Optional[TSSAReference] = None):
        self.vc = vc
        self.sa = sa
        self.mirror = mirror
        self.size = sa.size
        self._pending_meta: List[Dict] = []
        self._out_read_idx = 0
        self._warmup = max(0, self.size - 1)
        self._flush_pending = 0
        self._zero_row = [0.0] * self.size

    def _pack_rsp(self, out_row: List[float], meta: Dict) -> Dict:
        vec = [0.0] * self.vc.vector_len
        for i, val in enumerate(out_row[: self.size]):
            vec[i] = _fp16_bits(val)
        return {"vdata": vec, "meta": dict(meta), "dtype": meta.get("dtype")}

    def tick(self) -> None:
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
            else:
                did_req = True
                expect_output = bool(req.get("expect_output", True))
                assert self.sa.enqueue(row, dtype=req.get("dtype"))
                assert self.sa.enqueue_psums([0.0] * self.size, dtype=req.get("dtype"))
                self.sa.set_control(weight_en=False, mac_shift=True, start=True, stall=False)
                if expect_output:
                    self._pending_meta.append(dict(req.get("meta", {})))
                self._flush_pending = self.size - 1
        elif self._flush_pending > 0:
            if self.sa.enqueue(self._zero_row, dtype=self.sa.dtype):
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
            meta = self._pending_meta.pop(0)
            assert self.vc.push_systolic_response(self._pack_rsp(out_row, meta))
            self._out_read_idx += 1


class SysArrTSSASystem:
    def __init__(self, size: int = 32, dtype: str = "fp16", mirror: bool = True):
        self.size = int(size)
        self.dtype = str(dtype)
        self.eq, self.clk, self.sim = build_sim()

        self.vc = VectorCore(
            veggie_size=self.size * 16,
            lane_count=4,
            vls_count=1,
            fu_latencies={"alu": 1},
            dtype=self.dtype,
        )
        self.spad = Scratchpad(
            num_banks=32,
            bank_size=128,
            read_latency=1,
            write_latency=1,
            xbar_delay=1,
            elem_bytes=2,
            frontend_queue_size=4,
        )
        self.sa = SystolicArrayTSSA(size=self.size, dtype=self.dtype)
        self.dram = DRAM(block_bytes=256)

        self.DRAM_ACT = 0x1000
        self.DRAM_WGT = 0x2000
        self.DRAM_OUT = 0x3000
        self.SPAD_ACT_BASE = 0
        self.SPAD_WGT_BASE = self.size
        self.SPAD_OUT_BASE = self.size * 2

        self.W_REG = 1
        self.A_REG = 2
        self.OUT_REG = 3

        self.mirror = TSSAReference(size=self.size, dtype=self.dtype) if mirror else None
        self.vls_bridge = VLSFrontendBridge(self.vc, self.spad, vls_id=0, frontend_id=0)
        self.sysarr_bridge = GSAUTSSABridge(self.vc, self.sa, mirror=self.mirror)

    def load_inputs(self, act: List[List[int]], wgt_stream: List[List[int]]) -> None:
        row_bytes = self.size * 2
        for r in range(self.size):
            self.dram.write(self.DRAM_ACT + r * row_bytes, _encode_row_u16(act[r]))
            self.dram.write(self.DRAM_WGT + r * row_bytes, _encode_row_u16(wgt_stream[r]))
        for r in range(self.size):
            act_row = _decode_row_u16(self.dram.read(self.DRAM_ACT + r * row_bytes, row_bytes), self.size)
            wgt_row = _decode_row_u16(self.dram.read(self.DRAM_WGT + r * row_bytes, row_bytes), self.size)
            slot = int(self.SPAD_ACT_BASE + r) % self.spad.bank_size
            for lane, value in enumerate(act_row):
                bank = _xor_bank(slot, lane, self.spad.num_banks)
                self.spad.tiles[0].banks[bank].mem[slot] = int(value).to_bytes(2, "little", signed=False)
            slot = int(self.SPAD_WGT_BASE + r) % self.spad.bank_size
            for lane, value in enumerate(wgt_row):
                bank = _xor_bank(slot, lane, self.spad.num_banks)
                self.spad.tiles[0].banks[bank].mem[slot] = int(value).to_bytes(2, "little", signed=False)

    def run(self, max_cycles: int = 20000) -> Tuple[List[List[int]], Optional[List[List[int]]]]:
        row_bytes = self.size * 2
        observed_rows: List[Optional[List[int]]] = [None for _ in range(self.size)]
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
        }

        def _issue_weight_load():
            if state["pending_weight_load"] or state["weight_row"] >= self.size:
                return
            assert self.vc.enqueue_memory(
                {
                    "kind": "load",
                    "vls": 0,
                    "dst": self.W_REG,
                    "addr": self.SPAD_WGT_BASE + state["weight_row"],
                    "dtype": self.dtype,
                }
            )
            state["pending_weight_load"] = True

        def _issue_act_load():
            if state["pending_act_load"] or state["act_row"] >= self.size:
                return
            assert self.vc.enqueue_memory(
                {
                    "kind": "load",
                    "vls": 0,
                    "dst": self.A_REG,
                    "addr": self.SPAD_ACT_BASE + state["act_row"],
                    "dtype": self.dtype,
                }
            )
            state["pending_act_load"] = True

        def _step(time: float):
            self.spad.now = time
            self.vc.tick()
            self.vls_bridge.tick()
            self.sysarr_bridge.tick()
            self.spad.tick(time)

            if self.vc.wb_valid and self.vc.last_wb is not None:
                wb = self.vc.last_wb
                src = wb.get("source")
                dst = wb.get("dst")

                if src == "vlsu" and dst == self.W_REG and state["pending_weight_load"]:
                    state["pending_weight_load"] = False
                    assert self.vc.enqueue_scheduler_instruction(
                        {
                            "unit": "gsau",
                            "vdata": list(wb.get("data", [])),
                            "is_weight": True,
                            "expect_output": False,
                            "dtype": self.dtype,
                        }
                    )
                    state["weight_row"] += 1
                    if state["weight_row"] < self.size:
                        _issue_weight_load()
                    else:
                        state["weights_done"] = True

                elif src == "vlsu" and dst == self.A_REG and state["pending_act_load"]:
                    state["pending_act_load"] = False
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
                    state["act_row"] += 1
                    if state["act_row"] < self.size:
                        _issue_act_load()

                elif src == "gsau" and dst == self.OUT_REG:
                    row_idx = state["next_out_row"]
                    if row_idx < self.size:
                        if observed_rows[row_idx] is None:
                            observed_rows[row_idx] = [int(x) for x in list(wb.get("data", []))[: self.size]]
                        state["pending_output_rows"].append(
                            {
                                "row": row_idx,
                                "data": list(wb.get("data", [])),
                            }
                        )
                        state["next_out_row"] += 1

            if state["weights_done"] and (not state["pending_act_load"]) and state["act_row"] < self.size:
                _issue_act_load()

            if (not state["store_inflight"]) and state["pending_output_rows"]:
                next_item = state["pending_output_rows"][0]
                row_idx = next_item["row"]
                row_data = next_item["data"]
                assert self.vc.enqueue_memory(
                    {
                        "kind": "store",
                        "vls": 0,
                        "data": row_data,
                        "addr": self.SPAD_OUT_BASE + row_idx,
                        "dtype": self.dtype,
                    }
                )
                state["store_inflight"] = True
                state["store_row_idx"] = row_idx
                state["store_row_data"] = row_data
                state["store_age"] = 0

            if state["store_inflight"] and state["store_row_idx"] is not None:
                row_idx = state["store_row_idx"]
                slot = int(self.SPAD_OUT_BASE + row_idx) % self.spad.bank_size
                spad_vec = []
                for lane in range(self.vc.vector_len):
                    bank = _xor_bank(slot, lane, self.spad.num_banks)
                    blob = self.spad.tiles[0].banks[bank].mem[slot]
                    blob = bytes(blob) if blob is not None else b"\x00\x00"
                    if len(blob) < 2:
                        blob = blob + (b"\x00" * (2 - len(blob)))
                    spad_vec.append(int.from_bytes(blob[:2], "little", signed=False))
                state["store_age"] += 1
                if spad_vec == (state["store_row_data"] or []):
                    self.dram.write(self.DRAM_OUT + row_idx * row_bytes, _encode_row_u16(spad_vec))
                    state["completed_rows"].add(row_idx)
                    state["store_inflight"] = False
                    state["store_row_idx"] = None
                    state["store_row_data"] = None
                    state["store_age"] = 0
                    if state["pending_output_rows"] and state["pending_output_rows"][0]["row"] == row_idx:
                        state["pending_output_rows"].pop(0)
                elif state["store_age"] > 1000:
                    raise AssertionError("store did not commit to scratchpad")

            state["cycles"] += 1
            if len(state["completed_rows"]) >= self.size:
                return
            if state["cycles"] >= max_cycles:
                return
            self.eq.schedule(time + 1.0, _step, time + 1.0)

        _issue_weight_load()
        self.eq.schedule(0.0, _step, 0.0)
        self.sim.run()

        if len(state["completed_rows"]) < self.size:
            raise AssertionError("timed out waiting for DRAM->SPAD->VRF->GSAU->TSSA->VRF->SPAD->DRAM")

        out = []
        for r in range(self.size):
            blob = self.dram.read(self.DRAM_OUT + r * row_bytes, row_bytes)
            out.append(_decode_row_u16(blob, self.size))
        mirror_out = self.mirror.outputs if self.mirror is not None else None
        return out, mirror_out
