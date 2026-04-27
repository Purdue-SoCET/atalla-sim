import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, List, Dict, Optional, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from base.clock_domain import ClockDomain
from base.clocked_object import Clocked
from base.core import Core
from base.eventq import EventQueue
from base.sim import Sim
from memory.backend import Backend, SharedDRAMBurstChannel
from memory.dram import DRAM
from memory.sc_sram_banks import _xor_bank
from memory.scratchpad import Scratchpad
from systolic_array.systolic_array_tpu import SystolicArrayTPU
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


class TPUMetrics:
    def __init__(self, size: int):
        self.size = int(size)
        self.cycles = 0
        self.bytes_moved = 0
        self.flops = 0

    def count_weight_load(self) -> None:
        self.bytes_moved += self.size * 2

    def count_act_load(self) -> None:
        self.bytes_moved += self.size * 2

    def count_store_row(self) -> None:
        self.bytes_moved += self.size * 2

    def count_output_row(self) -> None:
        self.flops += 2 * self.size * self.size

    def count_cycle(self) -> None:
        self.cycles += 1

    def arithmetic_intensity(self) -> float:
        return (self.flops / self.bytes_moved) if self.bytes_moved else 0.0


class VLSFrontendBridge(Clocked):
    def __init__(self, vc: VectorCore, spad: Scratchpad, vls_id: int = 0, frontend_id: int = 0):
        super().__init__()
        self.vc = vc
        self.spad = spad
        self.vls_id = int(vls_id)
        self.frontend_id = int(frontend_id)
        self._next_load_id = 0
        self._completed_load_ids = set()
        self.bytes_load = 0
        self.bytes_store = 0
        self.activity_this_cycle = False
        self.now = 0
        self.trace_hook = None

    def start_cycle(self) -> None:
        self.activity_this_cycle = False

    def _on_frontend_read(self, load_id: int, addr: int, lanes, meta: Optional[Dict] = None) -> None:
        if load_id in self._completed_load_ids:
            return
        self._completed_load_ids.add(load_id)
        data = _decode_lanes_u16(lanes, self.vc.vector_len)
        self.bytes_load += len(data) * 2
        self.activity_this_cycle = True
        meta_dict = dict(meta or {})
        if self.trace_hook is not None:
            self.trace_hook(
                {
                    "kind": "load_rsp",
                    "cycle": int(self.now),
                    "vls": self.vls_id,
                    "frontend": self.frontend_id,
                    "addr": addr,
                    "meta": meta_dict,
                }
            )
        assert self.vc.push_scratchpad_response(self.vls_id, {"addr": addr, "data": data, "meta": meta_dict})

    def tick(self, time: Optional[float] = None) -> None:
        self.activity_this_cycle = False
        if time is not None:
            self.now = time

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
            meta = dict(req.get("meta", {}) or {})
            self.bytes_store += len(req.get("data", [])) * 2
            self.activity_this_cycle = True
            if self.trace_hook is not None:
                self.trace_hook(
                    {
                        "kind": "store_req",
                        "cycle": int(self.now),
                        "vls": self.vls_id,
                        "frontend": self.frontend_id,
                        "addr": addr,
                        "meta": meta,
                    }
                )
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
            meta = dict(req.get("meta", {}) or {})
            if self.trace_hook is not None:
                self.trace_hook(
                    {
                        "kind": "load_req",
                        "cycle": int(self.now),
                        "vls": self.vls_id,
                        "frontend": self.frontend_id,
                        "addr": addr,
                        "meta": meta,
                    }
                )
            load_id = self._next_load_id
            self._next_load_id += 1
            assert self.spad.frontends[self.frontend_id].read(
                addr,
                0,
                lambda lanes, _lid=load_id, _addr=addr, _meta=meta: self._on_frontend_read(
                    _lid, _addr, lanes, _meta
                ),
            )
            return

        raise ValueError("unsupported request kind: %s" % req["kind"])


class TPUReference:
    def __init__(self, size: int, dtype: str = "fp16"):
        self.sa = SystolicArrayTPU(size=size, dtype=dtype)
        self.size = size
        self.dtype = dtype
        self._pending = 0
        self._out_read_idx = 0
        self._warmup = self.sa.warmup_cycles()
        self._flush_pending = 0
        self._zero_row = [0.0] * size
        self.outputs: List[List[int]] = []
        self._input_done = False

    def finish_inputs(self) -> None:
        self._input_done = True
        self._flush_pending = self.sa.flush_cycles()

    def tick_from_bridge(
        self,
        *,
        did_req: bool,
        row: List[float],
        is_weight: bool,
        expect_output: bool,
        did_flush: bool,
        time: Optional[float] = None,
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

        self.sa.tick(time)

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


class GSAUTPUBridge(Clocked):
    def __init__(self, vc: VectorCore, sa: SystolicArrayTPU, mirror: Optional[TPUReference] = None):
        super().__init__()
        self.vc = vc
        self.sa = sa
        self.mirror = mirror
        self.size = sa.size
        self._pending_meta: List[Dict] = []
        self._out_read_idx = 0
        self._warmup = self.sa.warmup_cycles()
        self._flush_pending = 0
        self._zero_row = [0.0] * self.size
        self._input_done = False
        self.now = 0
        self.trace_hook = None

    def finish_inputs(self) -> None:
        self._input_done = True
        self._flush_pending = self.sa.flush_cycles()
        if self.mirror is not None:
            self.mirror.finish_inputs()

    def _pack_rsp(self, out_row: List[float], meta: Dict) -> Dict:
        vec = [0.0] * self.vc.vector_len
        for i, val in enumerate(out_row[: self.size]):
            vec[i] = _fp16_bits(val)
        return {"vdata": vec, "meta": dict(meta), "dtype": meta.get("dtype")}

    def tick(self, time: Optional[float] = None) -> None:
        if time is not None:
            self.now = time

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
            if self.trace_hook is not None:
                self.trace_hook(
                    {
                        "stage": "request",
                        "cycle": int(self.now),
                        "is_weight": is_weight,
                        "expect_output": expect_output,
                        "meta": dict(req.get("meta", {}) or {}),
                    }
                )
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
                time=time,
            )

        self.sa.tick(time)

        buf = self.sa.get_buffer()
        while self._out_read_idx < len(buf):
            if self._out_read_idx < self._warmup:
                self._out_read_idx += 1
                continue
            if not self._pending_meta:
                break
            out_row = [float(x) for x in buf[self._out_read_idx]]
            meta = self._pending_meta.pop(0)
            if self.trace_hook is not None:
                self.trace_hook(
                    {
                        "stage": "response",
                        "cycle": int(self.now),
                        "meta": dict(meta),
                    }
                )
            assert self.vc.push_systolic_response(self._pack_rsp(out_row, meta))
            self._out_read_idx += 1


class RoundRobinBackendTicker(Clocked):
    def __init__(self, backends: List[Backend]):
        super().__init__()
        self.backends = list(backends)

    def tick(self, time: Optional[float] = None) -> None:
        if not self.backends:
            return
        cycle = int(time) if time is not None else 0
        start_backend = cycle % len(self.backends)
        for offset in range(len(self.backends)):
            backend = self.backends[(start_backend + offset) % len(self.backends)]
            backend.tick(time)


@dataclass
class TPUPlatform:
    eq: EventQueue
    clk: ClockDomain
    sim: Sim
    vc: VectorCore
    spad: Scratchpad
    sa: SystolicArrayTPU
    dram: DRAM
    backend: Optional[Backend]
    backends: List[Backend]
    vls_bridge: Any
    vls_bridges: List[Any]
    sysarr_bridge: GSAUTPUBridge
    mirror: Optional[TPUReference]


def _build_shared_dram_backends(
    *,
    spad: Scratchpad,
    dram: DRAM,
    dram_latency: int,
    dram_q_depth: int,
    dram_burst_bytes: int,
    delay_cycles: int,
) -> List[Backend]:
    backends: List[Backend] = []
    shared_burst_channel = SharedDRAMBurstChannel()
    for tile_id in range(len(spad.tiles)):
        backend_obj = Backend(
            dram_latency=int(dram_latency),
            dram_q_depth=int(dram_q_depth),
            dram_burst_bytes=int(dram_burst_bytes),
            elem_bytes=2,
            delay_cycles=int(delay_cycles),
            shared_burst_channel=shared_burst_channel,
        )
        spad.attach_backend(backend_obj, tile_id=tile_id)
        backend_obj.attach_dram(dram)
        backends.append(backend_obj)
    return backends


def _build_vls_frontend_bridges(
    *,
    vc: VectorCore,
    spad: Scratchpad,
    bridge_cls: Callable[..., Any],
    bridge_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    kwargs = dict(bridge_kwargs or {})
    bridge_count = min(len(vc.vls_units), len(spad.frontends))
    bridges: List[Any] = []
    for bridge_id in range(bridge_count):
        bridges.append(
            bridge_cls(
                vc,
                spad,
                vls_id=bridge_id,
                frontend_id=bridge_id,
                **kwargs,
            )
        )
    return bridges


def build_tpu_compute_path(
    *,
    vc: VectorCore,
    size: int = 32,
    dtype: str = "fp16",
    mirror: Optional[TPUReference] = None,
) -> Tuple[SystolicArrayTPU, GSAUTPUBridge]:
    sa = SystolicArrayTPU(size=int(size), dtype=str(dtype))
    sysarr_bridge = GSAUTPUBridge(vc, sa, mirror=mirror)
    return sa, sysarr_bridge


def build_tpu_platform(
    *,
    size: int = 32,
    dtype: str = "fp16",
    lane_count: int = 4,
    vls_count: int = 1,
    spad_num_banks: int = 32,
    spad_bank_size: int = 128,
    spad_read_latency: int = 1,
    spad_write_latency: int = 1,
    spad_xbar_delay: int = 1,
    spad_frontend_queue_size: int = 4,
    dram_block_bytes: int = 256,
    backend_dram_latency: Optional[int] = None,
    backend_dram_q_depth: int = 16,
    backend_dram_burst_bytes: int = 32,
    backend_delay_cycles: int = 1,
    mirror: bool = False,
    vls_bridge_cls: Callable[..., Any] = VLSFrontendBridge,
    vls_bridge_kwargs: Optional[Dict[str, Any]] = None,
) -> TPUPlatform:
    eq, clk, sim = build_sim()
    vc = VectorCore(
        veggie_size=int(size) * 16,
        lane_count=int(lane_count),
        vls_count=int(vls_count),
        fu_latencies={"alu": 1},
        dtype=str(dtype),
    )
    spad = Scratchpad(
        num_banks=int(spad_num_banks),
        bank_size=int(spad_bank_size),
        read_latency=int(spad_read_latency),
        write_latency=int(spad_write_latency),
        xbar_delay=int(spad_xbar_delay),
        elem_bytes=2,
        frontend_queue_size=int(spad_frontend_queue_size),
    )
    dram = DRAM(block_bytes=int(dram_block_bytes))
    backends: List[Backend] = []
    backend = None
    if backend_dram_latency is not None:
        backends = _build_shared_dram_backends(
            spad=spad,
            dram=dram,
            dram_latency=int(backend_dram_latency),
            dram_q_depth=int(backend_dram_q_depth),
            dram_burst_bytes=int(backend_dram_burst_bytes),
            delay_cycles=int(backend_delay_cycles),
        )
        backend = backends[0]

    mirror_obj = TPUReference(size=int(size), dtype=str(dtype)) if mirror else None
    vls_bridges = _build_vls_frontend_bridges(
        vc=vc,
        spad=spad,
        bridge_cls=vls_bridge_cls,
        bridge_kwargs=vls_bridge_kwargs,
    )
    vls_bridge = vls_bridges[0] if vls_bridges else None
    sa, sysarr_bridge = build_tpu_compute_path(vc=vc, size=int(size), dtype=str(dtype), mirror=mirror_obj)
    return TPUPlatform(
        eq=eq,
        clk=clk,
        sim=sim,
        vc=vc,
        spad=spad,
        sa=sa,
        dram=dram,
        backend=backend,
        backends=backends,
        vls_bridge=vls_bridge,
        vls_bridges=vls_bridges,
        sysarr_bridge=sysarr_bridge,
        mirror=mirror_obj,
    )


class SysArrTPUSystem:
    def __init__(self, size: int = 32, dtype: str = "fp16", mirror: bool = True):
        self.size = int(size)
        self.dtype = str(dtype)
        platform = build_tpu_platform(size=self.size, dtype=self.dtype, mirror=mirror)
        self.eq = platform.eq
        self.clk = platform.clk
        self.sim = platform.sim
        self.vc = platform.vc
        self.spad = platform.spad
        self.sa = platform.sa
        self.dram = platform.dram
        self.backend = platform.backend
        self.backends = platform.backends
        self.vls_bridge = platform.vls_bridge
        self.vls_bridges = platform.vls_bridges

        self.DRAM_ACT = 0x1000
        self.DRAM_WGT = 0x2000
        self.DRAM_OUT = 0x3000
        self.SPAD_ACT_BASE = 0
        self.SPAD_WGT_BASE = self.size
        self.SPAD_OUT_BASE = self.size * 2

        self.W_REG = 1
        self.A_REG = 2
        self.OUT_REG = 3

        self.mirror = platform.mirror
        self.sysarr_bridge = platform.sysarr_bridge
        self.metrics = TPUMetrics(size=self.size)

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

    def run(self, max_cycles: int = 20000) -> Tuple[List[List[int]], Optional[List[List[int]]], int, "TPUMetrics"]:
        row_bytes = self.size * 2
        observed_rows: List[Optional[List[int]]] = [None for _ in range(self.size)]
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
            "load_issue_window": self.spad.frontends[self.vls_bridge.frontend_id].readq.max_size + 1,
            "store_issue_window": self.spad.frontends[self.vls_bridge.frontend_id].writeq.max_size + 1,
        }

        class RunHarness(Clocked):
            def __init__(self):
                super().__init__()
                self.done = False

            def _finish(self) -> None:
                self.done = True
                self_outer.clk.stop()

            def tick(self, time: float) -> None:
                if self.done:
                    return

                if self_outer.vc.wb_valid and self_outer.vc.last_wb is not None:
                    wb = self_outer.vc.last_wb
                    src = wb.get("source")
                    dst = wb.get("dst")

                    if src == "vlsu" and dst == self_outer.W_REG and state["weight_loads_inflight"] > 0:
                        state["weight_loads_inflight"] -= 1
                        self_outer.metrics.count_weight_load()
                        assert self_outer.vc.enqueue_scheduler_instruction(
                            {
                                "unit": "gsau",
                                "vdata": list(wb.get("data", [])),
                                "is_weight": True,
                                "expect_output": False,
                                "dtype": self_outer.dtype,
                            }
                        )
                        state["weight_row"] += 1
                        if state["weight_row"] >= self_outer.size:
                            state["weights_done"] = True

                    elif src == "vlsu" and dst == self_outer.A_REG and state["act_loads_inflight"] > 0:
                        state["act_loads_inflight"] -= 1
                        self_outer.metrics.count_act_load()
                        assert self_outer.vc.enqueue_scheduler_instruction(
                            {
                                "unit": "gsau",
                                "vdata": list(wb.get("data", [])),
                                "dst": self_outer.OUT_REG,
                                "is_weight": False,
                                "expect_output": True,
                                "dtype": self_outer.dtype,
                            }
                        )
                        state["act_row"] += 1

                    elif src == "gsau" and dst == self_outer.OUT_REG:
                        row_idx = state["next_out_row"]
                        if row_idx < self_outer.size:
                            self_outer.metrics.count_output_row()
                            if observed_rows[row_idx] is None:
                                observed_rows[row_idx] = [int(x) for x in list(wb.get("data", []))[: self_outer.size]]
                            state["pending_output_rows"].append(
                                {
                                    "row": row_idx,
                                    "data": list(wb.get("data", [])),
                                }
                            )
                            state["next_out_row"] += 1

                while (
                    state["weight_issue_row"] < self_outer.size
                    and state["weight_loads_inflight"] < state["load_issue_window"]
                ):
                    if not _issue_weight_load():
                        break

                if state["weights_done"]:
                    while (
                        state["act_issue_row"] < self_outer.size
                        and state["act_loads_inflight"] < state["load_issue_window"]
                    ):
                        if not _issue_act_load():
                            break

                completed_store_rows = []
                for row_idx, store_meta in list(state["store_inflight"].items()):
                    slot = int(self_outer.SPAD_OUT_BASE + row_idx) % self_outer.spad.bank_size
                    spad_vec = []
                    for lane in range(self_outer.vc.vector_len):
                        bank = _xor_bank(slot, lane, self_outer.spad.num_banks)
                        blob = self_outer.spad.tiles[0].banks[bank].mem[slot]
                        blob = bytes(blob) if blob is not None else b"\x00\x00"
                        if len(blob) < 2:
                            blob = blob + (b"\x00" * (2 - len(blob)))
                        spad_vec.append(int.from_bytes(blob[:2], "little", signed=False))
                    store_meta["age"] += 1
                    if spad_vec == (store_meta["data"] or []):
                        self_outer.dram.write(self_outer.DRAM_OUT + row_idx * row_bytes, _encode_row_u16(spad_vec))
                        self_outer.metrics.count_store_row()
                        state["completed_rows"].add(row_idx)
                        completed_store_rows.append(row_idx)
                    elif store_meta["age"] > 1000:
                        raise AssertionError("store did not commit to scratchpad")
                for row_idx in completed_store_rows:
                    state["store_inflight"].pop(row_idx, None)

                while state["pending_output_rows"] and len(state["store_inflight"]) < state["store_issue_window"]:
                    next_item = state["pending_output_rows"].pop(0)
                    row_idx = next_item["row"]
                    row_data = next_item["data"]
                    assert self_outer.vc.enqueue_memory(
                        {
                            "kind": "store",
                            "vls": 0,
                            "data": row_data,
                            "addr": self_outer.SPAD_OUT_BASE + row_idx,
                            "dtype": self_outer.dtype,
                        }
                    )
                    state["store_inflight"][row_idx] = {
                        "data": row_data,
                        "age": 0,
                    }

                state["cycles"] += 1
                self_outer.metrics.count_cycle()
                if len(state["completed_rows"]) >= self_outer.size:
                    self._finish()
                    return
                if state["cycles"] >= max_cycles:
                    self._finish()

        self_outer = self
        harness = RunHarness()

        def _issue_weight_load():
            if state["weight_issue_row"] >= self.size:
                return False
            assert self.vc.enqueue_memory(
                {
                    "kind": "load",
                    "vls": 0,
                    "dst": self.W_REG,
                    "addr": self.SPAD_WGT_BASE + state["weight_issue_row"],
                    "dtype": self.dtype,
                }
            )
            state["weight_issue_row"] += 1
            state["weight_loads_inflight"] += 1
            return True

        def _issue_act_load():
            if state["act_issue_row"] >= self.size:
                return False
            assert self.vc.enqueue_memory(
                {
                    "kind": "load",
                    "vls": 0,
                    "dst": self.A_REG,
                    "addr": self.SPAD_ACT_BASE + state["act_issue_row"],
                    "dtype": self.dtype,
                }
            )
            state["act_issue_row"] += 1
            state["act_loads_inflight"] += 1
            if state["act_issue_row"] >= self.size:
                self.sysarr_bridge.finish_inputs()
            return True

        _issue_weight_load()
        self.clk.objects = [self.vc, self.vls_bridge, self.sysarr_bridge, self.spad, harness]
        self.clk.schedule_next(0.0)
        self.sim.run()

        if len(state["completed_rows"]) < self.size:
            raise AssertionError("timed out waiting for DRAM->SPAD->VRF->GSAU->TPU->VRF->SPAD->DRAM")

        out = []
        for r in range(self.size):
            blob = self.dram.read(self.DRAM_OUT + r * row_bytes, row_bytes)
            out.append(_decode_row_u16(blob, self.size))
        mirror_out = self.mirror.outputs if self.mirror is not None else None
        return out, mirror_out, state["cycles"], self.metrics


def _act_u16(size: int) -> List[List[int]]:
    return [[((j * size + i) % 4) + 1 for j in range(size)] for i in range(size)]


def _weights_u16(size: int) -> List[List[int]]:
    return [[((i * size + j) % 8) + 1 for j in range(size)] for i in range(size)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TPU system end-to-end.")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--mirror", action="store_true", default=True)
    args = parser.parse_args()

    system = SysArrTPUSystem(size=args.size, dtype=args.dtype, mirror=args.mirror)
    act = _act_u16(args.size)
    wgt = _weights_u16(args.size)
    wgt_stream = [[wgt[r][c] for r in range(args.size)] for c in range(args.size - 1, -1, -1)]

    system.load_inputs(act, wgt_stream)
    got, mirror, cycles, metrics = system.run()

    result = {
        "cycles": cycles,
        "flops": metrics.flops,
        "bytes_moved": metrics.bytes_moved,
        "arithmetic_intensity": metrics.arithmetic_intensity(),
        "match_mirror": (got == mirror) if mirror is not None else None,
        "output_rows": len(got),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
