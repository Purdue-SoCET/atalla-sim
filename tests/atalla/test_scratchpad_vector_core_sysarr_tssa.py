import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..//..", "src")))

from base.clock_domain import ClockDomain
from base.core import Core
from base.eventq import EventQueue
from base.sim import Sim
from scratchpad.sc_sram_banks import _xor_bank
from scratchpad.scratchpad import Scratchpad
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


class GSAUTSSABridge:
    """Consumes VectorCore GSAU requests, drives TSSA model, returns responses."""

    def __init__(self, vc: VectorCore, sa: SystolicArrayTSSA):
        self.vc = vc
        self.sa = sa
        self.size = sa.size
        self._pending_meta: List[Dict] = []
        self._out_read_idx = 0

    def _pack_rsp(self, out_row: List[float], meta: Dict) -> Dict:
        vec = [0.0] * self.vc.vector_len
        for i, val in enumerate(out_row[: self.size]):
            vec[i] = float(val)
        return {"vdata": vec, "meta": dict(meta)}

    def tick(self) -> None:
        # Default: idle compute controls.
        self.sa.set_control(weight_en=False, mac_shift=False, start=False, stall=False)

        req = self.vc.pop_systolic_request()
        if req is not None:
            vdata = [float(x) for x in req.get("vdata", [])]
            row = vdata[: self.size]
            if len(row) < self.size:
                row += [0.0] * (self.size - len(row))

            if bool(req.get("is_weight", False)):
                assert self.sa.enqueue_weights(row)
                self.sa.set_control(weight_en=True, mac_shift=False, start=False, stall=False)
            else:
                assert self.sa.enqueue(row)
                assert self.sa.enqueue_psums([0.0] * self.size)
                self.sa.set_control(weight_en=False, mac_shift=True, start=True, stall=False)
                if bool(req.get("expect_output", True)):
                    self._pending_meta.append(dict(req.get("meta", {})))

        self.sa.tick()

        buf = self.sa.get_buffer()
        while self._out_read_idx < len(buf) and self._pending_meta:
            out_row = [float(x) for x in buf[self._out_read_idx]]
            meta = self._pending_meta.pop(0)
            assert self.vc.push_systolic_response(self._pack_rsp(out_row, meta))
            self._out_read_idx += 1


def test_scratchpad_vector_core_sysarr_tssa_end_to_end():
    eq, clk, sim = build_sim()

    vc = VectorCore(veggie_size=128, lane_count=4, vls_count=1, fu_latencies={"alu": 1})
    spad = Scratchpad(
        num_banks=8,
        bank_size=32,
        read_latency=1,
        write_latency=1,
        xbar_delay=1,
        elem_bytes=2,
        frontend_queue_size=4,
    )
    sa = SystolicArrayTSSA(size=1)

    vls_bridge = VLSFrontendBridge(vc, spad, vls_id=0, frontend_id=0)
    sysarr_bridge = GSAUTSSABridge(vc, sa)

    # DRAM model for this integration test.
    dram: Dict[int, List[int]] = {}
    DRAM_ACT = 0x100
    DRAM_OUT = 0x108

    SPAD_ACT = 6
    SPAD_OUT = 10

    activation = [7, 0, 0, 0, 0, 0, 0, 0]
    weight = [3, 0, 0, 0, 0, 0, 0, 0]
    expected = [21, 0, 0, 0, 0, 0, 0, 0]

    # DRAM -> scratchpad
    dram[DRAM_ACT] = list(activation)
    _write_slot_vector_u16(spad, SPAD_ACT, dram[DRAM_ACT])

    # scratchpad -> VRF
    assert vc.enqueue_memory({"kind": "load", "vls": 0, "dst": 2, "addr": SPAD_ACT})

    state = {
        "cycles": 0,
        "act_loaded": False,
        "gsau_weight_issued": False,
        "gsau_input_issued": False,
        "gsau_wb_seen": False,
        "store_issued": False,
        "spad_written": False,
    }

    def _step(time: float):
        vc.tick()
        vls_bridge.tick()
        sysarr_bridge.tick()
        spad.tick(time)

        if vc.wb_valid and vc.last_wb is not None:
            wb = vc.last_wb
            src = wb.get("source")
            dst = wb.get("dst")

            if src == "vlsu" and dst == 2:
                state["act_loaded"] = True

            if state["act_loaded"] and (not state["gsau_weight_issued"]):
                assert vc.enqueue_scheduler_instruction(
                    {
                        "unit": "gsau",
                        "vdata": weight,
                        "is_weight": True,
                        "expect_output": False,
                    }
                )
                state["gsau_weight_issued"] = True

            if state["gsau_weight_issued"] and (not state["gsau_input_issued"]):
                assert vc.enqueue_scheduler_instruction(
                    {"unit": "gsau", "src": 2, "dst": 3, "is_weight": False, "expect_output": True}
                )
                state["gsau_input_issued"] = True

            if src == "gsau" and dst == 3:
                state["gsau_wb_seen"] = True

            if state["gsau_wb_seen"] and (not state["store_issued"]):
                assert vc.enqueue_memory({"kind": "store", "vls": 0, "src": 3, "addr": SPAD_OUT})
                state["store_issued"] = True

        if state["store_issued"]:
            spad_vec = _read_slot_vector_u16(spad, SPAD_OUT, vc.vector_len)
            if spad_vec == expected:
                state["spad_written"] = True
                # scratchpad -> DRAM
                dram[DRAM_OUT] = list(spad_vec)
                return

        state["cycles"] += 1
        if state["cycles"] >= 256:
            return
        eq.schedule(time + 1.0, _step, time + 1.0)

    eq.schedule(0.0, _step, 0.0)
    sim.run()

    if not state["spad_written"]:
        raise AssertionError("timed out waiting for DRAM->SPAD->VRF->GSAU->TSSA->VRF->SPAD->DRAM")

    assert vc.dump_vreg(2) == [float(x) for x in activation]
    assert vc.dump_vreg(3) == [float(x) for x in expected]
    assert _read_slot_vector_u16(spad, SPAD_OUT, vc.vector_len) == expected
    assert dram[DRAM_OUT] == expected


if __name__ == "__main__":
    test_scratchpad_vector_core_sysarr_tssa_end_to_end()
