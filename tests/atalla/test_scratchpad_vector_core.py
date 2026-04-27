import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..//..", "src")))

from base.eventq import EventQueue
from base.clock_domain import ClockDomain
from base.clocked_object import Clocked
from base.core import Core
from base.sim import Sim

from memory.sc_sram_banks import _xor_bank
from memory.scratchpad import Scratchpad
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


class VLSFrontendBridge(Clocked):
    def __init__(self, vc: VectorCore, spad: Scratchpad, vls_id: int = 0, frontend_id: int = 0):
        super().__init__()
        self.vc = vc
        self.spad = spad
        self.vls_id = int(vls_id)
        self.frontend_id = int(frontend_id)
        self._next_load_id = 0
        self._completed_load_ids = set()
        self.stores_seen = 0
        self.store_addrs = []

    def _on_frontend_read(self, load_id: int, addr: int, lanes) -> None:
        if load_id in self._completed_load_ids:
            return
        self._completed_load_ids.add(load_id)
        data = _decode_lanes_u16(lanes, self.vc.vector_len)
        assert self.vc.push_scratchpad_response(self.vls_id, {"addr": addr, "data": data})

    def tick(self, time: float = None) -> None:
        while True:
            req = self.vc.pop_scratchpad_request(self.vls_id)
            if req is None:
                break

            addr = int(req.get("addr", 0))
            if req["kind"] == "store":
                self.stores_seen += 1
                self.store_addrs.append(addr)
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


def test_scratchpad_vector_core_load_compute_store_back():
    eq, clk, sim = build_sim()
    vc = VectorCore(veggie_size=128, lane_count=4, vls_count=1, fu_latencies={"alu": 1}, dtype="fp16")
    spad = Scratchpad(
        num_banks=8,
        bank_size=32,
        read_latency=1,
        write_latency=1,
        xbar_delay=1,
        elem_bytes=2,
        frontend_queue_size=4,
    )

    bridge = VLSFrontendBridge(vc, spad, vls_id=0, frontend_id=0)
    src = [3, 6, 9, 12, 15, 18, 21, 24]
    computed = [x + 1 for x in src]
    src_addr = 6
    dst_addr = 10

    # Preload source vector directly into scratchpad slot, then load it via VLS.
    _write_slot_vector_u16(spad, src_addr, src)
    assert vc.enqueue_memory({"kind": "load", "vls": 0, "dst": 5, "addr": src_addr, "dtype": "fp16"})

    state = {
        "cycles": 0,
        "load_wb_seen": False,
        "compute_issued": False,
        "store_issued": False,
        "store_committed": False,
    }

    class LoadComputeStoreHarness(Clocked):
        def __init__(self):
            super().__init__()
            self.done = False

        def tick(self, time: float) -> None:
            if self.done:
                return

            if vc.wb_valid and vc.last_wb is not None:
                source = vc.last_wb.get("source")
                dst = vc.last_wb.get("dst")
                if source == "vlsu" and dst == 5:
                    state["load_wb_seen"] = True
                if state["load_wb_seen"] and (not state["compute_issued"]):
                    assert vc.enqueue_compute("add", dst=6, src0=5, src1=[1] * vc.vector_len)
                    state["compute_issued"] = True
                if state["compute_issued"] and (not state["store_issued"]) and source == "datapath" and dst == 6:
                    assert vc.enqueue_memory({"kind": "store", "vls": 0, "src": 6, "addr": dst_addr, "dtype": "fp16"})
                    state["store_issued"] = True

            if state["store_issued"]:
                written = _read_slot_vector_u16(spad, dst_addr, vc.vector_len)
                if written == computed:
                    state["store_committed"] = True
                    self.done = True
                    clk.stop()
                    return

            state["cycles"] += 1
            if state["cycles"] >= 256:
                self.done = True
                clk.stop()

    clk.objects = [vc, bridge, spad, LoadComputeStoreHarness()]
    clk.schedule_next(0.0)
    sim.run()

    if not state["store_committed"]:
        raise AssertionError("timed out waiting for load->compute->store-back flow")

    assert vc.dump_vreg(5) == src
    assert vc.dump_vreg(6) == computed
    assert _read_slot_vector_u16(spad, dst_addr, vc.vector_len) == computed


if __name__ == "__main__":
    test_scratchpad_vector_core_load_compute_store_back()
