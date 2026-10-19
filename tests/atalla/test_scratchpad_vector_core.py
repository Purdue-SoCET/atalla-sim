import pytest

from base.testing import build_sim
from base.clocked_object import Clocked

from memory.scratchpad import Scratchpad
from vector_core.vector_core import VectorCore
from atalla.sysarr_tpu_system import VLSFrontendBridge
from atalla.sysarr_tpu_experiment import _read_slot_vector_u16


def _write_slot_vector_u16(spad: Scratchpad, addr: int, values):
    tile = 0
    slot = int(addr) % spad.bank_size
    for lane, value in enumerate(list(values)):
        bank = lane
        spad.tiles[tile].banks[bank].mem[slot] = int(value).to_bytes(2, "little", signed=False)


def test_scratchpad_vector_core_load_compute_store_back():
    eq, clk, sim = build_sim()
    vc = VectorCore(veggie_size=128, lane_count=4, vls_count=1, fu_latencies={"alu": 1}, dtype="fp16")
    spad = Scratchpad(
        num_banks=8,
        bank_size=32,
        elem_bytes=2,
        num_tiles=1,
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
                    assert vc.enqueue_scheduler_instruction({
                        "unit": "datapath",
                        "op": "add",
                        "dst": 6,
                        "src0": 5,
                        "src1": [1] * vc.vector_len,
                    })
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

    assert vc.read_vreg(5) == src
    assert vc.read_vreg(6) == computed
    assert _read_slot_vector_u16(spad, dst_addr, vc.vector_len) == computed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
