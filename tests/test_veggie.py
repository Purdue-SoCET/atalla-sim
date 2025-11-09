# tests/test_veggie.py
from base.eventq import EventQueue
from base.clock_domain import ClockDomain
from base.core import Core
from base.sim import Sim
from vector_core.veggie_file import Veggie, OpBuffer

class IO:
    def __init__(self):
        # Veggie <-> OpBuffer signals
        self.read_reqs = []
        self.write_reqs = []
        self.vreg = {}
        self.vmask = {}
        self.dvalid = {}
        self.mvalid = {}
        self.ready = True
        self.ivalid = []
        # outputs container for op buffer
        self.vreg = {}
        self.vmask = {}

def build_sim():
    eq = EventQueue()
    clk = ClockDomain(eq, period=1.0)
    core = Core(eq)
    core.add_clock_domain(clk)
    sim = Sim()
    sim.init(eq, core)
    return eq, clk, sim

def test_vector_pipeline():
    eq, clk, sim = build_sim()

    veggie = Veggie(bank_count=2, regs_per_bank=8, dread_ports=1, dwrite_ports=1, mask_banks=1)
    opbuf = OpBuffer(num_pairs=1)

    veg_in = IO()
    veg_out = IO()
    op_in = veg_out
    op_out = IO()

    veggie.connect(veg_in, veg_out)
    opbuf.connect(op_in, op_out)

    # register to clock domain (optional here, we schedule ticks explicitly)
    clk.add_clocked(veggie)
    clk.add_clocked(opbuf)

    # schedule a write at t=0.0
    veg_in.write_reqs = [{"port": 0, "bank": 0, "addr": 2, "data": 99}]
    veg_in.read_reqs = []
    eq.schedule(0.0, veggie.Tick, 0.0)

    # schedule a read at t=1.0 (we set the request just before scheduling)
    def place_read(time):
        veg_in.write_reqs = []
        veg_in.read_reqs = [{"port": 0, "bank": 0, "addr": 2}]
        print(f"[{time}] test: placed read_req")
    eq.schedule(1.0, place_read, 1.0)

    # schedule veggie to service the read shortly after placement
    eq.schedule(1.01, veggie.Tick, 1.01)

    # schedule opbuf to sample veggie output after veggie produced it
    eq.schedule(1.02, opbuf.Tick, 1.02)

    # inject mask so op buffer can combine it with data
    def inject_mask(time):
        op_in.mvalid = {0: True}
        op_in.vmask = {0: 0xFF}
        print(f"[{time}] Injected mask into op_in")
    eq.schedule(1.03, inject_mask, 1.03)

    # call opbuf again to observe the combined result
    eq.schedule(1.04, opbuf.Tick, 1.04)

    # run sim
    sim.run(until=2.0)

    # results
    print("\n--- RESULTS ---")
    print("veg_out.vreg:", veg_out.vreg)
    print("veg_out.dvalid:", veg_out.dvalid)
    print("op_out.ivalid:", op_out.ivalid)
    print("op_out.vreg:", getattr(op_out, "vreg", {}))
    print("op_out.vmask:", getattr(op_out, "vmask", {}))
    print("----------------\n")

    # assertions
    assert veg_out.vreg.get(0, None) == 99, f"Veggie readback failed: {veg_out.vreg}"
    assert op_out.ivalid and op_out.ivalid[0] is True, f"OpBuffer didn't mark ready: {op_out.ivalid}"
    print("Test passed")

if __name__ == "__main__":
    test_vector_pipeline()
