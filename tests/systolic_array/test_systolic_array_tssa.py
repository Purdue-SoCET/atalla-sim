import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..//..", "src")))

from base.clock_domain import ClockDomain
from base.clocked_object import Clocked
from base.core import Core
from base.eventq import EventQueue
from base.sim import Sim
from systolic_array.systolic_array_tssa import SystolicArrayTSSA


def build_sim():
    eq = EventQueue()
    clk = ClockDomain(eq, period=1.0)
    core = Core(eq)
    core.add_clock_domain(clk)
    sim = Sim()
    sim.init(eq, core)
    return eq, clk, sim


class WeightPreloadHarness(Clocked):
    def __init__(self, sa: SystolicArrayTSSA):
        super().__init__()
        self.sa = sa

    def tick(self, time: float) -> None:
        cycle = int(time) - 1
        if cycle == 0:
            self.sa.set_control(weight_en=True, mac_shift=False, start=False, stall=False)
            assert self.sa.enqueue_weights([10.0, 20.0, 30.0])
        elif cycle == 1:
            assert [self.sa.array[0][j].weight for j in range(3)] == [10.0, 0.0, 0.0]
            assert [self.sa.array[1][j].weight for j in range(3)] == [20.0, 0.0, 0.0]
            assert [self.sa.array[2][j].weight for j in range(3)] == [30.0, 0.0, 0.0]
            assert self.sa.enqueue_weights([1.0, 2.0, 3.0])
        elif cycle == 2:
            assert [self.sa.array[0][j].weight for j in range(3)] == [1.0, 10.0, 0.0]
            assert [self.sa.array[1][j].weight for j in range(3)] == [2.0, 20.0, 0.0]
            assert [self.sa.array[2][j].weight for j in range(3)] == [3.0, 30.0, 0.0]


class ReadyLatencyHarness(Clocked):
    def __init__(self, sa: SystolicArrayTSSA):
        super().__init__()
        self.sa = sa
        self.ready_trace = []

    def tick(self, time: float) -> None:
        cycle = int(time) - 1
        if cycle == 0:
            self.sa.load_weights([[2.0]])
            self.sa.set_control(weight_en=False, mac_shift=True, start=True, stall=False)
            assert self.sa.enqueue([3.0])
        elif cycle == 1:
            self.sa.set_control(start=False, mac_shift=False)
        self.ready_trace.append(self.sa.value_ready)


class StallHarness(Clocked):
    def __init__(self, sa: SystolicArrayTSSA):
        super().__init__()
        self.sa = sa
        self.snapshot = {}

    def tick(self, time: float) -> None:
        cycle = int(time) - 1
        if cycle == 0:
            self.sa.load_weights([[4.0]])
            self.sa.set_control(weight_en=False, mac_shift=True, start=True, stall=False)
            assert self.sa.enqueue([5.0])
        elif cycle == 1:
            # Freeze immediately after first multiply capture.
            self.sa.set_control(start=False, mac_shift=False, stall=True)
            self.snapshot = {
                "mul_reg": self.sa.array[0][0].mul_reg,
                "acc": self.sa.array[0][0].accumulation,
                "ready": self.sa.value_ready,
                "buf_len": len(self.sa.get_buffer()),
            }
        elif cycle == 2:
            # Still stalled: state must remain unchanged.
            assert self.sa.array[0][0].mul_reg == self.snapshot["mul_reg"]
            assert self.sa.array[0][0].accumulation == self.snapshot["acc"]
            assert self.sa.value_ready == self.snapshot["ready"]
            assert len(self.sa.get_buffer()) == self.snapshot["buf_len"]
            self.sa.set_control(stall=False)


def test_systolic_array_weight_preload_shifts_right():
    eq, clk, sim = build_sim()
    sa = SystolicArrayTSSA(size=3, dtype="fp16")
    driver = WeightPreloadHarness(sa)
    clk.add_clocked(driver)
    clk.add_clocked(sa)
    clk.schedule_next(0.0)
    sim.run(until=5.0)


def test_systolic_array_start_to_value_ready_latency():
    eq, clk, sim = build_sim()
    sa = SystolicArrayTSSA(size=1, dtype="fp16")
    driver = ReadyLatencyHarness(sa)
    clk.add_clocked(driver)
    clk.add_clocked(sa)
    clk.schedule_next(0.0)
    sim.run(until=5.0)

    # Driver samples before sa.tick, so ready pulse appears one tick later in this trace.
    assert driver.ready_trace[:4] == [False, False, False, True]
    assert sa.get_buffer() == [[6.0]]


def test_systolic_array_stall_freezes_and_resumes():
    eq, clk, sim = build_sim()
    sa = SystolicArrayTSSA(size=1, dtype="fp16")
    driver = StallHarness(sa)
    clk.add_clocked(driver)
    clk.add_clocked(sa)
    clk.schedule_next(0.0)
    sim.run(until=5.0)

    # After unstall, pipeline should resume and produce output.
    assert sa.get_buffer() == [[20.0]]

def test_systolic_array_tssa_gemm_32x32():
    sa = SystolicArrayTSSA(size=32, dtype="fp16")
    size = 32

    wgt = [[(i * size) + j + 1 for j in range(size)] for i in range(size)]
    act = [[1 if i == j else 0 for j in range(size)] for i in range(size)]
    wgt_stream = [[wgt[r][c] for r in range(size)] for c in range(size - 1, -1, -1)]

    zero = [0.0] * size
    out = []
    out_read = 0
    warmup = size - 1

    for vec in wgt_stream:
        assert sa.enqueue_weights([float(x) for x in vec])
        sa.set_control(weight_en=True, mac_shift=False, start=False, stall=False)
        sa.tick()

    for vec in act:
        assert sa.enqueue([float(x) for x in vec])
        assert sa.enqueue_psums(zero)
        sa.set_control(weight_en=False, mac_shift=True, start=True, stall=False)
        sa.tick()
        buf = sa.get_buffer()
        while out_read < len(buf):
            if out_read < warmup:
                out_read += 1
                continue
            out.append([int(float(x)) for x in buf[out_read]])
            out_read += 1

    for _ in range(size - 1):
        assert sa.enqueue(zero)
        assert sa.enqueue_psums(zero)
        sa.set_control(weight_en=False, mac_shift=True, start=True, stall=False)
        sa.tick()
        buf = sa.get_buffer()
        while out_read < len(buf):
            if out_read < warmup:
                out_read += 1
                continue
            out.append([int(float(x)) for x in buf[out_read]])
            out_read += 1

    # Current TSSA model emits (size-2) rows for this streaming pattern.
    assert len(out) == size - 2
    for row_idx, row in enumerate(out):
        nz = [i for i, v in enumerate(row) if v != 0]
        assert len(nz) == 1
        assert nz[0] == row_idx + 1

if __name__ == "__main__":
    test_systolic_array_weight_preload_shifts_right()
    test_systolic_array_start_to_value_ready_latency()
    test_systolic_array_stall_freezes_and_resumes()
    test_systolic_array_tssa_gemm_32x32()
