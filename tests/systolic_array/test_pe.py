import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..//..", "src")))

from base.clock_domain import ClockDomain
from base.clocked_object import Clocked
from base.core import Core
from base.eventq import EventQueue
from base.sim import Sim
from systolic_array.systolic_array import PE


def build_sim():
    eq = EventQueue()
    clk = ClockDomain(eq, period=1.0)
    core = Core(eq)
    core.add_clock_domain(clk)
    sim = Sim()
    sim.init(eq, core)
    return eq, clk, sim


class PEHorizontalShiftHarness(Clocked):
    def __init__(self, src: PE, dst: PE):
        super().__init__()
        self.src = src
        self.dst = dst

    def tick(self, time: float) -> None:
        cycle = int(time) - 1
        if cycle == 0:
            self.src._input(3.25)
            self.src.shift(2, "activation")
            assert self.dst.activation_latch == 3.25
        elif cycle == 1:
            self.src._weight(7.5)
            self.src.shift(2, "weight")
            assert self.dst.weight == 7.5


class PEVerticalAccumHarness(Clocked):
    def __init__(self, src: PE, dst: PE):
        super().__init__()
        self.src = src
        self.dst = dst

    def tick(self, time: float) -> None:
        if int(time) - 1 == 0:
            self.src._accumulation(9.75)
            self.src.shift(3, "accumulation")
            assert self.dst.accumulation == 9.75


def test_pe_horizontal_activation_and_weight_shift():
    eq, clk, sim = build_sim()
    left = PE()
    right = PE()
    left.links[2] = right

    clk.add_clocked(PEHorizontalShiftHarness(left, right))
    clk.schedule_next(0.0)
    sim.run()

    assert right.activation_latch == 3.25
    assert right.weight == 7.5


def test_pe_vertical_accumulation_shift():
    eq, clk, sim = build_sim()
    top = PE()
    bottom = PE()
    top.links[3] = bottom

    clk.add_clocked(PEVerticalAccumHarness(top, bottom))
    clk.schedule_next(0.0)
    sim.run()

    assert bottom.accumulation == 9.75
