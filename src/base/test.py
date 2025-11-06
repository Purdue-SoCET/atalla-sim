from eventq import EventQueue
from clock_domain import ClockDomain
from core import Core
from clocked_object import Clocked
from sim import Sim

eq = EventQueue()
clk = ClockDomain(eq, period=1.0)
core = Core(eq)
core.add_clock_domain(clk)
sim = Sim()
sim.init(eq, core)

class MyCPU(Clocked):
    def _Tick(self, time): print(f"CPU tick {time}")

cpu = MyCPU()
clk.add_clocked(cpu)
clk.schedule_next(0)
sim.run(until=5)
