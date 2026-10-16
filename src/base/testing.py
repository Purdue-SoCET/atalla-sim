"""build_sim: the event queue, clock and simulator, wired together.

Every test and harness that needs a simulated clock uses this one, so the
wiring can't drift between them.
"""

from base.clock_domain import ClockDomain
from base.core import Core
from base.eventq import EventQueue
from base.sim import Sim


def build_sim(period: float = 1.0, name: str = "clk"):
    """Return (event_queue, clock_domain, sim) wired together and ready to run."""
    eq = EventQueue()
    clk = ClockDomain(eq, period=period, name=name)
    core = Core(eq)
    core.add_clock_domain(clk)
    sim = Sim()
    sim.init(eq, core)
    return eq, clk, sim
