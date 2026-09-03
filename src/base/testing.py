"""Shared helpers for tests and scratch drivers.

`build_sim` was copy-pasted, byte-identical, into 17 test files; it lives here
now so the wiring cannot drift between them.
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


def tick_n(obj, n: int, start: int = 0):
    """Tick `obj` for `n` cycles, returning the next unused cycle number."""
    for t in range(start, start + n):
        obj.tick(float(t))
    return start + n


def run_until(obj, done, max_cycles: int = 200, start: int = 0, hook=None):
    """Tick until `done()` is true; return the cycle it became true.

    `hook(cycle)` runs after each tick, for driving peripherals. Raises rather
    than returning a sentinel, so a hung test fails loudly.
    """
    for t in range(start, start + max_cycles):
        obj.tick(float(t))
        if hook is not None:
            hook(t)
        if done():
            return t
    raise AssertionError("condition not reached within %d cycles" % max_cycles)
