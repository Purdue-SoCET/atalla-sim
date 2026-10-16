"""Core: holds an event queue and the clock domains run from it.

A container for base/sim.py's Sim; it does no work of its own.
"""

from base.eventq import EventQueue
from base.clock_domain import ClockDomain

Time = float

class Core:

    """An event queue and the clock domains attached to it."""
    def __init__(self, event_queue: EventQueue) -> None:
        self.event_queue = event_queue
        self.domains = []

    def add_clock_domain(self, domain: ClockDomain) -> None:
        self.domains.append(domain)
    
    def reset(self) -> None:
        pass