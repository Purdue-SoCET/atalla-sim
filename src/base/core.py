from typing import Callable, Any, Optional, Tuple, Iterable, List
from base.eventq import EventQueue
from base.clock_domain import ClockDomain

Time = float
EventHandle = Tuple[Time, int]

class Core:

    def __init__(self, event_queue: EventQueue) -> None:
        self.event_queue = event_queue
        self.domains = []

    def add_clock_domain(self, domain: ClockDomain) -> None:
        self.domains.append(domain)
    
    def reset(self) -> None:
        pass