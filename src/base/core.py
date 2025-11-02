from dataclasses import dataclass
from typing import Callable, Any, Optional, Tuple, Iterable, List
from eventq import EventQueue
from clock_domain import ClockDomain

Time = float
EventHandle = Tuple[Time, int]

@dataclass
class Core:

    def __init__(self, event_queue: EventQueue) -> None:
        pass

    def add_clock_domain(self, domain: ClockDomain) -> None:
        pass
    
    def reset(self) -> None:
        pass