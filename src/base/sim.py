from eventq import EventQueue
from core import Core
from typing import Optional, Tuple

Time = float
EventHandle = Tuple[Time, int]

class Sim:
    def __init__(self) -> None:
        self.event_queue = None
        self.core = None

    def init(self, event_queue: EventQueue, core: Core) -> None:
        self.event_queue = event_queue
        self.core = core

    def run(self, until: Optional[Time] = None) -> None:
        if until is None:
            self.event_queue.run_all()
        else:
            self.event_queue.run_until(until)

    def stop(self) -> None:
        pass