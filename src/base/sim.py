from dataclasses import dataclass
from eventq import EventQueue
from core import Core
from typing import Optional, Tuple

Time = float
EventHandle = Tuple[Time, int]

@dataclass
class Sim:
    def __init__(self) -> None:
        pass

    def init(self, event_queue: EventQueue, core: Core) -> None:
        pass

    def run(self, until: Optional[Time] = None) -> None:
        pass

    def stop(self) -> None:
        pass