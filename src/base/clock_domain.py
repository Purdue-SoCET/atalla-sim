from eventq import EventQueue
from clocked_object import Clocked
from typing import Tuple

Time = float
EventHandle = Tuple[Time, int]

class ClockDomain:

    def __init__(self, event_queue: EventQueue, period: Time, name: str = "clk") -> None:
        pass

    def add_clocked(self, obj: Clocked) -> None:
        pass

    def remove_clocked(self, obj: Clocked) -> None:
        pass

    def __Tick(self, time: Time) -> None:
        pass

    def schedule_next(self, time: Time) -> None:
        pass