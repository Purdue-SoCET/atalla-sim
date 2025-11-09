from base.eventq import EventQueue
from base.clocked_object import Clocked
from typing import Tuple

Time = float
EventHandle = Tuple[Time, int]

class ClockDomain:

    def __init__(self, event_queue: EventQueue, period: Time, name: str = "clk") -> None:
        self.event_queue = event_queue
        self.period = period
        self.name = name
        self.objects = []
        self.next_time = 0.0

    def add_clocked(self, obj: Clocked) -> None:
        self.objects.append(obj)

    def remove_clocked(self, obj: Clocked) -> None:
        self.objects.remove(obj)

    def _Tick(self, time: Time) -> None:
        for obj in self.objects:
            obj._Tick(time)
        self.schedule_next(time)

    def schedule_next(self, time: Time) -> None:
        next_time = time + self.period
        self.event_queue.schedule(next_time, self._Tick, next_time)