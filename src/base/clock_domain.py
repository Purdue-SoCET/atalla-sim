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
        self.running = False

    def add_clocked(self, obj: Clocked) -> None:
        self.objects.append(obj)

    def remove_clocked(self, obj: Clocked) -> None:
        self.objects.remove(obj)

    def start(self, time: Time = 0.0) -> None:
        self.running = True
        self.schedule_next(time)

    def stop(self) -> None:
        self.running = False

    def tick(self, time: Time) -> None:
        if not self.running:
            return
        for obj in list(self.objects):
            obj.tick(time)
        if self.running:
            self.schedule_next(time)

    def schedule_next(self, time: Time) -> None:
        if not self.running:
            self.running = True
        next_time = time + self.period
        self.event_queue.schedule(next_time, self.tick, next_time)