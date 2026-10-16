"""A clock: ticks a list of objects every `period` time units.

ClockDomain puts its own tick on the EventQueue, ticks every object added
with add_clocked() in the order they were added, and schedules the next tick.
Most of atalla-sim doesn't use this (the platforms tick a CompositeClocked
tree by hand, base/sched.py); the event-queue harnesses do. To clock a new
component from a domain, give it a tick(time) method and add_clocked() it.
"""

from base.eventq import EventQueue
from base.clocked_object import Clocked

Time = float

class ClockDomain:

    """A clock with a period, ticking `objects` in order on each edge."""
    def __init__(self, event_queue: EventQueue, period: Time, name: str = "clk") -> None:
        self.event_queue = event_queue
        self.period = period
        self.name = name
        self.objects = []
        self.next_time = 0.0
        self.running = False

    def add_clocked(self, obj: Clocked) -> None:
        self.objects.append(obj)

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