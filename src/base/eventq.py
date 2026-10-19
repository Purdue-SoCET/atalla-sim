"""EventQueue: a time-ordered queue of callbacks.

schedule(time, fn, *args) adds an event; run_until(t) runs every event due at
or before t, in time order (ties in scheduling order); run_all() runs
everything. Used by ClockDomain and Sim.
"""

from typing import Callable, Any, Tuple
import heapq

Time = float
EventHandle = Tuple[Time, int]

class EventQueue:
    """Callbacks ordered by the time they are due."""
    def __init__(self):
        self._queue = []
        self._time = 0.0
        self._next_id = 0

    def now(self) -> Time:
        return self._time

    def schedule(self, time: Time, callback: Callable[..., Any], *args, **kwargs) -> EventHandle:
        handle = (time, self._next_id)
        heapq.heappush(self._queue, (time, self._next_id, callback, args, kwargs))
        self._next_id += 1
        return handle

    def run_until(self, time: Time) -> None:
        while (self._queue and self._queue[0][0] <= time):
            t, eid, callback, args, kwargs = heapq.heappop(self._queue)
            self._time = t
            callback(*args, **(kwargs or {}))
        self._time = time


    def run_all(self) -> None:
        while (self._queue and self._queue[0][0] >= 0):
            t, eid, callback, args, kwargs = heapq.heappop(self._queue)
            self._time = t
            callback(*args, **(kwargs or {}))