from typing import Callable, Any, Optional, Tuple, Iterable, List
import heapq

Time = float
EventHandle = Tuple[Time, int]

class Event:
    time: Time
    callback: Callable[..., Any]
    args: Tuple[Any, ...] = ()
    kwargs: dict = None
    cancelled: bool = False


class EventQueue:
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

    def cancel(self, handle: EventHandle) -> bool:
        pass

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