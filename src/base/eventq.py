from dataclasses import dataclass
from typing import Callable, Any, Optional, Tuple, Iterable, List

Time = float
EventHandle = Tuple[Time, int]

@dataclass
class Event:
    time: Time
    callback: Callable[..., Any]
    args: Tuple[Any, ...] = ()
    kwargs: dict = None
    cancelled: bool = False


class EventQueue:
    def now(self) -> Time:
        pass

    def schedule(self, time: Time, callback: Callable[..., Any], *args, **kwargs) -> EventHandle:
        pass

    def cancel(self, handle: EventHandle) -> bool:
        pass

    def run_until(self, time: Time) -> None:
        pass

    def run_all(self) -> None:
        pass