from typing import Optional, Tuple
from base.debug import dprintf

Time = float
EventHandle = Tuple[Time, int]

class Clocked:
    def __init__(self) -> None:
        self._clocked_last_tick = -1

    def _consume_tick(self, time: Optional[Time], attr_name: str = "_clocked_last_tick") -> Optional[int]:
        current = int(getattr(self, attr_name, -1))
        if time is None:
            current += 1
            setattr(self, attr_name, current)
            return current

        cycle = int(time)
        if isinstance(time, float):
            rounded = int(round(time))
            if abs(time - rounded) < 1e-6:
                cycle = rounded

        if cycle < current:
            current = cycle - 1
            setattr(self, attr_name, current)

        if cycle <= current:
            return None

        setattr(self, attr_name, cycle)
        return cycle

    def tick(self, time: Optional[Time] = None) -> None:
        dprintf("Clocked", f"{self.__class__.__name__} tick at {time}")

    def __Cycles(self, time: Time)-> None:
        pass

    def __update(self, time: Time)-> None:
        pass
