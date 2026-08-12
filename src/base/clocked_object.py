from typing import Optional, Tuple
from base.debug import dprintf

Time = float
EventHandle = Tuple[Time, int]

class Clocked:
    # Set when the object is registered with a base.sched.WakeGroup.
    _wake_group = None
    _wake_index = -1

    def __init__(self) -> None:
        self._clocked_last_tick = -1

    def next_wake(self, now: int) -> Optional[int]:
        """Earliest cycle at which this object must run; None if quiescent.

        The default says "tick me every cycle", which is exactly what the
        simulator did before scheduling existed -- so an object that has not
        been converted behaves identically. Overrides may only ever be
        CONSERVATIVE: returning a cycle earlier than strictly necessary just
        costs a wasted tick, but returning one too late drops work silently.
        """
        return now + 1

    def request_wake(self, cycle: int) -> None:
        """Ask to be ticked at `cycle` (used by whoever hands this object work)."""
        group = self._wake_group
        if group is not None:
            group.request_wake(self._wake_index, int(cycle))

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
