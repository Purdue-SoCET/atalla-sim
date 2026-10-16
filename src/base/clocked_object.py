"""Clocked: the base class of every component that does work each cycle.

A subclass overrides tick(time). What comes with it:
  _consume_tick(time)  turns the caller's time into a cycle number and
                       returns None if this cycle was already ticked, so a
                       component reached twice in one cycle runs once.
  next_wake(now)       the next cycle this component has work, or None to
                       sleep (base/sched.py). The default never sleeps.
  wake()               called by whoever hands this component work, so a
                       sleeping component ticks again.
  add_child(c)         make `c` a child, ticked by tick_children(time) only
                       on the cycles it is due.

To let a component sleep: override next_wake to return None exactly when a
tick would change nothing, and call its wake() everywhere work reaches it
from outside its own tick. base/sched.py states the rules in full.
"""

from typing import Optional
from base.debug import dprintf

Time = float

class Clocked:
    """A component that is ticked once a cycle. Override tick(time)."""
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
        """Ask to be ticked at `cycle`."""
        group = self._wake_group
        if group is not None:
            group.request_wake(self._wake_index, int(cycle))

    def wake(self) -> None:
        """Work has arrived from outside this component's tick: tick it again
        as soon as it would have seen the work had it never slept."""
        group = self._wake_group
        if group is not None:
            group.wake(self._wake_index)

    # -- children -----------------------------------------------------------------
    #: Order children by round robin, starting from child (cycle mod count),
    #: instead of the order they were added.
    CHILDREN_ROTATE = False
    _children = None

    def add_child(self, child):
        """Make `child` one of the components tick_children() ticks."""
        if self._children is None:
            from base.sched import WakeGroup
            self._children = WakeGroup(type(self).__name__, rotate=self.CHILDREN_ROTATE)
        return self._children.add(child)

    def tick_children(self, time: Optional[Time] = None) -> None:
        """Tick the children that are due, in order."""
        if self._children is not None:
            self._children.tick(time)

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
