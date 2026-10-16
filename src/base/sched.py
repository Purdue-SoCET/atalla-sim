"""Wake scheduling: tick a component only on the cycles it has work.

Most components are idle most cycles (a VLSU with empty queues, a DMA backend
with no transfer, the systolic array between GEMMs), and ticking them does
nothing but cost simulation time. A `WakeGroup` holds sibling components and
ticks only the ones that are due; the rest sleep until something wakes them.
Results never change: a component only sleeps when its tick would be a no-op.

How a component takes part (base/clocked_object.py):

    next_wake(now)   after each tick: the next cycle it has work, or None to
                     sleep. The default, now + 1, never sleeps -- so a
                     component that does nothing special behaves as before.
    wake()           whoever hands it work calls this, so it ticks again: in
                     this cycle if its group has not reached it yet, else in
                     the next one -- exactly when it would have seen the work
                     had it been ticked every cycle.

and a parent puts its children in a group with Clocked.add_child, ticking
them with tick_children (CompositeClocked below does this for the platform).

The rule that keeps this exact: next_wake may say None only when ticking
would change nothing at all -- not a value, not a counter -- until someone
calls wake(). Returning a cycle too early is merely a wasted tick; too late
silently loses work. A component that sleeps also stops counting cycles, so
it must not use its own last-tick cycle for timing while asleep.

ATALLA_SCHED=legacy turns every group off (every member ticked every cycle),
so the two modes can be compared: they must give identical results.
"""

import heapq
import os
from bisect import insort
from typing import List, Optional

from base.clocked_object import Clocked

SCHED_MODE = os.environ.get("ATALLA_SCHED", "event").strip().lower()
SCHED_ENABLED = SCHED_MODE != "legacy"


class WakeGroup:
    """Sibling components, ticked in a fixed order but only when due.

    Due members tick in registration order, or with `rotate` in round-robin
    order starting from member (cycle mod count), as RoundRobinBackendTicker
    needs. A member woken during the group's tick runs in the same cycle if
    it comes later in this cycle's order, else in the next cycle.
    """

    __slots__ = ("name", "enabled", "rotate", "_members", "_heap", "_soon", "_next",
                 "cycle", "_ticking", "_pos", "_due", "ticks", "skipped")

    def __init__(self, name: str = "", enabled: Optional[bool] = None,
                 rotate: bool = False) -> None:
        self.name = name
        self.enabled = SCHED_ENABLED if enabled is None else bool(enabled)
        self.rotate = bool(rotate)
        self._members: List[object] = []
        #: When each member is next due. Most are due again next cycle, so
        #: those are listed in _soon; later cycles go on the heap.
        self._next: List[Optional[int]] = []  # each member's due cycle, None asleep
        self._soon: List[int] = []            # indices due at cycle + 1
        self._heap: List[tuple] = []          # (cycle, index) due later
        #: The last cycle this group ticked (-1: never).
        self.cycle = -1
        self._ticking = False
        self._pos = -1                        # order position being ticked
        self._due: List[tuple] = []           # (position, index) due this cycle
        self.ticks = 0          # member ticks run
        self.skipped = 0        # member ticks skipped

    def add(self, obj):
        idx = len(self._members)
        self._members.append(obj)
        self._next.append(None)
        self._schedule(idx, self.cycle + 1)      # every member starts awake
        obj._wake_group = self
        obj._wake_index = idx
        return obj

    @property
    def members(self):
        return self._members

    def _position(self, idx: int, cycle: int) -> int:
        if not self.rotate:
            return idx
        return (idx - cycle) % len(self._members)

    def _schedule(self, idx: int, cycle: int) -> None:
        """Bring member `idx` forward to `cycle` if that is sooner than planned."""
        if self._ticking and cycle <= self.cycle:
            pos = self._position(idx, self.cycle)
            if pos > self._pos:
                if self._next[idx] != self.cycle:
                    self._next[idx] = self.cycle
                    insort(self._due, (pos, idx))
                return
            cycle = self.cycle + 1
        cur = self._next[idx]
        if cur is None or cycle < cur:
            self._next[idx] = cycle
            if cycle == self.cycle + 1:
                self._soon.append(idx)
            else:
                heapq.heappush(self._heap, (cycle, idx))

    def request_wake(self, idx: int, cycle: int) -> None:
        self._schedule(idx, int(cycle))

    def wake(self, idx: int) -> None:
        """Tick member `idx` as soon as it would have seen new work."""
        self._schedule(idx, self.cycle if self._ticking else self.cycle + 1)

    def tick(self, now) -> None:
        cycle = self.cycle + 1 if now is None else int(round(now))
        members = self._members
        if not self.enabled:
            self.cycle = cycle
            if self.rotate and members:
                n = len(members)
                start = cycle % n
                for off in range(n):
                    members[(start + off) % n].tick(now)
            else:
                for m in members:
                    m.tick(now)
            self.ticks += len(members)
            return

        # Gather who is due, skipping entries a sooner wake superseded.
        nxt = self._next
        due = []
        if self._soon:
            c = self.cycle + 1
            for idx in self._soon:
                if nxt[idx] == c:
                    nxt[idx] = cycle
                    due.append(idx)
            self._soon = []
        heap = self._heap
        while heap and heap[0][0] <= cycle:
            c, idx = heapq.heappop(heap)
            if nxt[idx] == c:
                nxt[idx] = cycle
                due.append(idx)
        self.cycle = cycle
        if not due:
            self.skipped += len(members)
            return

        if self.rotate:
            n = len(members)
            self._due = sorted(((idx - cycle) % n, idx) for idx in due)
        else:
            due.sort()
            self._due = [(idx, idx) for idx in due]
        ran: List[int] = []
        self._ticking = True
        try:
            # A member woken mid-tick is inserted into _due ahead of `i`.
            i = 0
            order = self._due
            while i < len(order):
                self._pos, idx = order[i]
                nxt[idx] = None
                members[idx].tick(now)
                ran.append(idx)
                i += 1
        finally:
            self._ticking = False
            self._pos = -1
            self._due = []
        # After every member has ticked, so work handed over later in the
        # cycle is seen. A member cannot run twice in one cycle.
        for idx in ran:
            w = members[idx].next_wake(cycle)
            if w is not None:
                self._schedule(idx, w if w > cycle else cycle + 1)
        self.ticks += len(ran)
        self.skipped += len(members) - len(ran)

    def stats(self) -> dict:
        total = self.ticks + self.skipped
        return {
            "name": self.name,
            "members": len(self._members),
            "ticked": self.ticks,
            "skipped": self.skipped,
            "skipped_pct": (100.0 * self.skipped / total) if total else 0.0,
        }


class CompositeClocked(Clocked):
    """A component whose tick() ticks its children in a declared order.

    Used for the platform itself, so that `platform.tick()` advances every
    component and the order is stated once. Children run phase by phase
    (lowest first), in the order they were added within a phase; children
    with nothing to do sleep (see WakeGroup).
    """

    def __init__(self, name: str = "") -> None:
        super().__init__()
        self.name = name
        self._entries: List[tuple] = []      # (phase, order, child)

    def add_child(self, child, *, phase: int = 0):
        self._entries.append((int(phase), len(self._entries), child))
        self._entries.sort(key=lambda e: (e[0], e[1]))
        # Children are added while the platform is built, before it runs, so
        # the group is simply rebuilt in the new order.
        self._children = None
        for _, _, c in self._entries:
            super().add_child(c)
        return child

    @property
    def children(self):
        return [c for _, _, c in self._entries]

    def tick(self, time=None) -> None:
        self.tick_children(time)
