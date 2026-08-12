"""Wake scheduling for clocked objects.

The simulator used to advance by ticking every object every cycle. Measured on
a 16x16 GEMM, 92% of those tick calls changed no state: an SRAM bank with an
empty ready-heap, a functional-unit pipeline with nothing in flight, and so on.

This module lets a parent skip children that have nothing to do. A `WakeGroup`
owns a set of sibling objects and a min-heap keyed by the cycle each one next
needs to run; ticking the group wakes only the members that are due.

ORDERING INVARIANT
------------------
Within a cycle, due members are ticked in registration order -- the same order
the old flat loops used. The one rule a member must respect:

    a member may wake a member of a group that ticks LATER in the same cycle,
    or itself/anyone for a FUTURE cycle, but never an earlier-ticking group
    for the current cycle.

Waking an already-passed group for the current cycle would run that member one
cycle sooner than the old code did. The existing pipeline satisfies this
naturally: within Scratchpad the order is frontends -> crossbars -> banks, and
data always flows in that direction.

STALE TIME
----------
A sleeping object stops observing cycles, so anything it derives from "when did
I last run" goes stale. `SimClock` gives such objects a cheap, always-correct
view of the current cycle without having to be ticked. SRAMBank needs this: it
computes operation due-cycles from the clock rather than from its own last
tick, which is only equivalent while it is ticked unconditionally.
"""

import heapq
import os
from typing import List, Optional

# ATALLA_SCHED=legacy ticks every member every cycle, as the simulator did
# before wake groups existed. Both paths must produce identical results; the
# toggle exists so that equivalence stays testable.
SCHED_MODE = os.environ.get("ATALLA_SCHED", "event").strip().lower()
SCHED_ENABLED = SCHED_MODE != "legacy"


class SimClock:
    """The current simulated cycle, readable by objects that are asleep."""

    __slots__ = ("cycle",)

    def __init__(self, cycle: int = -1) -> None:
        self.cycle = int(cycle)

    def advance_to(self, cycle: int) -> None:
        if cycle > self.cycle:
            self.cycle = int(cycle)


class WakeGroup:
    """A set of sibling clocked objects, ticked only when they have work.

    Set ``enabled = False`` to fall back to ticking every member every cycle,
    which is exactly the old behaviour and is what ATALLA_SCHED=legacy selects.
    """

    __slots__ = ("name", "enabled", "_members", "_heap", "_next", "ticks", "skipped")

    def __init__(self, name: str = "", enabled: Optional[bool] = None) -> None:
        self.name = name
        self.enabled = SCHED_ENABLED if enabled is None else bool(enabled)
        self._members: List[object] = []
        self._heap: List[tuple] = []
        self._next: List[Optional[int]] = []
        self.ticks = 0          # members actually ticked
        self.skipped = 0        # member-cycles skipped

    def add(self, obj):
        idx = len(self._members)
        self._members.append(obj)
        self._next.append(0)
        heapq.heappush(self._heap, (0, idx))     # every member starts awake
        obj._wake_group = self
        obj._wake_index = idx
        return obj

    @property
    def members(self):
        return self._members

    def _schedule(self, idx: int, cycle: int) -> None:
        """Bring member `idx` forward to `cycle` if that is sooner than planned."""
        cur = self._next[idx]
        if cur is None or cycle < cur:
            self._next[idx] = cycle
            heapq.heappush(self._heap, (cycle, idx))

    def request_wake(self, idx: int, cycle: int) -> None:
        self._schedule(idx, int(cycle))

    def tick(self, now) -> None:
        if not self.enabled:
            for m in self._members:
                m.tick(now)
            self.ticks += len(self._members)
            return

        cycle = int(now)
        due: List[int] = []
        heap = self._heap
        nxt = self._next
        while heap and heap[0][0] <= cycle:
            c, idx = heapq.heappop(heap)
            if nxt[idx] != c:
                continue                     # superseded by an earlier entry
            nxt[idx] = None
            due.append(idx)

        if not due:
            self.skipped += len(self._members)
            return

        # Registration order, so a partially-woken group matches the old loop.
        due.sort()
        members = self._members
        for idx in due:
            members[idx].tick(now)
        for idx in due:
            # A member cannot ask to run twice in one cycle; _consume_tick would
            # ignore the second call anyway.
            w = members[idx].next_wake(cycle)
            if w is not None:
                self._schedule(idx, w if w > cycle else cycle + 1)

        self.ticks += len(due)
        self.skipped += len(members) - len(due)

    def stats(self) -> dict:
        total = self.ticks + self.skipped
        return {
            "name": self.name,
            "members": len(self._members),
            "ticked": self.ticks,
            "skipped": self.skipped,
            "skipped_pct": (100.0 * self.skipped / total) if total else 0.0,
        }


class CompositeClocked:
    """A component whose tick() ticks its children in a declared order.

    Used for the platform itself, so that `platform.tick()` advances every
    component and the order is stated once instead of being re-specified at
    each call site (which is how `dram` and the backends came to be left out
    of the driving list entirely).
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._entries: List[tuple] = []      # (phase, order, child)
        self._order = 0
        self._clocked_last_tick = -1

    def add_child(self, child, *, phase: int = 0):
        self._entries.append((int(phase), self._order, child))
        self._order += 1
        self._entries.sort(key=lambda e: (e[0], e[1]))
        return child

    @property
    def children(self):
        return [c for _, _, c in self._entries]

    def tick(self, time=None) -> None:
        for _, _, child in self._entries:
            child.tick(time)

    def next_wake(self, now: int) -> Optional[int]:
        best = None
        for _, _, child in self._entries:
            w = child.next_wake(now) if hasattr(child, "next_wake") else now + 1
            if w is not None and (best is None or w < best):
                best = w
        return best
