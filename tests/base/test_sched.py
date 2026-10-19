"""Tests for wake-group scheduling and platform composition.

The scheduler must be invisible: skipping a tick that would have done nothing
cannot change a single reported number. These tests check the mechanism in
isolation, then check that a whole platform run is identical with scheduling
on and off.
"""

import os
import subprocess
import sys


from base.clocked_object import Clocked
from base.sched import CompositeClocked, WakeGroup


class Counter(Clocked):
    """Ticks are recorded; wakes are whatever the test dictates."""

    def __init__(self, wake_every=None):
        super().__init__()
        self.ticks = []
        self.wake_every = wake_every

    def tick(self, time=None):
        self.ticks.append(int(time))

    def next_wake(self, now):
        if self.wake_every is None:
            return None                      # quiescent forever
        return now + self.wake_every


class Unconverted(Clocked):
    """A component that has not opted in: inherits Clocked.next_wake."""

    def __init__(self):
        super().__init__()
        self.ticks = []

    def tick(self, time=None):
        self.ticks.append(int(time))


def test_group_ticks_unconverted_member_every_cycle():
    """The default next_wake must preserve the old tick-everything behaviour,
    so components can be converted one at a time."""
    g = WakeGroup("t", enabled=True)
    a = g.add(Unconverted())
    for c in range(5):
        g.tick(c)
    assert a.ticks == [0, 1, 2, 3, 4]
    assert g.skipped == 0


def test_group_sleeps_a_quiescent_member():
    g = WakeGroup("t", enabled=True)
    a = g.add(Counter(wake_every=None))
    for c in range(5):
        g.tick(c)
    # Every member starts awake, so cycle 0 runs; after that it declares
    # itself quiescent and is never woken again.
    assert a.ticks == [0]
    assert g.skipped == 4


def test_request_wake_revives_a_sleeping_member():
    g = WakeGroup("t", enabled=True)
    a = g.add(Counter(wake_every=None))
    g.tick(0)
    for c in range(1, 4):
        g.tick(c)
    assert a.ticks == [0]
    a.request_wake(6)
    for c in range(4, 9):
        g.tick(c)
    assert a.ticks == [0, 6]


def test_request_wake_only_moves_a_member_earlier():
    g = WakeGroup("t", enabled=True)
    a = g.add(Counter(wake_every=None))
    g.tick(0)
    a.request_wake(9)
    a.request_wake(4)      # earlier: takes effect
    a.request_wake(7)      # later: must not push it back
    for c in range(1, 6):
        g.tick(c)
    assert a.ticks == [0, 4]


def test_due_members_run_in_registration_order():
    """Ordering within a cycle is load-bearing and must not depend on the heap."""
    order = []

    class Recorder(Clocked):
        def __init__(self, tag):
            super().__init__()
            self.tag = tag

        def tick(self, time=None):
            order.append(self.tag)

        def next_wake(self, now):
            return now + 1

    g = WakeGroup("t", enabled=True)
    for tag in "abcd":
        g.add(Recorder(tag))
    # Wake them out of order; they must still tick a, b, c, d.
    for idx in (3, 1, 0, 2):
        g._schedule(idx, 1)
    g.tick(1)
    assert order[-4:] == ["a", "b", "c", "d"]


def test_disabled_group_ticks_everything():
    g = WakeGroup("t", enabled=False)
    a = g.add(Counter(wake_every=None))
    for c in range(4):
        g.tick(c)
    assert a.ticks == [0, 1, 2, 3]


def test_composite_ticks_children_in_phase_order():
    order = []

    class Rec(Clocked):
        def __init__(self, tag):
            super().__init__()
            self.tag = tag

        def tick(self, time=None):
            order.append(self.tag)

    root = CompositeClocked("root")
    root.add_child(Rec("late"), phase=90)
    root.add_child(Rec("early"), phase=10)
    root.add_child(Rec("mid"), phase=50)
    root.tick(0)
    assert order == ["early", "mid", "late"]


def test_composite_preserves_insertion_order_within_a_phase():
    order = []

    class Rec(Clocked):
        def __init__(self, tag):
            super().__init__()
            self.tag = tag

        def tick(self, time=None):
            order.append(self.tag)

    root = CompositeClocked("root")
    for tag in "xyz":
        root.add_child(Rec(tag), phase=10)
    root.tick(0)
    assert order == ["x", "y", "z"]


class Handoff(Clocked):
    """Sleeps when it has no work; `give` hands it some, as a unit would."""

    def __init__(self, tag, order):
        super().__init__()
        self.tag, self.order, self.work, self.target = tag, order, 0, None

    def give(self):
        self.work += 1
        self.wake()

    def tick(self, time=None):
        self.order.append((int(time), self.tag))
        if self.work:
            self.work -= 1
            if self.target is not None:
                self.target.give()

    def next_wake(self, now):
        return now + 1 if self.work else None


def test_clocked_children_tick_only_when_due():
    order = []

    class Parent(Clocked):
        def tick(self, time=None):
            self.tick_children(time)

    parent = Parent()
    a = parent.add_child(Handoff("a", order))
    for c in range(3):
        parent.tick(c)
    assert order == [(0, "a")], "ticks once (everyone starts awake), then sleeps"
    a.give()
    for c in range(3, 6):
        parent.tick(c)
    assert order == [(0, "a"), (3, "a")], "woken, does its work, sleeps again"


def test_work_handed_forward_runs_the_same_cycle():
    """A member that comes later in the order sees work from an earlier one in
    the same cycle -- as it would if every member ticked every cycle."""
    order = []
    g = WakeGroup("t", enabled=True)
    first, second = g.add(Handoff("first", order)), g.add(Handoff("second", order))
    first.target = second
    g.tick(0)
    order.clear()
    first.give()
    g.tick(1)
    assert order == [(1, "first"), (1, "second")]


def test_work_handed_back_runs_the_next_cycle():
    """An earlier member has already ticked this cycle: it sees the work next
    cycle, again as it would if ticked every cycle."""
    order = []
    g = WakeGroup("t", enabled=True)
    first, second = g.add(Handoff("first", order)), g.add(Handoff("second", order))
    second.target = first
    g.tick(0)
    order.clear()
    second.give()
    g.tick(1)
    g.tick(2)
    g.tick(3)
    assert order == [(1, "second"), (2, "first")]


def test_a_rotating_group_starts_from_member_cycle_mod_count():
    order = []

    class Rec(Clocked):
        def __init__(self, tag):
            super().__init__()
            self.tag = tag

        def tick(self, time=None):
            order.append(self.tag)

    g = WakeGroup("t", enabled=True, rotate=True)
    for tag in "abc":
        g.add(Rec(tag))
    for c in range(3):
        g.tick(c)
    assert order == list("abc" "bca" "cab")


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_platform_tick_advances_every_component():
    """platform.tick() must reach every component, not just a chosen list."""
    from atalla.sysarr_tpu_system import build_tpu_platform

    platform = build_tpu_platform(size=8, dtype="fp16")
    children = platform.root.children
    assert platform.vc in children
    assert platform.spad in children
    assert platform.sysarr_bridge in children
    for bridge in platform.vls_bridges:
        assert bridge in children

    platform.tick(1.0)
    # Reaching the scratchpad proves the tick propagated through the tree.
    assert platform.spad.now == 1


def test_an_idle_scratchpad_sleeps():
    """No queued request and nothing in flight: the scratchpad asks for no
    tick, and wakes again for the next request."""
    from atalla.sysarr_tpu_system import (
        SysArrTPUSystem, _act_u16, _weights_u16)

    size = 16
    system = SysArrTPUSystem(size=size, dtype="fp16", mirror=False)
    act = _act_u16(size)
    wgt = _weights_u16(size)
    wgt_stream = [[wgt[r][c] for r in range(size)] for c in range(size - 1, -1, -1)]
    system.load_inputs(act, wgt_stream)
    system.run(max_cycles=200000)

    spad = system.spad
    assert sum(p.reads + p.writes for p in spad.tiles) > 0, "the pads were used"
    now = int(spad.now)
    assert spad.next_wake(now) is None
    assert spad.submit_read(0, 0, lambda _lanes: None, now=now + 1)
    assert spad.next_wake(now + 1) == now + 2


def _run_platform(mode):
    """Run a fixed workload in a subprocess so ATALLA_SCHED applies at import."""
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "from atalla.sysarr_tpu_system import SysArrTPUSystem, _act_u16, _weights_u16;"
        "size=16;"
        "s=SysArrTPUSystem(size=size, dtype='fp16', mirror=True);"
        "a=_act_u16(size); w=_weights_u16(size);"
        "ws=[[w[r][c] for r in range(size)] for c in range(size-1,-1,-1)];"
        "s.load_inputs(a, ws);"
        "out, mirror, cycles, m = s.run(max_cycles=200000);"
        "print(repr((out, mirror, cycles, m.cycles, m.bytes_moved, m.flops,"
        " [(p.reads, p.writes, p.stalls) for p in s.spad.tiles],"
        " dict(s.sa.metrics), dict(s.sa.internal_bytes),"
        " dict(s.sa.internal_bytes_valid))))"
    )
    env = dict(os.environ)
    env["ATALLA_SCHED"] = mode
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env=env, cwd=root)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout.strip()


def test_event_and_legacy_scheduling_agree():
    """The whole point: scheduling changes speed, never results.

    Compares outputs, cycle count, every pad's read, write and stall counts,
    and every systolic array counter -- the stall count shifts if any request
    is accepted on a different cycle.
    """
    assert _run_platform("event") == _run_platform("legacy")
