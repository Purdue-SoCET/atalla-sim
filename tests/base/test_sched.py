"""Tests for wake-group scheduling and platform composition.

The scheduler must be invisible: skipping a tick that would have done nothing
cannot change a single reported number. These tests check the mechanism in
isolation, then check that a whole platform run is identical with scheduling
on and off.
"""

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..//..', 'src')))

from base.clocked_object import Clocked
from base import sched
from base.sched import CompositeClocked, SimClock, WakeGroup


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


def test_sim_clock_only_moves_forward():
    c = SimClock()
    c.advance_to(5)
    assert c.cycle == 5
    c.advance_to(3)
    assert c.cycle == 5


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
    assert platform.spad.now == 1.0
    assert platform.spad.clock.cycle == 1


@pytest.mark.skipif(not sched.SCHED_ENABLED,
                    reason="ATALLA_SCHED=legacy disables skipping by design")
def test_scratchpad_actually_skips_idle_banks():
    from atalla.sysarr_tpu_system import (
        SysArrTPUSystem, _act_u16, _weights_u16)

    size = 16
    system = SysArrTPUSystem(size=size, dtype="fp16", mirror=False)
    act = _act_u16(size)
    wgt = _weights_u16(size)
    wgt_stream = [[wgt[r][c] for r in range(size)] for c in range(size - 1, -1, -1)]
    system.load_inputs(act, wgt_stream)
    system.run(max_cycles=200000)

    bank_group = system.spad._bank_group
    stats = bank_group.stats()
    assert stats["ticked"] > 0, "banks never ran"
    assert stats["skipped_pct"] > 50.0, (
        f"expected most bank ticks to be skipped, got {stats['skipped_pct']:.1f}%")


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
        " [b.cycles_busy for t in s.spad.tiles for b in t.banks],"
        " [b.enqueue_stalls for t in s.spad.tiles for b in t.banks],"
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

    Compares outputs, cycle count, per-bank utilisation and every systolic
    array counter -- per-bank cycles_busy is the sharpest probe available,
    since it shifts if any operation completes on a different cycle.
    """
    assert _run_platform("event") == _run_platform("legacy")
