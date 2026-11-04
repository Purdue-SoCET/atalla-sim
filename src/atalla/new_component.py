"""`atalla-sim new-component NAME`: a new clocked component and its test.

    atalla-sim new-component DotProductUnit --package vector_core

writes src/vector_core/dot_product_unit.py -- a Clocked unit with an input
queue, a fixed latency and an output queue, following
docs/architecture/base-classes.md section 8 (tick, next_wake, wake) -- and
tests/vector_core/test_dot_product_unit.py, which runs it with wake-group
scheduling on and off and checks that the results are identical and that
it sleeps when idle. Both pass as generated; replace the latency pipeline
with the unit's behaviour and keep the test green.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODULE = '''"""{name}: TODO one line on what this unit models.

TODO: what it takes in, what it produces, its timing, and the RTL it
follows (file and branch), or that atalla-sim's model is the reference.

    unit = {name}(latency=2, depth=4)
    unit.submit(payload)          # False when the input queue is full
    ...                           # tick it as part of the platform
    unit.out_q.dequeue()          # results, `latency` cycles after they start

Written from docs/architecture/base-classes.md section 8: it opens tick()
with _consume_tick, says when it next has work in next_wake (None when
idle), and is woken by whoever hands it work, so the platform skips its
ticks while it has nothing to do.
"""

from typing import Any, List, Optional, Tuple

from base.clocked_object import Clocked
from base.queue import SimQueue


class {name}(Clocked):
    def __init__(self, latency: int = 2, depth: int = 4):
        super().__init__()
        self.latency = int(latency)
        self.in_q = SimQueue(depth)
        self.out_q = SimQueue(depth)
        #: (cycle it completes, payload), oldest first
        self._pending: List[Tuple[int, Any]] = []
        self._tick = -1
        self.completed = 0

    def submit(self, payload) -> bool:
        """Hand the unit work; False when its input queue is full (retry)."""
        if not self.in_q.enqueue(payload):
            return False
        self.wake()                       # it may be asleep
        return True

    def next_wake(self, now) -> Optional[int]:
        if self._pending:
            return self._pending[0][0]    # the next completion
        if not self.in_q.is_empty():
            return now + 1                # work waiting to start
        return None                       # idle: skip my ticks until woken

    def tick(self, time: Optional[float] = None) -> None:
        cycle = self._consume_tick(time, attr_name="_tick")
        if cycle is None:
            return
        while self._pending and self._pending[0][0] <= cycle and not self.out_q.is_full():
            _, payload = self._pending.pop(0)
            self.out_q.enqueue(payload)
            self.completed += 1
        item = self.in_q.dequeue()
        if item is not None:
            # TODO: the unit's behaviour; this one passes payloads through.
            self._pending.append((cycle + self.latency, item))
'''

TEST = '''"""{name} ({module_path}): the behaviour, and that wake-group scheduling
changes nothing (docs/architecture/base-classes.md section 5)."""

from base.sched import WakeGroup
from {package}.{module} import {name}


def run(enabled: bool, submits, cycles: int = 40):
    """Tick the unit in a wake group for `cycles` cycles, submitting
    `submits` ({{cycle: payload}}); return (cycle, payload) for every result."""
    unit = {name}(latency=3, depth=4)
    group = WakeGroup("{module}", enabled=enabled)
    group.add(unit)
    out = []
    for cycle in range(cycles):
        if cycle in submits:
            assert unit.submit(submits[cycle])
        group.tick(cycle)
        item = unit.out_q.dequeue()
        if item is not None:
            out.append((cycle, item))
    return out, group


def test_results_arrive_after_the_latency():
    out, _ = run(True, {{0: "a", 1: "b"}})
    assert [p for _, p in out] == ["a", "b"]
    assert out[0][0] >= 3


def test_scheduling_changes_nothing_and_an_idle_unit_sleeps():
    submits = {{0: "a", 1: "b", 2: "c", 20: "d"}}
    on, group = run(True, submits)
    off, _ = run(False, submits)
    assert on == off
    assert group.skipped > 0                    # it slept while idle
'''


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


def write(name: str, package: str, root: Path = ROOT, force: bool = False):
    if not re.fullmatch(r"[A-Z][A-Za-z0-9]*", name):
        raise ValueError("NAME is a class name in CamelCase, e.g. DotProductUnit; not %r" % name)
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", package):
        raise ValueError("--package is a package under src/, e.g. vector_core; not %r" % package)
    module = snake(name)
    src = root / "src" / package / ("%s.py" % module)
    test = root / "tests" / package / ("test_%s.py" % module)
    for p in (src, test):
        if p.exists() and not force:
            raise FileExistsError("%s exists (--force overwrites it)" % p)
    src.parent.mkdir(parents=True, exist_ok=True)
    test.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(MODULE.format(name=name))
    test.write_text(TEST.format(name=name, package=package, module=module,
                                module_path="src/%s/%s.py" % (package, module)))
    return src, test


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="atalla-sim new-component",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("name", help="the class name, CamelCase")
    ap.add_argument("--package", default="atalla",
                    help="the package under src/ it belongs to (default atalla)")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args(argv)
    try:
        src, test = write(args.name, args.package, force=args.force)
    except (ValueError, FileExistsError) as exc:
        sys.exit(str(exc))
    print("wrote %s\nwrote %s\nrun: pytest %s" % (src.relative_to(ROOT), test.relative_to(ROOT),
                                               test.relative_to(ROOT)))
    print("then: give it its behaviour, attach it to the platform with a phase "
          "(src/atalla/sysarr_tpu_system.py lists them), and keep both modes equal "
          "(ATALLA_SCHED=legacy pytest).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
