"""A per-cycle execution trace atalla-sim and an RTL testbench can both write,
and a comparison that finds where two traces part ways.

The format (text, one event per line, `#` starts a comment):

    # atalla-trace v1
    <cycle> issue <pc>              decode 2 issued the packet at <pc> (hex)
    <cycle> wb x<r> <value>         scalar register r written: 8 hex digits
    <cycle> wb m<r> <value>         mask register r written: 8 hex digits
    <cycle> wb v<r> <h0>,...,<h31>  vector register r written: 32 BF16 bit
                                    patterns, 4 hex digits each, element 0 first
    <cycle> halt                    the program finished (halt.s retired, work drained)

<cycle> is a decimal cycle count from reset. A write's cycle is the edge it
reaches the register file. Writes to x0 and v0 are not traced. An RTL
tracer samples decode 2's issue (valid and pc) and the register files'
write ports on each clock edge (docs/rtl-traces.md has the signal list).
Event kinds one trace lacks are not compared, so a tracer can start with
`issue` alone.

    atalla-sim run prog.in --trace sim.trace
    atalla-sim compare-rtl rtl.trace sim.trace

The comparison reports, for the issue stream, the first packet issued at a
different pc (control flow) and the first issued at a different cycle
(timing, with the offset), and for each register, the first write whose
value or cycle differs, earliest first, with the events around it.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from base.clocked_object import Clocked

HEADER = "# atalla-trace v1"


def bf16_hex(values) -> str:
    from atalla.tensors import to_bf16_bits
    return ",".join("%04x" % int(b) for b in to_bf16_bits(list(values)))


class TraceWriter(Clocked):
    """Records issue and writeback events of an AtallaPlatform (opt-in:
    build_atalla_platform(trace=path) / run_kernel --trace)."""

    def __init__(self, platform):
        super().__init__()
        self.p = platform
        self._tick = -1
        self.events: List[Tuple[int, int, str]] = []         # (cycle, order, text)
        self._n = 0
        self._issued = 0
        core = platform.core
        inner = core.writeback_at

        def writeback_at(cycle, scalar=(), vector=(), mask=(), sdma=(), vst=(), _inner=inner):
            for r, v in scalar:
                if int(r):
                    self._add(int(cycle), "wb x%d %08x" % (int(r), int(v) & 0xFFFFFFFF))
            for e in mask:
                if isinstance(e, tuple):
                    self._add(int(cycle), "wb m%d %08x" % (int(e[0]), int(e[1]) & 0xFFFFFFFF))
            for e in vector:
                if isinstance(e, tuple) and int(e[0]):
                    self._add(int(cycle), "wb v%d %s" % (int(e[0]), bf16_hex(e[1])))
            return _inner(cycle, scalar=scalar, vector=vector, mask=mask, sdma=sdma, vst=vst)
        core.writeback_at = writeback_at

    def _add(self, cycle: int, text: str) -> None:
        self.events.append((cycle, self._n, text))
        self._n += 1

    def tick(self, time: Optional[float] = None) -> None:
        cycle = self._consume_tick(time, attr_name="_tick")
        if cycle is None:
            return
        core = self.p.core
        if core.done_at is not None and cycle > core.done_at:
            return
        if core.d2ex is not None and core.issued_count > self._issued:
            self._issued = core.issued_count
            self._add(cycle, "issue %x" % core.d2ex.pc)

    def text(self) -> str:
        events = list(self.events)
        if self.p.core.done_at is not None:
            events.append((self.p.core.done_at, self._n, "halt"))
        lines = [HEADER] + ["%d %s" % (c, t) for c, _, t in sorted(events)]
        return "\n".join(lines) + "\n"

    def write(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.text())


# -- reading and comparing ----------------------------------------------------------------

@dataclass
class Event:
    cycle: int
    kind: str               # "issue", "wb", "halt"
    key: str                # "issue", "x5", "v3", "m1", "halt"
    value: str
    line: int


def read(path: str) -> List[Event]:
    out = []
    with open(path) as f:
        for n, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            try:
                cycle = int(parts[0])
            except ValueError:
                raise ValueError("%s:%d: a line starts with a decimal cycle: %r" % (path, n, raw))
            kind = parts[1] if len(parts) > 1 else ""
            if kind == "issue" and len(parts) == 3:
                out.append(Event(cycle, "issue", "issue", "%x" % int(parts[2], 16), n))
            elif kind == "wb" and len(parts) == 4:
                out.append(Event(cycle, "wb", parts[2], parts[3].lower(), n))
            elif kind == "halt":
                out.append(Event(cycle, "halt", "halt", "", n))
            else:
                raise ValueError("%s:%d: not an event: %r" % (path, n, raw))
    return out


def _streams(events: List[Event]) -> Dict[str, List[Event]]:
    s: Dict[str, List[Event]] = defaultdict(list)
    for e in events:
        s[e.key].append(e)
    return s


@dataclass
class Difference:
    cycle: int              # in trace A
    what: str


def compare(a: List[Event], b: List[Event], names=("A", "B")) -> List[Difference]:
    """Differences between traces a and b, earliest (in a) first."""
    sa, sb = _streams(a), _streams(b)
    kinds_a = {e.kind for e in a}
    kinds_b = {e.kind for e in b}
    shared = kinds_a & kinds_b
    out: List[Difference] = []
    na, nb = names
    if "issue" in shared:
        ia, ib = sa.get("issue", []), sb.get("issue", [])
        timing_done = False
        for k in range(max(len(ia), len(ib))):
            ea = ia[k] if k < len(ia) else None
            eb = ib[k] if k < len(ib) else None
            if ea is None or eb is None:
                more, fewer = (na, nb) if eb is None else (nb, na)
                e = ea or eb
                out.append(Difference(e.cycle if ea else (ia[-1].cycle if ia else 0),
                                      "%s issued %d packets, %s %d: %s has more from packet "
                                      "#%d (pc %s)" % (na, len(ia), nb, len(ib), more, k + 1,
                                                       "0x" + e.value)))
                break
            if ea.value != eb.value:
                out.append(Difference(ea.cycle, "control flow: packet #%d issued at pc 0x%s "
                                      "in %s (cycle %d) but 0x%s in %s (cycle %d)"
                                      % (k + 1, ea.value, na, ea.cycle, eb.value, nb, eb.cycle)))
                break
            if ea.cycle != eb.cycle and not timing_done:
                timing_done = True
                out.append(Difference(ea.cycle, "timing: packet #%d (pc 0x%s) issued at cycle "
                                      "%d in %s, %d in %s (%+d)"
                                      % (k + 1, ea.value, ea.cycle, na, eb.cycle, nb,
                                         eb.cycle - ea.cycle)))
    if "wb" in shared:
        regs = sorted({k for k in set(sa) | set(sb) if k[0] in "xvm"})
        for reg in regs:
            wa, wb_ = sa.get(reg, []), sb.get(reg, [])
            for k in range(max(len(wa), len(wb_))):
                ea = wa[k] if k < len(wa) else None
                eb = wb_[k] if k < len(wb_) else None
                if ea is not None and eb is not None and ea.value == eb.value \
                        and ea.cycle == eb.cycle:
                    continue
                nth = "write #%d to %s" % (k + 1, reg)
                if ea is None or eb is None:
                    e = ea or eb
                    out.append(Difference(e.cycle, "%s: only in %s (cycle %d, %s)"
                                          % (nth, na if ea else nb, e.cycle, e.value[:40])))
                elif ea.value != eb.value:
                    out.append(Difference(ea.cycle, "%s: value %s in %s (cycle %d), %s in %s "
                                          "(cycle %d)" % (nth, ea.value[:40], na, ea.cycle,
                                                          eb.value[:40], nb, eb.cycle)))
                else:
                    out.append(Difference(ea.cycle, "%s: same value, cycle %d in %s, %d in %s "
                                          "(%+d)" % (nth, ea.cycle, na, eb.cycle, nb,
                                                     eb.cycle - ea.cycle)))
                break
    if "halt" in shared:
        ha, hb = sa.get("halt", []), sb.get("halt", [])
        if ha and hb and ha[0].cycle != hb[0].cycle:
            out.append(Difference(ha[0].cycle, "halt at cycle %d in %s, %d in %s (%+d)"
                                  % (ha[0].cycle, na, hb[0].cycle, nb, hb[0].cycle - ha[0].cycle)))
    out.sort(key=lambda d: d.cycle)
    return out


def context(events: List[Event], cycle: int, width: int = 3) -> List[str]:
    """The events within `width` cycles of `cycle`."""
    return ["%6d %-5s %s %s" % (e.cycle, e.kind, e.key if e.kind == "wb" else "", e.value[:48])
            for e in events if abs(e.cycle - cycle) <= width]


def report(path_a: str, path_b: str, limit: int = 10, window: int = 3,
           align: bool = True) -> Tuple[str, int]:
    """The comparison as text, and 0 if identical else 1. With `align`, B's
    cycles are shifted so its first issue lines up with A's (testbenches
    count reset cycles differently)."""
    import os
    a, b = read(path_a), read(path_b)
    na, nb = os.path.basename(path_a), os.path.basename(path_b)
    if na == nb:
        na, nb = path_a, path_b
    shift = 0
    if align:
        fa = next((e.cycle for e in a if e.kind == "issue"), None)
        fb = next((e.cycle for e in b if e.kind == "issue"), None)
        if fa is not None and fb is not None:
            shift = fa - fb
            for e in b:
                e.cycle += shift
    diffs = compare(a, b, (na, nb))
    kinds = sorted({e.kind for e in a} & {e.kind for e in b})
    lines = ["comparing %s (%d events) with %s (%d events); compared: %s"
             % (na, len(a), nb, len(b), ", ".join(kinds) or "nothing in common")]
    if shift:
        lines.append("%s's cycles shifted by %+d so the first packets issue together "
                     "(--no-align keeps them)" % (nb, shift))
    if not diffs:
        lines.append("identical: every compared event at the same cycle with the same value")
        return "\n".join(lines), 0
    lines.append("%d difference%s; earliest first:" % (len(diffs), "s" * (len(diffs) != 1)))
    for d in diffs[:limit]:
        lines.append("  cycle %d: %s" % (d.cycle, d.what))
    first = diffs[0].cycle
    lines.append("around cycle %d in %s:" % (first, na))
    lines += ["  " + t for t in context(a, first, window)]
    lines.append("around cycle %d in %s:" % (first, nb))
    lines += ["  " + t for t in context(b, first, window)]
    return "\n".join(lines), 1
