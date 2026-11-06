"""The first register write where atalla-sim and the functional sim part ways.

Comparing final states says *that* a run went wrong; this says *where*.
Both simulators record every register write (scalar x, vector v, mask m):
the functional sim in program order with the pc that made it
(golden.run_golden_state(trace=...)), atalla-sim as each write reaches the
register file, with its cycle (`attach(plat)` before the run).

In-order issue and the scoreboard's write-after-write check mean each
register is written in program order on both sides, so the k-th write to a
register is the same instruction's on both. The first divergence is the
register and k with the earliest atalla-sim cycle whose values differ, or
where atalla-sim never makes the k-th write at all (a lost write). The
report names the instruction from the functional sim's pc.

    rec = lockstep.attach(plat)
    plat.run_until_done()
    print(lockstep.report(plat, rec, instr, data, init_sregs))

Vector values are compared after rounding the functional sim's to BF16 (the
register file holds 16 bits); reductions use the lane model's order.
DRAM is not compared here: golden.differences() still checks it at the end.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from scheduler import golden
from scheduler.decode1 import decode1
from scheduler.decode2 import DecodedPacket
from scheduler.disasm import format_packet
from scheduler.vector import bf16_round


@dataclass
class Divergence:
    kind: str                       # "x", "v" or "m"
    reg: int
    index: int                      # which write to the register, from 0
    cycle: Optional[int]            # atalla-sim's, None if it never wrote it
    mine: object
    theirs: object
    pc: Optional[int]               # the functional sim's, None if it never wrote it


class WriteRecorder:
    def __init__(self):
        #: (kind, reg) -> [(cycle, value), ...] in the order they landed
        self.writes: Dict[Tuple[str, int], List[Tuple[int, object]]] = defaultdict(list)


def attach(plat) -> WriteRecorder:
    """Record every register write of `plat` from now on (wraps the core's
    writeback_at; costs nothing unless attached)."""
    rec = WriteRecorder()
    core = plat.core
    inner = core.writeback_at

    def writeback_at(cycle, scalar=(), vector=(), mask=(), sdma=(), vst=(), _inner=inner):
        for r, v in scalar:
            if int(r) != 0:
                rec.writes[("x", int(r))].append((int(cycle), int(v) & 0xFFFFFFFF))
        for e in vector:
            if isinstance(e, tuple) and int(e[0]) != 0:
                rec.writes[("v", int(e[0]))].append((int(cycle), [float(x) for x in e[1]]))
        for e in mask:
            if isinstance(e, tuple):
                rec.writes[("m", int(e[0]))].append((int(cycle), int(e[1]) & 0xFFFFFFFF))
        return _inner(cycle, scalar=scalar, vector=vector, mask=mask, sdma=sdma, vst=vst)
    core.writeback_at = writeback_at
    return rec


def _same(kind: str, mine, theirs) -> bool:
    if kind != "v":
        return mine == theirs
    if len(mine) != len(theirs):
        return False
    # Both rounded: a reduction's sum reaches the register file unrounded,
    # and the 16-bit file rounds it there.
    for a, b in zip((bf16_round(x) for x in mine), (bf16_round(x) for x in theirs)):
        if a != b and not (math.isnan(a) and math.isnan(b)):
            return False
    return True


def first_divergence(rec: WriteRecorder, trace) -> Optional[Divergence]:
    theirs: Dict[Tuple[str, int], List[Tuple[int, object]]] = defaultdict(list)
    for kind, reg, value, pc in trace:
        theirs[(kind, reg)].append((pc, value))
    best: Optional[Divergence] = None
    never = float("inf")
    for key in set(theirs) | set(rec.writes):
        mine_w, their_w = rec.writes.get(key, []), theirs.get(key, [])
        for k in range(max(len(mine_w), len(their_w))):
            m = mine_w[k] if k < len(mine_w) else None
            t = their_w[k] if k < len(their_w) else None
            if m is not None and t is not None and _same(key[0], m[1], t[1]):
                continue
            d = Divergence(key[0], key[1], k, m[0] if m else None, m[1] if m else None,
                           t[1] if t else None, t[0] if t else None)
            when = d.cycle if d.cycle is not None else never
            best_when = (best.cycle if best is not None and best.cycle is not None else never)
            if best is None or when < best_when:
                best = d
            break
    return best


def _value(kind: str, v) -> str:
    if v is None:
        return "nothing"
    if kind == "v":
        head = ", ".join("%g" % bf16_round(x) for x in v[:4])
        return "[%s, ...]" % head
    return "%#010x" % v


def _packet_at(instr, pc) -> str:
    word = instr.get(pc)
    if word is None:
        return "?"
    return format_packet(DecodedPacket.from_words(*decode1(word)))


def describe(d: Optional[Divergence], instr, describe_pc=None) -> str:
    if d is None:
        return "every register write matches the functional sim's, in order"
    name = "%s%d" % (d.kind, d.reg)
    nth = "write #%d to %s" % (d.index + 1, name)
    where = ""
    if d.pc is not None:
        where = "%#x" % d.pc
        if describe_pc is not None and describe_pc(d.pc) != where:
            where += " (%s)" % describe_pc(d.pc)
        where = " by the packet at %s: %s" % (where, _packet_at(instr, d.pc))
    if d.cycle is None:
        return ("first divergence: atalla-sim never made %s; the functional sim wrote %s%s"
                % (nth, _value(d.kind, d.theirs), where))
    if d.pc is None:
        return ("first divergence: atalla-sim made %s at cycle %d (%s), which the functional "
                "sim never made" % (nth, d.cycle, _value(d.kind, d.mine)))
    return ("first divergence: %s, at cycle %d%s:\n  atalla-sim      %s\n  functional sim  %s"
            % (nth, d.cycle, where, _value(d.kind, d.mine), _value(d.kind, d.theirs)))


def report(plat, rec: WriteRecorder, instr, data, init_sregs=None,
           reductions: str = "lanes") -> str:
    """Run the functional sim on the same image and describe the first divergence."""
    trace: list = []
    golden.run_golden_state(instr, data, init_sregs=init_sregs, reductions=reductions,
                            trace=trace)
    return describe(first_divergence(rec, trace), instr, plat.describe_pc)
