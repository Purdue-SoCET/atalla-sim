"""Where the cycles went, packet by packet: a per-PC profile.

The performance monitor puts every cycle in one bucket (issue, a stall and
its cause, a frontend wait, the halt drain). This observer charges each
cycle to the packet responsible:

    issue               the packet decode 2 issued
    stall:...           the packet decode 2 is holding
    frontend:...        the packet being fetched (the fetch pc)
    halt:drain          a row of its own, pc -1: after halt.s, waiting for
                        work in flight and the dcache's writebacks

so a hot loop shows which packet waits, on what, and for how long:

    plat = build_atalla_platform(..., profile=True)     # attaches the monitor too
    plat.run_until_done()
    print(plat.profile.summary(top=15))
    plat.profile.to_json("profile.json")

Opt-in and read-only, like the monitor; the totals over all packets equal
the monitor's cycle buckets.
"""

import json
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from base.clocked_object import Clocked
from scheduler.decode1 import decode1
from scheduler.decode2 import DecodedPacket
from scheduler.disasm import format_packet


HALT_ROW = -1


class Profiler(Clocked):
    def __init__(self, platform):
        super().__init__()
        self.p = platform
        self._tick = -1
        #: pc -> Counter(bucket -> cycles)
        self.by_pc: Dict[int, Counter] = defaultdict(Counter)

    def tick(self, time: Optional[float] = None) -> None:
        cycle = self._consume_tick(time, attr_name="_tick")
        if cycle is None:
            return
        bucket = self.p.perf.last_bucket
        if bucket is None:
            return
        core = self.p.core
        if bucket.startswith("halt"):
            pc = HALT_ROW
        elif bucket == "issue" and core.d2ex is not None:
            pc = core.d2ex.pc
        elif bucket.startswith("stall") and core.d1d2.valid:
            pc = core.d1d2.pc
        else:
            pc = core.pc
        self.by_pc[pc][bucket] += 1

    # -- results ------------------------------------------------------------------------
    def _text(self, pc: int) -> str:
        if pc == HALT_ROW:
            return "(after halt.s: draining work in flight and dcache writebacks)"
        word = self.p.core.program.get(pc)
        if word is None:
            return "?"
        return format_packet(DecodedPacket.from_words(*decode1(word)))

    def _where(self, pc: int) -> str:
        if pc == HALT_ROW:
            return "halt"
        where = "%#x" % pc
        d = self.p.describe_pc
        if d is not None and d(pc) != where:
            where += " (%s)" % d(pc)
        return where

    def report(self) -> List[dict]:
        rows = []
        for pc, c in self.by_pc.items():
            total = sum(c.values())
            rows.append(dict(pc=pc, where=self._where(pc), packet=self._text(pc), cycles=total,
                             issued=c.get("issue", 0),
                             waits=dict(sorted(((k, v) for k, v in c.items() if k != "issue"),
                                               key=lambda kv: -kv[1]))))
        rows.sort(key=lambda r: (-r["cycles"], r["pc"]))
        return rows

    def summary(self, top: int = 15) -> str:
        rows = self.report()
        total = sum(r["cycles"] for r in rows) or 1
        lines = ["where the cycles went, by packet (top %d of %d packets; %d cycles):"
                 % (min(top, len(rows)), len(rows), total),
                 "  %6s  %5s  %7s  %-24s  %s" % ("cycles", "%", "issued", "pc", "packet / waits")]
        for r in rows[:top]:
            lines.append("  %6d  %4.1f%%  %7d  %-24s  %s"
                         % (r["cycles"], 100.0 * r["cycles"] / total, r["issued"],
                            r["where"], r["packet"]))
            if r["waits"]:
                lines.append("  %6s  %5s  %7s  %-24s    %s" % (
                    "", "", "", "", ", ".join("%s %d" % kv for kv in list(r["waits"].items())[:4])))
        return "\n".join(lines)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.report(), f, indent=1)
