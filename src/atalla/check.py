"""The dependency checker: accesses nothing in the hardware orders.

An opt-in observer (`build_atalla_platform(check=True)`, `atalla-sim run
--check`). It never changes timing; it reads each packet decode 2 issues,
with its operand values, and reports:

- **scpad.ld -> vreg.ld.** A `vreg.ld` reads a pad row an in-flight
  `scpad.ld` is still filling. The hardware makes a `vreg.ld` wait for a DMA
  only through the register the DMA holds busy (its rs1): a load that
  addresses the pad through another register issues at once and reads the
  row before the data lands.
- **vreg.st -> scpad.st.** A `scpad.st` reads a pad row a `vreg.st` hasn't
  written yet. The vreg.st reservations order an SDMA only after the vreg.st
  on the same rs1.
- **The same rows from the other side:** a `scpad.ld` into rows a vreg.st
  still has to write, and a `vreg.st` into rows an in-flight `scpad.ld`
  writes or `scpad.st` reads.
- **Misaligned scalar loads and stores.** The word path aligns the address
  down, so the access overlaps the word below it.
- **Packets the hardware can't run** as one packet (decode2's contract
  violations: two ops for one scalar EX unit, too many register reads); the
  RTL drops part of such a packet.

Each finding is reported once per program counter, with how often it
happened. A finding is a dependency the program relies on luck (timing) for:
it may still compute the right answer in this run.

Not checked: a `scpad.ld` overwriting rows an earlier `vreg.ld` has yet to
read (the DMA would have to outrun the load queue).
"""

from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from base.clocked_object import Clocked
from scheduler.control import LD, ST, V_VLSU
from scheduler.disasm import format_packet

MASK32 = 0xFFFFFFFF

KINDS = OrderedDict([
    ("sdma_then_vreg_ld", "scpad.ld -> vreg.ld: a vreg.ld read a row an in-flight scpad.ld "
                          "was still filling"),
    ("vreg_st_then_sdma_st", "vreg.st -> scpad.st: a scpad.st read a row a vreg.st hadn't "
                             "written yet"),
    ("vreg_st_then_sdma_ld", "vreg.st -> scpad.ld: a scpad.ld filled rows a vreg.st still had "
                             "to write"),
    ("sdma_then_vreg_st", "SDMA -> vreg.st: a vreg.st wrote a row an in-flight SDMA was "
                          "still using"),
    ("misaligned", "a scalar load or store at an address that isn't a multiple of 4"),
    ("packet_rules", "a packet the hardware can't run as one packet"),
])


@dataclass
class Finding:
    kind: str
    pc: int
    first_cycle: int
    detail: str
    packet: str
    count: int = 1


@dataclass
class _Dma:
    sid: int
    first: int
    rows: int
    store: bool
    pc: int
    rs1: int
    seen_busy: bool = False

    def covers(self, row: int) -> bool:
        return self.first <= row < self.first + self.rows


@dataclass
class _Vst:
    row: int
    rs1: int
    pc: int


class DependencyChecker(Clocked):
    """See the module docstring. Ticked last, after the core."""

    def __init__(self, platform):
        super().__init__()
        self.p = platform
        self._tick = -1
        self.findings: Dict[Tuple[str, int], Finding] = {}
        self._dma: Dict[int, _Dma] = {}
        #: pad -> vreg.st issued and not yet written, oldest first
        self._vst: Dict[int, deque] = defaultdict(deque)
        self._violations_seen = 0
        for bridge in platform.vls_bridges:
            inner = bridge.on_store_commit
            if inner is not None:
                bridge.on_store_commit = self._wrap_commit(inner)

    # -- hooks --------------------------------------------------------------------------
    def _wrap_commit(self, inner):
        def on_store_commit(sid, rd, _inner=inner):
            q = self._vst[int(sid)]
            for k, v in enumerate(q):
                if v.rs1 == int(rd):
                    del q[k]
                    break
            return _inner(sid, rd)
        return on_store_commit

    # -- one cycle ----------------------------------------------------------------------
    def tick(self, time: Optional[float] = None) -> None:
        cycle = self._consume_tick(time, attr_name="_tick")
        if cycle is None:
            return
        core = self.p.core
        vs = core.vector
        busy = vs.sdma_busy if vs is not None else []
        for sid, dma in list(self._dma.items()):
            if sid < len(busy) and busy[sid]:
                dma.seen_busy = True
            elif dma.seen_busy:
                del self._dma[sid]
        viol = core.decode2.violations
        while self._violations_seen < len(viol):
            pc, why = viol[self._violations_seen]
            self._violations_seen += 1
            self._report("packet_rules", pc, cycle, "; ".join(why), "")
        ip = core.d2ex
        if ip is not None and vs is not None:
            self._packet(ip, cycle, vs.row_bytes)

    def _packet(self, ip, cycle: int, row_bytes: int) -> None:
        pkt = ip.packet
        text = None

        def ptext():
            nonlocal text
            if text is None:
                text = format_packet(pkt)
            return text

        for o in pkt.scalar:
            if o.valid and o.fu in (LD, ST):
                rs1 = ip.operands.get(("scalar", o.slot), {}).get("rs1", 0)
                addr = (rs1 + o.imm) & MASK32
                if addr % 4:
                    self._report("misaligned", ip.pc, cycle,
                                 "%s at %#x: the word path uses %#x, overlapping the word "
                                 "below" % (o.mnemonic, addr, addr & ~3), ptext())
        # SDMAs first: a vreg op in the same packet runs alongside them.
        for o in pkt.sdma:
            if not o.valid:
                continue
            ops = ip.operands.get(("sdma", o.slot), {})
            meta = int(ops.get("rs3", 0)) & MASK32
            sid, rows = (meta >> 30) & 3, ((meta >> 25) & 0x1F) + 1
            first = int(ops.get("rs1", 0)) // row_bytes
            dma = _Dma(sid, first, rows, o.store, ip.pc, o.rs1_rd)
            for v in self._vst[sid]:
                if dma.covers(v.row):
                    kind = "vreg_st_then_sdma_st" if o.store else "vreg_st_then_sdma_ld"
                    self._report(kind, ip.pc, cycle,
                                 "%s of pad %d rows %d-%d (rs1 x%d) while the vreg.st at %s to "
                                 "row %d (rs1 x%d) hasn't landed; different rs1, so no "
                                 "reservation orders them"
                                 % (o.mnemonic, sid, first, first + rows - 1, o.rs1_rd,
                                    self._where(v.pc), v.row, v.rs1), ptext())
                    break
            self._dma[sid] = dma
        for o in pkt.vector:
            if not (o.valid and o.fu == V_VLSU):
                continue
            ops = ip.operands.get(("vector", o.slot), {})
            row = int(ops.get("rs1", 0)) // row_bytes + int(ops.get("rs2", 0))
            dma = self._dma.get(o.sid)
            load = o.vector_reg_write
            if dma is not None and dma.covers(row) and (load and not dma.store or not load):
                if load:
                    self._report("sdma_then_vreg_ld", ip.pc, cycle,
                                 "vreg.ld of pad %d row %d (rs1 x%d) while the scpad.ld at %s "
                                 "(rs1 x%d) still fills rows %d-%d; the load doesn't read x%d, "
                                 "so nothing makes it wait"
                                 % (o.sid, row, o.rs1, self._where(dma.pc), dma.rs1,
                                    dma.first, dma.first + dma.rows - 1, dma.rs1), ptext())
                else:
                    self._report("sdma_then_vreg_st", ip.pc, cycle,
                                 "vreg.st to pad %d row %d while the %s at %s (rs1 x%d) still "
                                 "uses rows %d-%d"
                                 % (o.sid, row, "scpad.st" if dma.store else "scpad.ld",
                                    self._where(dma.pc), dma.rs1, dma.first,
                                    dma.first + dma.rows - 1), ptext())
            if not load:
                self._vst[o.sid].append(_Vst(row, o.rs1, ip.pc))

    # -- results ------------------------------------------------------------------------
    def _where(self, pc: int) -> str:
        where = "%#x" % pc
        d = self.p.describe_pc
        if d is not None and d(pc) != where:
            where += " (%s)" % d(pc)
        return where

    def _report(self, kind: str, pc: int, cycle: int, detail: str, packet: str) -> None:
        key = (kind, pc)
        f = self.findings.get(key)
        if f is None:
            self.findings[key] = Finding(kind, pc, cycle, detail, packet)
        else:
            f.count += 1

    def report(self) -> List[dict]:
        return [dict(kind=f.kind, pc=f.pc, where=self._where(f.pc), first_cycle=f.first_cycle,
                     count=f.count, detail=f.detail, packet=f.packet)
                for f in sorted(self.findings.values(),
                                key=lambda f: (list(KINDS).index(f.kind), f.pc))]

    def summary(self) -> str:
        if not self.findings:
            return "dependency check: nothing found"
        rows = self.report()
        lines = ["dependency check: %d finding%s at %d pc%s"
                 % (sum(r["count"] for r in rows), "s" * (sum(r["count"] for r in rows) != 1),
                    len(rows), "s" * (len(rows) != 1))]
        kind = None
        for r in rows:
            if r["kind"] != kind:
                kind = r["kind"]
                lines.append("  %s" % KINDS[kind])
            lines.append("    %s, first at cycle %d%s: %s"
                         % (r["where"], r["first_cycle"],
                            " (%d times)" % r["count"] if r["count"] > 1 else "", r["detail"]))
            if r["packet"]:
                lines.append("      packet: %s" % r["packet"])
        return "\n".join(lines)
