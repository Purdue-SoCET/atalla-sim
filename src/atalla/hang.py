"""Why a run stopped making progress: what each part of the machine waits on.

`AtallaPlatform.run_until_done` raises `SimulationHang` when nothing has
moved for `hang_after` cycles -- no packet issued or entered decode, no
lane op, no array multiply, no DMA burst or cache burst, no AXI beat -- and
adds the same report to its "no halt after N cycles" error. The check reads
a handful of counters every CHECK_EVERY cycles, so it costs nothing a plain
run would notice.

The report names the packet decode 2 holds and why (a busy register, a busy
unit, a vreg.st reservation, a register-file bank conflict), the registers
the scoreboard holds busy, the scratchpad DMAs in flight, and every block
the activity probe sees holding work (atalla/activity.py: one definition of
"busy" for the viewer, the probe and this report).
"""

from typing import List, Optional

from scheduler.disasm import format_packet

#: How often run_until_done reads the progress counters.
CHECK_EVERY = 4096
#: Cycles without progress before a run is declared hung. Long DMAs and
#: DRAM refresh keep counters moving; nothing legitimate is still this long.
HANG_AFTER = 50_000


class SimulationHang(RuntimeError):
    """Nothing in the machine moved for `hang_after` cycles."""


def progress_signature(plat) -> tuple:
    """Counters that move whenever any part of the machine does work."""
    core = plat.core
    sig: List[int] = [core.issued_count, core.d1d2_count,
                      int(plat.sa.metrics.get("mul_ops", 0))]
    for be in plat.backends:
        sig += [be.total_dram_bursts_issued, be.total_dram_bursts_completed,
                be.total_tx_completed]
    for port in plat.ports.values():
        sig += [port.stats.get("read_bursts", 0), port.stats.get("write_bursts", 0)]
    if plat.axi is not None:
        st = plat.axi.stats
        sig += [st.get("r_beats", 0), st.get("w_beats", 0)]
    sig.append(sum(getattr(lane, "total_ops", 0) for lane in plat.vc.datapath.lanes))
    sig.append(sum(core.dcache.stats.values()) if isinstance(core.dcache.stats, dict) else 0)
    return tuple(sig)


#: Activity-probe blocks left out of "holding work": decode slots and bank
#: reads (the decode 2 lines say it better), and blocks whose state comes
#: from a counter moving this cycle, which says nothing about a stopped run.
_SKIP_PREFIXES = ("dec1_", "d2_", "srf_b", "vrf_b")
_SKIP = {"pc", "bpu", "decode1_xbar", "srf_read", "vrf_read", "array", "array_in",
         "channel", "axi_r", "axi_w"}


def _where(plat, pc: int) -> str:
    where = "%#x" % pc
    if plat.describe_pc is not None and plat.describe_pc(pc) != where:
        where += " (%s)" % plat.describe_pc(pc)
    return where


def _regs(kind: str, regs) -> str:
    return ", ".join("%s%d" % (kind, r) for r in sorted(regs)) or "-"


def _decode2_lines(plat) -> List[str]:
    core, d2 = plat.core, plat.core.decode2
    if not core.d1d2.valid:
        return ["decode 2 holds no packet (fetch: icache %s)"
                % ("filling a line" if core.icache.state != 0 else "idle")]
    pkt = getattr(d2, "_pkt", None)
    lines = ["decode 2 holds the packet at %s: %s"
             % (_where(plat, d2.in_pc), format_packet(pkt) if pkt is not None else "?")]
    why = d2.last_stall
    if why == "hazard" and d2.hazard_reg is not None:
        kind, r = d2.hazard_reg
        lines.append("  waiting on %s%d, which an op in flight still has to write"
                     % ({"s": "x"}.get(kind, kind), r))
    elif why == "unit":
        lines.append("  waiting for a busy unit: %s" % (d2.blocked_by or "?"))
    elif why == "vst":
        lines.append("  an SDMA waiting for the vreg.st reservations on its rs1")
    elif why == "vst_full":
        lines.append("  a vreg.st waiting for room in the reservation table")
    elif why == "regfile":
        lines.append("  waiting for a register-file bank conflict to clear")
    elif core.halt_latch or core.halted_at is not None:
        lines.append("  halted, waiting for work in flight to drain")
    return lines


def _block_lines(plat) -> List[str]:
    from atalla.activity import IDLE, STALLED, ActivityProbe
    probe = ActivityProbe(plat)
    cycle = plat.cycle
    probe._sample(cycle)                    # prime its counters, then read states only
    states = probe._sample(cycle)
    by_id = {b.id: b for b in probe.blocks}
    held, lanes = [], []
    for bid, state in states.items():
        if state == IDLE or bid in _SKIP or bid.startswith(_SKIP_PREFIXES) or bid not in by_id:
            continue
        if bid.startswith("lane"):
            lanes.append(int(bid[4:]))
            continue
        b = by_id[bid]
        what = b.stalled if state == STALLED and b.stalled else b.active
        held.append("  %-22s %s%s" % (b.label, "stalled: " if state == STALLED else "", what))
    if lanes:
        held.insert(0, "  %-22s a functional unit holds work"
                    % ("Lane %d" % lanes[0] if len(lanes) == 1
                       else "Lanes %d-%d" % (min(lanes), max(lanes))))
    return held


def diagnose(plat, stalled_for: Optional[int] = None) -> str:
    """A report of what the machine is waiting on, for a person to read."""
    core, d2 = plat.core, plat.core.decode2
    lines = ["at cycle %d, fetch pc %s%s" % (plat.cycle, _where(plat, core.pc),
                                               "; nothing has moved for %d cycles" % stalled_for
                                               if stalled_for else "")]
    lines += _decode2_lines(plat)
    if d2.violations:
        pc, why = d2.violations[0]
        lines.append("%d packet%s broke the hardware's packet rules, first at %s: %s. The RTL "
                     "drops part of such a packet, which can lose a register write and leave "
                     "it busy forever." % (len(d2.violations), "s" * (len(d2.violations) != 1),
                                           _where(plat, pc), "; ".join(why)))
    sb = d2.scoreboard
    lines.append("busy registers: scalar %s; vector %s; mask %s"
                 % (_regs("x", sb.scalar), _regs("v", sb.vector), _regs("m", sb.mask)))
    if d2.vst.cnt:
        lines.append("vreg.st reservations: %s" % ", ".join(
            "x%d (%d)" % (r, n) for r, n in sorted(d2.vst.cnt.items())))
    vs = core.vector
    if vs is not None and any(vs.sdma_busy):
        lines.append("scratchpad DMAs in flight: %s" % ", ".join(
            "pad %d" % i for i, b in enumerate(vs.sdma_busy) if b))
    held = _block_lines(plat)
    if held:
        lines.append("blocks holding work:")
        lines += held
    else:
        lines.append("no block holds work: nothing in flight can finish, so whatever decode 2 "
                     "waits on will never come")
    return "\n".join(lines)
