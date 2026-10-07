"""Performance monitor: metrics for workload and system analysis.

A read-only observer of AtallaPlatform, ticked last in every cycle (it
changes nothing). It gathers:

- **Where the cycles go.** Each cycle is put in exactly one bucket, from the
  scheduler's point of view: a packet issued; decode 2 held one back (a
  register hazard, a register-file bank conflict, or a busy unit, named);
  or decode 2 had nothing to issue (the icache filling, the front end
  refilling after a flush, or the halt waiting for work in flight). The
  buckets add up to the run's cycles.
- **How busy each unit was:** cycles each scalar execute unit, the
  load/store unit, the data cache, the lanes, the GSAU, the systolic array,
  each VLSU, the transpose unit and each scratchpad's DMA had work. A unit
  near 100% is the bottleneck; one near 0% is idle hardware.
- **The instruction mix** retired, by class, and how full the packets were.
- **Memory:** bursts and bytes each DRAM master moved (icache, dcache, the
  four scratchpad backends), how busy the shared channel was, and the
  cycles each master lost waiting for it.
- **Work and intensity:** floating-point operations done (lanes and the
  array), throughput per cycle, DRAM bandwidth used against its peak, and
  arithmetic intensity against bytes really moved over DRAM -- unlike the
  functional sim's, which counts only SDMA reads.

    mon = PerfMonitor(platform)            # build_atalla_platform adds one
    ...run...
    mon.report()                           # nested dict
    print(mon.summary())                   # a readable table
    mon.to_json(path, timeline=True)       # with the per-cycle timeline

With `timeline=True` it also keeps, per cycle, the bucket and the set of
busy units, for plotting where time goes over a run.
"""

import json
from collections import Counter
from typing import Dict, List, Optional

from base.clocked_object import Clocked
from scheduler.control import (
    ALU, BF_ADD, BF_DIV, BF_MULT, BF_SLT, BF_SUB, BF_TO_INT, CONTROL, HALT, INT_TO_BF, LD,
    S_DIV, S_MOD, S_MULT, SQRT, ST, V_EXP, V_GSAU, V_MUL, V_MVMT, V_REDU, V_TRANS, V_VLSU,
    VECTOR_ALU_FUS)
from scheduler.execute import START
from scheduler.isa import PACKET_SIZE

_SCALAR_CLASS = {ALU: "scalar_alu", CONTROL: "branch", HALT: "halt",
                 BF_ADD: "scalar_bf16", BF_SUB: "scalar_bf16", BF_MULT: "scalar_bf16",
                 BF_SLT: "scalar_bf16", BF_DIV: "scalar_bf16", SQRT: "scalar_bf16",
                 BF_TO_INT: "convert", INT_TO_BF: "convert",
                 S_DIV: "scalar_muldiv", S_MOD: "scalar_muldiv", S_MULT: "scalar_muldiv",
                 LD: "scalar_load", ST: "scalar_store"}


def _vector_class(o) -> str:
    if o.fu in VECTOR_ALU_FUS or o.fu in (V_MUL, V_EXP):
        return "vector_lane"
    if o.fu == V_REDU:
        return "vector_reduce"
    if o.fu == V_GSAU:
        return "weight_load" if o.mnemonic == "lw.vi" else "gemm"
    if o.fu == V_VLSU:
        return "vreg_load" if o.mnemonic == "vreg.ld" else "vreg_store"
    if o.fu == V_MVMT:
        return "move"
    if o.fu == V_TRANS:
        return "transpose"
    return "vector_other"


class PerfMonitor(Clocked):
    def __init__(self, platform, *, timeline: bool = False):
        super().__init__()
        self.p = platform
        self.timeline_on = bool(timeline)
        self._tick = -1
        self.cycles = 0
        self.buckets: Counter = Counter()
        self.busy: Counter = Counter()
        self.mix: Counter = Counter()
        self.packets = 0
        self.slots_filled = 0
        self._issued_seen = 0
        #: ("s"|"v"|"m", register) -> the class of the op last issued to
        #: write it, so a hazard can be put down to the unit it waits on.
        self._producer: Dict = {}
        self._mul_ops_seen = 0
        self.timeline: List[str] = []
        self.timeline_busy: List[List[str]] = []

    # -- one cycle --------------------------------------------------------------------
    def tick(self, time: Optional[float] = None) -> None:
        cycle = self._consume_tick(time, attr_name="_tick")
        if cycle is None:
            return
        core = self.p.core
        if core.done_at is not None and cycle > core.done_at:
            return
        self.cycles += 1
        bucket = self._bucket(core)
        self.buckets[bucket] += 1
        busy = self._busy_units(core, cycle)
        self.busy.update(busy)
        if self.timeline_on:
            self.timeline.append(bucket)
            self.timeline_busy.append(busy)

    def _bucket(self, core) -> str:
        issued = core.issued
        if len(issued) > self._issued_seen:
            for _, ip in issued[self._issued_seen:]:
                self._count_packet(ip.packet)
            self._issued_seen = len(issued)
            return "issue"
        d2 = core.decode2
        if d2.last_stall == "unit":
            return "stall:" + (d2.blocked_by or "unit")
        if d2.last_stall == "hazard":
            return "stall:hazard:" + self._hazard_producer(d2)
        if d2.last_stall:
            return "stall:" + d2.last_stall
        if core.halt_latch or core.halted_at is not None:
            return "halt:drain"
        if core.icache.state != 0:                 # FILL
            return "frontend:icache"
        return "frontend:refill"

    def _hazard_producer(self, d2) -> str:
        """The unit writing the busy register decode 2 waited on."""
        return self._producer.get(d2.hazard_reg, "unknown")

    def _count_packet(self, pkt) -> None:
        s, v, d = pkt.valid_ops()
        for o in s:
            w = o.scalar_write()
            if w is not None:
                self._producer[("s", w)] = _SCALAR_CLASS.get(o.fu, "scalar_other")
            w = o.mask_write()
            if w is not None:
                self._producer[("m", w)] = "move"
        for o in v:
            c = _vector_class(o)
            for kind, w in (("s", o.scalar_write()), ("v", o.vector_write()),
                            ("m", o.mask_write())):
                if w is not None:
                    self._producer[(kind, w)] = c
        for o in d:
            w = o.scalar_write()
            if w is not None:
                self._producer[("s", w)] = "sdma"
        self.packets += 1
        self.slots_filled += len(s) + len(v) + len(d)
        for o in s:
            self.mix[_SCALAR_CLASS.get(o.fu, "scalar_other")] += 1
        for o in v:
            self.mix[_vector_class(o)] += 1
        for o in d:
            self.mix["sdma_load" if o.mnemonic == "scpad.ld" else "sdma_store"] += 1

    def _busy_units(self, core, cycle: int) -> List[str]:
        out = []
        for name in ("ex2", "ex3", "ex4"):
            if getattr(core, name).state != START:
                out.append(name)
        if not core.lsu.idle:
            out.append("lsu")
        if not core.dcache.idle():
            out.append("dcache")
        vs = core.vector
        if vs is None:
            return out
        vc = vs.vc
        if vs._lane_ops or vs._lane_results:
            out.append("lanes")
        g = vc.gsau
        if g.has_pending() or not g.rd_queue.is_empty() or not g.writebacks.is_empty():
            out.append("gsau")
        mul_ops = self.p.sa.metrics.get("mul_ops", 0)
        if mul_ops > self._mul_ops_seen:
            out.append("array")
        self._mul_ops_seen = mul_ops
        for s, v in enumerate(vc.vls_units):
            if not (v.issue_q.is_empty() and v.req_q.is_empty() and v.rsp_q.is_empty()
                    and v.wb_q.is_empty()) or v.outstanding_loads():
                out.append("vlsu%d" % s)
        if vc.transpose.busy or vs._tp_reqs or vs._tp_vds:
            out.append("transpose")
        for s, b in enumerate(vs.sdma_busy[:4]):
            if b:
                out.append("sdma%d" % s)
        ch = self.p.backends[0].shared_burst_channel if self.p.backends else None
        if ch is not None and ch.last_issue_tick == cycle:
            out.append("dram_channel")
        return out

    # -- the report ---------------------------------------------------------------------
    def report(self) -> Dict:
        p, core = self.p, self.p.core
        cyc = max(1, self.cycles)
        n = p.vc.vector_len
        vs = core.vector
        burst = p.backends[0].dram_burst_bytes if p.backends else 0

        masters = {}
        for name, port in (p.ports or {}).items():
            st = port.stats
            masters[name] = dict(bursts=st["read_bursts"] + st["write_bursts"],
                                 bytes=st["read_bytes"] + st["write_bytes"],
                                 read_bytes=st["read_bytes"], write_bytes=st["write_bytes"],
                                 wait_cycles=st["contention_cycles"])
        for s, b in enumerate(p.backends):
            st = b.get_stats()
            masters["scpad%d" % s] = dict(bursts=st["issued_bursts"],
                                          bytes=st["issued_bursts"] * burst,
                                          wait_cycles=st["backend_stalls"])
        dram_bursts = sum(m["bursts"] for m in masters.values())
        dram_bytes = sum(m["bytes"] for m in masters.values())

        lane_flops = (vs.stats["lane_ops"] * n) if vs else 0
        array_flops = 2 * p.sa.metrics.get("mac_ops", 0)
        flops = lane_flops + array_flops
        ops = sum(self.mix.values())

        return {
            "cycles": self.cycles,
            "cycle_breakdown": {k: v for k, v in self.buckets.most_common()},
            "issue": {
                "packets": self.packets,
                "packets_per_cycle": self.packets / cyc,
                "ops": ops,
                "ops_per_cycle": ops / cyc,
                "slot_utilization": self.slots_filled / max(1, self.packets * PACKET_SIZE),
                "flushes": core.flushes,
                "packets_fetched": core.packets_fetched,
            },
            "instruction_mix": dict(self.mix.most_common()),
            "unit_busy": {k: {"cycles": v, "fraction": v / cyc}
                          for k, v in self.busy.most_common()},
            "memory": {
                "dram_bursts": dram_bursts,
                "dram_bytes": dram_bytes,
                "dram_bytes_per_cycle": dram_bytes / cyc,
                "dram_peak_bytes_per_cycle": burst,
                "dram_channel_busy": dram_bursts / cyc,
                "masters": masters,
                "icache": {"fills": core.icache.fills, "iwait_cycles": core.icache.iwait_cycles},
                "dcache": dict(core.dcache.stats),
                "scratchpad": p.spad.get_stats()["tiles"],
            },
            "work": {
                "flops": flops,
                "lane_flops": lane_flops,
                "array_flops": array_flops,
                "flops_per_cycle": flops / cyc,
                "array_mac_utilization": p.sa.metrics.get("mac_ops", 0)
                                         / (cyc * p.sa.size * p.sa.size),
                "arithmetic_intensity": flops / dram_bytes if dram_bytes else None,
            },
        }

    def summary(self) -> str:
        r = self.report()
        cyc = max(1, r["cycles"])
        lines = ["%d cycles" % r["cycles"], "", "where the cycles went:"]
        for k, v in r["cycle_breakdown"].items():
            lines.append("  %-22s %8d  %5.1f%%" % (k, v, 100.0 * v / cyc))
        iss = r["issue"]
        lines += ["", "issue: %d packets (%.3f/cycle), %d ops (%.3f/cycle), slots %.1f%% full"
                  % (iss["packets"], iss["packets_per_cycle"], iss["ops"],
                     iss["ops_per_cycle"], 100 * iss["slot_utilization"]),
                  "instruction mix: " + ", ".join("%s %d" % kv
                                                  for kv in r["instruction_mix"].items()),
                  "", "unit busy:"]
        for k, v in r["unit_busy"].items():
            lines.append("  %-14s %8d  %5.1f%%" % (k, v["cycles"], 100 * v["fraction"]))
        m = r["memory"]
        lines += ["", "DRAM: %d bytes in %d bursts, %.2f B/cycle of %d peak, channel busy %.1f%%"
                  % (m["dram_bytes"], m["dram_bursts"], m["dram_bytes_per_cycle"],
                     m["dram_peak_bytes_per_cycle"], 100 * m["dram_channel_busy"])]
        for k, v in m["masters"].items():
            if v["bursts"] or v["wait_cycles"]:
                lines.append("  %-8s %6d bursts %8d bytes  waited %d cycles"
                             % (k, v["bursts"], v["bytes"], v["wait_cycles"]))
        w = r["work"]
        ai = w["arithmetic_intensity"]
        lines += ["", "work: %d FLOPs (lanes %d, array %d), %.1f FLOP/cycle, array MAC use %.2f%%"
                  % (w["flops"], w["lane_flops"], w["array_flops"], w["flops_per_cycle"],
                     100 * w["array_mac_utilization"]),
                  "arithmetic intensity: %s FLOP/DRAM byte"
                  % ("%.2f" % ai if ai is not None else "n/a")]
        return "\n".join(lines)

    def to_json(self, path: str, timeline: bool = False) -> None:
        out = self.report()
        if timeline and self.timeline_on:
            out["timeline"] = {"bucket": self.timeline, "busy": self.timeline_busy}
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
