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
  cycles each master lost waiting for it. On the AXI bus: bursts are AXI
  transactions, the read and write paths' bytes a cycle against their 8
  each, and the cycles a master waited for a full bus FIFO.
- **The vloop sequencer** (only when a loop ran): rows, its ARM / RUN /
  DRAIN cycles and stalls, rows per cycle against the 1/II ceiling, its
  micro-ops, measured load latency, peak buffer depth, the tables it ran,
  and how long ordinary vector ops were held back by it. It shows in "unit
  busy" as `vloop` and `vloop:<FU row/group>`, in the instruction mix as
  `vloop`, and a lane op held out of a cycle the loop takes (v1) as the
  bucket `stall:alu:vloop` / `stall:mul:vloop`.
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
    VECTOR_ALU_FUS, V_CISC)
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
    if o.fu == V_CISC:
        return "vloop"
    return "vector_other"


class PerfMonitor(Clocked):
    """Watches an AtallaPlatform each cycle and tallies the metrics above.

    To add a metric: count it in tick() (read the platform, never change it)
    and add it to report(); summary() prints from report().
    """
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
        self._vloop_ticked = 0
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
        # The vloop sequencer: "vloop" from start to the last store leaving,
        # "vloop:<kind>" while its micro-ops are in that FU row / group.
        seq = getattr(vs, "vloop", None)
        # Busy in every cycle it ticked in ARM / RUN / DRAIN -- including the
        # last DRAIN cycle, after which it is already IDLE when sampled.
        ticked = (seq.stats["arm_cycles"] + seq.stats["run_cycles"]
                  + seq.stats["drain_cycles"]) if seq is not None else 0
        loop_cycle, self._vloop_ticked = ticked > self._vloop_ticked, ticked
        if loop_cycle:
            out.append("vloop")
            for kind, row in seq.rows.items():
                if row.busy:
                    out.append("vloop:" + kind)
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
        """Every metric, as a nested dict (the keys are the sections of the
        module docstring). To add a metric, add its key here.
        """
        p, core = self.p, self.p.core
        cyc = max(1, self.cycles)
        n = p.vc.vector_len
        vs = core.vector
        burst = p.backends[0].dram_burst_bytes if p.backends else 0

        axi = getattr(p, "axi", None)
        masters = {}
        for name, port in ({} if axi is not None else (p.ports or {})).items():
            st = port.stats
            masters[name] = dict(bursts=st["read_bursts"] + st["write_bursts"],
                                 bytes=st["read_bytes"] + st["write_bytes"],
                                 read_bytes=st["read_bytes"], write_bytes=st["write_bytes"],
                                 wait_cycles=st["contention_cycles"])
        for s, b in enumerate(p.backends if axi is None else ()):
            st = b.get_stats()
            masters["scpad%d" % s] = dict(bursts=st["issued_bursts"],
                                          bytes=st["issued_bursts"] * burst,
                                          wait_cycles=st["backend_stalls"])
        if axi is not None:
            # On the AXI bus (memory/axi.py) a burst is a transaction of beats.
            bb = axi.params.beat_bytes
            for port in axi.ports:
                st = port.stats
                name = "scpad" + port.name[2:] if port.name.startswith("sp") else port.name
                masters[name] = dict(bursts=st["reads"] + st["writes"],
                                     bytes=(st["read_beats"] + st["write_beats"]) * bb,
                                     read_bytes=st["read_beats"] * bb,
                                     write_bytes=st["write_beats"] * bb,
                                     wait_cycles=st["ar_full"] + st["aw_full"])
        dram_bursts = sum(m["bursts"] for m in masters.values())
        dram_bytes = sum(m["bytes"] for m in masters.values())

        lane_flops = (vs.stats["lane_ops"] * n) if vs else 0
        array_flops = 2 * p.sa.metrics.get("mac_ops", 0)
        flops = lane_flops + array_flops
        ops = sum(self.mix.values())

        out = {
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
                # The shared channel launches one burst a cycle; the AXI bus has a
                # read path and a write path, a beat a cycle each.
                "dram_peak_bytes_per_cycle": (2 * axi.params.beat_bytes if axi is not None
                                              else burst),
                "dram_channel_busy": dram_bursts / cyc,
                "axi": None if axi is None else {
                    "read_bytes_per_cycle": axi.stats["r_beats"] * axi.params.beat_bytes / cyc,
                    "write_bytes_per_cycle": axi.stats["w_beats"] * axi.params.beat_bytes / cyc,
                    "path_peak_bytes_per_cycle": axi.params.beat_bytes,
                },
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
        # Only when a loop ran, so programs without one report as before (the
        # equivalence snapshots hold whole reports).
        seq = getattr(vs, "vloop", None) if vs else None
        if seq is not None and seq.stats["loops"]:
            out["vloop"] = self._vloop_report(seq, n)
        return out

    def _vloop_report(self, seq, n: int) -> Dict:
        """The vloop sequencer: what ran, how well the pipeline kept up, what
        held it back. Its CMP/MUL/... element ops are counted here, apart
        from "work"'s FLOPs (relu's MUL by 1.0 is a select, not arithmetic)."""
        st = seq.stats
        run, stall = st["run_cycles"], st["stall_cycles"]
        tables = {name: {"ii": t.ii, "ii_wanted": t.ii_wanted, "stages": t.stages,
                         "offsets": {op.name: off for op, off in zip(t.ops, t.offsets)}}
                  for name, t in seq.tables_run.items()}
        best = max((1.0 / t.ii for t in seq.tables_run.values()), default=None)
        vc_stall = getattr(self.p.vc, "stats", {}).get("stall", {})
        return {
            "loops": st["loops"],
            "rows": st["rows"],
            "tables": tables,
            "cycles": {"arm": st["arm_cycles"], "run": run, "drain": st["drain_cycles"],
                       "total": st["arm_cycles"] + run + st["drain_cycles"]},
            "stall_cycles": stall,
            "stall_fraction": stall / run if run else 0.0,
            # Rows finished per RUN cycle (prologue and epilogue included), and
            # the steady-state ceiling 1/II it approaches as n grows.
            "rows_per_run_cycle": st["rows"] / run if run else 0.0,
            "rows_per_cycle_ceiling": best,
            "loads": st["loads"],
            "stores": st["stores"],
            "fu_ops": st["fu_ops"],
            "element_ops": st["fu_ops"] * n,
            "load_latency": {"min": st["load_latency_min"], "max": st["load_latency_max"]},
            "peak_buffer": dict(seq.peak_buffer),
            # The buffers' size the tables need in steady state (the area
            # side of a variant's trade; peak_buffer is what this run
            # reached), and scratchpad reads per row (the other side).
            "buffer_bits": dict(seq.buffer_bits()),
            "buffer_bits_total": sum(seq.buffer_bits().values()),
            "loads_per_row": st["loads"] / st["rows"] if st["rows"] else 0.0,
            # Ordinary vector ops held back by the loop: v1 holds them in
            # decode 2 ("stall:alu:vloop"); the M x N core stalls their slot.
            "ops_stalled_by_loop": {
                "decode2_cycles": sum(v for k, v in self.buckets.items()
                                      if k.startswith("stall:") and k.endswith(":vloop")),
                "slot_stall_cycles": sum(vc_stall.get("kernel", [])),
            },
        }

    def summary(self) -> str:
        """report() as a readable text table."""
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
        if m["axi"] is not None:
            a = m["axi"]
            lines += ["", "DRAM: %d bytes in %d bursts over the AXI bus: reads %.2f B/cycle, "
                      "writes %.2f B/cycle, of %d each"
                      % (m["dram_bytes"], m["dram_bursts"], a["read_bytes_per_cycle"],
                         a["write_bytes_per_cycle"], a["path_peak_bytes_per_cycle"])]
        else:
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
        vl = r.get("vloop")
        if vl is not None:
            c, lat, held = vl["cycles"], vl["load_latency"], vl["ops_stalled_by_loop"]
            lines += ["", "vloop: %d loop(s), %d rows; %d cycles (arm %d, run %d, drain %d), "
                      "%d stalled (%.1f%% of run)"
                      % (vl["loops"], vl["rows"], c["total"], c["arm"], c["run"], c["drain"],
                         vl["stall_cycles"], 100 * vl["stall_fraction"]),
                      "  %.3f rows/run cycle (ceiling 1/II = %s), %d micro-ops "
                      "(%d element ops), load latency %s..%s"
                      % (vl["rows_per_run_cycle"],
                         "%.3f" % vl["rows_per_cycle_ceiling"]
                         if vl["rows_per_cycle_ceiling"] is not None else "n/a",
                         vl["fu_ops"], vl["element_ops"], lat["min"], lat["max"])]
            for name, t in vl["tables"].items():
                lines.append("  table %s: II %d, %d stages, offsets %s"
                             % (name, t["ii"], t["stages"],
                                " ".join("%s@%d" % kv for kv in t["offsets"].items())))
            lines.append("  peak buffer entries: %s; ordinary ops held back: decode 2 %d "
                         "cycles, M x N slots %d cycles"
                         % (", ".join("%s %d" % kv for kv in vl["peak_buffer"].items()),
                            held["decode2_cycles"], held["slot_stall_cycles"]))
            lines.append("  buffers %d bits (%.2f KB), %.2f loads/row"
                         % (vl["buffer_bits_total"], vl["buffer_bits_total"] / 8192,
                            vl["loads_per_row"]))
        return "\n".join(lines)

    def to_json(self, path: str, timeline: bool = False) -> None:
        out = self.report()
        if timeline and self.timeline_on:
            out["timeline"] = {"bucket": self.timeline, "busy": self.timeline_busy}
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
