"""SchedulerCore: the one Clocked object the platform registers.

Owns every pipeline stage as an RTLModule and drives them in three phases per
cycle -- settle readiness, settle data and next state, commit -- so that no
stage's view of another depends on the order they were ticked in.

Stages 2-4 cover the front end, decode 2, and scalar execute and writeback:

    PC --> [ icache | BTB ] --> IF/D1 --> decode1 --> D1/D2 --> decode2 --> D2/EX
                                                                 ^            |
                                          EX/WB <-- scalar WB <-- EX1-EX5 <---+
                                                                     |
                                                                   dcache

With `execute=True` (the default) the execute stage drives the redirect, the
halt, the units' readiness and the scalar writebacks itself. Given a vector
core and the scratchpad backends (stage 5, scheduler/vector.py), it also
dispatches the vector and SDMA slots and owns vector writeback; without
them those signals stay attributes, set by whoever drives the core:

    vector_ready, vlsu_ready, scpad_busy, and writeback_at() for vector,
    mask and SDMA writebacks

With `execute=False` (the stage 2 and 3 tests) every execute-side signal is
an attribute too:

    redirect_valid     execute's redirect: flush and jump, and write the BTB
    redirect_target    where to jump
    redirect_pc        the branch's own PC, which is what the BTB is written at
    internal_halt      execute's internal_halt
    ex_ready           {1..5: bool}, each scalar execute unit's ready
    vector_ready       {"alu", "mul", "reduction", "gsau", "movement": bool}
    vlsu_ready         [bool] * 4, one VLSU per scratchpad
    scpad_busy         [bool] * 4
    writeback_at()     queue a writeback (scalar value, vector/mask register,
                       or an SDMA's completion) for a given cycle's EX/WB latch
    decode2_override   None, or force decode 2's ready -- for front-end tests
                       that have no execute stage to clear their hazards

Each latch accepts when the stage after it is ready OR when it is empty, so
bubbles collapse instead of blocking:

    D1D2 ready = decode2_ready || !D1D2.valid
    IFD1 ready = D1D2 ready    || !IFD1.valid
    PC moves   when flush || (ihit && IFD1 ready)
"""

from typing import Dict, Iterable, Optional

from base.clocked_object import Clocked
from base.dtype import DType
from scheduler.dcache import DCache, DCacheConfig, WordMemory
from scheduler.decode1 import D1D2Latch
from scheduler.decode2 import Decode2
from scheduler.execute import (
    DEFAULT_LATENCY, Ex1, ExOp, LoadStoreUnit, MultiCycleUnit, arbitrate)
from scheduler.vector import VectorSide
from scheduler.fetch import BTB, Fetch, IFD1Latch
from scheduler.icache import ICache
from scheduler.isa import PACKET_BYTE_W

Time = float


def load_program_text(text: str):
    """Parse the functional sim's program format into (instr, data) images.

    Mirrors src/misc/memory.py Memory.load_from_file: `addr: w0 w1 w2 w3`
    with spaces removed and the whole thing read as one hex number, so the
    first word is the most significant -- slot 0. A `.data` line switches to
    32-bit data words. `#` starts a comment.
    """
    instr: Dict[int, int] = {}
    data: Dict[int, int] = {}
    mode = instr
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.split("#")[0].strip()
        if not line:
            continue
        if line.startswith(".data"):
            mode = data
            continue
        try:
            addr_s, data_s = [x.strip() for x in line.split(":")]
            value = int(data_s.replace(" ", "").replace("_", ""), 16)
            addr = int(addr_s, 16)
        except ValueError:
            raise ValueError("line %d: not '<addr>: <hex>': %r" % (lineno, line))
        limit = 160 if mode is instr else 32
        if value.bit_length() > limit:
            raise ValueError("line %d: wider than %d bits" % (lineno, limit))
        mode[addr] = value
    return instr, data


class VeggieStorage:
    """Decode 2's vector register file kept in the vector core's Veggie
    banks, so the registers exist once."""

    def __init__(self, vc):
        self.vc = vc

    def read(self, r: int):
        return self.vc.read_vreg(r)

    def write(self, r: int, value) -> None:
        # Straight into the banks: values are already BF16, and the vector
        # core's cast would round them to FP16 (numpy has no bfloat16).
        bank, addr = self.vc._reg_to_bank_addr(r)
        self.vc.veggie.data_banks[bank][addr] = [float(x) for x in value]
        self.vc.veggie.dtype_banks[bank][addr] = DType.BF16


class SchedulerCore(Clocked):

    def __init__(self, program: Optional[Dict[int, int]] = None, *,
                 icache_first_beat_wait: int = 0, icache_beat_wait: int = 0,
                 strict: bool = True, execute: bool = True,
                 data: Optional[Dict[int, int]] = None,
                 dcache_config: Optional[DCacheConfig] = None,
                 lsu_depth: int = 4,
                 ex_latency: Optional[Dict[str, int]] = None,
                 vector_core=None, backends=(), memory=None):
        super().__init__()
        self._tick = -1
        self.program: Dict[int, int] = dict(program or {})

        self.icache = ICache(first_beat_wait=icache_first_beat_wait,
                             beat_wait=icache_beat_wait)
        self.btb = BTB("btb")
        self.fetch = Fetch("fetch")
        self.ifd1 = IFD1Latch("ifd1")
        self.d1d2 = D1D2Latch()
        self.vector_core = vector_core
        self.decode2 = Decode2(
            strict=strict,
            vector_storage=VeggieStorage(vector_core) if vector_core is not None else None,
            vector_len=vector_core.vector_len if vector_core is not None else 32)
        self._modules = (self.icache, self.btb, self.fetch, self.ifd1, self.d1d2,
                         self.decode2)

        # Inputs from stages that do not exist yet.
        self.decode2_override: Optional[bool] = None
        self.ex_ready = {u: True for u in range(1, 6)}
        self.vector_ready = {}
        self.vlsu_ready = [True] * 4
        self.scpad_busy = [False] * 4
        self._wb_due: Dict[int, Dict[str, list]] = {}
        self.redirect_valid = False
        self.redirect_target = 0
        self.redirect_pc = 0
        self.internal_halt = False

        self.cycles = 0
        self.packets_fetched = 0
        self.fetch_stall_cycles = 0
        self.btb_hits = 0
        self.predicted_taken = 0
        self.flushes = 0
        #: Every packet that entered the D1/D2 latch, as (cycle, pc).
        self.issued_to_d1d2 = []
        #: The D2/EX latch: the packet decode 2 issued last cycle, or None.
        self.d2ex = None
        #: Every packet decode 2 issued, as (cycle, IssuedPacket).
        self.issued = []

        # Stage 4: execute and writeback.
        self.execute = bool(execute)
        self.memory = memory if memory is not None else \
            WordMemory(lambda a, d=dict(data or {}): d.get(a, 0))
        self.dcache = DCache(config=dcache_config or DCacheConfig(), memory=self.memory)
        latency = dict(DEFAULT_LATENCY, **(ex_latency or {}))
        self.ex1 = Ex1()
        self.ex2 = MultiCycleUnit("ex2", latency)
        self.ex3 = MultiCycleUnit("ex3", latency)
        self.ex4 = MultiCycleUnit("ex4", latency)
        self.lsu = LoadStoreUnit(self.dcache, depth=lsu_depth)
        #: Stage 5: the vector side, when there is a vector core to drive.
        self.vector: Optional[VectorSide] = None
        if vector_core is not None:
            d2 = self.decode2
            self.vector = VectorSide(vector_core, backends,
                                     read_vreg=d2.vrf.read, read_mask=d2.mrf.read)
        self.halt_latch = False
        #: The cycle execute raised halt_out, and the cycle the dcache had
        #: written back its dirty lines after it.
        self.halted_at: Optional[int] = None
        self.done_at: Optional[int] = None
        #: Every scalar write, as (cycle it reached the EX/WB latch, rd, value).
        self.scalar_writes = []

    @property
    def pc(self) -> int:
        return self.fetch.pc

    def load(self, program: Dict[int, int]) -> None:
        self.program = dict(program)

    def warm_icache(self, addresses: Optional[Iterable[int]] = None) -> None:
        self.icache.warm(self.program if addresses is None else addresses)

    def writeback_at(self, cycle: int, scalar=(), vector=(), mask=(), sdma=()) -> None:
        """Put writebacks in the EX/WB latch for `cycle`: scalar (reg, value)
        pairs; vector and mask registers, each a register (busy bit only) or
        (register, value); and SDMA completions (the rs1 an SDMA held). They
        reach the register file and clear their busy bits on that cycle's
        edge."""
        due = self._wb_due.setdefault(int(cycle), {"scalar": [], "vector": [],
                                                   "mask": [], "sdma": []})
        due["scalar"] += [(int(r), int(v)) for r, v in scalar]
        due["vector"] += [(int(e[0]), e[1]) if isinstance(e, tuple) else int(e) for e in vector]
        due["mask"] += [(int(e[0]), e[1]) if isinstance(e, tuple) else int(e) for e in mask]
        due["sdma"] += [int(r) for r in sdma]

    # -- one cycle ---------------------------------------------------------
    def tick(self, time: Optional[Time] = None) -> None:
        cycle = self._consume_tick(time, attr_name="_tick")
        if cycle is None:
            return
        self._step(cycle)

    # -- execute (stage 4) ----------------------------------------------------------
    def _crossbar(self):
        """xbar_4x5_exec_comb.sv: each valid scalar op in the D2/EX latch to its
        unit, slot 0 first. Returns ({unit: ExOp}, halt)."""
        routed = {}
        pkt = self.d2ex
        if pkt is None:
            return routed, False
        for o in pkt.packet.scalar:
            if not o.valid or o.ex is None or o.ex in routed:
                continue
            ops = pkt.operands.get(("scalar", o.slot), {})
            routed[o.ex] = ExOp(o, ops.get("rs1", 0), ops.get("rs2", 0), pkt.pc,
                                pkt.pc_pred_addr, pkt.predict_taken)
        return routed, pkt.halt

    def _execute_eval(self, cycle: int) -> None:
        """Everything execute drives combinationally this cycle: results,
        writeback grants, readiness, the redirect and the halt."""
        routed, halt_in = self._crossbar()
        self._routed = routed
        ex1_res, redirect = self.ex1.offer(routed.get(1))
        self.lsu.present(routed.get(5))
        dc = self.dcache
        dc.in_halt = self.halted_at is not None
        dc.eval_ready()
        dc.eval_data()
        offers = {"ex1": ex1_res, "ex2": self.ex2.offer(), "ex3": self.ex3.offer(),
                  "ex4": self.ex4.offer(), "ex5": self.lsu.offer()}
        vs = self.vector
        if vs is not None:
            mts_op, mts_ops = vs.mts_op(self.d2ex)
            offers["mts"] = vs.mts.offer(mts_op, mts_ops)
        granted, writes, masks = arbitrate(offers)
        self._granted, self._redirect = granted, redirect
        if writes or masks:
            self.writeback_at(cycle + 1, scalar=[(w.rd, w.value) for w in writes],
                              mask=[(m.rd, m.value) for m in masks])
            for w in writes:
                self.scalar_writes.append((cycle + 1, w.rd, w.value))
        if vs is not None:
            self._vector_eval(cycle, granted.get("mts", False))
        self.ex_ready = {1: self.ex1.ready_in(granted.get("ex1", False)),
                         2: self.ex2.ready_in(routed.get(2)),
                         3: self.ex3.ready_in(routed.get(3)),
                         4: self.ex4.ready_in(routed.get(4)),
                         5: self.lsu.ready_in()}
        self.redirect_valid = redirect is not None
        if redirect is not None:
            self.redirect_target, self.redirect_pc = redirect.target, redirect.pc
        # execute_stage.sv: halt latches when a halt packet reaches EX, and
        # holds fetch and decode from that cycle on.
        self._halt_in = halt_in
        self.internal_halt = halt_in or self.halt_latch

    def _vector_eval(self, cycle: int, mts_granted: bool) -> None:
        """Stage 5's part of the cycle: readiness, vector writeback, SDMA
        completions, then the D2/EX latch's vector and SDMA ops to their units."""
        vs = self.vector
        vec, vlsu, busy = vs.unit_ready()
        vec["movement"] = vs.mts.ready(mts_granted)
        self.vector_ready, self.vlsu_ready, self.scpad_busy = vec, vlsu, busy
        granted, vw, mw = vs.arbitrate(vs.offers())
        done = vs.take_sdma_done()
        if vw or mw or done:
            self.writeback_at(cycle + 1, vector=[(w.reg, w.value) for w in vw],
                              mask=[(w.reg, w.value) for w in mw], sdma=done)
            vs.stats["vector_writes"] += len(vw)
            vs.stats["mask_writes"] += len(mw)
        vs.retire(granted)
        self._mts_granted = mts_granted
        vs.dispatch(self.d2ex)

    def _execute_commit(self, cycle: int) -> None:
        g, r = self._granted, self._routed
        if self.vector is not None:
            self.vector.mts.advance(self._mts_granted)
            self.vector.tick(cycle)
        self.ex1.advance(g.get("ex1", False), self._redirect is not None)
        self.ex2.advance(r.get(2), g.get("ex2", False))
        self.ex3.advance(r.get(3), g.get("ex3", False))
        self.ex4.advance(r.get(4), g.get("ex4", False))
        self.lsu.advance(g.get("ex5", False))
        self.dcache.commit()
        if (self.halt_latch and self.halted_at is None and self.decode2.scoreboard.idle
                and self.lsu.idle and not any(self.scpad_busy)
                and (self.vector is None or self.vector.idle)):
            self.halted_at = cycle          # halt_out
        if self.halted_at is not None and self.done_at is None and self.dcache.out_flushed:
            self.done_at = cycle
        self.halt_latch = self.halt_latch or self._halt_in

    @property
    def done(self) -> bool:
        """Halted, with every dirty dcache line written back to memory."""
        return self.done_at is not None

    def run_until_done(self, limit: int = 1_000_000) -> int:
        """Tick until done; return the number of cycles run."""
        c = self._tick + 1
        start = c
        while not self.done:
            if c - start >= limit:
                raise RuntimeError("no halt after %d cycles (pc %#x)" % (limit, self.pc))
            self.tick(float(c))
            c += 1
        return c - start

    def scalar_reg(self, r: int) -> int:
        return self.decode2.srf.read(r)

    def _step(self, cycle: int) -> None:
        if self.execute:
            self._execute_eval(cycle)
        flush, halt = bool(self.redirect_valid), bool(self.internal_halt)

        # Decode 2 first: its ready closes the backpressure chain.
        d2, d = self.decode2, self.d1d2
        wb = self._wb_due.pop(cycle, {})
        d2.in_scalar, d2.in_vector, d2.in_sdma = d.scalar, d.vector, d.sdma
        d2.in_valid, d2.in_pc = d.valid, d.pc
        d2.in_pc_pred_addr, d2.in_predict_taken = d.pc_pred_addr, d.predict_taken
        d2.in_flush, d2.in_halt = flush, halt
        d2.in_ex_ready, d2.in_vector_ready = self.ex_ready, self.vector_ready
        d2.in_vlsu_ready, d2.in_scpad_busy = self.vlsu_ready, self.scpad_busy
        d2.in_wb_scalar = wb.get("scalar", [])
        d2.in_wb_vector = wb.get("vector", [])
        d2.in_wb_mask = wb.get("mask", [])
        d2.in_wb_sdma = wb.get("sdma", [])
        d2.eval_ready()
        if self.decode2_override is not None:
            d2.out_ready = bool(self.decode2_override)
        decode2_ready = d2.out_ready

        # Readiness, back to front.
        d1d2_ready = bool(decode2_ready or not self.d1d2.valid)
        ifd1_ready = bool(d1d2_ready or not self.ifd1.valid)

        f = self.fetch
        f.in_flush, f.in_halt, f.in_ready = flush, halt, ifd1_ready
        f.in_pc_branch = self.redirect_target
        f.eval_ready()

        ic = self.icache
        ic.in_imemaddr, ic.in_imemREN, ic.in_halt = f.pc, f.out_imemREN, halt
        ic.eval_data()

        bhit, target = self.btb.read(f.pc)
        f.in_ihit = ic.out_ihit
        f.in_imemload = self.program.get(f.pc, 0) if ic.out_ihit else 0
        f.in_bhit, f.in_predict_target = bhit, target
        f.eval_data()

        b = self.btb
        b.in_update_en = flush
        b.in_pc_update, b.in_true_target = self.redirect_pc, self.redirect_target
        b.eval_data()

        # Decode 2 reads the D1/D2 latch's current contents; D1/D2 reads the
        # IF/D1 latch's. Both are settled from registers alone, so the order
        # against the latches' updates does not matter.
        d2.eval_data()
        i = self.ifd1
        d.in_flush, d.in_halt, d.in_ready = flush, halt, d1d2_ready
        d.in_packet, d.in_pc, d.in_valid = i.inst_packet, i.pc, i.valid
        d.in_predict_taken, d.in_pc_pred_addr = i.predict_taken, i.pc_pred_addr
        d.eval_data()

        i.in_flush, i.in_halt, i.in_ready = flush, halt, ifd1_ready
        i.in_pc, i.in_inst_packet = f.out_pc, f.out_inst_packet
        i.in_predict_taken, i.in_pc_pred_addr = (
            f.out_predict_taken, f.out_pc_pred_addr)
        i.in_valid = f.out_valid
        i.eval_data()

        self._count(cycle, flush, halt, d1d2_ready, ifd1_ready, bhit)
        self.d2ex = d2.out_issued
        if self.d2ex is not None:
            self.issued.append((cycle, self.d2ex))
        for m in self._modules:
            m.commit()
        if self.execute:
            self._execute_commit(cycle)

    def _count(self, cycle, flush, halt, d1d2_ready, ifd1_ready, bhit) -> None:
        self.cycles += 1
        f = self.fetch
        if f.out_valid and ifd1_ready and not flush:
            self.packets_fetched += 1
            if bhit:
                self.btb_hits += 1
            if f.out_predict_taken:
                self.predicted_taken += 1
        elif not flush and not halt:
            self.fetch_stall_cycles += 1
        if flush:
            self.flushes += 1
        if (d1d2_ready and self.ifd1.valid and not flush and not halt):
            self.issued_to_d1d2.append((cycle, self.ifd1.pc))

    def run(self, cycles: int, start: int = 0) -> None:
        for c in range(start, start + cycles):
            self.tick(float(c))
