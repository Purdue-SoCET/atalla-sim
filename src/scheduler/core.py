"""SchedulerCore: the one Clocked object the platform registers.

Owns every pipeline stage as an RTLModule and drives them in three phases per
cycle -- settle readiness, settle data and next state, commit -- so that no
stage's view of another depends on the order they were ticked in.

Stage 2 covers the front end:

    PC --> [ icache | BTB ] --> IF/D1 latch --> decode1 --> D1/D2 latch
                                                            |
                                                   (decode2 in stage 3)

The signals the later stages will produce are attributes for now, set by
whoever drives the core -- tests, today:

    decode2_ready      decode_2.sv's `ready`: may D1/D2 hand a packet on
    redirect_valid     execute's redirect: flush and jump, and write the BTB
    redirect_target    where to jump
    redirect_pc        the branch's own PC, which is what the BTB is written at
    internal_halt      execute's internal_halt

Each latch accepts when the stage after it is ready OR when it is empty, so
bubbles collapse instead of blocking:

    D1D2 ready = decode2_ready || !D1D2.valid
    IFD1 ready = D1D2 ready    || !IFD1.valid
    PC moves   when flush || (ihit && IFD1 ready)
"""

from typing import Dict, Iterable, Optional

from base.clocked_object import Clocked
from scheduler.decode1 import D1D2Latch
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


class SchedulerCore(Clocked):

    def __init__(self, program: Optional[Dict[int, int]] = None, *,
                 icache_first_beat_wait: int = 0, icache_beat_wait: int = 0):
        super().__init__()
        self._tick = -1
        self.program: Dict[int, int] = dict(program or {})

        self.icache = ICache(first_beat_wait=icache_first_beat_wait,
                             beat_wait=icache_beat_wait)
        self.btb = BTB("btb")
        self.fetch = Fetch("fetch")
        self.ifd1 = IFD1Latch("ifd1")
        self.d1d2 = D1D2Latch()
        self._modules = (self.icache, self.btb, self.fetch, self.ifd1, self.d1d2)

        # Inputs from stages that do not exist yet.
        self.decode2_ready = True
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

    @property
    def pc(self) -> int:
        return self.fetch.pc

    def load(self, program: Dict[int, int]) -> None:
        self.program = dict(program)

    def warm_icache(self, addresses: Optional[Iterable[int]] = None) -> None:
        self.icache.warm(self.program if addresses is None else addresses)

    # -- one cycle ---------------------------------------------------------
    def tick(self, time: Optional[Time] = None) -> None:
        cycle = self._consume_tick(time, attr_name="_tick")
        if cycle is None:
            return
        self._step(cycle)

    def _step(self, cycle: int) -> None:
        flush, halt = bool(self.redirect_valid), bool(self.internal_halt)

        # Readiness, back to front.
        d1d2_ready = bool(self.decode2_ready or not self.d1d2.valid)
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

        # D1/D2 reads the IF/D1 latch's *current* contents, so it is settled
        # from registers alone and the order against the IF/D1 update does
        # not matter.
        d = self.d1d2
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
        for m in self._modules:
            m.commit()

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
