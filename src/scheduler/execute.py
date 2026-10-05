"""Execute: the five scalar units, the load/store unit, and scalar writeback.

Transcribes rtl/modules/scheduler/execution_units/ and writeback/s_wb_arbiter.sv
(atalla scheduler_integration_SP26_joshklug) for timing. Values come from
scheduler/semantics.py.

A packet that decode 2 issues sits in the D2/EX latch for one cycle; call
that cycle the op's EX cycle. The crossbar (xbar_4x5_exec_comb.sv) sends each
scalar op to its unit in that cycle; the packet contract guarantees at most
one op per unit.

    unit  ops                                    result      next op
    EX1   ALU, branches, jumps                   EX cycle    every cycle
    EX2   rcp.bf, sqrt.bf 11; div, mod 66;       EX + L      EX + L + 2
          bfts.s, stbf.s, mv.stm 1
    EX3   BF16 add, sub, mul, slt                EX + 1      EX + 3
    EX4   mul.s                                  EX + 2      EX + 4
    EX5   loads and stores                       see LoadStoreUnit

EX2-EX4 share one FSM (start / latch / done): `ready_in` falls in the EX cycle
itself, the result is offered from EX + L until the writeback arbiter takes
it, and the unit is ready again the cycle after. EX1 is combinational: its
result and any redirect are out in the EX cycle, and it only holds an op
when writeback is not ready.

A result offered in cycle v that the arbiter grants is in the EX/WB latch in
v + 1, which writes the register file and clears the busy bit on that edge.

The scalar writeback arbiter grants one write per register-file bank
(rd[1:0]) per cycle, by fixed priority EX5, EX1, EX4, EX3, EX2 (move-to-
scalar, from the vector side, comes last). A unit that loses holds its result.
mv.stm's result goes to the mask register file instead, which is always ready.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from scheduler.control import ScalarOp
from scheduler.dcache import (
    FILL, HIT, MISS, RETRY, STORE_ACK, DCache, DCacheReq)
from scheduler.semantics import (
    control_outcome, is_control, load_value, mem_address, scalar_value, store_word)


#: Cycles from the EX cycle to the result, per EX2-EX4 mnemonic
#: (bfD_sD_bfInt_intBF.sv, bfA_bfM_bfS_bfSLT.sv, sMult.sv).
DEFAULT_LATENCY = {
    "rcp.bf": 11, "sqrt.bf": 11,
    "div.s": 66, "divi.s": 66, "mod.s": 66, "modi.s": 66,
    "bfts.s": 1, "stbf.s": 1, "mv.stm": 1,
    "add.bf": 1, "sub.bf": 1, "mul.bf": 1, "slt.bf": 1,
    "mul.s": 2, "muli.s": 2,
}


@dataclass
class ExOp:
    """One scalar op in the D2/EX latch, with what decode 2 read for it."""
    op: ScalarOp
    rs1: int
    rs2: int
    pc: int
    pc_pred_addr: int = 0
    predict_taken: bool = False


@dataclass
class Result:
    """A result a unit offers to writeback."""
    rd: int
    value: int
    to_mask: bool = False
    source: str = ""


@dataclass
class Redirect:
    target: int
    pc: int                     # the control op's own pc: where the BTB is written


# ---------------------------------------------------------------------------
# EX1
# ---------------------------------------------------------------------------

class Ex1:
    """alu_control.sv: scalar_alu + control, combinational, held on backpressure."""

    def __init__(self):
        self.held: Optional[Tuple[ExOp, Result]] = None   # latch state
        self.pulse_spent = False                         # alu_control's pulse/stall FSM
        self.redirects = 0

    def offer(self, ex: Optional[ExOp]) -> Tuple[Optional[Result], Optional[Redirect]]:
        """This cycle's result and redirect, before the arbiter decides."""
        self._in = ex
        if self.held is not None:
            return self.held[1], None
        if ex is None:
            self._cur = None
            return None, None
        op = ex.op
        redirect = None
        if is_control(op):
            out = control_outcome(op, ex.pc, ex.rs1, ex.rs2)
            result = Result(out.rd, out.value, source="ex1")
            if op.mnemonic in ("jal", "jalr"):
                wrong = not (ex.pc_pred_addr and ex.predict_taken
                             and out.target == ex.pc_pred_addr)
            else:
                wrong = (ex.predict_taken != out.taken
                         or (ex.predict_taken and ex.pc_pred_addr != out.target))
            if wrong and not self.pulse_spent:
                redirect = Redirect(out.target, ex.pc)
        else:
            result = Result(op.rd, scalar_value(op, ex.rs1, ex.rs2), source="ex1")
        self._cur = (ex, result)
        return result, redirect

    def ready_in(self, granted: bool) -> bool:
        if self.held is not None:
            return False
        return not (self._in is not None and not granted)

    def advance(self, granted: bool, redirected: bool) -> None:
        if redirected:
            self.redirects += 1
        if self.held is not None:
            if granted:
                self.held, self.pulse_spent = None, False
            return
        if self._in is not None and not granted:
            self.held = self._cur
            self.pulse_spent = self.pulse_spent or redirected

    @property
    def busy(self) -> bool:
        return self.held is not None


# ---------------------------------------------------------------------------
# EX2, EX3, EX4
# ---------------------------------------------------------------------------

START, LATCH, DONE = "start", "latch", "done"


class MultiCycleUnit:
    """The start / latch / done FSM of div_bf_scalar_convert_wrapper,
    addsub_bf16_wrapper and mult_wrapper."""

    def __init__(self, name: str, latency: Dict[str, int]):
        self.name = name
        self.latency = latency
        self.state = START
        self.count = 0
        self.cur: Optional[Result] = None
        self.L = 1
        self.ops = 0
        self.busy_cycles = 0

    def offer(self) -> Optional[Result]:
        return self.cur if self.state == DONE else None

    def ready_in(self, ex: Optional[ExOp]) -> bool:
        return self.state == START and ex is None

    def advance(self, ex: Optional[ExOp], granted: bool) -> None:
        if self.state != START:
            self.busy_cycles += 1
        if self.state == START:
            if ex is not None:
                op = ex.op
                self.L = int(self.latency[op.mnemonic])
                to_mask = op.mask_reg_write
                rd = op.rd & 0xF if to_mask else op.rd
                self.cur = Result(rd, scalar_value(op, ex.rs1, ex.rs2), to_mask, self.name)
                self.state = DONE if self.L <= 1 else LATCH
                self.count = 0
                self.ops += 1
        elif self.state == LATCH:
            if self.count == self.L - 2:
                self.state = DONE
            self.count += 1
        elif self.state == DONE:
            if granted:
                self.state, self.count, self.cur = START, 0, None

    @property
    def busy(self) -> bool:
        return self.state != START


# ---------------------------------------------------------------------------
# EX5: a non-blocking load/store unit
# ---------------------------------------------------------------------------

@dataclass
class _Access:
    id: int
    op: ScalarOp
    req: DCacheReq


class LoadStoreUnit:
    """EX5, rebuilt to use the lockup-free data cache.

    The RTL's ld_st_unit.sv is blocking: it waits out every miss and replays
    it, so the cache's MSHRs are never used. This one keeps going:

      * Ops enter a queue of `depth` entries; `ready_in` is "the queue has
        room". An op arriving with the queue empty and no lookup in progress
        goes to the cache in its EX cycle, as the RTL's does.
      * The queue head is presented to the cache and held until taken, in
        program order. One lookup is outstanding at a time (the cache's
        lookups are serial anyway). A retry (no MSHR free) goes back to the
        front of the queue, so the cache still sees program order.
      * A load hit writes back from the hit cycle; a load miss waits in the
        MSHR and writes back when its fill answers, matched by id. Any number
        of loads can be waiting; finished loads queue for the one writeback
        port.
      * A store is done when the cache acknowledges it: at the hit's SRAM
        write, or at once on a miss (the MSHR holds the data).

    Addresses are rs1 + imm (the RTL uses imm alone; see semantics.py).
    """

    def __init__(self, cache: DCache, depth: int = 4):
        self.cache = cache
        self.depth = int(depth)
        self.queue: Deque[_Access] = deque()
        self.in_lookup: Optional[_Access] = None
        self.waiting: Dict[int, _Access] = {}         # load misses, by id
        self.wb: Deque[Result] = deque()
        self._next_id = 0
        #: Ids in the order the cache accepted them (retries excluded). Ids
        #: are handed out in program order, so this must be increasing.
        self.accepted: List[int] = []
        self.stats = dict(loads=0, stores=0, hits=0, misses=0, retries=0,
                          max_waiting=0, full_cycles=0)

    def _make(self, ex: ExOp) -> _Access:
        op = ex.op
        store = op.mnemonic in ("sw.s", "shw.s")
        self._next_id += 1
        req = DCacheReq(self._next_id, mem_address(op, ex.rs1), store,
                        store_word(op, ex.rs2) if store else 0)
        return _Access(self._next_id, op, req)

    def ready_in(self) -> bool:
        """Room for one more op next cycle, counting what this cycle does to
        the queue: the head taken by the cache, the op arriving now, and a
        lookup coming back as a retry. Call after offer()."""
        occ = len(self.queue)
        if self._taken and self._presented is not self._arriving:
            occ -= 1
        if self._arriving is not None and not (self._taken and self._presented is self._arriving):
            occ += 1
        if self._outcome is not None and self._outcome[0] == RETRY:
            occ += 1
        return occ < self.depth

    # -- phase 1: what goes to the cache this cycle ------------------------------
    def present(self, ex: Optional[ExOp]) -> None:
        """Drive the cache's request port. `ex` is this cycle's EX5 op."""
        self._arriving = self._make(ex) if ex is not None else None
        head = None
        if self.in_lookup is None:
            head = self.queue[0] if self.queue else self._arriving
        self._presented = head
        self.cache.in_req_valid = head is not None
        self.cache.in_req = head.req if head is not None else None

    # -- phase 2: after the cache has evaluated ------------------------------------
    def offer(self) -> Optional[Result]:
        """Take in this cycle's cache answers; return the writeback on offer."""
        self._taken = self._presented is not None and bool(self.cache.out_req_ready)
        lookup = self._presented if self._taken else self.in_lookup
        self._outcome: Optional[Tuple[str, _Access]] = None
        self._new_wb: List[Result] = []
        self._filled: List[int] = []
        for r in self.cache.out_resp:
            if r.kind == FILL:
                a = self.waiting.get(r.id)
                if a is not None:
                    self._new_wb.append(Result(a.op.rd, load_value(a.op, r.data), source="ex5"))
                    self._filled.append(r.id)
                continue
            if lookup is None or r.id != lookup.id:
                continue
            self._outcome = (r.kind, lookup)
            if r.kind == HIT and not r.store:
                self._new_wb.append(Result(lookup.op.rd, load_value(lookup.op, r.data),
                                           source="ex5"))
        pending = list(self.wb) + self._new_wb
        return pending[0] if pending else None

    # -- phase 3: the edge ----------------------------------------------------------
    def advance(self, granted: bool) -> None:
        if self._outcome is not None:
            kind, a = self._outcome
            self.in_lookup = None
            if kind != RETRY:
                self.accepted.append(a.id)
            if kind == RETRY:
                self.stats["retries"] += 1
                self.queue.appendleft(a)
            elif kind == MISS:
                self.stats["misses"] += 1
                self.waiting[a.id] = a
                self.stats["max_waiting"] = max(self.stats["max_waiting"], len(self.waiting))
            elif kind == STORE_ACK:
                self.stats["misses"] += 1
                self.stats["stores"] += 1
            elif kind == HIT:
                self.stats["hits"] += 1
                self.stats["stores" if a.req.store else "loads"] += 1
        for i in self._filled:
            self.waiting.pop(i, None)
            self.stats["loads"] += 1
        if self._taken:
            if self._presented is not self._arriving:
                self.queue.popleft()
            self.in_lookup = self._presented
        if self._arriving is not None and not (self._taken and self._presented is self._arriving):
            self.queue.append(self._arriving)
        if len(self.queue) >= self.depth:
            self.stats["full_cycles"] += 1
        self.wb.extend(self._new_wb)
        if granted:
            self.wb.popleft()

    @property
    def idle(self) -> bool:
        return not (self.queue or self.in_lookup or self.waiting or self.wb)


# ---------------------------------------------------------------------------
# Scalar writeback
# ---------------------------------------------------------------------------

#: s_wb_arbiter.sv's fixed priority.
WB_PRIORITY = ("ex5", "ex1", "ex4", "ex3", "ex2", "mts")


def arbitrate(offers: Dict[str, Optional[Result]], banks: int = 4) -> Tuple[Dict[str, bool], List[Result], List[Result]]:
    """Grant one scalar write per bank, by priority. Returns (granted per
    source, scalar writes, mask writes). Mask writes bypass the banks."""
    used = set()
    granted: Dict[str, bool] = {}
    writes: List[Result] = []
    masks: List[Result] = []
    for src in WB_PRIORITY:
        r = offers.get(src)
        if r is None:
            continue
        if r.to_mask:
            granted[src] = True
            masks.append(r)
            continue
        bank = r.rd % banks
        if bank in used:
            granted[src] = False
            continue
        used.add(bank)
        granted[src] = True
        writes.append(r)
    return granted, writes, masks
