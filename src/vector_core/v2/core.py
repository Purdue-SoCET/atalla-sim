"""VectorCoreV2: the M x N vector core (docs/vector_core_v2_change_plan.md).

    rounds --> slots V1..VS --> bank reads (each slot owns a bank pair)
           --> per-destination arbiters (one slot per destination per cycle)
           --> FU groups (+ RTree) | VLSUs | GSAU | transpose | move-to-scalar
           --> WB arbiter --> per-bank WB FIFOs --> banks

The core is driven once per cycle by step(cycle), which decides everything
from the state at the start of the cycle and then applies it, so no unit's
view depends on the order it is looked at. The VLSUs, GSAU and transpose
unit are the v1 models, ticked by tick_units() after step().

Timing (Q10: t_read 1, t_xbar 0, t_arb 0, t_wbbuf 1, t_write_visible 1):

    c        the slot reads its operands and its destination accepts the token
    c+L+1    the FU group's output is offered to writeback and granted
    c+L+2    the bank FIFO writes it, at the end of the cycle
    c+L+3    a dependent slot may read it              D = L + 3

Hazards in single mode: a busy count per vector register (RAW and WAW), and
rounds that fire in order (WAR across rounds); inside a round an op may not
fire a write to a register another op of the round has yet to read. Mask
registers are written through `on_mask_write` (the scheduler's mask file and
busy bits, see scheduler/vector_v2.py) and read through `read_mask`.
"""

import struct
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from base.clocked_object import Clocked
from base.dtype import DType, cast_scalar
from vector_core.transpose import TransposeUnit
from vector_core.vector_core import GSAU
from vector_core.vector_load_store import VLSU
from vector_core.v2.config import VectorCoreConfig
from vector_core.v2.fu_group import (
    ALUGroup, FUGroup, Token, compare_mask, elementwise, reduction_vector)
from vector_core.v2.slots import SlotOp, _OpState
from vector_core.v2.vrf_banks import BankedVRF, BankedVRFStorage, SlotContractError
from vector_core.v2.writeback import WBArbiter, WBBufferV2, WBEntry

MASK32 = 0xFFFFFFFF
STALL_REASONS = ("raw", "waw", "war", "bank", "mask", "compat", "fu_busy", "xbar", "round")


def bits_bf16(b: int) -> float:
    return struct.unpack(">f", struct.pack(">I", (int(b) & 0xFFFF) << 16))[0]


def bf16_bits(x: float) -> int:
    return struct.unpack(">I", struct.pack(">f", cast_scalar(float(x), DType.BF16)))[0] >> 16


def fp32_bits(x: float) -> int:
    return struct.unpack(">I", struct.pack(">f", float(x)))[0]


def mask_bools(m: int, n: int) -> List[bool]:
    return [bool((int(m) >> i) & 1) for i in range(n)]


class VectorCoreV2(Clocked):

    arch = "fu_groups"

    def __init__(self, cfg: Optional[VectorCoreConfig] = None, vls_count: int = 4):
        super().__init__()
        cfg = cfg or VectorCoreConfig(arch="fu_groups")
        self.cfg = cfg
        self._tick = -1
        self.vector_len = int(cfg.vector_len)
        self.vrf = BankedVRF(cfg.num_banks, cfg.regs_per_bank, self.vector_len)

        self.groups: Dict[str, FUGroup] = {}
        self.alu: Optional[ALUGroup] = None
        for g in cfg.fu_groups:
            if g.rtree:
                self.alu = ALUGroup(g.latency, g.ii, cfg.rtree_levels)
                self.alu.name = g.name
                self.groups[g.name] = self.alu
            else:
                self.groups[g.name] = FUGroup(g.name, g.latency, g.ii)

        self.wbbuf = WBBufferV2(cfg.num_banks, cfg.wb_fifo_depth, cfg.bank_write_ports,
                                cfg.t_wbbuf)
        self.arbiter = WBArbiter(cfg.wb_priority, cfg.wb_grants_per_cycle, cfg.mask_banks)

        # The v1 units, unchanged (section 8): stores always carry inline data.
        self.vls_units = [VLSU(scratchpad_count=1, read_vreg_cb=self.read_vreg)
                          for _ in range(int(vls_count))]
        self.gsau = GSAU(max_vregs=self.vrf.num_regs)
        self.transpose = TransposeUnit(vec_len=self.vector_len)
        for unit in (*self.vls_units, self.gsau, self.transpose):
            self.add_child(unit)

        # Rounds waiting for the slots; the head one is firing.
        self.rounds: Deque[List[_OpState]] = deque()
        self._round_start = 0                # first cycle the head round may fire
        self._seq = 0
        # Single-mode scoreboard (section 5): pending writes per vector
        # register, and the cycle each one's latest value becomes readable.
        self.busy: Dict[int, int] = {}
        self.readable_at: Dict[int, int] = {}
        # Transpose requests, as scheduler/vector.py keeps them.
        self.tp_reqs: Deque[Tuple[str, Any]] = deque()
        self.tp_vds: Deque[int] = deque()
        self.tp_cols_left = 0
        # Move-to-scalar's one-entry hold register: (rd, value).
        self.mts_hold: Optional[Tuple[int, int]] = None
        # Mask file access, given by the owner; standalone, a local file.
        self._masks = [0] * 16
        self.read_mask: Callable[[int], int] = self._read_local_mask
        self.on_mask_write: Callable[[int, int, int], None] = self._write_local_mask
        self.blocked_mask_banks = frozenset()
        self._pending_local_masks: List[Tuple[int, int, int]] = []

        self.cycle = -1
        self._last_progress = 0
        #: Set to a list to record every fire as (cycle, slot, name).
        self.fire_log: Optional[List[Tuple[int, int, str]]] = None
        S = cfg.num_slots
        self.stats = dict(
            cycles=0, rounds=0, fires=[0] * S,
            stall={k: [0] * S for k in STALL_REASONS},
            compat_cross_pair_reads=[0] * S, compat_cross_pair_cycles=[0] * S,
            strict_violations=0, bank_read_conflicts=[0] * cfg.num_banks,
            mask_read_conflicts=[0] * cfg.mask_banks, xbar_conflicts={},
            fu_ops=0, wb_writes=0)

    # -- untimed register access ---------------------------------------------------
    def read_vreg(self, reg: int) -> List[float]:
        return self.vrf.read(reg)

    def read_vreg_with_dtype(self, reg: int):
        return self.vrf.read_with_dtype(reg)

    def write_vreg(self, reg: int, data: Sequence[float], dtype=None) -> None:
        self.vrf.write(reg, data)

    def vrf_storage(self) -> BankedVRFStorage:
        return BankedVRFStorage(self.vrf)

    def _read_local_mask(self, m: int) -> int:
        return MASK32 if int(m) == 0 else self._masks[int(m)]

    def _write_local_mask(self, cycle: int, m: int, value: int) -> None:
        # Standalone: lands at the end of the next cycle, as the scheduler's
        # writeback_at(cycle + 1) does.
        self._pending_local_masks.append((cycle + 1, int(m), int(value) & MASK32))

    # -- the bridges' interface (unchanged from VectorCore) ------------------------
    def pop_scratchpad_request(self, vls_id: int) -> Optional[Dict]:
        return self.vls_units[vls_id].pop_request()

    def push_scratchpad_response(self, vls_id: int, rsp: Dict) -> bool:
        local = dict(rsp)
        local["scratchpad"] = 0
        return self.vls_units[vls_id].push_response(local)

    def pop_systolic_request(self) -> Optional[Dict]:
        return self.gsau.pop_systolic_request()

    def push_systolic_response(self, rsp: Dict) -> bool:
        return self.gsau.push_systolic_response(rsp)

    # -- issuing rounds ------------------------------------------------------------
    def can_accept_round(self) -> bool:
        return len(self.rounds) < self.cfg.round_queue_depth

    def issue_round(self, ops: Sequence[SlotOp]) -> None:
        """Queue one round: at most one op per slot."""
        ops = [o for o in ops if o is not None]
        if not ops:
            return
        if len({o.slot for o in ops}) != len(ops):
            raise SlotContractError("two ops for one slot in a round: %s"
                                    % [(o.slot, o.name) for o in ops])
        if not self.can_accept_round():
            raise AssertionError("round issued while the round queue is full")
        states = []
        for o in sorted(ops, key=lambda o: o.slot):
            if not 0 <= o.slot < self.cfg.num_slots:
                raise SlotContractError("slot %d out of range" % o.slot)
            self._seq += 1
            o.seq = self._seq
            pair = self.cfg.slot_banks(o.slot)
            reads, cross = [], False
            for name, reg in o.vector_reads():
                bank = self.vrf.bank_of(reg)
                if bank not in pair:
                    cross = True
                    if self.cfg.read_mode == "strict":
                        if self.cfg.strict:
                            raise SlotContractError(
                                "%s in slot V%d reads v%d (bank %d), outside its pair %s"
                                % (o.name or o.kind, o.slot + 1, reg, bank, pair))
                        self.stats["strict_violations"] += 1
                reads.append((name, reg, bank))
            needs_mask = (o.mask_reg is not None
                          and (o.mask_reg != 0 or self.cfg.m0_uses_port))
            states.append(_OpState(op=o, reads_left=reads, mask_left=needs_mask,
                                   cross_pair=cross))
        self.rounds.append(states)

    # -- destination readiness -----------------------------------------------------
    def _group_of(self, dest: str) -> Optional[FUGroup]:
        return self.groups.get(dest[3:]) if dest.startswith("fu:") else None

    def _dest_ready(self, st: _OpState, granted: set) -> bool:
        o, d = st.op, st.op.dest
        g = self._group_of(d)
        if g is not None:
            if g is self.alu:
                return self.alu.ready_for("reduce" if o.kind == "reduce" else "plain",
                                          "fu:" + g.name in granted, "fu:rtree" in granted)
            return g.ready("fu:" + g.name in granted)
        if d.startswith("vlsu"):
            return self.vls_units[int(d[4:])].can_accept_issue()
        if d == "gsau":
            if self.gsau.to_systolic.is_full():
                return False
            return o.kind == "weight" or not self.gsau.rd_queue.is_full()
        if d == "transpose":
            return len(self.tp_reqs) < self.cfg.transpose_queue
        if d == "mts":
            return self.mts_hold is None
        raise ValueError("unknown destination %r" % d)

    def _vreg_ready(self, r: int, c: int) -> bool:
        return self.busy.get(r, 0) == 0 and self.readable_at.get(r, -1) <= c

    # -- one cycle -----------------------------------------------------------------
    def step(self, cycle: int) -> None:
        c = int(cycle)
        self.cycle = c
        cfg = self.cfg
        self.stats["cycles"] += 1
        progress = False

        # Local mask writes (standalone use) landing this cycle.
        if self._pending_local_masks:
            keep = []
            for when, m, v in self._pending_local_masks:
                if when <= c:
                    if m:
                        self._masks[m] = v
                else:
                    keep.append((when, m, v))
            self._pending_local_masks = keep

        # 1. What the bank FIFOs write at the end of this cycle.
        occ = [len(f) for f in self.wbbuf.fifos]
        due = self.wbbuf.pop_due(c)

        # 2. Writeback arbitration over the source heads.
        offers = self._wb_offers()
        granted = self.arbiter.arbitrate(offers, self.vrf.bank_of, lambda b: occ[b],
                                         cfg.wb_fifo_depth, set(self.blocked_mask_banks))
        granted_src = {e.source for e in granted}

        # 3. The slots: the head round, if its start cycle has come.
        sends: List[_OpState] = []
        collect: List[Tuple[_OpState, List[Tuple[str, int, int]], bool]] = []
        round_states = self.rounds[0] if self.rounds and c >= self._round_start else None
        if round_states is not None:
            sends, collect = self._slots(round_states, c, granted_src)
        for later in list(self.rounds)[1 if round_states is not None else 0:]:
            for st in later:
                self.stats["stall"]["round"][st.op.slot] += 1

        # -- apply --------------------------------------------------------------
        for e in due:
            self.vrf.write(e.reg, e.value, e.enable)
            self.vrf.writes[self.vrf.bank_of(e.reg)] += 1
            self.busy[e.reg] = self.busy.get(e.reg, 1) - 1
            if self.busy[e.reg] <= 0:
                del self.busy[e.reg]
            self.readable_at[e.reg] = c + cfg.t_write_visible
            self.stats["wb_writes"] += 1
            progress = True
        for e in granted:
            progress = True
            if e.is_mask:
                self.on_mask_write(c, e.reg, e.value)
            else:
                self.wbbuf.push(e, self.vrf.bank_of(e.reg), c)
            if e.source.startswith("vlsu"):
                self.vls_units[int(e.source[4:])].pop_writeback()
            elif e.source == "gsau":
                self.gsau.pop_writeback()
            elif e.source == "transpose":
                self.transpose.pop_writeback()
                self.tp_vds.popleft()

        for st, reads, mask_got in collect:
            progress = True
            for name, reg, bank in reads:
                st.values[name] = self.vrf.read(reg)
                st.reads_left.remove((name, reg, bank))
                self.vrf.reads[bank] += 1
            if mask_got:
                st.mask_left = False
            if not st.reads_left and not st.mask_left and st.collected_at is None:
                st.collected_at = c
                penalty = (cfg.compat_read_penalty
                           if st.cross_pair and cfg.read_mode == "compat" else 0)
                st.send_ready = c + penalty
                if st.cross_pair:
                    self.stats["compat_cross_pair_reads"][st.op.slot] += 1
                    self.stats["compat_cross_pair_cycles"][st.op.slot] += penalty

        tokens: Dict[str, Token] = {}
        for st in sends:
            progress = True
            self._fire(st, c, tokens)

        for name, g in self.groups.items():
            if g is self.alu:
                self.alu.commit("fu:" + name in granted_src, tokens.get(name),
                                rtree_granted="fu:rtree" in granted_src)
            else:
                g.commit("fu:" + name in granted_src, tokens.get(name))

        if round_states is not None and all(st.sent for st in round_states):
            self.rounds.popleft()
            self.stats["rounds"] += 1
            self._round_start = c + 1
        self.wbbuf.sample()
        self._feed_transpose()

        if progress or not self.rounds:
            self._last_progress = c
        elif c - self._last_progress > cfg.watchdog_cycles:
            raise RuntimeError("vector core v2 made no progress for %d cycles:\n%s"
                               % (cfg.watchdog_cycles, self.describe()))

    def _wb_offers(self) -> List[WBEntry]:
        out = []
        for name, g in self.groups.items():
            if g.out is not None:
                out.append(self._token_entry("fu:" + name, g.out))
        if self.alu is not None and self.alu.rtree.out is not None:
            out.append(self._token_entry("fu:rtree", self.alu.rtree.out))
        n = self.vector_len
        for s, vls in enumerate(self.vls_units):
            wb = vls.wb_q.peek()
            if wb is not None:
                data = [bits_bf16(b) for b in wb["data"]][:n]
                data += [0.0] * (n - len(data))
                out.append(WBEntry("vlsu%d" % s, int(wb["vd"]), data))
        wb = self.gsau.writebacks.peek()
        if wb is not None:
            out.append(WBEntry("gsau", int(wb["dst"]), [float(x) for x in wb["data"]][:n]))
        if self.tp_vds and self.transpose.can_pop_writeback():
            col = self.transpose.peek_writeback()
            out.append(WBEntry("transpose", self.tp_vds[0],
                               [float(x) for x in col["data"]][:n]))
        return out

    @staticmethod
    def _token_entry(source: str, t: Token) -> WBEntry:
        if t.kind == "mask":
            return WBEntry(source, t.dst, t.value, is_mask=True)
        return WBEntry(source, t.dst, t.value, t.enable)

    def _slots(self, states: List[_OpState], c: int, granted_src: set):
        """This cycle's decisions for the head round: which reads each op is
        granted, and which ops fire (send to their destination)."""
        cfg = self.cfg
        # Ops whose sources are ready may read; RAW is checked before reading.
        reading = []
        for st in states:
            if st.sent:
                continue
            st.reason = ""
            if st.reads_left or st.mask_left:
                if not all(self._vreg_ready(reg, c) for _, reg, _ in st.reads_left):
                    st.reason = "raw"
                    continue
                reading.append(st)

        # Bank read ports: each bank serves bank_read_ports reads a cycle, the
        # slot that owns the bank first, then by slot (all ops of a round are
        # one iteration, so "oldest" is a tie).
        port_used: Dict[int, int] = {}
        mask_used: Dict[int, int] = {}
        grants: Dict[int, List[Tuple[str, int, int]]] = {}
        owner_first = sorted(
            ((st, r) for st in reading for r in st.reads_left),
            key=lambda x: (x[1][2] // 2 != x[0].op.slot, x[0].op.slot))
        wanted: Dict[int, int] = {}
        for st, r in owner_first:
            bank = r[2]
            wanted[bank] = wanted.get(bank, 0) + 1
            if any(g[2] == bank for g in grants.get(id(st), [])):
                continue                     # one read per bank per op per cycle
            if port_used.get(bank, 0) < cfg.bank_read_ports:
                port_used[bank] = port_used.get(bank, 0) + 1
                grants.setdefault(id(st), []).append(r)
        for bank, n in wanted.items():
            if n > cfg.bank_read_ports:
                self.stats["bank_read_conflicts"][bank] += 1
        mask_grant = set()
        mask_wanted: Dict[int, int] = {}
        for st in reading:
            if st.mask_left:
                mb = int(st.op.mask_reg) % cfg.mask_banks
                mask_wanted[mb] = mask_wanted.get(mb, 0) + 1
                if mask_used.get(mb, 0) < 1:
                    mask_used[mb] = 1
                    mask_grant.add(id(st))
        for mb, n in mask_wanted.items():
            if n > 1:
                self.stats["mask_read_conflicts"][mb] += 1

        collect = []
        for st in reading:
            got = grants.get(id(st), [])
            mask_got = id(st) in mask_grant
            if got or mask_got:
                collect.append((st, got, mask_got))
            done_reads = len(got) == len(st.reads_left)
            done_mask = (not st.mask_left) or mask_got
            if not (done_reads and done_mask):
                st.reason = "bank" if not done_reads else "mask"

        def collected_by_now(st):
            if st.collected_at is not None:
                return st.send_ready <= c
            got = grants.get(id(st), [])
            done = (len(got) == len(st.reads_left)
                    and ((not st.mask_left) or id(st) in mask_grant))
            if not done:
                return False
            return not (st.cross_pair and cfg.read_mode == "compat"
                        and cfg.compat_read_penalty > 0)

        # Registers the round still has to read (WAR inside the round).
        unread = {}
        for st in states:
            if st.sent:
                continue
            got = {r for r in grants.get(id(st), [])}
            for r in st.reads_left:
                if r not in got:
                    unread.setdefault(r[1], set()).add(id(st))
            if st.mask_left and id(st) not in mask_grant:
                unread.setdefault(("m", int(st.op.mask_reg)), set()).add(id(st))

        sends, dest_taken = [], {}
        for st in states:
            if st.sent or st.reason:
                if not st.sent and st.reason:
                    self.stats["stall"][st.reason][st.op.slot] += 1
                continue
            if not collected_by_now(st):
                st.reason = "compat"
                self.stats["stall"]["compat"][st.op.slot] += 1
                continue
            vd = st.op.vector_dst()
            if vd is not None and self.busy.get(vd, 0):
                st.reason = "waw"
            elif vd is not None and unread.get(vd, set()) - {id(st)}:
                st.reason = "war"
            elif (st.op.kind == "compare"
                  and unread.get(("m", int(st.op.vmd)), set()) - {id(st)}):
                st.reason = "war"
            elif st.op.dest in dest_taken:
                st.reason = "xbar"
                self.stats["xbar_conflicts"][st.op.dest] = \
                    self.stats["xbar_conflicts"].get(st.op.dest, 0) + 1
            elif not self._dest_ready(st, granted_src):
                st.reason = "fu_busy"
            if st.reason:
                self.stats["stall"][st.reason][st.op.slot] += 1
                continue
            dest_taken[st.op.dest] = st.op.slot
            sends.append(st)
        return sends, collect

    def _fire(self, st: _OpState, c: int, tokens: Dict[str, Token]) -> None:
        """The op's operands go through its crossbar to its destination."""
        o, v, n = st.op, st.values, self.vector_len
        st.sent = True
        self.stats["fires"][o.slot] += 1
        if self.fire_log is not None:
            self.fire_log.append((c, o.slot, o.name or o.kind))
        src0 = v.get("src0")
        if o.src1 is not None:
            src1 = v["src1"] if o.src1 != o.src0 else src0
        elif o.scalar is not None:
            # A .vs scalar is an fp32 bit pattern; the datapath works on BF16
            # operands, as v1's lanes cast theirs (VectorDatapath.enqueue).
            src1 = [cast_scalar(float(o.scalar), DType.BF16)] * n
        else:
            src1 = [0.0] * n
        mval = (self.read_mask(o.mask_reg) if o.mask_reg is not None else MASK32) & MASK32
        mbits = mask_bools(mval, n)
        k, d = o.kind, o.dest

        if k in ("elem", "compare", "reduce"):
            self.stats["fu_ops"] += 1
            if k == "elem":
                tok = Token("vreg", o.vd, elementwise(o.op, src0, src1, mbits), mbits,
                            o.seq, o.slot)
            elif k == "compare":
                tok = Token("mask", o.vmd,
                            compare_mask(o.op, src0, src1, mval, self.read_mask(o.vmd)),
                            None, o.seq, o.slot)
            else:
                out_mode, index = _reduce_placement(o.imm)
                tok = Token("reduce", o.vd, reduction_vector(src0, mbits, o.op, out_mode, index),
                            None, o.seq, o.slot)
            if tok.kind != "mask":
                self._claim(o.vd)
            tokens[d[3:]] = tok
        elif k == "load":
            self._claim(o.vd)
            self.vls_units[int(d[4:])].enqueue_issue(
                {"kind": "load", "scratchpad": 0, "vd": o.vd, "addr": o.vlsu["addr"],
                 "vl": o.vlsu.get("vl"), "dtype": "bf16"})
        elif k == "store":
            meta = {"vst_rd": o.vlsu["vst_rd"]} if o.vlsu.get("vst_rd") is not None else {}
            self.vls_units[int(d[4:])].enqueue_issue(
                {"kind": "store", "scratchpad": 0, "data": [bf16_bits(x) for x in src0],
                 "addr": o.vlsu["addr"], "vl": o.vlsu.get("vl"), "meta": meta})
        elif k in ("gemm", "weight"):
            weight = k == "weight"
            cmd = {"vdata": list(src0), "is_weight": weight, "expect_output": not weight,
                   "dtype": "bf16", "meta": {"dtype": "bf16", "kind": o.name}}
            if not weight:
                cmd["dst"] = o.vd
                self._claim(o.vd)
            assert self.gsau.issue(cmd), "GSAU refused a granted op"
        elif k == "tpush":
            self.tp_reqs.append(("push", list(src0)))
        elif k == "tpop":
            self._claim(o.vd)
            self.tp_reqs.append(("pop", o.vd))
        elif k == "vts":
            self.mts_hold = (o.rd, fp32_bits(src0[o.imm & 0x1F]))
        elif k == "mts":
            self.mts_hold = (o.rd, mval)
        else:
            raise ValueError("unknown slot op kind %r" % k)

    def _claim(self, vd: int) -> None:
        self.busy[vd] = self.busy.get(vd, 0) + 1

    def _feed_transpose(self) -> None:
        """Hand queued transpose requests to the unit, in order (as
        scheduler/vector.py's VectorSide does for v1)."""
        tp = self.transpose
        while self.tp_reqs:
            kind, arg = self.tp_reqs[0]
            if kind == "push":
                if self.tp_cols_left or not tp.ready_in:
                    return
                assert tp.push(arg)
            else:
                if self.tp_cols_left == 0:
                    if not tp.ready_in:
                        return
                    assert tp.pop()
                    self.tp_cols_left = self.vector_len
                self.tp_cols_left -= 1
                self.tp_vds.append(arg)
            self.tp_reqs.popleft()
            if kind == "push":
                return

    def take_mts(self) -> Optional[Tuple[int, int]]:
        return self.mts_hold

    def release_mts(self) -> None:
        self.mts_hold = None

    def tick_units(self, time: Optional[float] = None) -> None:
        """Advance the VLSUs, the GSAU and the transpose unit (call after step)."""
        cycle = self._consume_tick(time, attr_name="_tick")
        if cycle is None:
            return
        self.tick_children(cycle)

    def tick(self, time: Optional[float] = None) -> None:
        """Standalone use: step, then the units, once per cycle."""
        c = int(self._tick + 1 if time is None else time)
        self.step(c)
        self.tick_units(float(c))

    # -- status --------------------------------------------------------------------
    @property
    def fu_busy(self) -> bool:
        return any(not g.idle for g in self.groups.values())

    @property
    def idle(self) -> bool:
        return (not self.rounds and not self.fu_busy and self.wbbuf.empty
                and not self.busy and self.mts_hold is None
                and not self.gsau.has_pending() and self.gsau.rd_queue.is_empty()
                and self.gsau.writebacks.is_empty()
                and all(v.issue_q.is_empty() and v.req_q.is_empty() and v.rsp_q.is_empty()
                        and v.wb_q.is_empty() and v.outstanding_loads() == 0
                        for v in self.vls_units)
                and not self.tp_reqs and not self.tp_vds
                and not (self.transpose.busy and self.tp_cols_left == 0)
                and not self._pending_local_masks)

    def describe(self) -> str:
        lines = ["cycle %d, rounds queued %d, busy %s" % (self.cycle, len(self.rounds),
                                                         dict(self.busy))]
        for i, r in enumerate(self.rounds):
            for st in r:
                lines.append("  round %d V%d %s %s: sent=%s reads_left=%s mask_left=%s reason=%s"
                             % (i, st.op.slot + 1, st.op.name, st.op.dest, st.sent,
                                st.reads_left, st.mask_left, st.reason))
        for name, g in self.groups.items():
            lines.append("  %s: stages=%d out=%s" % (name, len(g.stages), g.out is not None))
        lines.append("  wb fifos: %s" % [len(f) for f in self.wbbuf.fifos])
        return "\n".join(lines)


def _reduce_placement(imm: int) -> Tuple[str, int]:
    """imm[6] broadcast, else imm[5] the result alone at imm[4:0], else the
    result at imm[4:0] over vs1 (scheduler/vector.py reduce_placement)."""
    if (imm >> 6) & 1:
        return "broadcast", 0
    if (imm >> 5) & 1:
        return "partial_zero", imm & 0x1F
    return "partial_passthru", imm & 0x1F
