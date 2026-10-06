"""Decode 2: read the operands, check the hazards, issue the whole packet.

Transcribes rtl/modules/scheduler/decode2/ (atalla
scheduler_integration_SP26_joshklug): decode_2.sv and the modules it wires
together -- the three control units (scheduler/control.py), the source
register allocator, the dependency checker and the scalar register file.

A packet issues only as a whole, and only in a cycle where all three hold:

    dependencies_ready   no source and no destination of any instruction is
                         busy in the scoreboard
    FU ready             every execute unit the packet needs is ready: scalar
                         EX1..EX5, the vector ALU/MUL/reduction lanes, the
                         VLSU of each scratchpad it names, the GSAU, the
                         move-to-scalar unit, and the scratchpad an SDMA's rs3
                         names (rs3[31:30], a register value, not a field)
    register files ready the scalar, vector and mask register files have
                         each served every read: a file whose busiest bank
                         has k >= 2 reads takes k cycles while reggie
                         serializes them, one per bank per cycle
                         (READY -> CONFLICT ... -> DONE)

Decode 2 owns all three register files and reads every operand at issue --
scalar values, vector registers and masks alike -- so the units downstream
get data, not register numbers. That makes write-after-read safe for every
register kind: nothing reads a register after its packet issues. The vector
file's storage can be the vector core's Veggie banks (`vector_storage`), so
the registers are never kept twice.

The scoreboard is a busy bit per register: set when the instruction that
writes it issues, cleared when its writeback reaches the register file. A
bit cleared and set on the same edge ends up set -- a new instruction has
just claimed the register. Branches write rs1 (rs1 += incr7); an SDMA holds
its rs1 until the scratchpad reports it done.

Packets the RTL cannot execute are a compiler contract, not a stall: the RTL
silently drops the extra instruction or zeroes the extra operand. Here they
raise PacketContractError (or are counted, with strict=False):

    one instruction per scalar EX unit   EX1 is ALU *and* control
    4 scalar, 4 vector, 2 mask reads     the source register allocator's ports
    1 GSAU, 1 move-to-scalar, 2 lane ops scheduler_core's routing
    1 VLSU op and 1 SDMA per scratchpad

Where the RTL is wrong this does what was meant (see also control.py):

  * decode_2 has no flush input, and the dependency checker sets busy bits
    whenever decode 2 is ready -- including for a wrong-path packet that the
    D2/EX latch then squashes, whose bits nothing ever clears. Here only a
    packet that actually issues reserves its registers.
  * The dependency checker tracks mask writes only for vector slots 0 and 1
    (its loops run to MASK_WRITE_PORTS = 2), and never WAW-checks mv.stm.
    Here every mask write is tracked and checked.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from base.rtl_module import RTLModule
from scheduler.control import (
    SdmaOp, ScalarOp, VectorOp, V_ALU_ADD, V_EXP, V_GSAU, V_MUL, V_MVMT,
    V_REDU, V_VLSU, VECTOR_ALU_FUS, decode_scalar, decode_sdma, decode_vector)
from scheduler.isa import INST_W, PACKET_SIZE

NUM_SCALAR_REGS = 256
NUM_VECTOR_REGS = 256
NUM_MASK_REGS = 16
SCALAR_READ_PORTS = 4
VECTOR_READ_PORTS = 4
MASK_READ_PORTS = 2
SCALAR_BANKS = 4
VECTOR_BANKS = 4
MASK_BANKS = 2
LANE_ISSUE_W = 2
MASK32 = 0xFFFFFFFF

READY, CONFLICT, DONE = "READY", "CONFLICT", "DONE"


class PacketContractError(ValueError):
    """A packet the RTL would silently mangle."""


# -- the packet --------------------------------------------------------------------
@dataclass
class DecodedPacket:
    scalar: List[ScalarOp]
    vector: List[VectorOp]
    sdma: List[SdmaOp]

    @classmethod
    def from_words(cls, scalar: Sequence[int], vector: Sequence[int],
                   sdma: Sequence[int]) -> "DecodedPacket":
        return cls([decode_scalar(w, i) for i, w in enumerate(scalar)],
                   [decode_vector(w, i) for i, w in enumerate(vector)],
                   [decode_sdma(w, i) for i, w in enumerate(sdma)])

    def valid_ops(self):
        return ([o for o in self.scalar if o.valid], [o for o in self.vector if o.valid],
                [o for o in self.sdma if o.valid])

    def scalar_reads(self) -> List[int]:
        """In source_reg_allocator.sv's port order."""
        s, v, d = self.valid_ops()
        return ([r for o in s for r in o.reads()] + [r for o in v for r in o.scalar_reads()]
                + [r for o in d for r in o.reads()])

    def vector_reads(self) -> List[int]:
        return [r for o in self.vector if o.valid for r in o.vector_reads()]

    def mask_reads(self) -> List[int]:
        return [r for o in self.vector if o.valid for r in o.mask_reads()]

    def scalar_writes(self) -> List[int]:
        s, v, d = self.valid_ops()
        return [w for w in ([o.scalar_write() for o in s] + [o.scalar_write() for o in v]
                            + [o.scalar_write() for o in d]) if w is not None]

    def vector_writes(self) -> List[int]:
        return [o.vector_write() for o in self.vector if o.vector_write() is not None]

    def mask_writes(self) -> List[int]:
        s, v, _ = self.valid_ops()
        return [w for w in ([o.mask_write() for o in s] + [o.mask_write() for o in v])
                if w is not None]

    @property
    def halt(self) -> bool:
        return any(o.valid and o.halt for o in self.scalar)

    @property
    def empty(self) -> bool:
        return not any(self.valid_ops())


def contract_violations(pkt: DecodedPacket, sdma_sids: Sequence[int] = ()) -> List[str]:
    """What the RTL would drop or zero in this packet. Empty if nothing."""
    out = []
    s, v, d = pkt.valid_ops()
    by_ex: Dict[int, List[str]] = {}
    for o in s:
        if o.ex is not None:
            by_ex.setdefault(o.ex, []).append(o.mnemonic)
    out += ["EX%d takes one instruction a packet: %s" % (ex, ", ".join(ms))
            for ex, ms in sorted(by_ex.items()) if len(ms) > 1]
    for what, n, limit in (("scalar", len(pkt.scalar_reads()), SCALAR_READ_PORTS),
                           ("vector", len(pkt.vector_reads()), VECTOR_READ_PORTS),
                           ("mask", len(pkt.mask_reads()), MASK_READ_PORTS)):
        if n > limit:
            out.append("%d %s register reads, %d ports" % (n, what, limit))
    for fu, name in ((V_GSAU, "GSAU"), (V_MVMT, "move-to-scalar")):
        if sum(o.fu == fu for o in v) > 1:
            out.append("more than one %s op" % name)
    lanes = sum(o.is_lane_op for o in v)
    if lanes > LANE_ISSUE_W:
        out.append("%d lane ops, %d lane issue ports" % (lanes, LANE_ISSUE_W))
    vlsu_sids = [o.sid for o in v if o.fu == V_VLSU]
    if len(vlsu_sids) != len(set(vlsu_sids)):
        out.append("two VLSU ops on one scratchpad")
    if len(sdma_sids) != len(set(sdma_sids)):
        out.append("two SDMAs on one scratchpad")
    return out


# -- dependency_checker ------------------------------------------------------------
class Scoreboard(RTLModule):
    """A busy bit per scalar, vector and mask register."""

    REGS = dict(scalar=frozenset(), vector=frozenset(), mask=frozenset())

    def hazard(self, pkt: DecodedPacket) -> bool:
        """RAW on any source, WAW on any destination. An SDMA's rs1 is both,
        so its read covers it, as the RTL's comment says."""
        return (any(r in self.scalar for r in pkt.scalar_reads())
                or any(w in self.scalar for w in pkt.scalar_writes())
                or any(r in self.vector for r in pkt.vector_reads())
                or any(w in self.vector for w in pkt.vector_writes())
                or any(r in self.mask for r in pkt.mask_reads())
                or any(w in self.mask for w in pkt.mask_writes()))

    def next_state(self, issued: Optional[DecodedPacket], wb_scalar=(), wb_vector=(),
                   wb_mask=(), wb_sdma=()) -> None:
        """Clear what wrote back, then set what issued: set wins."""
        s = set(self.scalar) - set(wb_scalar) - set(wb_sdma)
        v = set(self.vector) - set(wb_vector)
        m = set(self.mask) - set(wb_mask)
        if issued is not None:
            s |= set(issued.scalar_writes())
            v |= set(issued.vector_writes())
            m |= set(issued.mask_writes())
        self.scalar_n, self.vector_n, self.mask_n = frozenset(s), frozenset(v), frozenset(m)

    @property
    def scalar_halt_ready(self) -> bool:
        """The RTL's name; it is high while something is still outstanding."""
        return bool(self.scalar)

    @property
    def idle(self) -> bool:
        return not (self.scalar or self.vector or self.mask)


# -- reg_file / reggie -----------------------------------------------------------------
class ListStorage:
    """Plain register storage: one value per register."""

    def __init__(self, num_regs: int, reset=0):
        self.values = [reset] * int(num_regs)

    def read(self, r: int):
        return self.values[r]

    def write(self, r: int, value) -> None:
        self.values[r] = value


class RegFile(RTLModule):
    """reg_file.sv: banks by reg[log2(banks)-1:0], combinational reads,
    register 0 hardwired, and reggie's conflict FSM.

    decode_2 instantiates it three times: scalar (256 x 32 bits, 4 banks,
    4 read ports), vector (256 x 32 BF16, 4 banks, 4 read ports) and mask
    (16 x 32 bits, 2 banks, 2 read ports). Register 0 reads as all zeros, or
    all ones for the mask file (ZERO_REG_VAL = 1: m0 is "every lane"), and
    writes to it are dropped.

    reggie's conflict FSM: a cycle with more than one read (or write) on one
    bank is not ready; once the packet's dependencies are free it serves one
    read per bank per cycle until at most one is left, then DONE is ready
    with every operand buffered.

    Storage is pluggable so the vector file can live in the vector core's
    Veggie banks rather than in a second copy.
    """

    REGS = dict(state=READY, pending=None)

    def __init__(self, name: str, num_regs: int, banks: int, zero_value=0,
                 storage=None, reset=0):
        super().__init__(name)
        self.num_regs, self.banks, self.zero_value = int(num_regs), int(banks), zero_value
        self.storage = storage if storage is not None else ListStorage(num_regs, reset)
        self.pending = self.pending_n = [[] for _ in range(self.banks)]
        self._writes: List[Tuple[int, object]] = []

    def _banked(self, regs: Sequence[int]) -> List[List[int]]:
        out = [[] for _ in range(self.banks)]
        for port, r in enumerate(regs):
            out[r % self.banks].append(port)
        return out

    def eval_ready_for(self, reads: Sequence[int], writes: Sequence[int],
                       deps_ready: bool) -> bool:
        self._reads, self._deps_ready = list(reads), deps_ready
        rreqs, wreqs = self._banked(reads), self._banked(writes)
        if any(len(w) > 1 for w in wreqs):
            raise AssertionError("two %s writebacks on one bank; the writeback "
                                 "arbiter allows one per bank" % self.name)
        self._conflict = any(len(r) > 1 for r in rreqs)
        if self.state == READY:
            return not self._conflict
        if self.state == CONFLICT:
            return False
        return True                                  # DONE

    def next_state(self, dec2_ready: bool, writes: Sequence[Tuple[int, object]]) -> None:
        rreqs = self._banked(self._reads)
        if self.state == READY:
            grant_from = rreqs
            state = CONFLICT if (self._conflict and self._deps_ready) else READY
        elif self.state == CONFLICT:
            grant_from = [list(p) for p in self.pending]
            more = any(len(p) > 1 for p in self.pending)
            state = CONFLICT if more else DONE
        else:
            grant_from = rreqs
            state = READY if dec2_ready else DONE
        self.pending_n = [p[1:] for p in grant_from]        # lowest port wins
        self.state_n = state
        self._writes = [(int(r), v) for r, v in writes if int(r) != 0]

    def commit(self) -> None:
        for r, v in self._writes:
            self.storage.write(r, v)
        self._writes = []
        super().commit()

    def read(self, r: int):
        return self.zero_value if r == 0 else self.storage.read(r)


class ScalarRegFile(RegFile):
    """The scalar file: 32-bit values."""

    def __init__(self, name: str = "srf"):
        super().__init__(name, NUM_SCALAR_REGS, SCALAR_BANKS, zero_value=0)

    def next_state(self, dec2_ready: bool, writes: Sequence[Tuple[int, int]]) -> None:
        super().next_state(dec2_ready, [(r, int(v) & MASK32) for r, v in writes])


def _with_data(entries) -> Tuple[List[int], List[Tuple[int, object]]]:
    """Writeback entries are a register, or (register, value): busy-bit
    clears for all of them, register-file writes for those with a value."""
    regs, writes = [], []
    for e in entries:
        if isinstance(e, tuple):
            regs.append(int(e[0]))
            writes.append((int(e[0]), e[1]))
        else:
            regs.append(int(e))
    return regs, writes


# -- decode_2 ------------------------------------------------------------------------
#: Every vector unit decode 2 asks after; "vlsu" is one flag per scratchpad.
VECTOR_UNITS = ("alu", "mul", "exp", "reduction", "gsau", "movement")


@dataclass
class IssuedPacket:
    """What the D2/EX latch carries: the decoded packet with its scalar
    operand values read, and the fetch bookkeeping that rides along."""
    packet: DecodedPacket
    pc: int
    pc_pred_addr: int
    predict_taken: bool
    #: per op: {"rs1": value, "rs2": value, ...}, keyed by (kind, slot)
    operands: Dict[Tuple[str, int], Dict[str, int]] = field(default_factory=dict)
    sdma_sids: Dict[int, int] = field(default_factory=dict)

    @property
    def halt(self) -> bool:
        return self.packet.halt


class Decode2(RTLModule):

    INS = dict(scalar=[0] * PACKET_SIZE, vector=[0] * PACKET_SIZE,
               sdma=[0] * PACKET_SIZE, valid=False, pc=0, pc_pred_addr=0,
               predict_taken=False, flush=False, halt=False,
               ex_ready=None, vector_ready=None, vlsu_ready=None, scpad_busy=None,
               wb_scalar=(), wb_vector=(), wb_mask=(), wb_sdma=())
    OUTS = dict(ready=False, issued=None)
    #: The EX/WB latch's writebacks hold for one cycle only.
    CLEAR_ON_COMMIT = ("in_wb_scalar", "in_wb_vector", "in_wb_mask", "in_wb_sdma")

    def __init__(self, name: str = "decode2", *, strict: bool = True,
                 vector_storage=None, vector_len: int = 32):
        super().__init__(name)
        self.strict = bool(strict)
        self.scoreboard = Scoreboard("scoreboard")
        self.srf = ScalarRegFile("srf")
        zero_vec = [0.0] * int(vector_len)
        self.vrf = RegFile("vrf", NUM_VECTOR_REGS, VECTOR_BANKS, zero_value=zero_vec,
                           storage=vector_storage, reset=zero_vec)
        self.mrf = RegFile("mrf", NUM_MASK_REGS, MASK_BANKS, zero_value=MASK32, reset=0)
        self.violations: List[Tuple[int, List[str]]] = []
        self.stall_reasons: Dict[str, int] = {}

    # -- this cycle's view -----------------------------------------------------
    def _fu_ready(self, pkt: DecodedPacket, sids: Dict[int, int]) -> bool:
        ex_ready = self.in_ex_ready or {}
        vec = self.in_vector_ready or {}
        vlsu = self.in_vlsu_ready or [True] * 4
        busy = self.in_scpad_busy or [False] * 4
        s, v, d = pkt.valid_ops()
        for o in s:
            if o.ex is not None and not ex_ready.get(o.ex, True):
                return False
        for o in v:
            unit = ("alu" if o.fu in VECTOR_ALU_FUS else "mul" if o.fu == V_MUL
                    else "reduction" if o.fu == V_REDU else "gsau" if o.fu == V_GSAU
                    else "movement" if o.fu == V_MVMT else None)
            if o.fu == V_EXP:
                continue                 # decode_2: EXP is always ready (TODO in RTL)
            if o.fu == V_VLSU and not vlsu[o.sid]:
                return False
            if unit and not vec.get(unit, True):
                return False
        return not any(busy[sid] for sid in sids.values())

    def eval_ready(self) -> None:
        pkt = DecodedPacket.from_words(self.in_scalar, self.in_vector, self.in_sdma)
        self._pkt = pkt
        read = self.srf.read
        sids = {o.slot: (read(o.rs3) >> 30) & 3 for o in pkt.sdma if o.valid}
        self._sids = sids
        deps = not self.scoreboard.hazard(pkt)
        fus = self._fu_ready(pkt, sids)
        self._wb_v_regs, self._wb_v = _with_data(self.in_wb_vector)
        self._wb_m_regs, self._wb_m = _with_data(self.in_wb_mask)
        rf = [self.srf.eval_ready_for(pkt.scalar_reads(),
                                      [r for r, _ in self.in_wb_scalar], deps),
              self.vrf.eval_ready_for(pkt.vector_reads(), [r for r, _ in self._wb_v], deps),
              self.mrf.eval_ready_for(pkt.mask_reads(), [r for r, _ in self._wb_m], deps)]
        rf = all(rf)
        self.out_ready = deps and fus and rf
        if not self.out_ready and self.in_valid and not pkt.empty:
            why = "hazard" if not deps else "unit" if not fus else "regfile"
            self.stall_reasons[why] = self.stall_reasons.get(why, 0) + 1

    def eval_data(self) -> None:
        pkt, ready = self._pkt, self.out_ready
        issue = ready and self.in_valid and not self.in_flush and not self.in_halt \
            and not pkt.empty
        issued = None
        if issue:
            bad = contract_violations(pkt, list(self._sids.values()))
            if bad:
                if self.strict:
                    raise PacketContractError("packet at pc %#x: %s" % (self.in_pc, "; ".join(bad)))
                self.violations.append((self.in_pc, bad))
            issued = self._read_operands(pkt)
        self.out_issued = issued
        self.scoreboard.next_state(pkt if issue else None, [r for r, _ in self.in_wb_scalar],
                                   self._wb_v_regs, self._wb_m_regs, self.in_wb_sdma)
        self.srf.next_state(ready, self.in_wb_scalar)
        self.vrf.next_state(ready, self._wb_v)
        self.mrf.next_state(ready, self._wb_m)

    def _read_operands(self, pkt: DecodedPacket) -> IssuedPacket:
        read = self.srf.read
        out = IssuedPacket(packet=pkt, pc=self.in_pc, pc_pred_addr=self.in_pc_pred_addr,
                           predict_taken=self.in_predict_taken, sdma_sids=dict(self._sids))
        for o in pkt.scalar:
            if o.valid:
                out.operands[("scalar", o.slot)] = {"rs1": read(o.rs1) if o.use_rs1 else 0,
                                                     "rs2": read(o.rs2) if o.use_rs2 else 0}
        for o in pkt.vector:
            if o.valid:
                out.operands[("vector", o.slot)] = {
                    "rs1": read(o.rs1) if o.use_rs1 else 0,
                    "rs2": read(o.rs2) if o.use_rs2 else 0,
                    "vs1": self.vrf.read(o.vs1) if o.use_vs1 else None,
                    "vs2": self.vrf.read(o.vs2) if o.use_vs2 else None,
                    "vms": self.mrf.read(o.vms) if o.use_vms else MASK32}
        for o in pkt.sdma:
            if o.valid:
                out.operands[("sdma", o.slot)] = {"rs1": read(o.rs1_rd), "rs2": read(o.rs2),
                                                   "rs3": read(o.rs3)}
        return out

    def commit(self) -> None:
        self.scoreboard.commit()
        self.srf.commit()
        self.vrf.commit()
        self.mrf.commit()
        super().commit()

    # -- what execute asks before halting -----------------------------------------
    @property
    def scalar_halt_ready(self) -> bool:
        return bool(self.scoreboard.scalar)

    @property
    def vector_halt_ready(self) -> bool:
        return bool(self.scoreboard.vector)

    @property
    def mask_halt_ready(self) -> bool:
        return bool(self.scoreboard.mask)
