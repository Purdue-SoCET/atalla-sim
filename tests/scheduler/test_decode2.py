"""Stage 3: decode 2 -- control units, scoreboard, scalar register file, issue.

Cycle expectations are worked out from the RTL in each docstring, against a
warm instruction cache, where a packet at pc 0 reaches decode 2 at cycle 2:
fetched in 0, in IF/D1 during 1, in D1/D2 -- and so in decode 2 -- during 2.
"""
import random

import pytest

from scheduler.control import (
    ALU, CONTROL, LD, S_DIV, SQRT, V_GSAU, decode_scalar, decode_sdma, decode_vector)
from scheduler.core import SchedulerCore
from scheduler.decode1 import SCALAR, SCPAD, VECTOR, classify
from scheduler.decode2 import (
    DecodedPacket, PacketContractError, Scoreboard, contract_violations)
from scheduler.isa import INST_W, OPCODES, decode_instruction

OP = {m: op for op, (m, _) in OPCODES.items()}


# -- encoders --------------------------------------------------------------------
def r(m, rd=0, rs1=0, rs2=0):
    return OP[m] | rd << 7 | rs1 << 15 | rs2 << 23


def i(m, rd=0, rs1=0, imm=0):
    return OP[m] | rd << 7 | rs1 << 15 | (imm & 0xFFF) << 23


def mi(m, rd=0, imm=0):
    return OP[m] | rd << 7 | (imm & 0x1FFFFFF) << 15


def v3(m, d=0, s1=0, s2=0, mask=0):
    """VV, VS, VI, VMV, VMS: [14:7] [22:15] [30:23] [34:31]."""
    return OP[m] | d << 7 | s1 << 15 | s2 << 23 | mask << 31


def vm(m, vd=0, rs1=0, rs2=0, cols=31, sid=0):
    return OP[m] | vd << 7 | rs1 << 15 | rs2 << 23 | cols << 31 | sid << 36


def sdma(m, rs1=0, rs2=0, rs3=0):
    return OP[m] | rs1 << 7 | rs2 << 15 | rs3 << 23


def packet(*slots):
    slots = list(slots) + [0] * (4 - len(slots))
    return sum(w << (INST_W * (3 - k)) for k, w in enumerate(slots))


def slot0(word):
    """A packet holding one instruction in slot 0, padded with empty slots."""
    return packet(word)


def run(program, cycles=20, setup=None, strict=True):
    """A warm core running `program`; returns (core, {pc: issue cycle})."""
    core = SchedulerCore(program, strict=strict, execute=False)
    core.warm_icache()
    if setup:
        setup(core)
    core.run(cycles)
    return core, {ip.pc: c for c, ip in core.issued}


def nops(n, start):
    """Padding packets so fetch has somewhere to go."""
    return {start + 20 * k: packet(r("nop.s")) for k in range(n)}


# -- control units ---------------------------------------------------------------
def test_every_instruction_decodes_except_nop_and_the_li_pseudo_op():
    """li.s is lui.s + addi.s, expanded by the assembler; the RTL has no case
    for it and neither does the model."""
    dec = {SCALAR: decode_scalar, VECTOR: decode_vector, SCPAD: decode_sdma}
    not_valid = sorted(OPCODES[op][0] for op in OPCODES if not dec[classify(op)](op).valid)
    assert not_valid == ["li.s", "nop.s"]


def test_the_rtl_bit_slices_agree_with_the_isa_layout():
    """The control units take their own bit slices; every field they decode
    matches isa.py's layout, on random words for every opcode."""
    rng = random.Random(1)
    for op, (m, _t) in OPCODES.items():
        if m in ("li.s", "nop.s"):
            continue
        for _ in range(50):
            w = (rng.getrandbits(INST_W - 7) << 7) | op
            f = decode_instruction(w)
            cls = classify(op)
            if cls == SCALAR:
                d = decode_scalar(w)
                if m in ("sw.s", "shw.s"):
                    assert (d.rs1, d.rs2, d.imm) == (f["rs1"], f["rd"], f["imm"]), m
                elif m == "mv.stm":
                    assert (d.mask_write(), d.rs1) == (f["vmd"], f["rs1"]), m
                elif f["type"] == "BR":
                    assert (d.rs1, d.rs2, d.rd) == (f["rs1"], f["rs2"], f["rs1"]), m
                    assert (d.imm * 4, d.incr7 & 0x7F) == (f["imm"], f["incr_imm"]), m
                else:
                    for k in ("rd", "rs1", "rs2", "imm"):
                        if k in f:
                            got = getattr(d, k)
                            want = f[k] & 0x1FFFFFF if m == "lui.s" and k == "imm" else f[k]
                            assert got == want, (m, k)
            elif cls == VECTOR:
                d = decode_vector(w)
                names = {"mask": "vms", "imm8": "imm"}
                for k, val in f.items():
                    if k in ("opcode", "mnemonic", "type"):
                        continue
                    attr = names.get(k, k)
                    if f["type"] == "VS" and k == "rs1":
                        attr = "rs1"
                    if f["type"] in ("VTS", "MTS") and k == "rd":
                        attr = "rd"
                    if m == "vreg.st" and k == "vd":
                        attr = "vs1"
                    if m == "gemm.vv" and k == "vs2":
                        continue                    # gemm reads vs1 only
                    if m in ("tpus.vi", "tpop.vi") and k in ("vd", "imm", "mask"):
                        continue                    # one register, in the vs1 field
                    if m == "tpop.vi" and k == "vs1":
                        attr = "vd"                 # vs1 <= transpose_unit
                    assert getattr(d, attr) == val, (m, k)
            else:
                d = decode_sdma(w)
                assert (d.rs1_rd, d.rs2, d.rs3) == (f["rs1/rd1"], f["rs2"], f["rs3"]), m


@pytest.mark.parametrize("word,reads,writes", [
    # (scalar reads, vector reads, mask reads), (scalar, vector, mask writes)
    (v3("gemm.vv", 9, 3, 4, 2), ([], [3], []), ([], [9], [])),
    (v3("lw.vi", 9, 3, 0, 2), ([], [3], [2]), ([], [], [])),
    (vm("vreg.st", 9, 5, 6, sid=2), ([5, 6], [9], []), ([], [], [])),
    (vm("vreg.ld", 9, 5, 6, sid=2), ([5, 6], [], []), ([], [9], [])),
    (v3("vmov.vts", 7, 3, 4), ([], [3], []), ([7], [], [])),
    (v3("add.vs", 9, 3, 6, 1), ([6], [3], [1]), ([], [9], [])),
    (v3("mgt.mvv", 5, 3, 4, 1), ([], [3, 4], [1]), ([], [], [5])),
])
def test_vector_read_and_write_sets(word, reads, writes):
    """From vector_control_unit.sv, with lw.vi corrected: a weight load
    writes no vector register."""
    pkt = DecodedPacket.from_words([0] * 4, [word, 0, 0, 0], [0] * 4)
    assert (pkt.scalar_reads(), pkt.vector_reads(), pkt.mask_reads()) == reads
    assert (pkt.scalar_writes(), pkt.vector_writes(), pkt.mask_writes()) == writes


@pytest.mark.parametrize("word,reads,writes", [
    (r("beq.s", 0, 5, 6), [5, 6], [5]),          # rs1 += incr7
    (r("rcp.bf", 7, 5, 6), [5], [7]),
    (r("sqrt.bf", 7, 5, 6), [5], [7]),
    (r("stbf.s", 7, 5, 6), [5], [7]),
    (r("bfts.s", 7, 5, 6), [5], [7]),
    (i("sw.s", 7, 5, 4), [5, 7], []),            # data register in [14:7]
    (mi("jal", 7, 100), [], [7]),
    (mi("lui.s", 7, 3), [], [7]),
])
def test_scalar_read_and_write_sets(word, reads, writes):
    pkt = DecodedPacket.from_words([word, 0, 0, 0], [0] * 4, [0] * 4)
    assert (pkt.scalar_reads(), pkt.scalar_writes()) == (reads, writes)


def test_mv_stm_writes_the_mask_named_by_rd_and_an_sdma_holds_rs1():
    pkt = DecodedPacket.from_words([OP["mv.stm"] | 0x5 << 7 | 9 << 15, 0, 0, 0], [0] * 4,
                                   [sdma("scpad.ld", 11, 12, 13), 0, 0, 0])
    assert pkt.mask_writes() == [5] and pkt.scalar_reads() == [9, 11, 12, 13]
    assert pkt.scalar_writes() == [11]


def test_each_scalar_op_has_one_execute_unit():
    """xbar_4x5_exec_comb's routing; sqrt is EX2, which decode_2's structural
    check forgot."""
    ex = {m: decode_scalar(OP[m]).ex for m in
          ("add.s", "beq.s", "jal", "div.s", "bfts.s", "sqrt.bf", "add.bf", "slt.bf",
           "mul.s", "lw.s", "sw.s", "halt.s", "nop.s")}
    assert ex == {"add.s": 1, "beq.s": 1, "jal": 1, "div.s": 2, "bfts.s": 2, "sqrt.bf": 2,
                  "add.bf": 3, "slt.bf": 3, "mul.s": 4, "lw.s": 5, "sw.s": 5,
                  "halt.s": None, "nop.s": None}
    assert decode_scalar(OP["mv.stm"]).ex == 2


# -- the packet contract -----------------------------------------------------------
@pytest.mark.parametrize("scalar,vector,sdma_ops,why", [
    ([r("add.s", 1, 2, 3), r("beq.s", 0, 4, 5)], [], [], "EX1"),
    ([r("add.s", 1, 2, 3)], [], [sdma("scpad.ld", 4, 5, 6)], "5 scalar register reads"),
    ([], [v3("add.vv", 1, 2, 3), v3("mul.vv", 4, 5, 6), v3("sub.vv", 7, 8, 9)], [],
     "lane"),
    ([], [v3("gemm.vv", 1, 2), v3("gemm.vv", 3, 4)], [], "GSAU"),
    ([], [vm("vreg.ld", 1, 2, 3, sid=1), vm("vreg.ld", 4, 5, 6, sid=1)], [], "VLSU"),
])
def test_packets_the_rtl_would_mangle_are_contract_violations(scalar, vector, sdma_ops, why):
    pkt = DecodedPacket.from_words((scalar + [0] * 4)[:4], (vector + [0] * 4)[:4],
                                   (sdma_ops + [0] * 4)[:4])
    bad = contract_violations(pkt)
    assert bad and any(why in b for b in bad), bad


def test_a_legal_packet_has_no_violations():
    pkt = DecodedPacket.from_words(
        [r("add.s", 1, 2, 3), r("mul.s", 4, 5, 6), 0, 0],
        [vm("vreg.ld", 7, 8, 9, sid=0), vm("vreg.ld", 10, 11, 12, sid=1), 0, 0], [0] * 4)
    assert contract_violations(pkt) == ["8 scalar register reads, 4 ports"]
    pkt = DecodedPacket.from_words([r("add.s", 1, 2, 3), r("mul.s", 4, 5, 6), 0, 0],
                                   [v3("add.vv", 1, 2, 3), v3("gemm.vv", 4, 5), 0, 0],
                                   [0] * 4)
    assert contract_violations(pkt) == []


def test_strict_mode_refuses_a_bad_packet_and_lax_mode_counts_it():
    prog = {0: packet(r("add.s", 1, 2, 3), r("beq.s", 0, 4, 5))}
    with pytest.raises(PacketContractError, match="EX1"):
        run(prog, cycles=4)
    core, issued = run(prog, cycles=4, strict=False)
    assert issued == {0: 2} and len(core.decode2.violations) == 1


# -- scoreboard ----------------------------------------------------------------------
def test_a_dependent_packet_issues_the_cycle_after_the_writeback():
    """addi r5 issues at 2; its busy bit is set from 3. addi r6, r5 waits in
    decode 2 until the writeback in cycle 6 clears r5 on that edge, issues
    at 7, and reads the value written back. The independent packet behind it
    goes the next cycle."""
    prog = {0: packet(i("addi.s", 5, 0, 7)), 20: packet(i("addi.s", 6, 5, 1)),
            40: packet(i("addi.s", 9, 0, 3))}
    core, issued = run(prog, 12, setup=lambda c: c.writeback_at(6, scalar=[(5, 7)]))
    assert issued == {0: 2, 20: 7, 40: 8}
    second = next(ip for _, ip in core.issued if ip.pc == 20)
    assert second.operands[("scalar", 0)]["rs1"] == 7
    assert core.decode2.stall_reasons["hazard"] == 4


def test_write_after_write_waits_too():
    prog = {0: packet(i("addi.s", 5, 0, 7)), 20: packet(i("addi.s", 5, 0, 8))}
    _, issued = run(prog, 12, setup=lambda c: c.writeback_at(5, scalar=[(5, 7)]))
    assert issued == {0: 2, 20: 6}


def test_a_bit_cleared_and_set_on_one_edge_stays_set():
    sb = Scoreboard()
    sb.scalar = frozenset({5})
    pkt = DecodedPacket.from_words([i("addi.s", 5, 0, 1), 0, 0, 0], [0] * 4, [0] * 4)
    sb.next_state(pkt, wb_scalar=[5])
    sb.commit()
    assert sb.scalar == {5}


def test_an_sdma_holds_its_rs1_until_the_scratchpad_is_done():
    """A scalar writeback to r11 does not release it; the SDMA completion
    does, on cycle 9's edge."""
    prog = {0: slot0(sdma("scpad.ld", 11, 12, 13)), 20: packet(i("addi.s", 20, 11, 0))}
    _, issued = run(prog, 14, setup=lambda c: c.writeback_at(9, sdma=[11]))
    assert issued == {0: 2, 20: 10}


def test_a_mask_written_from_vector_slot_3_is_tracked():
    """The RTL tracks mask writes from vector slots 0 and 1 only; here a
    mask written from slot 3 still holds a reader until it writes back. The
    two loads' address registers sit on four different banks, so the packet
    itself issues without a register-file conflict."""
    vector = [vm("vreg.ld", 1, 2, 3, sid=0), vm("vreg.ld", 4, 4, 5, sid=1),
              v3("gemm.vv", 8, 9), v3("mgt.mvv", 5, 10, 11, 0)]
    prog = {0: packet(*vector), 20: packet(v3("add.vv", 12, 13, 14, 5))}
    _, issued = run(prog, 14, setup=lambda c: c.writeback_at(8, mask=[5]))
    assert issued == {0: 2, 20: 9}


def test_a_flushed_packet_reserves_nothing():
    """In the RTL a packet in decode 2 on a redirect cycle still sets its busy
    bits, and the D2/EX latch then drops it; nothing would ever clear them."""
    prog = {0: packet(i("addi.s", 5, 0, 7)), 200: packet(i("addi.s", 6, 5, 1))}

    def redirect_at_2(core):
        core.tick(0.0)
        core.tick(1.0)
        core.redirect_valid, core.redirect_pc, core.redirect_target = True, 0, 200
        core.tick(2.0)
        core.redirect_valid = False
        core.warm_icache([200])
    core = SchedulerCore(prog, execute=False)
    core.warm_icache()
    redirect_at_2(core)
    core.run(6, start=3)
    assert [ip.pc for _, ip in core.issued] == [200], "the flushed packet did not issue"
    assert core.decode2.scoreboard.scalar == {6}, "and r5 was never reserved"


def test_halt_ready_flags_say_something_is_outstanding():
    core, _ = run({0: packet(i("addi.s", 5, 0, 7))}, 4)
    d2 = core.decode2
    assert d2.scalar_halt_ready and not d2.vector_halt_ready
    core.writeback_at(4, scalar=[(5, 7)])
    core.run(1, start=4)
    assert not d2.scalar_halt_ready


# -- scalar register file -------------------------------------------------------------
def test_x0_reads_zero_and_ignores_writes():
    prog = {0: packet(i("addi.s", 0, 0, 7)), 20: packet(r("add.s", 6, 0, 1))}
    core, issued = run(prog, 10, setup=lambda c: c.writeback_at(4, scalar=[(0, 99), (1, 5)]))
    second = next(ip for _, ip in core.issued if ip.pc == 20)
    assert second.operands[("scalar", 0)] == {"rs1": 0, "rs2": 5}


@pytest.mark.parametrize("slots,issue", [
    ([r("add.s", 1, 4, 5), r("mul.s", 2, 6, 7)], 2),            # four banks, one read each
    ([r("add.s", 1, 4, 8)], 4),                                  # bank 0 twice: 2 late
    ([r("add.s", 1, 0, 4)], 4),                                  # x0 still occupies bank 0
    ([r("add.s", 1, 4, 8), i("muli.s", 2, 12, 1)], 5),           # bank 0 three times
    ([r("add.s", 1, 4, 8), r("mul.s", 2, 12, 16)], 6),           # bank 0 four times
])
def test_reads_on_one_bank_are_served_one_a_cycle(slots, issue):
    """reggie: READY sees the conflict and serves one per bank; CONFLICT
    serves one a cycle while more than one is left; then DONE is ready. k >= 2
    reads on the busiest bank cost k cycles."""
    _, issued = run({0: packet(*slots)}, 10)
    assert issued == {0: issue}


def test_conflicts_are_only_served_once_the_operands_are_free():
    """The FSM leaves READY only when dependencies are ready, so a packet
    that is waiting on a writeback pays for its bank conflict afterwards:
    deps free at 7, CONFLICT at 8, DONE and issue at 9."""
    prog = {0: packet(i("addi.s", 4, 0, 7)), 20: packet(r("add.s", 1, 4, 8))}
    _, issued = run(prog, 14, setup=lambda c: c.writeback_at(6, scalar=[(4, 7)]))
    assert issued == {0: 2, 20: 9}


# -- structural hazards -----------------------------------------------------------------
def test_a_busy_execute_unit_holds_the_packet():
    def setup(core):
        core.ex_ready[4] = False
    core = SchedulerCore({0: packet(r("mul.s", 1, 2, 3)), 20: packet(r("add.s", 4, 5, 6))}, execute=False)
    core.warm_icache()
    setup(core)
    core.run(5)
    assert core.issued == []
    core.ex_ready[4] = True
    core.run(3, start=5)
    assert [(c, ip.pc) for c, ip in core.issued] == [(5, 0), (6, 20)]


def test_each_scratchpad_has_its_own_vlsu():
    prog = {0: slot0(vm("vreg.ld", 1, 2, 3, sid=0)),
            20: slot0(vm("vreg.ld", 4, 5, 6, sid=1))}

    def setup(core):
        core.vlsu_ready = [True, False, True, True]
    core, issued = run(prog, 8, setup=setup)
    assert issued == {0: 2}, "sid 0 goes, sid 1 waits"


def test_an_sdma_waits_for_the_scratchpad_its_rs3_names():
    """The scratchpad is rs3[31:30] -- a register value. r13 = 2 << 30."""
    word = sdma("scpad.ld", 11, 12, 13)
    prog = {0: packet(i("addi.s", 30, 0, 0)), 20: slot0(word)}

    def busy(pad):
        def setup(core):
            core.writeback_at(1, scalar=[(13, 2 << 30)])
            core.scpad_busy = [p == pad for p in range(4)]
        return setup
    _, issued = run(prog, 8, setup=busy(1))
    assert issued == {0: 2, 20: 3}
    _, issued = run(prog, 8, setup=busy(2))
    assert issued == {0: 2}


def test_exp_is_always_ready():
    """decode_2 does not wait on the exp unit (TODO in the RTL)."""
    prog = {0: packet(0, v3("expi.vi", 1, 2, 0, 0))}
    _, issued = run(prog, 6, setup=lambda c: c.vector_ready.update(exp=False))
    assert issued == {0: 2}


# -- streaming ------------------------------------------------------------------------
def test_independent_packets_issue_one_a_cycle():
    prog = {20 * k: packet(i("addi.s", 10 + k, 0, k)) for k in range(8)}
    _, issued = run(prog, 12)
    assert issued == {20 * k: 2 + k for k in range(8)}
