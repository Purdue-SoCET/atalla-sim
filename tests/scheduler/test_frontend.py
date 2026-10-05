"""Stage 2: fetch, BTB, icache, decode1 and the two front-end latches.

Timing expectations here are derived by hand from the RTL, not read back off
the model, so a test that passes means the model agrees with arithmetic done
on icache.sv / fetch.sv -- not with itself.
"""
import pytest

from scheduler import golden
from scheduler.core import SchedulerCore as _SchedulerCore, load_program_text
from scheduler.decode1 import (
    NONE, SCALAR, SCPAD, VECTOR, classify, decode1, slot_word)
from scheduler.fetch import NOP_INST, NOP_PACKET, btb_index, btb_tag
from scheduler.icache import split_address
from scheduler.isa import INST_W, OPCODES

OP = {m: op for op, (m, _) in OPCODES.items()}


def SchedulerCore(*args, **kwargs):
    """The front end alone. With no execute stage, nothing writes back, so a
    real decode 2 would hold every packet that reads an earlier one's result;
    these tests force it ready (or stalled, per test) instead."""
    core = _SchedulerCore(*args, strict=False, **kwargs)
    core.decode2_override = True
    return core


def r_type(name, rd=1, rs1=2, rs2=3):
    return OP[name] | rd << 7 | rs1 << 15 | rs2 << 23


def packet(*slots):
    slots = list(slots) + [0] * (4 - len(slots))
    return sum(w << (INST_W * (3 - i)) for i, w in enumerate(slots))


def straight_line(n, start=0):
    return {start + 20 * i: packet(r_type("add.s", rd=1 + i % 200))
            for i in range(n)}


def fetch_hits(core, cycles, start=0):
    """(cycle, pc) of every packet fetch hands the IF/D1 latch."""
    hits = []
    for c in range(start, start + cycles):
        before = core.packets_fetched
        pc = core.pc
        core.tick(float(c))
        if core.packets_fetched > before:
            hits.append((c, pc))
    return hits


# --- icache timing ---------------------------------------------------------

def test_cold_start_follows_the_fill_and_early_restart():
    """8-beat fills, a beat a cycle. Early restart lets a packet go as soon as
    its beats are in: pc 0 needs beats 0-2, so it hits once fill_count > 2,
    four cycles in. pc 40 needs beat 7 and cannot early-restart; pc 60 is split
    across lines 0 and 1 and waits for line 1's second beat."""
    core = SchedulerCore(straight_line(10))
    hits = fetch_hits(core, 23)
    assert [c for c, _ in hits] == [4, 6, 9, 13, 16, 18, 22]
    assert [pc for _, pc in hits] == [0, 20, 40, 60, 80, 100, 120]


def test_warm_cache_streams_a_packet_a_cycle():
    core = SchedulerCore(straight_line(8))
    core.warm_icache()
    core.run(8)
    assert core.issued_to_d1d2 == [(c, 20 * (c - 1)) for c in range(1, 8)]
    assert core.icache.fills == 0


def test_a_split_packet_needs_both_lines():
    """pc 56 ends at byte 75, past the line. Line 0 is filled whole -- its
    last_chunk is 7, so no early restart -- over cycles 1-8; IDLE at 9 sees
    line 1 miss and starts it; line 1 needs beats 0-1 (last_b = 75>>3 & 7 = 1),
    in by cycle 12. Three times the cost of an unsplit cold fetch."""
    tag, idx, off = split_address(56)
    assert off == 56 and off + 19 >= 64
    core = SchedulerCore({56: packet(r_type("add.s"))})
    core.fetch.pc = core.fetch.pc_n = 56
    hits = fetch_hits(core, 13)
    assert hits == [(12, 56)]
    assert core.icache.fills == 2, "one fill per line, one after the other"


def test_memory_wait_states_stretch_the_fill():
    """With a wait cycle before every beat, each beat takes two cycles, so the
    first packet (needs 3 beats) arrives at cycle 7 instead of 4."""
    core = SchedulerCore(straight_line(4), icache_first_beat_wait=1,
                         icache_beat_wait=1)
    hits = fetch_hits(core, 12)
    assert hits[0] == (7, 0)


# --- backpressure ----------------------------------------------------------

def test_a_stalled_decode2_loses_and_duplicates_nothing():
    core = SchedulerCore(straight_line(20))
    core.warm_icache()
    core.run(3)
    core.decode2_override = False
    core.run(6, start=3)
    core.decode2_override = True
    core.run(10, start=9)
    pcs = [pc for _, pc in core.issued_to_d1d2]
    assert pcs == list(range(0, 20 * len(pcs), 20)), "a gap or a repeat"


def test_bubbles_collapse_into_empty_latches():
    """ready || !valid: with decode2 never ready, the two empty latches still
    fill -- one packet each -- before fetch stops."""
    core = SchedulerCore(straight_line(10))
    core.warm_icache()
    core.decode2_override = False
    core.run(10)
    assert core.d1d2.valid and core.ifd1.valid
    assert core.d1d2.pc == 0 and core.ifd1.pc == 20
    assert core.packets_fetched == 2
    assert core.pc == 40, "the PC stops once both latches are full"


# --- BTB ---------------------------------------------------------------------

def test_the_btb_indexes_on_a_4_byte_granule():
    """BTB_OFFSET = clog2(20 & -20) = 2, so packets 20 bytes apart land 5
    entries apart -- harmless, since 5 is coprime to 64: 64 consecutive
    packets still fill all 64 entries."""
    assert [btb_index(pc) for pc in (0, 20, 40, 60, 80)] == [0, 5, 10, 15, 20]
    assert len({btb_index(20 * k) for k in range(64)}) == 64
    assert btb_index(0) == btb_index(256) and btb_tag(0) != btb_tag(256)


def _redirect(core, cycle, branch_pc, target):
    core.redirect_valid, core.redirect_pc, core.redirect_target = (
        True, branch_pc, target)
    core.tick(float(cycle))
    core.redirect_valid = False


def test_a_backward_branch_is_predicted_taken_next_time():
    """BTFNT: after the BTB learns 100 -> 20, fetching 100 again predicts the
    back edge and goes straight to 20."""
    core = SchedulerCore(straight_line(10))
    core.warm_icache()
    core.run(2)
    _redirect(core, 2, branch_pc=100, target=20)
    assert core.btb.read(100) == (True, 20)
    core.fetch.pc = core.fetch.pc_n = 100
    core.tick(3.0)
    assert core.predicted_taken == 1
    assert core.pc == 20


def test_a_forward_branch_hits_but_is_not_predicted():
    core = SchedulerCore(straight_line(10))
    core.warm_icache()
    _redirect(core, 0, branch_pc=40, target=160)
    core.fetch.pc = core.fetch.pc_n = 40
    core.tick(1.0)
    assert core.btb.read(40) == (True, 160)
    assert core.predicted_taken == 0
    assert core.pc == 60, "forward target: fall through"


def test_the_btb_is_tag_checked():
    core = SchedulerCore(straight_line(4))
    _redirect(core, 0, branch_pc=0, target=0)
    assert core.btb.read(0)[0]
    assert not core.btb.read(256)[0], "same index, different tag"


# --- flush ---------------------------------------------------------------------

def test_a_redirect_costs_two_cycles_to_refill_d1d2():
    """Flush at F clears IF/D1 and points the PC at the target; F+1 fetches it
    (warm); F+2 it is in D1/D2."""
    core = SchedulerCore(straight_line(20))
    core.warm_icache()
    core.run(4)
    _redirect(core, 4, branch_pc=60, target=200)
    assert not core.ifd1.valid, "flush clears IF/D1"
    core.run(3, start=5)
    after = [(c, pc) for c, pc in core.issued_to_d1d2 if c > 4]
    assert after[0] == (6, 200)


def test_flush_clears_every_slot_in_d1d2():
    """A flush empties the whole latch -- scalar, vector and SDMA -- and drops
    valid, so nothing from the wrong path reaches decode2. (The RTL clears
    only the scalar slots; the simulator models the intent, not the bug.)"""
    ld = OP["scpad.ld"] | 1 << 7
    add_vv = r_type("add.vv")
    core = SchedulerCore({0: packet(r_type("add.s"), add_vv, ld)})
    core.warm_icache()
    core.decode2_override = False
    core.run(3)
    assert core.d1d2.valid and core.d1d2.vector[1] == add_vv
    assert core.d1d2.sdma[0] == ld
    _redirect(core, 3, branch_pc=0, target=400)
    assert core.d1d2.scalar == [NOP_INST] * 4
    assert core.d1d2.vector == [0] * 4
    assert core.d1d2.sdma == [0] * 4
    assert not core.d1d2.valid


def test_halt_freezes_the_pc():
    """halt drops imemREN, so the icache cannot hit, so the PC -- which only
    moves on flush or ihit -- stays put, and nothing more is fetched."""
    core = SchedulerCore(straight_line(10))
    core.warm_icache()
    core.run(3)
    pc, fetched = core.pc, core.packets_fetched
    core.internal_halt = True
    core.run(5, start=3)
    assert core.pc == pc and core.packets_fetched == fetched
    assert not core.ifd1.valid


# --- decode1 -----------------------------------------------------------------

def test_every_opcode_lands_in_exactly_one_class():
    counts = {SCALAR: 0, VECTOR: 0, SCPAD: 0, NONE: 0}
    for op in OPCODES:
        counts[classify(op)] += 1
    assert counts == {SCALAR: 51, VECTOR: 24, SCPAD: 2, NONE: 0}


def test_scalar_and_vector_keep_their_slot_scpad_compacts():
    ld = OP["scpad.ld"] | 1 << 7
    st = OP["scpad.st"] | 2 << 7
    add_s, add_vv = r_type("add.s"), r_type("add.vv")
    scalar, vector, scpad = decode1(packet(ld, add_s, st, add_vv))
    assert scalar == [0, add_s, 0, 0], "scalar stays in slot 1"
    assert vector == [0, 0, 0, add_vv], "vector stays in slot 3"
    assert scpad == [ld, st, 0, 0], "scpad packed to the front, in order"


def test_a_zero_word_is_empty_but_a_nop_is_scalar():
    assert classify(0) == NONE
    assert classify(NOP_INST) == SCALAR
    scalar, vector, scpad = decode1(NOP_PACKET)
    assert scalar == [NOP_INST] * 4 and vector == [0] * 4


def test_slot_zero_is_the_first_word():
    assert slot_word(packet(111, 222, 333, 444), 0) == 111
    assert slot_word(packet(111, 222, 333, 444), 3) == 444


# --- programs ----------------------------------------------------------------

def test_the_program_loader_reads_the_functional_sim_format():
    text = """
    00000000: 0000000001 0000000031 0000000031 0000000031   # add.s + nops
    00000014: 0000000032 0000000031 0000000031 0000000031
    .data
    00000100: DEADBEEF
    """
    instr, data = load_program_text(text)
    assert sorted(instr) == [0x00, 0x14]
    assert slot_word(instr[0], 0) == 1 and slot_word(instr[0], 1) == 0x31
    assert data == {0x100: 0xDEADBEEF}


def test_branching_smoke_streams_through_the_front_end():
    """The one live program in the functional sim's corpus that exercises
    control flow. Without decode2 or execute, branches never resolve, so this
    checks fetch walks the image in order and every word classifies."""
    golden.require(pytest)
    path = golden.SUBMODULE_PATH / "tests" / "unit" / "branching_smoke.txt"
    instr, _ = load_program_text(path.read_text())
    core = SchedulerCore(instr)
    core.warm_icache()
    core.run(len(instr) + 2)
    pcs = [pc for _, pc in core.issued_to_d1d2]
    assert pcs[:len(instr)] == sorted(instr)[:len(pcs)]
    for word in (slot_word(p, s) for p in instr.values() for s in range(4)):
        assert word == 0 or classify(word) != NONE
