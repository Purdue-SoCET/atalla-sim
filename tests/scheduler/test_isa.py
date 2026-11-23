"""The ISA table and packet decoder.

The table is generated from the RTL's opcode_t enum; these tests pin the parts
that a regeneration could silently get wrong, and -- when the functional sim
submodule is present -- that the port still agrees with it.
"""
import random

import pytest

from scheduler import golden
from scheduler.isa import (
    FIELDS, INST_W, OPCODES, OP_W, PACKET_BYTE_W, PACKET_SIZE, PACKET_W,
    VLOOP_FUNCT_BITS, VLOOP_FUNCTS, VLOOP_OPCODE,
    decode_instruction, decode_packet, get_bits, sign_extend)


def test_packet_geometry_matches_the_rtl():
    """PACKET_W is INST_W * PACKET_SIZE = 160, not the 192 the RTL comment
    claims; PACKET_BYTE_W = 20 is the functional sim's packet stride."""
    assert (INST_W, PACKET_SIZE, OP_W) == (40, 4, 7)
    assert PACKET_W == 160
    assert PACKET_BYTE_W == 20


#: tpop.vi, tpus.vi: from the spec sheet; the functional sim doesn't have them.
TRANSPOSE = {0b1001110, 0b1001111}
#: vloop, the CISC vector loop: ahead of the RTL and the spec sheet. (The
#: functional sim has it.)
AHEAD_OF_SHEET = {VLOOP_OPCODE}


def test_the_table_covers_every_opcode_once():
    assert len(OPCODES) == 80                 # the RTL's 77, tpop.vi / tpus.vi, vloop
    mnemonics = [m for m, _ in OPCODES.values()]
    assert len(set(mnemonics)) == len(mnemonics), "duplicate mnemonic"
    assert all(0 <= op < (1 << OP_W) for op in OPCODES), "opcode wider than OP_W"
    # every type in the table has a field layout, or is the special-cased BR
    types = {t for _, t in OPCODES.values()}
    assert types - set(FIELDS) == {"BR"}


def test_fields_stay_inside_an_instruction():
    for ty, fields in FIELDS.items():
        for name, high, low, signed in fields:
            assert 0 <= low <= high < INST_W, "%s.%s out of range" % (ty, name)
            if signed is not None:
                assert signed == high - low + 1, \
                    "%s.%s sign width %d != field width" % (ty, name, signed)


def test_slot_zero_is_the_most_significant():
    """Packets are laid out slot 0 first, so a value in the top INST_W bits
    must decode into slot 0 and nowhere else."""
    add_s = 0b0000001
    packet = add_s << (INST_W * (PACKET_SIZE - 1))
    decoded = decode_packet(packet)
    assert [d["slot"] for d in decoded] == [0, 1, 2, 3]
    assert decoded[0]["mnemonic"] == "add.s"
    assert all(d["mnemonic"] != "add.s" for d in decoded[1:])


def test_an_unknown_opcode_decodes_rather_than_raises():
    """A packet slot can hold anything; the pipeline has to report a bad
    instruction, not die decoding it."""
    unused = next(op for op in range(1 << OP_W) if op not in OPCODES)
    d = decode_instruction(unused)
    assert d["mnemonic"] == "unknown" and d["type"] == "UNKNOWN"
    assert d["raw"] == unused


def test_immediates_sign_extend():
    assert sign_extend(0b0111, 4) == 7
    assert sign_extend(0b1000, 4) == -8
    # addi.s with a negative 12-bit immediate
    addi_s = next(op for op, (m, _) in OPCODES.items() if m == "addi.s")
    instr = addi_s | (0xFFF << 23)
    assert decode_instruction(instr)["imm"] == -1


def test_branch_immediate_is_split_and_counts_instructions():
    """The BR immediate is bit 14 as its top bit, bits 39:31 below it, counting
    5-byte instructions; decode reports it in bytes."""
    beq = next(op for op, (m, t) in OPCODES.items() if t == "BR" and m.startswith("beq"))
    instr = beq | (1 << 14) | (0b000000001 << 31)
    d = decode_instruction(instr)
    assert d["type"] == "BR"
    assert d["imm"] == sign_extend((1 << 9) | 1, 10) * 5


def test_get_bits_is_inclusive():
    assert get_bits(0b1011, 3, 0) == 0b1011
    assert get_bits(0b1011, 1, 1) == 1


# --- agreement with the functional simulator -------------------------------

@pytest.mark.golden
def test_the_table_matches_the_functional_sim():
    assert set(OPCODES) - TRANSPOSE == set(golden.GOLDEN_OPCODES), \
        "opcode sets differ"
    for op, (mnemonic, ty) in OPCODES.items():
        if op in TRANSPOSE:
            continue
        g_mn, g_ty = golden.GOLDEN_OPCODES[op]
        assert (mnemonic, ty) == (g_mn.lower(), g_ty), \
            "opcode %d: ours %r golden %r" % (op, (mnemonic, ty), (g_mn, g_ty))


@pytest.mark.golden
def test_random_packets_decode_identically():
    """Field for field, over random bit patterns -- the layouts are where a
    port goes wrong, and only a differential test finds it."""
    rng = random.Random(20260925)
    for _ in range(2000):
        opcode = rng.choice(sorted(set(OPCODES) - TRANSPOSE))
        instr = opcode | (rng.getrandbits(INST_W - OP_W) << OP_W)
        ours = decode_instruction(instr)
        theirs = golden.golden_decode_instruction(instr)
        assert ours == theirs, "instr 0x%010x\n ours   %s\n golden %s" % (
            instr, ours, theirs)


# --- agreement with the ISA spreadsheet ------------------------------------

import csv
from pathlib import Path

SHEET = Path(__file__).parent / "data" / "atalla_isa_sheet.csv"

#: Where the spec and isa.py knowingly disagree. Decided: the model follows
#: the RTL on scheduler_integration_SP26_joshklug, not the spec, wherever the
#: two differ. When the RTL adds an opcode, move it out of this table and into
#: isa.py -- this test then fails until you do. Anything not listed here that
#: differs is a failure.
#: Every spec opcode is in isa.py now: tpop.vi (78) and tpus.vi (79) were
#: added ahead of the RTL enum and the functional sim.
SPEC_AHEAD_OF_RTL = {}
SPEC_NAMES = {
    46: ("jalr", "jalr.s"),   # RTL enum JALR and the functional sim say jalr
}


def _sheet():
    rows = {}
    for r in csv.DictReader(SHEET.open()):
        code = (r.get("OPCODE[6:0]") or "").strip()
        if code and set(code) <= {"0", "1"}:
            rows[int(code, 2)] = r
    return rows


def test_the_table_matches_the_spec_sheet():
    sheet = _sheet()
    assert set(OPCODES) - AHEAD_OF_SHEET <= set(sheet), "isa.py has opcodes the spec does not"
    missing = {op: sheet[op]["Instruction Name"].strip()
               for op in set(sheet) - set(OPCODES)}
    assert missing == SPEC_AHEAD_OF_RTL, (
        "spec opcodes not in isa.py changed: %s" % missing)
    for op, (mnemonic, _) in OPCODES.items():
        if op in AHEAD_OF_SHEET:
            continue
        spec = sheet[op]["Instruction Name"].strip().lower()
        if op in SPEC_NAMES:
            assert (mnemonic, spec) == SPEC_NAMES[op]
            continue
        assert mnemonic == spec, "opcode %d: isa.py %r, spec %r" % (op, mnemonic, spec)


def test_gemm_reads_one_vector_register():
    """vd = vs1 * weights_sys_array. The VV encoding still carries a vs2
    field, but GEMM does not read it -- a scoreboard that trusted the field
    names would invent a false dependency on vs2."""
    sheet = _sheet()
    gemm = next(op for op, (m, _) in OPCODES.items() if m == "gemm.vv")
    assert gemm == 54
    pseudo = sheet[gemm]["Pseudo-code"]
    assert "vs1" in pseudo and "vs2" not in pseudo


# --- vloop: the CISC vector loop -------------------------------------------

def _vloop(funct, rs1, rs2, rs3, sid_in, sid_out):
    hi, lo = VLOOP_FUNCT_BITS
    return (VLOOP_OPCODE | rs1 << 7 | rs2 << 15 | rs3 << 23 | sid_in << 31
            | sid_out << 33 | funct << lo)


def test_vloop_decodes_its_funct_as_the_mnemonic():
    d = decode_instruction(_vloop(0, rs1=5, rs2=6, rs3=255, sid_in=1, sid_out=3))
    assert d == {"opcode": VLOOP_OPCODE, "mnemonic": "relu.vl", "type": "VL",
                 "funct": 0, "rs1": 5, "rs2": 6, "rs3": 255, "sid_in": 1, "sid_out": 3}


def test_vloop_with_an_unassigned_funct_is_unknown():
    unused = next(f for f in range(16) if f not in VLOOP_FUNCTS)
    assert decode_instruction(_vloop(unused, 1, 2, 3, 0, 0))["mnemonic"] == "unknown"


@pytest.mark.golden
def test_vloop_decodes_as_the_functional_sim_does():
    """Field for field over random VL words, every funct (assigned or not)."""
    rng = random.Random(20261112)
    for _ in range(500):
        instr = VLOOP_OPCODE | (rng.getrandbits(INST_W - OP_W) << OP_W)
        ours = decode_instruction(instr)
        theirs = golden.golden_decode_instruction(instr)
        assert ours == theirs, "instr 0x%010x\n ours   %s\n golden %s" % (instr, ours, theirs)


def test_vloop_fields_fit_and_do_not_overlap():
    hi, lo = VLOOP_FUNCT_BITS
    spans = [(h, l) for _, h, l, _ in FIELDS["VL"]] + [(OP_W - 1, 0), (hi, lo)]
    bits = [b for h, l in spans for b in range(l, h + 1)]
    assert len(bits) == len(set(bits)) and max(bits) < INST_W
