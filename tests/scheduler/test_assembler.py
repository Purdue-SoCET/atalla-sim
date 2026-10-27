"""The assembler (src/scheduler/assembler.py): compiler packets to a program image.

Encoding is checked three ways: every field decodes back (scheduler/isa.py's
decoder is the inverse), the layout rules hold (bootstrap, li.s, padding,
labels), and -- when both submodules are present -- every instruction of the
compiled programs encodes to exactly the word the functional sim's assembler
gives.
"""
import sys
from pathlib import Path

import pytest

from scheduler import assembler as A
from scheduler.core import load_program_text
from scheduler.isa import INST_BYTES, INST_W, PACKET_BYTE_W, PACKET_SIZE, decode_instruction

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import c_kernels  # noqa: E402


def slots(packet):
    return [(packet >> (INST_W * (PACKET_SIZE - 1 - s))) & ((1 << INST_W) - 1)
            for s in range(PACKET_SIZE)]


def program(*lines):
    """A main of the given packets (each a list of up to four instructions)."""
    text = ["main:"]
    for pkt in lines:
        text += list(pkt) + ["nop"] * (PACKET_SIZE - len(pkt))
    return "\n".join(text)


@pytest.mark.parametrize("line, fields", [
    ("add_s x3, x1, x2", {"rd": 3, "rs1": 1, "rs2": 2}),
    ("addi_s x9, x2, -32", {"rd": 9, "rs1": 2, "imm": -32}),
    ("lw_s x9, 20(x8)", {"rd": 9, "rs1": 8, "imm": 20}),
    ("sw_s x1, 4(x2)", {"rd": 1, "rs1": 2, "imm": 4}),
    ("lui_s x32, 15625", {"rd": 32, "imm": 15625}),
    ("add_vv v1, v2, v3, m1", {"vd": 1, "vs1": 2, "vs2": 3, "mask": 1}),
    ("mul_vs v1, v1, x11, m1", {"vd": 1, "vs1": 1, "rs1": 11, "mask": 1}),
    ("rsum_vi v6, v4, 0, m1", {"vd": 6, "vs1": 4, "imm": 0, "mask": 1}),
    ("vreg_ld v2, x10, x9, 31, 1", {"vd": 2, "rs1": 10, "rs2": 9, "num_cols": 31, "sid": 1}),
    ("mgt_mvs m1, v1, x10, m2", {"vmd": 1, "vs1": 1, "rs1": 10, "mask": 2}),
    ("vmov_vts x9, v1, 3", {"rd": 9, "vs1": 1, "imm8": 3}),
    ("mv_mts x10, m1", {"rd": 10, "vms": 1}),
    ("mv_stm m1, x10", {"vmd": 1, "rs1": 10}),
    ("scpad_ld x11, x10, x9", {"rs1/rd1": 11, "rs2": 10, "rs3": 9}),
])
def test_every_field_decodes_back(line, fields):
    ins = A.parse(line)[0]
    d = decode_instruction(A.encode(ins))
    assert d["mnemonic"] == ins.mnemonic
    assert {k: d[k] for k in fields} == fields


def test_main_is_called_and_halts_on_return():
    prog = A.assemble(program(["addi_s x1, x0, 1"]))
    jal, halt = decode_instruction(slots(prog.instr[0])[0]), decode_instruction(slots(prog.instr[20])[0])
    # jal's immediate counts instructions: main is 40 bytes on, 8 instructions.
    assert (jal["mnemonic"], jal["rd"], jal["imm"] * INST_BYTES) == ("jal", 1, prog.labels["main"])
    assert halt["mnemonic"] == "halt.s"
    assert prog.labels["main"] == A.BOOTSTRAP_PACKETS * PACKET_BYTE_W


def test_branches_and_jumps_are_offsets_from_their_packet():
    prog = A.assemble(program(["blt_s x9, x10, main"], ["jal x0, main"]))
    main = prog.labels["main"]
    br = decode_instruction(slots(prog.instr[main])[0])
    j = decode_instruction(slots(prog.instr[main + PACKET_BYTE_W])[0])
    assert br["imm"] == 0 and j["imm"] * INST_BYTES == -PACKET_BYTE_W


def test_a_branch_out_of_range_is_an_error():
    far = [["nop"]] * 200
    with pytest.raises(A.AssemblyError, match="out of range"):
        A.assemble(program(["blt_s x1, x2, end"], *far) + "\nend:\nhalt\nnop\nnop\nnop")


def test_a_small_li_becomes_addi_in_place():
    prog = A.assemble(program(["li_s x9, -5", "li_s x10, 60"]))
    s = [decode_instruction(w) for w in slots(prog.instr[prog.labels["main"]])]
    assert [(d["mnemonic"], d["rd"], d["rs1"], d["imm"]) for d in s[:2]] == \
        [("addi.s", 9, 0, -5), ("addi.s", 10, 0, 60)]
    assert prog.added_packets == A.BOOTSTRAP_PACKETS


def test_a_large_li_becomes_lui_then_addi_in_a_new_packet():
    value = 1177550855
    text = program(["li_s x9, %d" % value, "add_s x3, x1, x2"], ["blt_s x9, x3, next"])
    prog = A.assemble(text + "\nnext:\nhalt\nnop\nnop\nnop")
    main = prog.labels["main"]
    first = [decode_instruction(w) for w in slots(prog.instr[main])]
    second = decode_instruction(slots(prog.instr[main + PACKET_BYTE_W])[0])
    assert first[0]["mnemonic"] == "lui.s" and first[1]["mnemonic"] == "add.s"
    assert (second["mnemonic"], second["rd"], second["rs1"]) == ("addi.s", 9, 9)
    assert (first[0]["imm"] << 7) + second["imm"] == value
    # The label after it moved past the inserted packet.
    assert prog.labels["next"] == main + 3 * PACKET_BYTE_W


def test_nop_padding_outside_packets_is_dropped():
    text = "nop\nnop\nnop\n" + program(["addi_s x1, x0, 1"]) + "\n.section code\nnop\nnop"
    assert len(A.assemble(text).instr) == A.BOOTSTRAP_PACKETS + 1


def test_a_partial_packet_with_an_instruction_is_an_error():
    with pytest.raises(A.AssemblyError, match="packet of 2"):
        A.assemble("main:\naddi_s x1, x0, 1\naddi_s x2, x0, 2\nnext:\nhalt\nnop\nnop\nnop")


def test_operand_errors_name_the_line():
    with pytest.raises(A.AssemblyError, match="line 2: addi.s: imm=4096 does not fit 12 bits"):
        A.assemble("main:\naddi_s x1, x0, 4096\nnop\nnop\nnop")


def test_the_image_round_trips_through_the_in_format():
    prog = A.assemble(program(["li_s x9, 1177550855"], ["halt"]))
    data = {0x1000: 0xDEADBEEF}
    assert load_program_text(A.to_in_text(prog.instr, data)) == (prog.instr, data)


@pytest.mark.golden
@pytest.mark.skipif(not (c_kernels.COMPILER / "ppci").is_dir(), reason="no compiler submodule")
@pytest.mark.parametrize("name", ["gemm_tiled_baseline.c", "softmax.c", "scpad_store_roundtrip.c"])
def test_encoding_matches_the_functional_sims_assembler(name):
    """Every instruction the compiler emits, in place, with its labels."""
    from scheduler import golden
    sys.path.insert(0, str(golden.SUBMODULE_PATH))
    import build_compiler as fs
    path = next(d / name for d in (c_kernels.KERNELS, c_kernels.UNIT_TESTS) if (d / name).is_file())
    packets, labels = A._packetize(A.parse(c_kernels.compile_c(path)))
    packets, labels, _ = A._expand_li(packets, labels)
    packets = [[A.Instr("jal", ["x1", "main"], 0)], [A.Instr("halt.s", [], 0)]] + packets
    addr = {k: (v + A.BOOTSTRAP_PACKETS) * PACKET_BYTE_W for k, v in labels.items()}
    for i, pkt in enumerate(packets):
        pc = i * PACKET_BYTE_W
        for ins in pkt:
            d = fs.asm_to_instr_dict(ins.mnemonic, [fs.normalize_operand(o) for o in ins.ops],
                                     labels=addr, pc=pc)
            assert A.encode(ins, pc, addr) == int(fs.base.encode_instruction(d), 16), ins
