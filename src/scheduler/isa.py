"""Atalla instruction set: opcodes, field layouts and packet decode.

The scheduler fetches one VLIW packet per cycle. A packet is PACKET_SIZE
instructions of INST_W bits, laid out most-significant slot first, so slot 0
occupies the top INST_W bits.

    packet  = [ slot 0 | slot 1 | slot 2 | slot 3 ]   160 bits, 20 bytes
    instr   = 40 bits, opcode in [6:0]

Two independent sources define this, and they agree: the RTL's opcode_t enum in
rtl/include/scheduler/atalla_isa_types.vh on branch
scheduler_integration_SP26_joshklug, and the functional simulator's opcode
table. All 77 opcodes match on value and mnemonic; the table below is generated
from the RTL enum with the type column taken from the functional sim, and the
trailing comment on each row is the RTL enum name it came from.
tests/scheduler/test_isa.py re-checks that agreement against the submodule
whenever it is present, so the two cannot drift silently.

Where the ISA spreadsheet disagrees, the RTL wins -- this module models the
hardware as built -- with one addition: the spreadsheet's transpose
instructions, tpop.vi (78) and tpus.vi (79), which neither RTL branch nor
the functional sim decodes yet (docs/modelling-rules.md). The spreadsheet
also names opcode 46 jalr.s; the RTL calls it jalr.

The RTL computes `PACKET_W = INST_W * PACKET_SIZE` and then comments it as
"192 bits". The arithmetic gives 160, PACKET_BYTE_W is therefore 20, and the
functional simulator advances the PC by 20 bytes per packet. The comment is
stale; 160 is correct.
"""

from typing import Dict, List

#: Bits per instruction. RTL: INST_W.
INST_W = 40
#: Instructions per VLIW packet. RTL: PACKET_SIZE.
PACKET_SIZE = 4
#: RTL: PACKET_W = INST_W * PACKET_SIZE.
PACKET_W = INST_W * PACKET_SIZE
#: Bytes the PC advances per packet. RTL: PACKET_BYTE_W.
PACKET_BYTE_W = PACKET_W // 8
#: Opcode field width. RTL: OP_W.
OP_W = 7

#: opcode -> (mnemonic, instruction type). Generated from the RTL enum.
OPCODES: Dict[int, tuple] = {
    # ---- R ----
    0b0000001: ("add.s", "R"),            # ADD_S
    0b0000010: ("sub.s", "R"),            # SUB_S
    0b0000011: ("mul.s", "R"),            # MUL_S
    0b0000100: ("div.s", "R"),            # DIV_S
    0b0000101: ("mod.s", "R"),            # MOD_S
    0b0000110: ("or.s", "R"),             # OR_S
    0b0000111: ("and.s", "R"),            # AND_S
    0b0001000: ("xor.s", "R"),            # XOR_S
    0b0001001: ("sll.s", "R"),            # SLL_S
    0b0001010: ("srl.s", "R"),            # SRL_S
    0b0001011: ("sra.s", "R"),            # SRA_S
    0b0001100: ("slt.s", "R"),            # SLT_S
    0b0001101: ("sltu.s", "R"),           # SLTU_S
    0b0001110: ("bfts.s", "R"),           # BFTS_S
    0b0001111: ("add.bf", "R"),           # ADD_BF
    0b0010000: ("sub.bf", "R"),           # SUB_BF
    0b0010001: ("mul.bf", "R"),           # MUL_BF
    0b0010010: ("rcp.bf", "R"),           # RCP_BF
    0b0010011: ("slt.bf", "R"),           # SLT_BF
    0b0010100: ("sqrt.bf", "R"),          # SQRT_BF
    0b0010101: ("stbf.s", "R"),           # STBF_S
    0b0110001: ("nop.s", "R"),            # NOP_S
    0b0110010: ("halt.s", "R"),           # HALT_S
    # ---- I ----
    0b0010110: ("addi.s", "I"),           # ADDI_S
    0b0010111: ("subi.s", "I"),           # SUBI_S
    0b0011000: ("muli.s", "I"),           # MULI_S
    0b0011001: ("divi.s", "I"),           # DIVI_S
    0b0011010: ("modi.s", "I"),           # MODI_S
    0b0011011: ("ori.s", "I"),            # ORI_S
    0b0011100: ("andi.s", "I"),           # ANDI_S
    0b0011101: ("xori.s", "I"),           # XORI_S
    0b0011110: ("slli.s", "I"),           # SLLI_S
    0b0011111: ("srli.s", "I"),           # SRLI_S
    0b0100000: ("srai.s", "I"),           # SRAI_S
    0b0100001: ("slti.s", "I"),           # SLTI_S
    0b0100010: ("sltui.s", "I"),          # SLTUI_S
    0b0101110: ("jalr", "I"),             # JALR
    # ---- BR ----
    0b0100011: ("beq.s", "BR"),           # BEQ_S
    0b0100100: ("bne.s", "BR"),           # BNE_S
    0b0100101: ("blt.s", "BR"),           # BLT_S
    0b0100110: ("bge.s", "BR"),           # BGE_S
    0b0100111: ("bgt.s", "BR"),           # BGT_S
    0b0101000: ("ble.s", "BR"),           # BLE_S
    # ---- M ----
    0b0101001: ("lw.s", "M"),             # LW_S
    0b0101010: ("sw.s", "M"),             # SW_S
    0b0101011: ("lhw.s", "M"),            # LHW_S
    0b0101100: ("shw.s", "M"),            # SHW_S
    # ---- MI ----
    0b0101101: ("jal", "MI"),             # JAL
    0b0101111: ("li.s", "MI"),            # LI_S
    0b0110000: ("lui.s", "MI"),           # LUI_S
    # ---- VV ----
    0b0110011: ("add.vv", "VV"),          # ADD_VV
    0b0110100: ("sub.vv", "VV"),          # SUB_VV
    0b0110101: ("mul.vv", "VV"),          # MUL_VV
    0b0110110: ("gemm.vv", "VV"),         # GEMM_VV
    # ---- VS ----
    0b1001011: ("add.vs", "VS"),          # ADD_VS
    0b1001100: ("sub.vs", "VS"),          # SUB_VS
    0b1001101: ("mul.vs", "VS"),          # MUL_VS
    # ---- VI ----
    0b0110111: ("expi.vi", "VI"),         # EXPI_VI
    0b0111000: ("lw.vi", "VI"),           # LW_VI
    0b0111001: ("rsum.vi", "VI"),         # RSUM_VI
    0b0111010: ("rmin.vi", "VI"),         # RMIN_VI
    0b0111011: ("rmax.vi", "VI"),         # RMAX_VI
    0b1001110: ("tpop.vi", "VI"),         # TPOP_VI
    0b1001111: ("tpus.vi", "VI"),         # TPUS_VI
    # ---- VM ----
    0b1000100: ("vreg.ld", "VM"),         # VREG_LD
    0b1000101: ("vreg.st", "VM"),         # VREG_ST
    # ---- VMV ----
    0b0111100: ("mgt.mvv", "VMV"),        # MGT_MVV
    0b0111101: ("mlt.mvv", "VMV"),        # MLT_MVV
    0b0111110: ("meq.mvv", "VMV"),        # MEQ_MVV
    0b0111111: ("mneq.mvv", "VMV"),       # MNEQ_MVV
    # ---- VMS ----
    0b1000000: ("mgt.mvs", "VMS"),        # MGT_MVS
    0b1000001: ("mlt.mvs", "VMS"),        # MLT_MVS
    0b1000010: ("meq.mvs", "VMS"),        # MEQ_MVS
    0b1000011: ("mneq.mvs", "VMS"),       # MNEQ_MVS
    # ---- VTS ----
    0b1001000: ("vmov.vts", "VTS"),       # VMOV_VTS
    # ---- MTS ----
    0b1001001: ("mv.mts", "MTS"),         # MV_MTS
    # ---- STM ----
    0b1001010: ("mv.stm", "STM"),         # MV_STM
    # ---- SDMA ----
    0b1000110: ("scpad.ld", "SDMA"),      # SCPAD_LD
    0b1000111: ("scpad.st", "SDMA"),      # SCPAD_ST
}

#: Field layout per instruction type, as (name, high_bit, low_bit, signed_width).
#: signed_width is None for unsigned fields; BR is special-cased below because
#: its immediate is split across two ranges.
FIELDS: Dict[str, tuple] = {
    "R":    (("rd", 14, 7, None), ("rs1", 22, 15, None), ("rs2", 30, 23, None)),
    "I":    (("rd", 14, 7, None), ("rs1", 22, 15, None), ("imm", 34, 23, 12)),
    "M":    (("rd", 14, 7, None), ("rs1", 22, 15, None), ("imm", 34, 23, 12)),
    "MI":   (("rd", 14, 7, None), ("imm", 39, 15, 25)),
    "VV":   (("vd", 14, 7, None), ("vs1", 22, 15, None), ("vs2", 30, 23, None),
             ("mask", 34, 31, None)),
    "VS":   (("vd", 14, 7, None), ("vs1", 22, 15, None), ("rs1", 30, 23, None),
             ("mask", 34, 31, None)),
    "VI":   (("vd", 14, 7, None), ("vs1", 22, 15, None), ("imm", 30, 23, None),
             ("mask", 34, 31, None)),
    "VM":   (("vd", 14, 7, None), ("rs1", 22, 15, None), ("rs2", 30, 23, None),
             ("num_cols", 35, 31, None), ("sid", 37, 36, None)),
    "VMV":  (("vmd", 10, 7, None), ("vs1", 22, 15, None), ("vs2", 30, 23, None),
             ("mask", 34, 31, None)),
    "VMS":  (("vmd", 10, 7, None), ("vs1", 22, 15, None), ("rs1", 30, 23, None),
             ("mask", 34, 31, None)),
    "VTS":  (("rd", 14, 7, None), ("vs1", 22, 15, None), ("imm8", 30, 23, None)),
    "MTS":  (("rd", 14, 7, None), ("vms", 18, 15, None)),
    "STM":  (("vmd", 10, 7, None), ("rs1", 22, 15, None)),
    "SDMA": (("rs1/rd1", 14, 7, None), ("rs2", 22, 15, None),
             ("rs3", 30, 23, None)),
}


def get_bits(value: int, high: int, low: int) -> int:
    return (value >> low) & ((1 << (high - low + 1)) - 1)


def sign_extend(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return (value & (sign - 1)) - (value & sign)


def decode_instruction(instr: int) -> Dict:
    """One 40-bit instruction to a field dict.

    An unknown opcode decodes to mnemonic "unknown" and carries the raw bits
    rather than raising: a packet slot can hold anything, and the pipeline has
    to be able to report a bad instruction rather than die decoding it.
    """
    opcode = get_bits(instr, OP_W - 1, 0)
    if opcode not in OPCODES:
        return {"opcode": opcode, "mnemonic": "unknown", "type": "UNKNOWN",
                "raw": instr}

    mnemonic, instr_type = OPCODES[opcode]
    decoded = {"opcode": opcode, "mnemonic": mnemonic, "type": instr_type}

    if instr_type == "BR":
        # The branch immediate is split: bit 14 is its top bit, bits 39:31 the
        # rest, and the whole thing is shifted left by 2 (packet-aligned).
        imm1 = get_bits(instr, 14, 14)
        imm9 = get_bits(instr, 39, 31)
        decoded.update({
            "incr_imm": get_bits(instr, 13, 7),
            "rs1": get_bits(instr, 22, 15),
            "rs2": get_bits(instr, 30, 23),
            "imm": sign_extend(((imm1 << 9) | imm9) << 2, 12),
        })
        return decoded

    for name, high, low, signed in FIELDS.get(instr_type, ()):
        raw = get_bits(instr, high, low)
        decoded[name] = sign_extend(raw, signed) if signed else raw
    return decoded


def decode_packet(packet: int, packet_length: int = PACKET_SIZE) -> List[Dict]:
    """One packet to a list of decoded slots, slot 0 first.

    Slot 0 is the most significant INST_W bits, so the shift counts down.
    """
    out = []
    for slot in range(packet_length):
        shift = ((packet_length - 1) - slot) * INST_W
        decoded = decode_instruction((packet >> shift) & ((1 << INST_W) - 1))
        decoded["slot"] = slot
        out.append(decoded)
    return out
