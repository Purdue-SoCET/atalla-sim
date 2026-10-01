"""Decode1 and the D1/D2 latch.

decode1 -- rtl/modules/scheduler/fetchdecode1/decode_1.sv -- is purely
combinational. It sorts each slot of the packet into scalar, vector or
scratchpad by opcode range, and routes it:

  * scalar and vector ops keep their slot index: slot i lands in
    scalar[i] or vector[i];
  * scratchpad ops are compacted: the first goes to scpad[0], the next to
    scpad[1], whatever slots they came from;
  * a slot whose whole 40-bit word is zero is empty and goes nowhere.

The classes, from the opcode_t enum's ordering:

    scalar   ADD_S .. HALT_S, plus MV_STM                     51 opcodes
    vector   ADD_VV .. MUL_VS, less MV_STM, SCPAD_LD/ST       24 opcodes
    scpad    SCPAD_LD, SCPAD_ST                                2 opcodes

Every opcode lands in exactly one class. NOP_S is in the scalar range, so a
NOP is a scalar instruction, not an empty slot.

The D1/D2 latch -- scheduler_core.sv, the "DEC1 outputs to latch" block.
On a flush or halt the whole latch is cleared: scalar slots to NOPs, vector
and SDMA slots emptied, valid dropped.

That is the intended behaviour, not what the RTL currently does. The RTL's
flush path assigns only `n_D1_D2_latch.scalar_instrs = NOP_PACKET`, so the
always_comb infers a latch for everything else and the wrong-path packet's
vector and SDMA ops survive into decode2 (which has no flush input). The
simulator does not reproduce RTL bugs; this one is recorded here so a
cycle comparison against the RTL knows where they diverge.
"""

from base.rtl_module import RTLModule
from scheduler.isa import INST_W, OPCODES, OP_W, PACKET_SIZE
from scheduler.fetch import NOP_INST

SCALAR_SLOTS = VECTOR_SLOTS = SCRATCH_SLOTS = 4

_BY_NAME = {m: op for op, (m, _) in OPCODES.items()}
ADD_S, HALT_S = _BY_NAME["add.s"], _BY_NAME["halt.s"]
ADD_VV, MUL_VS = _BY_NAME["add.vv"], _BY_NAME["mul.vs"]
MV_STM = _BY_NAME["mv.stm"]
SCPAD_LD, SCPAD_ST = _BY_NAME["scpad.ld"], _BY_NAME["scpad.st"]

NONE, SCALAR, VECTOR, SCPAD = "none", "scalar", "vector", "scpad"


def slot_word(packet: int, slot: int) -> int:
    """The raw 40-bit word in one slot. Slot 0 is the most significant."""
    return (packet >> (INST_W * (PACKET_SIZE - 1 - slot))) & ((1 << INST_W) - 1)


def classify(word: int) -> str:
    if word == 0:
        return NONE
    op = word & ((1 << OP_W) - 1)
    if ADD_S <= op <= HALT_S or op == MV_STM:
        return SCALAR
    if ADD_VV <= op <= MUL_VS and op not in (MV_STM, SCPAD_LD, SCPAD_ST):
        return VECTOR
    if op in (SCPAD_LD, SCPAD_ST):
        return SCPAD
    return NONE


def decode1(packet: int):
    """One packet to its (scalar, vector, scpad) slot arrays of raw words.
    Zero means an empty slot."""
    scalar = [0] * SCALAR_SLOTS
    vector = [0] * VECTOR_SLOTS
    scpad = [0] * SCRATCH_SLOTS
    wptr = 0
    for slot in range(PACKET_SIZE):
        word = slot_word(packet, slot)
        cat = classify(word)
        if cat == SCALAR and slot < SCALAR_SLOTS:
            scalar[slot] = word
        elif cat == VECTOR and slot < VECTOR_SLOTS:
            vector[slot] = word
        elif cat == SCPAD and wptr < SCRATCH_SLOTS:
            scpad[wptr] = word
            wptr += 1
    return scalar, vector, scpad


class D1D2Latch(RTLModule):
    """The decode1 -> decode2 pipeline register."""

    REGS = dict(scalar=[0] * SCALAR_SLOTS, vector=[0] * VECTOR_SLOTS,
                sdma=[0] * SCRATCH_SLOTS, pc=0, pc_pred_addr=0,
                predict_taken=False, valid=False)
    INS = dict(flush=False, halt=False, ready=False, packet=0, pc=0,
               pc_pred_addr=0, predict_taken=False, valid=False)

    def eval_data(self) -> None:
        for reg in self.REGS:
            setattr(self, reg + "_n", getattr(self, reg))

        if self.in_flush or self.in_halt:
            self.scalar_n = [NOP_INST] * SCALAR_SLOTS
            self.vector_n = [0] * VECTOR_SLOTS
            self.sdma_n = [0] * SCRATCH_SLOTS
            self.valid_n = False
            return
        if self.in_ready:
            scalar, vector, scpad = decode1(self.in_packet)
            self.scalar_n, self.vector_n, self.sdma_n = scalar, vector, scpad
            self.pc_n = self.in_pc
            self.predict_taken_n = self.in_predict_taken
            self.pc_pred_addr_n = self.in_pc_pred_addr
            self.valid_n = self.in_valid
