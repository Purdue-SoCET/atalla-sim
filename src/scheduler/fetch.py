"""Fetch, branch target buffer and the IF/D1 latch.

Transcribes three modules under rtl/modules/scheduler/fetchdecode1/:

  fetch.sv     the PC register and next-PC selection
  BTB.sv       64-entry direct-mapped branch target buffer
  if_dec1.sv   the fetch -> decode1 pipeline register

Next-PC priority, highest first:

    halt   -> 0
    flush  -> the redirect target from execute
    BTB    -> the predicted target, when prediction says taken
    else   -> pc + 20, the next packet

Prediction is backward-taken/forward-not-taken: taken only on a BTB hit whose
target is behind the PC, which is a loop's back edge. A forward target that
hits is still predicted not taken.

The BTB indexes on pc[7:2] and tags on pc[31:8]. BTB_OFFSET is
clog2(PACKET_BYTE_W & -PACKET_BYTE_W) = clog2(4) = 2, so it indexes on a
4-byte granule while packets sit 20 bytes apart: consecutive packets land on
entries 0, 5, 10, 15, ... That looks wrong but costs nothing. A packet at
pc = 20k indexes 5k mod 64, and 5 is coprime to 64, so any 64 consecutive
packets still use all 64 entries; index and tag together are pc >> 2, unique.

The BTB is written only on a redirect -- the execute stage's
redirect_valid -- with the resolved target.
"""

from base.rtl_module import RTLModule
from scheduler.isa import PACKET_BYTE_W, PACKET_SIZE, INST_W

BTB_ENTRIES = 64
BTB_IDX_W = 6
BTB_OFFSET = 2          # clog2(PACKET_BYTE_W & -PACKET_BYTE_W) = clog2(4)
BTB_TAG_W = 32 - BTB_IDX_W - BTB_OFFSET

#: NOP_S -- atalla_isa_types.vh: NOP_R has opcode NOP_S and every other field 0.
NOP_OPCODE = 49
NOP_INST = NOP_OPCODE
#: Four NOP_INSTs. Slot 0 is the most significant 40 bits, as in isa.py.
NOP_PACKET = sum(NOP_INST << (INST_W * (PACKET_SIZE - 1 - s))
                 for s in range(PACKET_SIZE))

MASK32 = 0xFFFFFFFF


def btb_index(pc: int) -> int:
    return (pc >> BTB_OFFSET) & (BTB_ENTRIES - 1)


def btb_tag(pc: int) -> int:
    return (pc >> (32 - BTB_TAG_W)) & ((1 << BTB_TAG_W) - 1)


class BTB(RTLModule):
    """Combinational read, write on the clock edge."""

    REGS = dict(valid=[False] * BTB_ENTRIES, tags=[0] * BTB_ENTRIES,
                targets=[0] * BTB_ENTRIES)
    INS = dict(update_en=False, pc_update=0, true_target=0)

    def read(self, pc: int):
        """(bhit, predict_target)"""
        i = btb_index(pc)
        if self.valid[i] and self.tags[i] == btb_tag(pc):
            return True, self.targets[i]
        return False, 0

    def eval_data(self) -> None:
        self.valid_n, self.tags_n, self.targets_n = (
            self.valid, self.tags, self.targets)
        if not self.in_update_en:
            return
        i = btb_index(self.in_pc_update)
        valid, tags, targets = list(self.valid), list(self.tags), list(self.targets)
        valid[i] = True
        tags[i] = btb_tag(self.in_pc_update)
        targets[i] = self.in_true_target & MASK32
        self.valid_n, self.tags_n, self.targets_n = valid, tags, targets


class Fetch(RTLModule):
    """fetch.sv -- the PC and what it hands the IF/D1 latch this cycle."""

    REGS = dict(pc=0)
    INS = dict(flush=False, ready=False, halt=False, pc_branch=0,
               ihit=False, imemload=0, bhit=False, predict_target=0)
    OUTS = dict(imemREN=False, pc=0, predict_taken=False, pc_pred_addr=0,
                inst_packet=NOP_PACKET, valid=False)

    def eval_ready(self) -> None:
        # imemREN only needs ready/flush/halt, and the icache needs it before
        # it can say whether this cycle hits -- so settle it first.
        self.out_imemREN = bool(self.in_ready and not self.in_flush
                                and not self.in_halt)

    def eval_data(self) -> None:
        pred_taken = bool(self.in_bhit and self.in_predict_target < self.pc)
        if self.in_halt:
            next_pc = 0
        elif self.in_flush:
            next_pc = self.in_pc_branch
        elif pred_taken:
            next_pc = self.in_predict_target
        else:
            next_pc = self.pc + PACKET_BYTE_W
        self.pc_n = (next_pc & MASK32
                     if self.in_flush or (self.in_ihit and self.in_ready)
                     else self.pc)

        ihit = self.in_ihit
        self.out_pc = self.pc
        self.out_predict_taken = pred_taken if ihit else False
        self.out_pc_pred_addr = self.in_predict_target if ihit else 0
        live = ihit and not self.in_halt
        self.out_inst_packet = self.in_imemload if live else NOP_PACKET
        self.out_valid = bool(live)


class IFD1Latch(RTLModule):
    """if_dec1.sv. Clears on flush or halt; loads when ready; holds otherwise."""

    REGS = dict(pc=0, inst_packet=0, predict_taken=False, pc_pred_addr=0,
                valid=False)
    INS = dict(flush=False, halt=False, ready=False, pc=0,
               inst_packet=NOP_PACKET, predict_taken=False, pc_pred_addr=0,
               valid=False)

    def eval_data(self) -> None:
        if self.in_flush or self.in_halt:
            self.pc_n, self.inst_packet_n = 0, NOP_PACKET
            self.predict_taken_n, self.pc_pred_addr_n, self.valid_n = False, 0, False
        elif self.in_ready:
            self.pc_n = self.in_pc
            self.inst_packet_n = self.in_inst_packet
            self.predict_taken_n = self.in_predict_taken
            self.pc_pred_addr_n = self.in_pc_pred_addr
            self.valid_n = self.in_valid
        else:
            for reg in self.REGS:
                setattr(self, reg + "_n", getattr(self, reg))
