from base.clocked_object import Clocked
from collections import defaultdict
from typing import Optional

Time = float

class VBank:
    def __init__(self, rows, width):
        self.mem = [[0.0] * width for _ in range(rows)]

    def read(self, addr):
        return self.mem[addr]

    def write(self, addr, data):
        self.mem[addr] = data

class Veggie(Clocked):
    def __init__(self, bank_count=4, regs_per_bank=64, dread_ports=4, dwrite_ports=2, mask_banks=2):
        super().__init__()
        self.bank_count = bank_count
        self.regs_per_bank = regs_per_bank
        self.dread_ports = dread_ports
        self.dwrite_ports = dwrite_ports
        self.mask_banks = mask_banks

        # register storage
        self.data_banks = [[0.0] * regs_per_bank for _ in range(bank_count)]
        self.dtype_banks = [[None] * regs_per_bank for _ in range(bank_count)]
        self.mask_banks_data = [[0.0] * regs_per_bank for _ in range(mask_banks)]

        # connection endpoints
        self.inp = None
        self.out = None

        # internal state
        self.conflict = False
        #: Read ports that lost their bank this cycle and must be re-driven.
        self.conflict_ports = []

    def connect(self, inp, out):
        self.inp = inp
        self.out = out

    def tick(self, time: Optional[Time] = None):
        if not self.inp:
            return

        read_reqs = getattr(self.inp, "read_reqs", [])
        write_reqs = getattr(self.inp, "write_reqs", [])

        bank_rreqs = defaultdict(list)
        bank_wreqs = defaultdict(list)

        for req in read_reqs:
            bank_rreqs[req["bank"]].append(req)
        for req in write_reqs:
            bank_wreqs[req["bank"]].append(req)

        # detect conflicts
        self.conflict = any(len(v) > 1 for v in bank_rreqs.values()) or \
                        any(len(v) > 1 for v in bank_wreqs.values())

        # One read and one write are granted per bank per cycle; the rest are
        # reported as not-granted and the caller re-drives them next cycle.
        # (This used to append to pending_reqs, which nothing ever read, so a
        # conflicting request was simply lost.)
        self.conflict_ports = [
            req["port"] for reqs in bank_rreqs.values() for req in reqs[1:]
        ]

        read_results = {}
        for bank_id, reqs in bank_rreqs.items():
            if reqs:
                req = reqs[0]
                read_results[req["port"]] = self.data_banks[bank_id][req["addr"]]

        for bank_id, reqs in bank_wreqs.items():
            if reqs:
                req = reqs[0]
                self.data_banks[bank_id][req["addr"]] = req["data"]

        if self.out:
            self.out.vreg = read_results
            self.out.dvalid = {p: (p in read_results) for p in range(self.dread_ports)}
            self.out.ready = True

class OpBuffer(Clocked):
    def __init__(self, num_pairs=1):
        super().__init__()
        self.num_pairs = num_pairs
        self.dready = [False] * (2 * num_pairs)
        self.mready = [False] * num_pairs
        self.vreg_tmp = [None] * (2 * num_pairs)
        self.vmask_tmp = [None] * num_pairs
        self.inp = None
        self.out = None

    def connect(self, inp, out):
        self.inp = inp
        self.out = out

    def tick(self, time: Optional[Time] = None):
        if not self.inp:
            return

        dvalid = getattr(self.inp, "dvalid", {})
        mvalid = getattr(self.inp, "mvalid", {})
        vreg = getattr(self.inp, "vreg", {})
        vmask = getattr(self.inp, "vmask", {})

        # Capture data valid operands
        for i in range(2 * self.num_pairs):
            if dvalid.get(i, False):
                self.vreg_tmp[i] = vreg[i]
                self.dready[i] = True

        # Capture mask valid
        for i in range(self.num_pairs):
            if mvalid.get(i, False):
                self.vmask_tmp[i] = vmask[i]
                self.mready[i] = True

        # A slot is complete only when BOTH its operands and its mask have
        # landed. Slot i owns read ports 2i and 2i+1 and mask i.
        ivalid = [
            self.dready[2 * i] and self.dready[2 * i + 1] and self.mready[i]
            for i in range(self.num_pairs)
        ]

        if self.out:
            self.out.ivalid = list(ivalid)
            self.out.vreg = self.vreg_tmp.copy()
            self.out.vmask = self.vmask_tmp.copy()
            self.out.ready = any(ivalid)

    def take(self, slot: int):
        """Hand a completed slot's operands over and free it.

        Kept separate from tick() so a consumer that cannot accept this cycle
        leaves the operands held, which is the point of a collector.
        """
        if not (0 <= slot < self.num_pairs):
            raise ValueError("slot out of range: %s" % slot)
        pair = (self.vreg_tmp[2 * slot], self.vreg_tmp[2 * slot + 1])
        mask = self.vmask_tmp[slot]
        self.dready[2 * slot] = self.dready[2 * slot + 1] = False
        self.mready[slot] = False
        self.vreg_tmp[2 * slot] = self.vreg_tmp[2 * slot + 1] = None
        self.vmask_tmp[slot] = None
        return pair, mask

    def present(self, port: int, value) -> None:
        """Hand the collector an operand directly.

        Immediates never go through a bank, but the slot still has to see both
        of its operands before it can report ready, so they are presented here.
        """
        self.vreg_tmp[port] = value
        self.dready[port] = True

    def present_mask(self, slot: int, mask) -> None:
        self.vmask_tmp[slot] = mask
        self.mready[slot] = True

    def slot_ready(self, slot: int) -> bool:
        return (self.dready[2 * slot] and self.dready[2 * slot + 1]
                and self.mready[slot])
