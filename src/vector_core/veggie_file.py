from base.clocked_object import Clocked
from collections import defaultdict

Time = float

class VBank:
    def __init__(self, rows, width):
        self.mem = [[0]*width for _ in range(rows)]

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
        self.data_banks = [[0] * regs_per_bank for _ in range(bank_count)]
        self.mask_banks_data = [[0] * regs_per_bank for _ in range(mask_banks)]

        # connection endpoints
        self.inp = None
        self.out = None

        # internal state
        self.conflict = False
        self.pending_reqs = []

    def connect(self, inp, out):
        self.inp = inp
        self.out = out

    def Tick(self, time):
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

        if self.conflict:
            # hold off and retry next tick
            self.pending_reqs.append((read_reqs, write_reqs))
            if self.out:
                self.out.ready = False
            return

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
        #     print(f"[{time}] Veggie wrote: {[(bank_id, req['addr'], req['data']) for bank_id, reqs in bank_wreqs.items() for req in reqs]}")
        #     print(f"[{time}] Veggie read_results: {read_results}")

        # print(f"[{time}] Veggie Tick: read_reqs={read_reqs} write_reqs={write_reqs}")

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

    def Tick(self, time):
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

        # For this test, mark valid if we have *any operand* and mask
        ivalid = [any(self.dready) and any(self.mready)]

        if self.out:
            self.out.ivalid = ivalid
            self.out.vreg = self.vreg_tmp.copy()
            self.out.vmask = self.vmask_tmp.copy()
            self.out.ready = all(ivalid)

        # Reset once consumed
        if all(ivalid):
            self.dready = [False] * (2 * self.num_pairs)
            self.mready = [False] * self.num_pairs

        # print(f"[{time}] OpBuffer Tick: dvalid={dvalid} mvalid={mvalid} -> ivalid={ivalid}")
