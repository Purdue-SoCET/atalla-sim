from base.clocked_object import Clocked
from base.queue import SimQueue
from typing import Dict, List, Optional, Sequence, Tuple, Union

from vector_core.vector_lanes import VectorDatapath
from vector_core.vector_load_store import VLSU
from vector_core.veggie_file import Veggie

Time = float


class WBBuffer(Clocked):
    """
    Shared writeback buffer for vector functional units.
    """

    def __init__(self, depth: int = 128):
        super().__init__()
        self.entries = SimQueue(max(1, depth))

    def can_accept(self) -> bool:
        return not self.entries.is_full()

    def enqueue(self, entry: Dict) -> bool:
        return self.entries.enqueue(dict(entry))

    def has_pending(self) -> bool:
        return not self.entries.is_empty()

    def peek(self) -> Optional[Dict]:
        return self.entries.peek()

    def pop(self) -> Optional[Dict]:
        return self.entries.dequeue()

    def __len__(self) -> int:
        return len(self.entries)


class GSAUStub(Clocked):
    """
    Placeholder Global Systolic Array Unit interface.
    TODO: Will update it to a new class on a new file after I implement Systolic Array
    """

    def __init__(self, queue_depth: int = 64):
        super().__init__()
        self.pending = SimQueue(max(1, queue_depth))

    def issue(self, cmd: Dict) -> bool:
        return self.pending.enqueue(dict(cmd))

    def pop_pending(self) -> Optional[Dict]:
        return self.pending.dequeue()

    def has_pending(self) -> bool:
        return not self.pending.is_empty()


class VectorCore(Clocked):
    """
    Integrated Vector Core shell.

    Integrates:
    - VectorDatapath (compute operations)
    - Veggie (vector register file storage backing)
    - VLSU(s) (scratchpad load/store path)
    - WBBuffer (common result sink before VRF write)

    TODO: GSAU is intentionally left as a stub and can be replaced later.
    """

    def __init__(
        self,
        veggie_size: int,
        lane_count: int = 1,
        dtype: Optional[object] = None,
        issue_width: int = 1,
        vls_count: int = 2,
        wb_depth: int = 128,
        scheduler_depth: int = 128,
        fu_latencies: Optional[Dict[str, int]] = None,
        veggie_bank_count: int = 4,
        veggie_regs_per_bank: int = 64,
    ):
        super().__init__()
        if vls_count <= 0:
            raise ValueError("vls_count must be > 0")

        self.datapath = VectorDatapath(
            veggie_size=veggie_size,
            lane_count=lane_count,
            dtype=dtype,
            issue_width=issue_width,
            fu_latencies=fu_latencies,
        )
        self.vector_len = self.datapath.vector_len

        self.veggie = Veggie(
            bank_count=veggie_bank_count,
            regs_per_bank=veggie_regs_per_bank,
        )
        self.max_vregs = self.veggie.bank_count * self.veggie.regs_per_bank

        self.wb_buffer = WBBuffer(depth=wb_depth)
        self.gsau = GSAUStub() # TODO
        self.scheduler_q = SimQueue(max(1, scheduler_depth))

        self.vls_units = []
        for _ in range(vls_count):
            # Each VLSU is wired to one scratchpad frontend.
            self.vls_units.append(VLSU(scratchpad_count=1, read_vreg_cb=self.read_vreg))

        self.last_wb = None
        self.wb_valid = False
        self.last_datapath_inst_id = None

    def _reg_to_bank_addr(self, reg: int) -> Tuple[int, int]:
        if reg < 0:
            raise ValueError("register index must be >= 0")
        bank = reg % self.veggie.bank_count
        addr = reg // self.veggie.bank_count
        if addr >= self.veggie.regs_per_bank:
            raise ValueError("register index out of range: %s" % reg)
        return bank, addr

    def _normalize_vector(self, data: Sequence[float]) -> List[float]:
        vec = [float(x) for x in data]
        if len(vec) != self.vector_len:
            raise ValueError(
                "vector length mismatch: expected %d, got %d"
                % (self.vector_len, len(vec))
            )
        return vec

    def read_vreg(self, reg: int) -> List[float]:
        bank, addr = self._reg_to_bank_addr(reg)
        raw = self.veggie.data_banks[bank][addr]
        if isinstance(raw, list):
            if len(raw) == self.vector_len:
                return [float(x) for x in raw]
            if len(raw) == 0:
                return [0.0] * self.vector_len
            if len(raw) < self.vector_len:
                return [float(x) for x in raw] + ([0.0] * (self.vector_len - len(raw)))
            return [float(x) for x in raw[: self.vector_len]]
        return [float(raw)] * self.vector_len

    def write_vreg(self, reg: int, data: Sequence[float]) -> None:
        bank, addr = self._reg_to_bank_addr(reg)
        self.veggie.data_banks[bank][addr] = self._normalize_vector(data)

    def load_vreg(self, reg: int, data: Sequence[float]) -> None:
        self.write_vreg(reg, data)

    def dump_vreg(self, reg: int) -> List[float]:
        return self.read_vreg(reg)

    def enqueue_scheduler_instruction(self, inst: Dict) -> bool:
        """
        Enqueue one scheduler-issued instruction.

        Supported instruction classes:
        - Compute: {"unit":"datapath", "op", "dst", "src0", "src1?", ...}
        - Memory:  {"unit":"vlsu", "kind":"load|store", "vls":0/1, ...}
        - GSAU:    {"unit":"gsau", ...}  # TODO
        """
        return self.scheduler_q.enqueue(dict(inst))

    def enqueue_compute(
        self,
        op: str,
        dst: int,
        src0: Union[Sequence[float], int],
        src1: Optional[Union[Sequence[float], int]] = None,
        mask: Optional[Union[Sequence[bool], int]] = None,
        reduce: bool = False,
        reduce_op: str = "sum",
        reduce_out_mode: str = "partial_zero",
    ) -> bool:
        return self.enqueue_scheduler_instruction(
            {
                "unit": "datapath",
                "op": op,
                "dst": dst,
                "src0": src0,
                "src1": src1,
                "mask": mask,
                "reduce": reduce,
                "reduce_op": reduce_op,
                "reduce_out_mode": reduce_out_mode,
            }
        )

    def enqueue_memory(self, op: Dict) -> bool:
        inst = dict(op)
        inst.setdefault("unit", "vlsu")
        return self.enqueue_scheduler_instruction(inst)

    def _resolve_operand(self, value, default_zero: bool = False) -> List[float]:
        if value is None:
            if default_zero:
                return [0.0] * self.vector_len
            raise ValueError("missing required operand")
        if isinstance(value, int):
            return self.read_vreg(value)
        return self._normalize_vector(value)

    def _resolve_mask(self, mask_value) -> List[bool]:
        if mask_value is None:
            return [True] * self.vector_len
        if isinstance(mask_value, int):
            mask_vec = self.read_vreg(mask_value)
            return [bool(x) for x in mask_vec]
        mask = list(mask_value)
        if len(mask) != self.vector_len:
            raise ValueError("mask length mismatch")
        return [bool(x) for x in mask]

    def _issue_datapath(self, inst: Dict) -> bool:
        op = inst.get("op")
        if op is None:
            raise ValueError("datapath instruction missing op")
        dst = inst.get("dst", inst.get("vd"))
        if dst is None:
            raise ValueError("datapath instruction missing dst")

        src0_spec = inst.get("src0", inst.get("vs1"))
        src1_spec = inst.get("src1", inst.get("vs2"))
        src0 = self._resolve_operand(src0_spec, default_zero=False)
        src1 = self._resolve_operand(src1_spec, default_zero=True)
        mask = self._resolve_mask(inst.get("mask"))

        inst_id = self.datapath.enqueue(
            src0=src0,
            src1=src1,
            mask=mask,
            op=op,
            dst=int(dst),
            reduce=bool(inst.get("reduce", False)),
            reduce_op=inst.get("reduce_op", "sum"),
            reduce_out_mode=inst.get("reduce_out_mode", "partial_zero"),
        )
        self.last_datapath_inst_id = inst_id
        return True

    def _issue_vlsu(self, inst: Dict) -> bool:
        vls_id = int(inst.get("vls", 0))
        if vls_id < 0 or vls_id >= len(self.vls_units):
            raise ValueError("invalid vls id: %s" % vls_id)

        op = dict(inst)
        op.pop("unit", None)
        op.pop("vls", None)
        # VLSU instance is tied to one frontend; force local frontend 0.
        op["scratchpad"] = 0
        if "dst" in op and "vd" not in op:
            op["vd"] = op["dst"]
        if "src" in op and "vs" not in op:
            op["vs"] = op["src"]
        return self.vls_units[vls_id].enqueue_issue(op)

    def _issue_gsau(self, inst: Dict) -> bool:
        cmd = dict(inst)
        cmd.pop("unit", None)
        return self.gsau.issue(cmd)

    def _try_issue_scheduler(self) -> None:
        inst = self.scheduler_q.peek()
        if inst is None:
            return
        unit = inst.get("unit", "datapath")
        if unit == "datapath":
            accepted = self._issue_datapath(inst)
        elif unit == "vlsu":
            accepted = self._issue_vlsu(inst)
        elif unit == "gsau":
            accepted = self._issue_gsau(inst)
        else:
            raise ValueError("unsupported scheduler unit: %s" % unit)
        if accepted:
            _ = self.scheduler_q.dequeue()

    def _collect_results_to_wb(self) -> None:
        if self.datapath.result_valid:
            pkt = self.datapath.last_result
            if pkt is not None and self.wb_buffer.can_accept():
                self.wb_buffer.enqueue(
                    {
                        "source": "datapath",
                        "dst": pkt["dst"],
                        "data": pkt["vector"],
                        "meta": {
                            "inst_id": pkt["inst_id"],
                            "op": pkt["op"],
                            "reduce_op": pkt["reduce_op"],
                            "reduce_out_mode": pkt["reduce_out_mode"],
                            "reduction": pkt["reduction"],
                        },
                    }
                )

        for vls_id, vls in enumerate(self.vls_units):
            while vls.can_pop_writeback() and self.wb_buffer.can_accept():
                wb = vls.pop_writeback()
                if wb is None:
                    break
                self.wb_buffer.enqueue(
                    {
                        "source": "vlsu",
                        "vls": vls_id,
                        "dst": wb["vd"],
                        "data": wb["data"],
                        "mask": wb.get("mask"),
                        "meta": wb,
                    }
                )

    def _commit_one_writeback(self) -> None:
        self.wb_valid = False
        self.last_wb = None

        wb = self.wb_buffer.pop()
        if wb is None:
            return

        dst = int(wb["dst"])
        new_vec = self._normalize_vector(wb["data"])
        mask = wb.get("mask")
        if mask is not None:
            mask_vec = [bool(x) for x in list(mask)]
            if len(mask_vec) != self.vector_len:
                raise ValueError("writeback mask length mismatch")
            old_vec = self.read_vreg(dst)
            merged = [new_vec[i] if mask_vec[i] else old_vec[i] for i in range(self.vector_len)]
            self.write_vreg(dst, merged)
        else:
            self.write_vreg(dst, new_vec)

        self.last_wb = wb
        self.wb_valid = True

    def tick(self) -> None:
        # 1) Consume one scheduler instruction when target unit can accept it.
        self._try_issue_scheduler()

        # 2) Advance compute and memory units.
        self.datapath.tick()
        for vls in self.vls_units:
            vls.tick()

        # 3) Funnel unit outputs into shared writeback buffer.
        self._collect_results_to_wb()

        # 4) Commit one writeback per cycle to Veggie.
        self._commit_one_writeback()

    def pop_scratchpad_request(self, vls_id: int) -> Optional[Dict]:
        if vls_id < 0 or vls_id >= len(self.vls_units):
            raise ValueError("invalid vls id: %s" % vls_id)
        return self.vls_units[vls_id].pop_request()

    def push_scratchpad_response(self, vls_id: int, rsp: Dict) -> bool:
        if vls_id < 0 or vls_id >= len(self.vls_units):
            raise ValueError("invalid vls id: %s" % vls_id)
        local_rsp = dict(rsp)
        local_rsp["scratchpad"] = 0
        return self.vls_units[vls_id].push_response(local_rsp)

    def scheduler_backlog(self) -> int:
        return len(self.scheduler_q)


VC = VectorCore
