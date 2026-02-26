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


class GSAU(Clocked):
    """
    Global Systolic Array Unit control block.

    Responsibilities:
    - Accept scheduler-issued vectors and stream them to systolic-array control.
    - Track destination register indices in-order via rd FIFO.
    - Pair returning systolic vectors with rd queue and emit WB packets.
    """

    def __init__(
        self,
        max_vregs: int = 256,
        instruction_latency_mac: int = 64,
        clocks_per_mac_cycle: int = 3,
        req_depth: int = 256,
        rsp_depth: int = 256,
    ):
        super().__init__()
        self.max_vregs = max(1, int(max_vregs))
        self.instruction_latency_mac = max(1, int(instruction_latency_mac))
        self.clocks_per_mac_cycle = max(1, int(clocks_per_mac_cycle))
        # Formula-driven destination queue depth in entries.
        self.rd_queue_depth = max(1, self.instruction_latency_mac * self.clocks_per_mac_cycle)

        self.to_systolic = SimQueue(max(1, int(req_depth)))
        self.from_systolic = SimQueue(max(1, int(rsp_depth)))
        self.rd_queue = SimQueue(self.rd_queue_depth)
        self.writebacks = SimQueue(max(1, int(rsp_depth)))

    def issue(self, cmd: Dict) -> bool:
        entry = dict(cmd)
        expects_output = bool(entry.get("expect_output", not bool(entry.get("is_weight", False))))
        if self.to_systolic.is_full():
            return False
        if expects_output:
            dst = entry.get("dst")
            if dst is None:
                raise ValueError("gsau command missing dst for expected output")
            if self.rd_queue.is_full():
                return False
            if not self.rd_queue.enqueue({"dst": int(dst), "meta": dict(entry.get("meta", {}))}):
                return False

        return self.to_systolic.enqueue(
            {
                "vdata": list(entry["vdata"]),
                "is_weight": bool(entry.get("is_weight", False)),
                "meta": dict(entry.get("meta", {})),
                "expect_output": expects_output,
            }
        )

    def pop_systolic_request(self) -> Optional[Dict]:
        return self.to_systolic.dequeue()

    def push_systolic_response(self, rsp: Dict) -> bool:
        packet = dict(rsp)
        if "vdata" not in packet and "data" in packet:
            packet["vdata"] = packet["data"]
        if "vdata" not in packet:
            raise ValueError("gsau response missing vdata")
        packet["vdata"] = list(packet["vdata"])
        return self.from_systolic.enqueue(packet)

    def can_pop_writeback(self) -> bool:
        return not self.writebacks.is_empty()

    def pop_writeback(self) -> Optional[Dict]:
        return self.writebacks.dequeue()

    def has_pending(self) -> bool:
        return (not self.to_systolic.is_empty()) or (not self.from_systolic.is_empty())

    def tick(self) -> None:
        while (not self.from_systolic.is_empty()) and (not self.rd_queue.is_empty()) and (not self.writebacks.is_full()):
            rsp = self.from_systolic.dequeue()
            rd = self.rd_queue.dequeue()
            if rsp is None or rd is None:
                break
            self.writebacks.enqueue(
                {
                    "dst": int(rd["dst"]),
                    "data": list(rsp["vdata"]),
                    "mask": rsp.get("mask"),
                    "meta": {
                        "rdq_depth": self.rd_queue_depth,
                        "rdq_entry": rd.get("meta", {}),
                        "rsp_meta": rsp.get("meta", {}),
                    },
                }
            )


class VectorCore(Clocked):
    """
    Integrated Vector Core shell.

    Integrates:
    - VectorDatapath (compute operations)
    - Veggie (vector register file storage backing)
    - VLSU(s) (scratchpad load/store path)
    - GSAU (systolic-array ingress/egress + rd queue tracking)
    - WBBuffer (common result sink before VRF write)
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
        self.gsau = GSAU(max_vregs=self.max_vregs)
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
        - GSAU:    {"unit":"gsau", ...}
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
        src_spec = cmd.get("src", cmd.get("vs"))
        vdata_spec = cmd.get("vdata")
        if vdata_spec is None:
            if src_spec is None:
                raise ValueError("gsau instruction requires src or vdata")
            if isinstance(src_spec, int):
                vdata = self.read_vreg(src_spec)
            else:
                vdata = self._normalize_vector(src_spec)
        else:
            if isinstance(vdata_spec, int):
                vdata = self.read_vreg(vdata_spec)
            else:
                vdata = self._normalize_vector(vdata_spec)

        expects_output = bool(cmd.get("expect_output", not bool(cmd.get("is_weight", False))))
        gsau_cmd = {
            "vdata": vdata,
            "is_weight": bool(cmd.get("is_weight", False)),
            "expect_output": expects_output,
            "meta": {
                "kind": cmd.get("kind"),
                "src": src_spec,
            },
        }
        if expects_output:
            dst = cmd.get("dst", cmd.get("vd"))
            if dst is None:
                raise ValueError("gsau instruction missing dst")
            gsau_cmd["dst"] = int(dst)
        return self.gsau.issue(gsau_cmd)

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

        while self.gsau.can_pop_writeback() and self.wb_buffer.can_accept():
            wb = self.gsau.pop_writeback()
            if wb is None:
                break
            self.wb_buffer.enqueue(
                {
                    "source": "gsau",
                    "dst": wb["dst"],
                    "data": wb["data"],
                    "mask": wb.get("mask"),
                    "meta": wb.get("meta", {}),
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
        self.gsau.tick()

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

    def pop_systolic_request(self) -> Optional[Dict]:
        return self.gsau.pop_systolic_request()

    def push_systolic_response(self, rsp: Dict) -> bool:
        return self.gsau.push_systolic_response(rsp)


VC = VectorCore
