import math
from base.clocked_object import Clocked
from base.queue import SimQueue
from typing import Callable, Dict, List, Optional, Sequence

from base.dtype import DType, cast_scalar, cast_vector, normalize_dtype

Time = float


FLOAT_SLOT_BITS = 16


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return float("inf") if a >= 0 else float("-inf")
    return a / b


def _op_lut() -> Dict[str, Callable[[float, float], float]]:
    return {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "max": lambda a, b: a if a >= b else b,
        "min": lambda a, b: a if a <= b else b,
        "and": lambda a, b: float(int(a) & int(b)),
        "or": lambda a, b: float(int(a) | int(b)),
        "xor": lambda a, b: float(int(a) ^ int(b)),
        "sqrt": lambda a, _b: math.sqrt(a) if a >= 0 else float("nan"),
        "exp": lambda a, _b: math.exp(a),
        "div": _safe_div,
        "shl": lambda a, b: float(int(a) << int(b)),
        "shr": lambda a, b: float(int(a) >> int(b)),
    }


def _fu_for_op(op: str) -> str:
    if op in ("sqrt",):
        return "sqrt"
    if op in ("exp",):
        return "exp"
    if op in ("div",):
        return "div"
    if op in ("shl", "shr"):
        return "shift"
    return "alu"


class FunctionalUnitPipeline:
    def __init__(self, name: str, latency: int, capacity: int = 8):
        self.name = name
        self.latency = max(1, latency)
        self.capacity = max(1, capacity)
        self.entries = SimQueue(self.capacity)
        self.completed = SimQueue(self.capacity * 4)

    def can_accept(self) -> bool:
        return not self.entries.is_full()

    def push(self, payload: dict) -> bool:
        if not self.can_accept():
            return False
        return self.entries.enqueue({"remain": self.latency, "payload": payload})

    def tick(self, time: Optional[float] = None) -> None:
        next_entries = SimQueue(self.capacity)
        while not self.entries.is_empty():
            entry = self.entries.dequeue()
            if entry is None:
                break
            entry["remain"] -= 1
            if entry["remain"] <= 0:
                self.completed.enqueue(entry["payload"])
            else:
                next_entries.enqueue(entry)
        self.entries = next_entries

    def has_completed(self) -> bool:
        return not self.completed.is_empty()

    def pop_completed(self) -> Optional[dict]:
        return self.completed.dequeue()


class LaneFUContext:
    def __init__(
        self,
        inst_id: int,
        op: str,
        src0: Sequence[float],
        src1: Sequence[float],
        mask: Sequence[bool],
        indices: List[int],
        dst: int,
        reduce: bool,
        dtype: DType,
    ):
        self.inst_id = inst_id
        self.op = op
        self.src0 = src0
        self.src1 = src1
        self.mask = mask
        self.indices = indices
        self.dst = dst
        self.reduce = reduce
        self.dtype = dtype
        self.cursor = 0
        self.last_sent = False
        self.pending_count = 0
        self.reduce_accum = 0.0


class GlobalReductionUnit:
    SUPPORTED_OPS = ("sum", "min", "max")
    SUPPORTED_OUT_MODES = ("partial_zero", "partial_passthru", "broadcast")

    def __init__(self, vector_len: int):
        if vector_len <= 0:
            raise ValueError("vector_len must be > 0")
        self.vector_len = vector_len
        self.tree_alus = max(1, vector_len // 2)

    def _reduce_pair(self, op: str, a: float, b: float) -> float:
        if op == "sum":
            return a + b
        if op == "min":
            return a if a <= b else b
        if op == "max":
            return a if a >= b else b
        raise ValueError("unsupported reduction op: %s" % op)

    def reduce_tree(self, values: Sequence[float], op: str) -> Optional[float]:
        if op not in self.SUPPORTED_OPS:
            raise ValueError("unsupported reduction op: %s" % op)
        if not values:
            return None

        level = list(values)
        while len(level) > 1:
            next_level = []
            i = 0
            while i + 1 < len(level):
                next_level.append(self._reduce_pair(op, level[i], level[i + 1]))
                i += 2
            if i < len(level):
                next_level.append(level[i])
            level = next_level
        return level[0]

    def format_output_vector(
        self,
        reduction: Optional[float],
        out_mode: str,
        seed_vector: Optional[Sequence[float]] = None,
    ) -> List[float]:
        if out_mode not in self.SUPPORTED_OUT_MODES:
            raise ValueError("unsupported reduction out_mode: %s" % out_mode)
        if seed_vector is None:
            seed = [0.0] * self.vector_len
        else:
            seed = list(seed_vector)
            if len(seed) != self.vector_len:
                raise ValueError("seed_vector length mismatch")

        if out_mode == "broadcast":
            if reduction is None:
                return seed
            return [reduction] * self.vector_len

        out = seed[:]
        if reduction is not None and len(out) > 0:
            out[0] = reduction
        return out


class ResultCollector(Clocked):
    def __init__(
        self,
        lane_count: int,
        vector_len: int,
        sink_capacity: int = 32,
        reduction_alu_latency: int = 4,
    ):
        super().__init__()
        self.lane_count = lane_count
        self.vector_len = vector_len
        self.sink_capacity = max(1, sink_capacity)
        self.reduction_alu_latency = max(1, reduction_alu_latency)
        self.reduction_unit = GlobalReductionUnit(vector_len)
        self.inflight = {}
        self.completed_vectors = SimQueue(self.sink_capacity)
        self.pending_reductions = SimQueue(max(1, self.sink_capacity * 4))

    def can_accept_result(self) -> bool:
        return not self.completed_vectors.is_full()

    def allocate_instruction(
        self,
        inst_id: int,
        dst: int,
        op: str,
        reduce: bool,
        reduce_op: str = "sum",
        reduce_out_mode: str = "partial_zero",
        seed_vector: Optional[List[float]] = None,
        dtype: Optional[DType] = None,
    ) -> None:
        if reduce:
            if reduce_op not in self.reduction_unit.SUPPORTED_OPS:
                raise ValueError("unsupported reduction op: %s" % reduce_op)
            if reduce_out_mode not in self.reduction_unit.SUPPORTED_OUT_MODES:
                raise ValueError("unsupported reduction out_mode: %s" % reduce_out_mode)
        self.inflight[inst_id] = {
            "dst": dst,
            "op": op,
            "reduce": reduce,
            "reduce_op": reduce_op,
            "reduce_out_mode": reduce_out_mode,
            "vector": (seed_vector[:] if seed_vector is not None else [0.0] * self.vector_len),
            "dtype": dtype,
            "lane_done": [False] * self.lane_count,
            "lane_pending": [0] * self.lane_count,
            "reduce_accum": [0.0] * self.lane_count,
            "reduce_seen": [False] * self.lane_count,
            "reduce_count": 0,
            "completion_scheduled": False,
        }

    def lane_dispatched(self, inst_id: int, lane_id: int) -> None:
        self.inflight[inst_id]["lane_pending"][lane_id] += 1

    def lane_reduce_accum(self, inst_id: int, lane_id: int, value: float) -> None:
        state = self.inflight[inst_id]
        reduce_op = state["reduce_op"]
        if reduce_op == "sum":
            state["reduce_accum"][lane_id] += value
        elif reduce_op == "min":
            if not state["reduce_seen"][lane_id]:
                state["reduce_accum"][lane_id] = value
            else:
                state["reduce_accum"][lane_id] = min(state["reduce_accum"][lane_id], value)
        elif reduce_op == "max":
            if not state["reduce_seen"][lane_id]:
                state["reduce_accum"][lane_id] = value
            else:
                state["reduce_accum"][lane_id] = max(state["reduce_accum"][lane_id], value)
        else:
            raise ValueError("unsupported reduction op: %s" % reduce_op)
        state["reduce_seen"][lane_id] = True
        state["reduce_count"] += 1

    def lane_result(self, inst_id: int, lane_id: int, lane_elem_idx: int, value: float) -> None:
        state = self.inflight[inst_id]
        vector_idx = lane_id + lane_elem_idx * self.lane_count
        if vector_idx < self.vector_len:
            state["vector"][vector_idx] = value
        state["lane_pending"][lane_id] -= 1
        self._try_complete(inst_id)

    def lane_done(self, inst_id: int, lane_id: int) -> None:
        self.inflight[inst_id]["lane_done"][lane_id] = True
        self._try_complete(inst_id)

    def _try_complete(self, inst_id: int) -> None:
        state = self.inflight.get(inst_id)
        if state is None:
            return
        if state["completion_scheduled"]:
            return
        if not all(state["lane_done"]):
            return
        if any(p != 0 for p in state["lane_pending"]):
            return

        reduction = None
        if state["reduce"]:
            lane_partials = [
                state["reduce_accum"][lane_id]
                for lane_id in range(self.lane_count)
                if state["reduce_seen"][lane_id]
            ]
            reduction = self.reduction_unit.reduce_tree(lane_partials, state["reduce_op"])
            state["vector"] = self.reduction_unit.format_output_vector(
                reduction=reduction,
                out_mode=state["reduce_out_mode"],
                seed_vector=(state["vector"] if state["reduce_out_mode"] == "partial_passthru" else None),
            )
        packet = {
            "inst_id": inst_id,
            "dst": state["dst"],
            "op": state["op"],
            "reduce_op": state["reduce_op"],
            "reduce_out_mode": state["reduce_out_mode"],
            "vector": state["vector"][:],
            "reduction": reduction,
            "dtype": state.get("dtype"),
        }

        if not state["reduce"]:
            if not self.completed_vectors.enqueue(packet):
                return
            del self.inflight[inst_id]
            return

        n = state["reduce_count"]
        remaining_cycles = max(0, (n - 1) * self.reduction_alu_latency)
        if remaining_cycles == 0:
            if not self.completed_vectors.enqueue(packet):
                return
            del self.inflight[inst_id]
            return

        if not self.pending_reductions.enqueue(
            {
                "inst_id": inst_id,
                "remain": remaining_cycles,
                "packet": packet,
            }
        ):
            return
        state["completion_scheduled"] = True

    def tick(self, time: Optional[float] = None) -> None:
        next_pending = SimQueue(self.pending_reductions.max_size)
        while not self.pending_reductions.is_empty():
            item = self.pending_reductions.dequeue()
            if item is None:
                break
            if item["remain"] > 0:
                item["remain"] -= 1
            if item["remain"] <= 0:
                if not self.completed_vectors.enqueue(item["packet"]):
                    next_pending.enqueue(item)
                else:
                    inst_id = item["inst_id"]
                    if inst_id in self.inflight:
                        del self.inflight[inst_id]
            else:
                next_pending.enqueue(item)
        self.pending_reductions = next_pending

    def pop_completed(self) -> Optional[dict]:
        item = self.completed_vectors.dequeue()
        if item is not None:
            for pending_inst_id in list(self.inflight.keys()):
                self._try_complete(pending_inst_id)
        return item


class VectorLane(Clocked):
    def __init__(
        self,
        lane_id: int,
        lane_count: int,
        fu_latencies: Optional[Dict[str, int]] = None,
        fu_capacity: int = 8,
    ):
        super().__init__()
        if lane_count <= 0:
            raise ValueError("lane_count must be > 0")
        if lane_id < 0 or lane_id >= lane_count:
            raise ValueError("lane_id must be in [0, lane_count)")

        self.lane_id = lane_id
        self.lane_count = lane_count
        self.ops = _op_lut()
        self.fu_latencies = {
            "alu": 4,
            "sqrt": 8,
            "exp": 14,
            "div": 11,
            "shift": 3, # not in the report? VC used an xbar here, but tbh we can abstract this
        }
        if fu_latencies:
            self.fu_latencies.update(fu_latencies)

        self.fus = {}
        for fu, lat in self.fu_latencies.items():
            self.fus[fu] = FunctionalUnitPipeline(fu, lat, capacity=fu_capacity)

        self.fu_ctx = {}
        self.meta_fifo = {}
        self.pending_outputs = SimQueue(2048)
        for fu in self.fu_latencies:
            self.fu_ctx[fu] = None
            self.meta_fifo[fu] = SimQueue(fu_capacity)

    def _lane_indices(self, vector_len: int) -> List[int]:
        return list(range(self.lane_id, vector_len, self.lane_count))

    def can_issue(self, fu_name: str) -> bool:
        return self.fu_ctx[fu_name] is None

    def issue(self, context: LaneFUContext, fu_name: str) -> bool:
        if not self.can_issue(fu_name):
            return False
        self.fu_ctx[fu_name] = context
        return True

    def tick(self, collector: ResultCollector) -> None:
        # Stage 1: sequencer routes one element/FU/cycle with ready/valid semantics.
        for fu_name, ctx in self.fu_ctx.items():
            if ctx is None:
                continue

            if ctx.cursor >= len(ctx.indices):
                if not ctx.last_sent:
                    collector.lane_done(ctx.inst_id, self.lane_id)
                    ctx.last_sent = True
                    self.fu_ctx[fu_name] = None
                continue

            if not self.fus[fu_name].can_accept():
                continue

            lane_elem_idx = ctx.cursor
            vector_idx = ctx.indices[ctx.cursor]
            ctx.cursor += 1
            active = bool(ctx.mask[vector_idx])

            if active:
                value = self.ops[ctx.op](ctx.src0[vector_idx], ctx.src1[vector_idx])
                value = cast_scalar(value, ctx.dtype)
                pushed = self.fus[fu_name].push({"value": value})
                if pushed:
                    self.meta_fifo[fu_name].enqueue(
                        {
                            "inst_id": ctx.inst_id,
                            "lane_elem_idx": lane_elem_idx,
                            "dst": ctx.dst,
                            "reduce": ctx.reduce,
                            "value": value,
                        }
                    )
                    collector.lane_dispatched(ctx.inst_id, self.lane_id)
                    ctx.pending_count += 1
                    if ctx.reduce:
                        ctx.reduce_accum += value
                        collector.lane_reduce_accum(ctx.inst_id, self.lane_id, value)
                else:
                    # Pipeline refused entry; retry this element next cycle.
                    ctx.cursor -= 1

            if ctx.cursor >= len(ctx.indices) and not ctx.last_sent:
                collector.lane_done(ctx.inst_id, self.lane_id)
                ctx.last_sent = True
                if ctx.pending_count == 0:
                    self.fu_ctx[fu_name] = None

        # Stage 2: execute pipelines.
        for fu in self.fus.values():
            fu.tick()

        # Stage 3: pair FU output with metadata FIFO.
        for fu_name, fu in self.fus.items():
            while fu.has_completed() and (not self.meta_fifo[fu_name].is_empty()):
                payload = fu.pop_completed()
                meta = self.meta_fifo[fu_name].dequeue()
                ctx = self.fu_ctx[fu_name]
                if ctx is not None:
                    ctx.pending_count -= 1
                    if ctx.last_sent and ctx.pending_count == 0:
                        self.fu_ctx[fu_name] = None
                if not self.pending_outputs.enqueue(
                    {
                        "inst_id": meta["inst_id"],
                        "lane_elem_idx": meta["lane_elem_idx"],
                        "value": payload["value"],
                    }
                ):
                    raise RuntimeError("lane pending output queue overflow")

        # Stage 4: forward to result collector with backpressure.
        while (not self.pending_outputs.is_empty()) and collector.can_accept_result():
            item = self.pending_outputs.dequeue()
            if item is None:
                break
            collector.lane_result(
                item["inst_id"],
                self.lane_id,
                item["lane_elem_idx"],
                item["value"],
            )


class VectorDatapath(Clocked):
    def __init__(
        self,
        veggie_size: int,
        lane_count: int = 1,
        dtype: Optional[object] = None,
        issue_width: int = 1,
        fu_latencies: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        if veggie_size <= 0:
            raise ValueError("veggie_size must be > 0")
        if lane_count <= 0:
            raise ValueError("lane_count must be > 0")
        if issue_width <= 0:
            raise ValueError("issue_width must be > 0")
        if veggie_size % FLOAT_SLOT_BITS != 0:
            raise ValueError("veggie_size must be divisible by %d" % FLOAT_SLOT_BITS)

        self.veggie_size = veggie_size
        self.dtype = normalize_dtype(dtype, default=None)
        self.vector_len = veggie_size // FLOAT_SLOT_BITS
        self.lane_count = min(lane_count, self.vector_len)
        self.issue_width = issue_width
        self.next_inst_id = 0
        self.alu_latency = 4
        if fu_latencies and ("alu" in fu_latencies):
            self.alu_latency = max(1, fu_latencies["alu"])

        self.lanes = [
            VectorLane(i, self.lane_count, fu_latencies=fu_latencies)
            for i in range(self.lane_count)
        ]
        self.collector = ResultCollector(
            self.lane_count,
            self.vector_len,
            reduction_alu_latency=self.alu_latency,
        )
        self.pending_issue = SimQueue(128)

        self.result_valid = False
        self.last_result = None

    def _mk_src1(self, src0: Sequence[float], src1: Optional[Sequence[float]]) -> Sequence[float]:
        if src1 is None:
            return [0.0] * len(src0)
        return [float(x) for x in src1]

    def enqueue(
        self,
        src0: Sequence[float],
        src1: Optional[Sequence[float]] = None,
        mask: Optional[Sequence[bool]] = None,
        op: str = "add",
        dst: int = 0,
        reduce: bool = False,
        reduce_op: str = "sum",
        reduce_out_mode: str = "partial_zero",
        dtype: Optional[object] = None,
    ) -> int:
        if len(src0) != self.vector_len:
            raise ValueError("src0 length mismatch")
        op_dtype = normalize_dtype(dtype, default=self.dtype)
        if op_dtype is None:
            raise ValueError("dtype must be specified")
        src0_full = cast_vector(src0, op_dtype)
        src1_full = self._mk_src1(src0_full, src1)
        src1_full = cast_vector(src1_full, op_dtype)
        if len(src1_full) != self.vector_len:
            raise ValueError("src1 length mismatch")
        mask_full = list(mask) if mask is not None else [True] * self.vector_len
        if len(mask_full) != self.vector_len:
            raise ValueError("mask length mismatch")
        if op not in _op_lut():
            raise ValueError("unsupported op: %s" % op)
        if reduce:
            if reduce_op not in self.collector.reduction_unit.SUPPORTED_OPS:
                raise ValueError("unsupported reduction op: %s" % reduce_op)
            if reduce_out_mode not in self.collector.reduction_unit.SUPPORTED_OUT_MODES:
                raise ValueError("unsupported reduction out_mode: %s" % reduce_out_mode)

        inst_id = self.next_inst_id
        self.next_inst_id += 1
        if not self.pending_issue.enqueue(
            {
                "inst_id": inst_id,
                "src0": src0_full,
                "src1": list(src1_full),
                "mask": mask_full,
                "op": op,
                "dst": dst,
                "reduce": reduce,
                "reduce_op": reduce_op,
                "reduce_out_mode": reduce_out_mode,
                "dtype": op_dtype,
            }
        ):
            raise RuntimeError("datapath pending issue queue overflow")
        return inst_id

    def _can_issue_to_all_lanes(self, fu_name: str) -> bool:
        for lane in self.lanes:
            if not lane.can_issue(fu_name):
                return False
        return True

    def tick(self, time: Optional[float] = None) -> None:
        self.collector.tick()
        issued = 0
        while (not self.pending_issue.is_empty()) and issued < self.issue_width:
            inst = self.pending_issue.peek()
            if inst is None:
                break
            fu_name = _fu_for_op(inst["op"])
            if not self._can_issue_to_all_lanes(fu_name):
                break

            self.pending_issue.dequeue()
            self.collector.allocate_instruction(
                inst["inst_id"],
                inst["dst"],
                inst["op"],
                inst["reduce"],
                reduce_op=inst["reduce_op"],
                reduce_out_mode=inst["reduce_out_mode"],
                seed_vector=(inst["src0"] if inst["reduce"] and inst["reduce_out_mode"] == "partial_passthru" else None),
                dtype=inst.get("dtype"),
            )
            for lane in self.lanes:
                ctx = LaneFUContext(
                    inst_id=inst["inst_id"],
                    op=inst["op"],
                    src0=inst["src0"],
                    src1=inst["src1"],
                    mask=inst["mask"],
                    indices=lane._lane_indices(self.vector_len),
                    dst=inst["dst"],
                    reduce=inst["reduce"],
                    dtype=inst.get("dtype", self.dtype),
                )
                lane.issue(ctx, fu_name)
            issued += 1

        for lane in self.lanes:
            lane.tick(self.collector)

        completed = self.collector.pop_completed()
        self.result_valid = completed is not None
        if completed is not None:
            self.last_result = completed
