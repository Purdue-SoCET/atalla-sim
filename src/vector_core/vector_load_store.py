from base.clocked_object import Clocked
from base.queue import SimQueue
from typing import Callable, Dict, List, Optional

Time = float


class VectorLoadStoreUnit(Clocked):
    """
    Vector Load Store Unit (VLSU) model.

    Design points:
    - One issue can be accepted per cycle when downstream can accept.
    - Two independent scratchpad frontends are supported by default.
    - Each scratchpad has a dedicated synchronous FIFO that tracks destination
      registers for outstanding loads.
    - Stores are pass-through requests (no destination tracking required).
    - Scratchpad responses are matched to destination registers by FIFO order.
    """

    def __init__(
        self,
        scratchpad_count: int = 2,
        issue_queue_depth: int = 64,
        req_queue_depth: int = 64,
        rsp_queue_depth: int = 64,
        dst_fifo_depth: int = 64,
        load_response_latency: int = 13,
        read_vreg_cb: Optional[Callable[[int], List[int]]] = None,
        write_vreg_cb: Optional[Callable[[int, List[int]], None]] = None,
    ):
        super().__init__()
        if scratchpad_count <= 0:
            raise ValueError("scratchpad_count must be > 0")

        self.scratchpad_count = scratchpad_count
        self.load_response_latency = max(1, load_response_latency)
        self.read_vreg_cb = read_vreg_cb
        self.write_vreg_cb = write_vreg_cb

        self.issue_q = SimQueue(max(1, issue_queue_depth))
        self.req_q = SimQueue(max(1, req_queue_depth))
        self.rsp_q = SimQueue(max(1, rsp_queue_depth))
        self.wb_q = SimQueue(max(1, rsp_queue_depth))

        self.load_dst_fifos = [SimQueue(max(1, dst_fifo_depth)) for _ in range(self.scratchpad_count)]
        self.req_valid = False
        self.rsp_valid = False
        self.wb_valid = False
        self.last_req = None
        self.last_rsp = None
        self.last_wb = None

    def _validate_spad(self, scratchpad: int) -> None:
        if scratchpad < 0 or scratchpad >= self.scratchpad_count:
            raise ValueError("scratchpad index out of range: %s" % scratchpad)

    def can_accept_issue(self) -> bool:
        return not self.issue_q.is_full()

    def enqueue_issue(self, op: Dict) -> bool:
        """
        Accept one vector memory instruction from scheduler.

        Required fields:
        - kind: "load" or "store"
        - scratchpad: frontend id (0..scratchpad_count-1)

        For load:
        - vd: destination vector register id

        For store:
        - vs: source vector register id or data must be provided inline as "data"

        Optional passthrough fields:
        - addr, size, eew, vl, stride, swizzle, mask
        """
        kind = op.get("kind")
        if kind not in ("load", "store"):
            raise ValueError("op.kind must be 'load' or 'store'")

        spad = int(op.get("scratchpad", 0))
        self._validate_spad(spad)

        if kind == "load" and ("vd" not in op):
            raise ValueError("load op requires vd")
        if kind == "store" and ("vs" not in op) and ("data" not in op):
            raise ValueError("store op requires vs or inline data")

        entry = dict(op)
        entry["scratchpad"] = spad
        return self.issue_q.enqueue(entry)

    def can_pop_request(self) -> bool:
        return not self.req_q.is_empty()

    def pop_request(self) -> Optional[Dict]:
        return self.req_q.dequeue()

    def can_accept_response(self) -> bool:
        return not self.rsp_q.is_full()

    def push_response(self, rsp: Dict) -> bool:
        """
        Accept one scratchpad response.
        Required fields:
        - scratchpad: frontend id
        - data: loaded vector payload
        Optional fields:
        - addr, meta
        """
        spad = int(rsp.get("scratchpad", -1))
        self._validate_spad(spad)
        if "data" not in rsp:
            raise ValueError("scratchpad response requires data")
        entry = dict(rsp)
        entry["scratchpad"] = spad
        return self.rsp_q.enqueue(entry)

    def can_pop_writeback(self) -> bool:
        return not self.wb_q.is_empty()

    def pop_writeback(self) -> Optional[Dict]:
        return self.wb_q.dequeue()

    def _issue_one(self) -> None:
        op = self.issue_q.peek()
        if op is None:
            return

        kind = op["kind"]
        spad = op["scratchpad"]

        if self.req_q.is_full():
            return

        if kind == "load":
            if self.load_dst_fifos[spad].is_full():
                return
            if not self.load_dst_fifos[spad].enqueue(
                {
                    "vd": op["vd"],
                    "mask": op.get("mask"),
                    "vl": op.get("vl"),
                    "eew": op.get("eew"),
                    "swizzle": op.get("swizzle"),
                }
            ):
                return

            req = {
                "kind": "load",
                "scratchpad": spad,
                "addr": op.get("addr"),
                "size": op.get("size"),
                "eew": op.get("eew"),
                "vl": op.get("vl"),
                "stride": op.get("stride"),
                "swizzle": op.get("swizzle"),
                "mask": op.get("mask"),
            }
            if not self.req_q.enqueue(req):
                _ = self.load_dst_fifos[spad].dequeue()
                return

        else:
            if "data" in op:
                store_data = op["data"]
            else:
                if self.read_vreg_cb is None:
                    raise RuntimeError("store requires read_vreg_cb when data is not inline")
                store_data = self.read_vreg_cb(op["vs"])

            req = {
                "kind": "store",
                "scratchpad": spad,
                "addr": op.get("addr"),
                "size": op.get("size"),
                "eew": op.get("eew"),
                "vl": op.get("vl"),
                "stride": op.get("stride"),
                "swizzle": op.get("swizzle"),
                "mask": op.get("mask"),
                "data": store_data,
                "vs": op.get("vs"),
            }
            if not self.req_q.enqueue(req):
                return

        _ = self.issue_q.dequeue()

    def _handle_one_response(self) -> None:
        rsp = self.rsp_q.peek()
        if rsp is None:
            return

        spad = rsp["scratchpad"]
        dst_tag = self.load_dst_fifos[spad].peek()
        if dst_tag is None:
            raise RuntimeError("received load response without pending destination tag")
        if self.wb_q.is_full():
            return

        _ = self.load_dst_fifos[spad].dequeue()
        _ = self.rsp_q.dequeue()
        wb = {
            "scratchpad": spad,
            "vd": dst_tag["vd"],
            "data": rsp["data"],
            "mask": dst_tag.get("mask"),
            "vl": dst_tag.get("vl"),
            "eew": dst_tag.get("eew"),
            "swizzle": dst_tag.get("swizzle"),
            "addr": rsp.get("addr"),
            "meta": rsp.get("meta"),
        }
        if not self.wb_q.enqueue(wb):
            raise RuntimeError("writeback queue overflow")

    def _apply_one_writeback(self) -> None:
        if self.write_vreg_cb is None:
            return
        wb = self.wb_q.peek()
        if wb is None:
            return
        self.write_vreg_cb(wb["vd"], wb["data"])
        _ = self.wb_q.dequeue()

    def _update_debug_signals(self) -> None:
        self.last_req = self.req_q.peek()
        self.last_rsp = self.rsp_q.peek()
        self.last_wb = self.wb_q.peek()
        self.req_valid = self.last_req is not None
        self.rsp_valid = self.last_rsp is not None
        self.wb_valid = self.last_wb is not None

    def tick(self) -> None:
        # One issue and one response are processed per cycle.
        self._issue_one()
        self._handle_one_response()
        self._apply_one_writeback()
        self._update_debug_signals()

    def fifo_occupancy(self) -> List[int]:
        return [len(fifo) for fifo in self.load_dst_fifos]

    def outstanding_loads(self, scratchpad: Optional[int] = None) -> int:
        if scratchpad is None:
            return sum(len(fifo) for fifo in self.load_dst_fifos)
        self._validate_spad(scratchpad)
        return len(self.load_dst_fifos[scratchpad])


VLSU = VectorLoadStoreUnit
