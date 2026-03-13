from math import ceil, log2
from typing import Dict, List, Optional, Sequence

import numpy as np

from base.clocked_object import Clocked
from base.queue import SimQueue
from base.dtype import DType, cast_vector, normalize_dtype, numpy_dtype


class SystolicArrayMEISSABlackbox(Clocked):
    """Cycle-level MEISSA blackbox model with formula-driven latency.

    Matrix contract:
    - Activations A are streamed left-to-right as n rows of length m.
    - Weights B are streamed up-to-down as m rows of length p.
    - Output C = A @ B is emitted as n rows of length p.

    Request/response contract (compatible with VectorCore GSAU packets):
    - request:  {"vdata": [...], "is_weight": bool, "expect_output": bool, "meta": {...}}
    - response: {"vdata": [...], "meta": {...}}
    """

    def __init__(
        self,
        n: int,
        m: int,
        p: int,
        req_depth: int = 256,
        rsp_depth: int = 256,
    ):
        super().__init__()
        self.n = int(n)
        self.m = int(m)
        self.p = int(p)
        if self.n <= 0 or self.m <= 0 or self.p <= 0:
            raise ValueError("n, m, p must be positive")

        self.to_array = SimQueue(max(1, int(req_depth)))
        self.from_array = SimQueue(max(1, int(rsp_depth)))

        self._cycle = 0
        self._pending_ready: List[Dict] = []

        self._reset_job()

    def _reset_job(self) -> None:
        self._job_start_cycle: Optional[int] = None
        self._weights_rows: List[List[float]] = []
        self._activation_rows: List[List[float]] = []
        self._activation_meta: List[Dict] = []
        self._scheduled = False
        self._job_dtype: Optional[DType] = None

    @property
    def t_load(self) -> int:
        return self.m

    @property
    def t_process(self) -> int:
        return self.n + self.p - 1

    @property
    def t_out(self) -> int:
        return self.n + ceil(log2(self.m)) + self.p - 1

    @property
    def t_total(self) -> int:
        return self.n + self.m + ceil(log2(self.m)) + self.p - 1

    def issue(self, req: Dict) -> bool:
        """Accept one GSAU-style request packet."""
        if self.to_array.is_full():
            return False
        entry = dict(req)
        if "vdata" not in entry:
            raise ValueError("request missing vdata")
        dtype = normalize_dtype(entry.get("dtype") or entry.get("meta", {}).get("dtype"), default=None)
        if dtype is None:
            raise ValueError("request missing dtype")
        if self._job_dtype is None:
            self._job_dtype = dtype
        if dtype != self._job_dtype:
            raise ValueError("meissa dtype mismatch: job=%s req=%s" % (self._job_dtype, dtype))
        entry["dtype"] = dtype
        entry["vdata"] = cast_vector(entry["vdata"], dtype)
        entry["is_weight"] = bool(entry.get("is_weight", False))
        entry["expect_output"] = bool(
            entry.get("expect_output", not entry["is_weight"])
        )
        entry["meta"] = dict(entry.get("meta", {}))
        return self.to_array.enqueue(entry)

    # Alias used for easier integration with VectorCore.GSAU naming.
    def push_request(self, req: Dict) -> bool:
        return self.issue(req)

    def pop_response(self) -> Optional[Dict]:
        return self.from_array.dequeue()

    def has_response(self) -> bool:
        return not self.from_array.is_empty()

    def pending_requests(self) -> int:
        return len(self.to_array)

    def pending_responses(self) -> int:
        return len(self.from_array)

    def _consume_one_request(self) -> None:
        req = self.to_array.dequeue()
        if req is None:
            return

        if self._job_start_cycle is None:
            self._job_start_cycle = self._cycle

        vec = req["vdata"]
        if req["is_weight"]:
            if len(vec) != self.p:
                raise ValueError(
                    "weight row length mismatch: expected %d, got %d"
                    % (self.p, len(vec))
                )
            if len(self._weights_rows) >= self.m:
                raise ValueError("received too many weight rows for one MEISSA job")
            self._weights_rows.append(vec)
            return

        if len(vec) != self.m:
            raise ValueError(
                "activation row length mismatch: expected %d, got %d"
                % (self.m, len(vec))
            )
        if len(self._activation_rows) >= self.n:
            raise ValueError("received too many activation rows for one MEISSA job")

        self._activation_rows.append(vec)
        if req["expect_output"]:
            self._activation_meta.append(dict(req.get("meta", {})))

    def _schedule_outputs_if_ready(self) -> None:
        if self._scheduled:
            return
        if len(self._weights_rows) < self.m or len(self._activation_rows) < self.n:
            return

        if self._job_dtype is None:
            raise ValueError("meissa job missing dtype")
        a = np.asarray(self._activation_rows, dtype=numpy_dtype(self._job_dtype))
        b = np.asarray(self._weights_rows, dtype=numpy_dtype(self._job_dtype))
        c = np.matmul(a, b)

        if self._job_start_cycle is None:
            self._job_start_cycle = self._cycle

        first_output_latency = self.t_total - (self.n - 1)
        # `_cycle` is checked before increment in `tick()`, so subtract one to
        # make an output with latency L visible on the L-th tick.
        first_due_cycle = self._job_start_cycle + first_output_latency - 1

        for row_idx in range(self.n):
            meta = (
                dict(self._activation_meta[row_idx])
                if row_idx < len(self._activation_meta)
                else {}
            )
            meta.update(
                {
                    "model": "meissa_blackbox",
                    "row": row_idx,
                    "shape": {"n": self.n, "m": self.m, "p": self.p},
                    "latency": {
                        "t_load": self.t_load,
                        "t_process": self.t_process,
                        "t_out": self.t_out,
                        "t_total": self.t_total,
                    },
                    "dtype": self._job_dtype,
                }
            )
            self._pending_ready.append(
                {
                    "due_cycle": first_due_cycle + row_idx,
                    "rsp": {
                        "vdata": cast_vector(c[row_idx].tolist(), self._job_dtype),
                        "dtype": self._job_dtype,
                        "meta": meta,
                    },
                }
            )

        self._scheduled = True

    def _flush_ready_responses(self) -> None:
        if not self._pending_ready:
            return

        # Preserve deterministic ordering by due cycle.
        self._pending_ready.sort(key=lambda x: int(x["due_cycle"]))
        keep: List[Dict] = []
        for item in self._pending_ready:
            if int(item["due_cycle"]) <= self._cycle and not self.from_array.is_full():
                _ = self.from_array.enqueue(dict(item["rsp"]))
            else:
                keep.append(item)
        self._pending_ready = keep

        # Job done once every scheduled response is emitted.
        if self._scheduled and not self._pending_ready:
            self._reset_job()

    def tick(self, time: Optional[float] = None) -> None:
        # Consume at most one ingress packet per cycle (hardware-like input port).
        self._consume_one_request()
        self._schedule_outputs_if_ready()
        self._flush_ready_responses()
        self._cycle += 1


MEISSABlackbox = SystolicArrayMEISSABlackbox
