from base.clocked_object import Clocked
from base.debug import dprintf
from base.queue import SimQueue
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple


class Xbar(Clocked):
    """
    Usage:
      x = Xbar(delay=3, num_banks=NUM_BANKS)
      x.submit(shift_mask, input_vals, callback=cb)   # returns op_id
      x.tick() -> List[(op_id, output_vals)]         # advance one cycle and collect completions

    The combinational routing is provided by Xbar.route

                Timing model:
                        - delay=N models an N-stage pipeline
                        - one request can advance into stage 0 per tick
                        - once full, the pipeline can retire one request per tick
                        - if the sink applies backpressure, the tail stage holds until the
                            callback accepts the routed output
    """

    @staticmethod
    def route(shift_mask: List[Optional[int]], input_vals: List[Any], num_banks: int = 32) -> List[Any]:
        assert len(shift_mask) == num_banks
        assert len(input_vals) == num_banks

        out = [0] * num_banks
        for i, b in enumerate(shift_mask):
            if b is not None and 0 <= b < num_banks:
                out[b] = input_vals[i]
        return out

    def __init__(self, delay: int = 3, num_banks: int = 32, max_size: Optional[int] = None):
        super().__init__()
        self.delay = max(1, int(delay))
        self.num_banks = int(num_banks)
        self.max_size = max(1, int(max_size)) if max_size is not None else self.delay
        self._tick = -1

        self._op_counter = 0
        self._issue_q: SimQueue[Dict[str, Any]] = SimQueue(self.max_size)
        self._inflight = deque()
        self._blocked_tail: Optional[Dict[str, Any]] = None

        # stats
        self.total_submitted = 0
        self.total_completed = 0
        self.total_retire_stalls = 0

    def inflight(self) -> int:
        return len(self._issue_q) + len(self._inflight) + (1 if self._blocked_tail is not None else 0)

    def next_wake(self, now: int) -> Optional[int]:
        """Idle only when the issue queue, the pipeline and the tail are empty.

        A blocked tail retries every cycle, and a queued entry has to be moved
        into the pipeline on the very next tick (that is where its due_cycle is
        stamped), so both force now+1. Only entries already in flight can be
        deferred to their own deadline.
        """
        if self._blocked_tail is not None or len(self._issue_q):
            return now + 1
        if self._inflight:
            return self._inflight[0]["due_cycle"]
        return None

    def can_accept(self) -> bool:
        return self.inflight() < self.max_size

    def enqueue(self, shift_mask: List[Optional[int]], input_vals: List[Any], callback: Optional[Callable[[List[Any]], None]] = None) -> int:
        """
        Submit a permutation request. The provided callback (if any) will be called with the
        routed output when the request completes.
        Returns an operation id, or -1 if the queue is full.
        """
        assert len(shift_mask) == self.num_banks
        assert len(input_vals) == self.num_banks
        self._op_counter += 1
        entry = {
            "shift": tuple(shift_mask),
            "vals": tuple(input_vals),
            "cb": callback,
            "op": self._op_counter,
            "due_cycle": -1,
        }
        if not self.can_accept() or not self._issue_q.enqueue(entry):
            # Queue is full, optionally call callback with False or handle overflow
            if callback:
                callback(False)
            dprintf("Xbar", f"enqueue dropped: queue full (op={self._op_counter})")
            return -1
        self.total_submitted += 1
        # Run on the next available tick so the entry is moved into the
        # pipeline exactly when it would have been before scheduling existed.
        # _tick may be stale if we have been asleep; asking too early is safe.
        self.request_wake(self._tick + 1)
        dprintf("Xbar", f"enqueue op={self._op_counter} delay={self.delay} inflight={self.inflight()}")
        return self._op_counter

    def tick(self, time: Optional[float] = None) -> List[Tuple[int, List[Any]]]:
        """
        Returns list of completed operations as (op_id, routed_output).
        Callbacks are invoked before returning.
        """
        cycle = self._consume_tick(time, attr_name="_tick")
        if cycle is None:
            return []

        completed: List[Tuple[int, List[Any]]] = []

        if self._blocked_tail is not None:
            tail = self._blocked_tail
            out = Xbar.route(tail["shift"], tail["vals"], self.num_banks)
            tail_accepted = tail["cb"](out) is not False if tail["cb"] else True
            if tail_accepted:
                completed.append((tail["op"], out))
                dprintf("Xbar", f"complete op={tail['op']}")
                self._blocked_tail = None
            else:
                self.total_retire_stalls += 1
                for entry in self._inflight:
                    entry["due_cycle"] += 1
                dprintf("Xbar", f"retire stalled op={tail['op']}")
                self.total_completed += len(completed)
                return completed
        elif self._inflight and self._inflight[0]["due_cycle"] <= cycle:
            tail = self._inflight[0]
            out = Xbar.route(tail["shift"], tail["vals"], self.num_banks)
            tail_accepted = tail["cb"](out) is not False if tail["cb"] else True
            if tail_accepted:
                self._inflight.popleft()
                completed.append((tail["op"], out))
                dprintf("Xbar", f"complete op={tail['op']}")
            else:
                self._blocked_tail = self._inflight.popleft()
                self.total_retire_stalls += 1
                for entry in self._inflight:
                    entry["due_cycle"] += 1
                dprintf("Xbar", f"retire stalled op={tail['op']}")
                self.total_completed += len(completed)
                return completed

        if len(self._inflight) + (1 if self._blocked_tail is not None else 0) < self.max_size:
            next_entry = self._issue_q.dequeue()
            if next_entry is not None:
                next_entry["due_cycle"] = cycle + self.delay
                self._inflight.append(next_entry)

        self.total_completed += len(completed)
        return completed

    def get_stats(self) -> Dict[str, Any]:
        return {
            "delay": self.delay,
            "num_banks": self.num_banks,
            "capacity": self.max_size,
            "pending": self.inflight(),
            "issue_queue": len(self._issue_q),
            "pipeline_occupancy": len(self._inflight) + (1 if self._blocked_tail is not None else 0),
            "total_submitted": self.total_submitted,
            "total_completed": self.total_completed,
            "total_retire_stalls": self.total_retire_stalls,
        }
