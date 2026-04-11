from base.clocked_object import Clocked
from base.debug import dprintf
from base.queue import SimQueue
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

        self._op_counter = 0
        self._issue_q: SimQueue[Dict[str, Any]] = SimQueue(self.max_size)
        self._pipeline: List[Optional[Dict[str, Any]]] = [None for _ in range(self.delay)]

        # stats
        self.total_submitted = 0
        self.total_completed = 0
        self.total_retire_stalls = 0

    def inflight(self) -> int:
        return len(self._issue_q) + sum(1 for stage in self._pipeline if stage is not None)

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
            "shift": list(shift_mask),
            "vals": list(input_vals),
            "cb": callback,
            "op": self._op_counter,
        }
        if not self.can_accept() or not self._issue_q.enqueue(entry):
            # Queue is full, optionally call callback with False or handle overflow
            if callback:
                callback(False)
            dprintf("Xbar", f"enqueue dropped: queue full (op={self._op_counter})")
            return -1
        self.total_submitted += 1
        dprintf("Xbar", f"enqueue op={self._op_counter} delay={self.delay} inflight={self.inflight()}")
        return self._op_counter

    def tick(self) -> List[Tuple[int, List[Any]]]:
        """
        Returns list of completed operations as (op_id, routed_output).
        Callbacks are invoked before returning.
        """
        completed: List[Tuple[int, List[Any]]] = []

        tail = self._pipeline[-1]
        tail_accepted = True
        if tail is not None:
            out = Xbar.route(tail["shift"], tail["vals"], self.num_banks)
            if tail["cb"]:
                tail_accepted = tail["cb"](out) is not False
            if tail_accepted:
                completed.append((tail["op"], out))
                dprintf("Xbar", f"complete op={tail['op']}")
            else:
                self.total_retire_stalls += 1
                dprintf("Xbar", f"retire stalled op={tail['op']}")

        if tail_accepted:
            for idx in range(self.delay - 1, 0, -1):
                self._pipeline[idx] = self._pipeline[idx - 1]
            self._pipeline[0] = None

            next_entry = self._issue_q.dequeue()
            if next_entry is not None:
                self._pipeline[0] = next_entry

        self.total_completed += len(completed)
        return completed

    def get_stats(self) -> Dict[str, Any]:
        return {
            "delay": self.delay,
            "num_banks": self.num_banks,
            "capacity": self.max_size,
            "pending": self.inflight(),
            "issue_queue": len(self._issue_q),
            "pipeline_occupancy": sum(1 for stage in self._pipeline if stage is not None),
            "total_submitted": self.total_submitted,
            "total_completed": self.total_completed,
            "total_retire_stalls": self.total_retire_stalls,
        }
