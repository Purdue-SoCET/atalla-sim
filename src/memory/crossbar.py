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

    def __init__(self, delay: int = 3, num_banks: int = 32, max_size: int = 1):
        super().__init__()
        self.delay = int(delay)
        self.num_banks = int(num_banks)

        self._op_counter = 0
        # pending queue entries: dicts with keys rem, shift, vals, cb, op
        self._pending: SimQueue[Dict[str, Any]] = SimQueue(max_size)  # or another limit        self._op_counter: int = 0

        # stats
        self.total_submitted = 0
        self.total_completed = 0

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
            "rem": int(self.delay),
            "shift": list(shift_mask),
            "vals": list(input_vals),
            "cb": callback,
            "op": self._op_counter,
        }
        if not self._pending.enqueue(entry):
            # Queue is full, optionally call callback with False or handle overflow
            if callback:
                callback(False)
            dprintf("Xbar", f"enqueue dropped: queue full (op={self._op_counter})")
            return -1
        self.total_submitted += 1
        dprintf("Xbar", f"enqueue op={self._op_counter} delay={self.delay}")
        return self._op_counter

    def tick(self) -> List[Tuple[int, List[Any]]]:
        """
        Returns list of completed operations as (op_id, routed_output).
        Callbacks are invoked before returning.
        """
        completed: List[Tuple[int, List[Any]]] = []

        # decrement remaining cycles
        for entry in self._pending.items:
            entry["rem"] -= 1

        # collect and remove finished entries (preserve FIFO order)
        to_remove: List[int] = []
        for idx, entry in enumerate(self._pending.items):
            if entry["rem"] <= 0:
                out = Xbar.route(entry["shift"], entry["vals"], self.num_banks)
                completed.append((entry["op"], out))
                if entry["cb"]:
                    entry["cb"](out)
                dprintf("Xbar", f"complete op={entry['op']}")
                to_remove.append(idx)

        # Remove completed entries by index (reverse order to avoid shifting)
        for idx in reversed(to_remove):
            del self._pending.items[idx]

        self.total_completed += len(completed)
        return completed

    def get_stats(self) -> Dict[str, Any]:
        return {
            "delay": self.delay,
            "num_banks": self.num_banks,
            "pending": len(self._pending),
            "total_submitted": self.total_submitted,
            "total_completed": self.total_completed,
        }
