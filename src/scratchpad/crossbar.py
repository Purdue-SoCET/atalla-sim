from base.clocked_object import Clocked
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple, Any

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

    def __init__(self, delay: int = 3, num_banks: int = 32):
        super().__init__()
        self.delay = int(delay)
        self.num_banks = int(num_banks)

        # pending queue entries: dicts with keys rem, shift, vals, cb, op
        self._pending: Deque[Dict[str, Any]] = deque()
        self._op_counter: int = 0

        # stats
        self.total_submitted = 0
        self.total_completed = 0

    def submit(self, shift_mask: List[Optional[int]], input_vals: List[Any], callback: Optional[Callable[[List[Any]], None]] = None) -> int:
        """
        Submit a permutation request. The provided callback (if any) will be called with the
        routed output when the request completes.
        Returns an operation id.
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
        self._pending.append(entry)
        self.total_submitted += 1
        return self._op_counter

    def tick(self) -> List[Tuple[int, List[Any]]]:
        """
        Returns list of completed operations as (op_id, routed_output).
        Callbacks are invoked before returning.
        """
        completed: List[Tuple[int, List[Any]]] = []

        # decrement remaining cycles
        for entry in list(self._pending):
            entry["rem"] -= 1

        # collect and remove finished entries (preserve FIFO order)
        to_remove: List[Dict[str, Any]] = []
        for entry in list(self._pending):
            if entry["rem"] <= 0:
                out = Xbar.route(entry["shift"], entry["vals"], self.num_banks)
                completed.append((entry["op"], out))
                if entry["cb"]:
                    entry["cb"](out)
                to_remove.append(entry)

        for entry in to_remove:
            self._pending.remove(entry)

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