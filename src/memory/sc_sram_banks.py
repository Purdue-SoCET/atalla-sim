from dataclasses import dataclass
import heapq
from typing import Callable, Optional, List, Dict, Tuple, Any

from base.sched import SimClock, WakeGroup

from base.clocked_object import Clocked
from base.queue import SimQueue

"""
  - SRAMBanks(bank_count, bank_size, read_latency, write_latency)
    Here bank_size is number of slots per bank (S in emulator).
  - enqueue_read(addr, length, callback=None)
  - enqueue_write(addr, data: bytes, callback=None)
  - tick() -> List[(bank_idx, op_id, Optional[bytes])]
  - get_stats() -> Dict
"""

def _noop_cb(_: Optional[bytes]) -> None:
    return None

@dataclass(slots=True)
class SRAMOperation:
    op_id: int
    is_write: bool
    addr: int  # per-bank slot index
    data: Optional[bytes]
    length: int
    remaining_cycles: int
    due_cycle: int = 0
    callback: Optional[Callable[[Optional[bytes]], None]] = None


class SRAMBank(Clocked):
    def __init__(self, slots: int, read_latency: int = 1, write_latency: int = 1, queue_size: int = 1,
                 clock: Optional["SimClock"] = None):
        super().__init__()
        # A bank that the scheduler lets sleep stops observing cycles, so it
        # cannot derive "now" from its own last tick any more. The shared clock
        # is the only always-correct source. Falls back to self-timing when no
        # clock is supplied, which keeps standalone construction working.
        self._clock = clock
        self.slots = int(slots)
        # each slot holds an arbitrary-bytes payload (empty by default)
        self.mem: List[bytes] = [b"" for _ in range(self.slots)]
        self.read_latency = int(read_latency)
        self.write_latency = int(write_latency)

        # Use SimQueue for pending operations
        self._pending: SimQueue[SRAMOperation] = SimQueue(max_size=queue_size)
        self._ready_heap: List[Tuple[int, int, SRAMOperation]] = []
        self._op_counter = 0

        # Stall / utilization counters
        self.cycles_busy: int = 0
        self.enqueue_stalls: int = 0
        self._curr_tick: int = -1
        self._last_enqueue_tick: int = -2
        # First cycle of the current busy stretch. cycles_busy used to be
        # incremented once per tick while ops were outstanding; now that the
        # bank sleeps between a op's enqueue and its due cycle, the skipped
        # cycles have to be added back or utilisation silently under-reports.
        self._busy_from: int = 0

    def _enqueue_base_cycle(self) -> int:
        """The cycle latencies are measured from.

        Callers enqueue during the crossbar phase, when this bank last advanced
        at cycle-1; the original code read that off self._curr_tick. Deriving it
        from the shared clock gives the identical value while keeping it correct
        when the bank has been asleep and _curr_tick is stale.
        """
        if self._clock is None:
            return self._curr_tick
        return self._clock.cycle - 1

    def can_accept_enqueue(self) -> bool:
        return self._last_enqueue_tick != self._enqueue_base_cycle() and not self._pending.is_full()

    def next_wake(self, now: int) -> Optional[int]:
        """Banks are purely deadline-driven: nothing due, nothing to do."""
        if self._ready_heap:
            return self._ready_heap[0][0]
        return None

    def _check_bounds(self, slot: int, length: int):
        if slot < 0 or slot >= self.slots:
            raise IndexError(f"SRAMBank slot out of bounds: slot={slot} slots={self.slots}")
        # length is advisory: reads will return min(length, len(slot_data))

    def enqueue_read(self, slot: int, length: int, callback: Optional[Callable[[bytes], None]] = None) -> int:
        if not self.can_accept_enqueue():
            self.enqueue_stalls += 1
            if self._last_enqueue_tick == self._enqueue_base_cycle():
                raise RuntimeError("SRAMBank enqueue stall: multiple enqueues in same cycle")
            raise RuntimeError("SRAMBank enqueue stall: pending queue full")
        base = self._enqueue_base_cycle()
        if self._pending.is_empty():
            self._busy_from = base + 1
        self._last_enqueue_tick = base
        self._check_bounds(slot, length)
        op = SRAMOperation(
            op_id=self._op_counter + 1,
            is_write=False,
            addr=int(slot),
            data=None,
            length=int(length),
            remaining_cycles=self.read_latency,
            due_cycle=base + self.read_latency,
            callback=callback,
        )
        if not self._pending.enqueue(op):
            self.enqueue_stalls += 1
            raise RuntimeError("SRAMBank enqueue stall: pending queue full")
        heapq.heappush(self._ready_heap, (op.due_cycle, op.op_id, op))
        self._op_counter += 1
        self.request_wake(op.due_cycle)
        return self._op_counter

    def enqueue_write(self, slot: int, data: bytes, callback: Optional[Callable[[None], None]] = None) -> int:
        if not self.can_accept_enqueue():
            self.enqueue_stalls += 1
            if self._last_enqueue_tick == self._enqueue_base_cycle():
                raise RuntimeError("SRAMBank enqueue stall: multiple enqueues in same cycle")
            raise RuntimeError("SRAMBank enqueue stall: pending queue full")
        base = self._enqueue_base_cycle()
        if self._pending.is_empty():
            self._busy_from = base + 1
        self._last_enqueue_tick = base
        self._check_bounds(slot, len(data))
        op = SRAMOperation(
            op_id=self._op_counter + 1,
            is_write=True,
            addr=int(slot),
            data=bytes(data),
            length=len(data),
            remaining_cycles=self.write_latency,
            due_cycle=base + self.write_latency,
            callback=callback,
        )
        if not self._pending.enqueue(op):
            self.enqueue_stalls += 1
            raise RuntimeError("SRAMBank enqueue stall: pending queue full")
        heapq.heappush(self._ready_heap, (op.due_cycle, op.op_id, op))
        self._op_counter += 1
        self.request_wake(op.due_cycle)
        return self._op_counter

    def tick(self, time: Optional[float] = None) -> List[Tuple[int, Optional[bytes]]]:
        """
        Advance one cycle. Return list of completed operations as tuples
        (op_id, result) where result is bytes for reads or None for writes.
        Callbacks are invoked before returning.
        """
        prev_tick = self._curr_tick
        cycle = self._consume_tick(time, attr_name="_curr_tick")
        if cycle is None:
            return []

        completed: List[Tuple[int, Optional[bytes]]] = []

        # Account every cycle spent with an op outstanding, including any the
        # scheduler let us sleep through. Reduces to "+= 1" when ticked every
        # cycle, which is what the unscheduled simulator did.
        if self._pending:
            start = max(prev_tick + 1, self._busy_from)
            if start <= cycle:
                self.cycles_busy += cycle - start + 1

        while self._ready_heap and self._ready_heap[0][0] <= cycle:
            _due_cycle, _op_id, op = heapq.heappop(self._ready_heap)
            if not self._pending.remove(op):
                continue
            if op.is_write:
                # perform write: replace the slot contents with provided bytes
                self.mem[op.addr] = op.data or b""
                result = None
            else:
                # perform read: return up to requested length of slot contents
                slot_data = self.mem[op.addr]
                result = bytes(slot_data[: op.length])
            completed.append((op.op_id, result))
            cb = op.callback or _noop_cb
            try:
                cb(result)
            except Exception:
                raise RuntimeError(f"Callback is None at tick: {self._curr_tick}")

        return completed


def _xor_bank(abs_row: int, col_id: int, bank_count: int) -> int:
    """
    Emulator XOR mapping for lane->bank permutation.
    Works correctly when bank_count is a power of two.
    Fallbacks to modulo mapping otherwise.
    """
    if bank_count & (bank_count - 1) == 0:
        low = abs_row & (bank_count - 1)
        return (col_id ^ low) & (bank_count - 1)
    else:
        # non-power-of-two fallback: simple interleave (not emulator-perfect)
        return (abs_row * bank_count + col_id) % bank_count


class SRAMBanks(Clocked):
    def __init__(self, bank_count: int = 4, bank_size: int = 1024, read_latency: int = 2, write_latency: int = 2,
                 clock: Optional["SimClock"] = None):
        """
        bank_count: number of banks (B)
        bank_size: number of slots per bank (S)
        read_latency / write_latency: per-bank latencies (cycles)
        clock: shared SimClock, required for banks that the scheduler may
               let sleep. Omitting it keeps the old self-timed behaviour.
        """
        super().__init__()
        self.bank_count = int(bank_count)
        self.bank_size = int(bank_size)  # interpreted as slots per bank
        self.read_latency = int(read_latency)
        self.write_latency = int(write_latency)
        self._clock = clock

        # list of SRAMBank objects accessible as .banks (tests access this)
        self.banks: List[SRAMBank] = [
            SRAMBank(slots=self.bank_size, read_latency=self.read_latency, write_latency=self.write_latency,
                     clock=clock)
            for _ in range(self.bank_count)
        ]
        self._tick = -1

    def _addr_to_bank_slot(self, addr: int) -> Tuple[int, int]:
        """
        Map a linear address to (bank, slot) using emulator XOR semantics:
          abs_row = addr // bank_count
          col_id  = addr %  bank_count
          bank = xor(abs_row, col_id)
          slot = abs_row
        """
        if addr < 0:
            raise IndexError("Negative address")
        abs_row = addr // self.bank_count
        col_id = addr % self.bank_count
        slot = abs_row
        if slot >= self.bank_size:
            raise IndexError(f"Address out of range: addr={addr} -> slot={slot} >= bank_size={self.bank_size}")
        bank = _xor_bank(abs_row, col_id, self.bank_count)
        return bank, slot

    def enqueue_read(self, addr: int, length: int, callback: Optional[Callable[[bytes], None]] = None) -> Tuple[int, int]:
        """
        Enqueue a read at linear address 'addr'. Uses emulator XOR mapping to pick bank & slot.
        Read will return up to 'length' bytes from the slot's payload.
        Returns (bank_idx, op_id).
        """
        bank, slot = self._addr_to_bank_slot(addr)
        op_id = self.banks[bank].enqueue_read(slot, length, callback=callback)
        return bank, op_id

    def enqueue_write(self, addr: int, data: bytes, callback: Optional[Callable[[None], None]] = None) -> Tuple[int, int]:
        """
        Enqueue a write at linear address 'addr'. Uses emulator XOR mapping to pick bank & slot.
        The slot's payload is replaced with 'data' on completion.
        Returns (bank_idx, op_id).
        """
        bank, slot = self._addr_to_bank_slot(addr)
        op_id = self.banks[bank].enqueue_write(slot, data, callback=callback)
        return bank, op_id

    def tick(self, time: Optional[float] = None) -> List[Tuple[int, int, Optional[bytes]]]:
        """
        Return aggregated list of completed operations as
        tuples (bank_idx, op_id, result).
        """
        cycle = self._consume_tick(time, attr_name="_tick")
        if cycle is None:
            return []

        all_completed: List[Tuple[int, int, Optional[bytes]]] = []
        for b_idx, b in enumerate(self.banks):
            completed = b.tick(time)
            for op_id, result in completed:
                all_completed.append((b_idx, op_id, result))
        return all_completed

    def get_stats(self) -> Dict[str, Any]:
        total_cycles_busy = sum(b.cycles_busy for b in self.banks)
        total_enqueue_stalls = sum(b.enqueue_stalls for b in self.banks)
        per_bank = [
            {
                "bank": i,
                "cycles_busy": b.cycles_busy,
                "enqueue_stalls": b.enqueue_stalls,
                "queue_len": len(getattr(b, "_pending", [])),
            }
            for i, b in enumerate(self.banks)
        ]
        return {
            "total_cycles_busy": total_cycles_busy,
            "total_enqueue_stalls": total_enqueue_stalls,
            "per_bank": per_bank,
        }
