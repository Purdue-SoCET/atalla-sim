from dataclasses import dataclass
from typing import Callable, Optional, List, Dict, Tuple, Any

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

@dataclass
class SRAMOperation:
    op_id: int
    is_write: bool
    addr: int  # per-bank slot index
    data: Optional[bytes]
    length: int
    remaining_cycles: int
    callback: Optional[Callable[[Optional[bytes]], None]] = None


class SRAMBank(Clocked):
    def __init__(self, slots: int, read_latency: int = 1, write_latency: int = 1, queue_size: int = 1):
        super().__init__()
        self.slots = int(slots)
        # each slot holds an arbitrary-bytes payload (empty by default)
        self.mem: List[bytes] = [b"" for _ in range(self.slots)]
        self.read_latency = int(read_latency)
        self.write_latency = int(write_latency)

        # Use SimQueue for pending operations
        self._pending: SimQueue[SRAMOperation] = SimQueue(max_size=queue_size)
        self._op_counter = 0

        # Stall / utilization counters
        self.cycles_busy: int = 0
        self.enqueue_stalls: int = 0

    def _check_bounds(self, slot: int, length: int):
        if slot < 0 or slot >= self.slots:
            raise IndexError(f"SRAMBank slot out of bounds: slot={slot} slots={self.slots}")
        # length is advisory: reads will return min(length, len(slot_data))

    # DEBUGGAR: self._curr_tick updated at every tick. if two jobs try to happen at the same tick, raise flag
    def enqueue_read(self, slot: int, length: int, callback: Optional[Callable[[bytes], None]] = None) -> Optional[int]:
        self._check_bounds(slot, length)
        if not self._pending.enqueue(SRAMOperation(
            op_id=self._op_counter + 1,
            is_write=False,
            addr=int(slot),
            data=None,
            length=int(length),
            remaining_cycles=self.read_latency,
            callback=callback,
        )):
            self.enqueue_stalls += 1 # DEBUGGAR: THE OTHER UNIT IS STALLING, NOT SRAM BANK
            return None
        self._op_counter += 1
        return self._op_counter

    def enqueue_write(self, slot: int, data: bytes, callback: Optional[Callable[[None], None]] = None) -> Optional[int]:
        self._check_bounds(slot, len(data))
        if not self._pending.enqueue(SRAMOperation(
            op_id=self._op_counter + 1,
            is_write=True,
            addr=int(slot),
            data=bytes(data),
            length=len(data),
            remaining_cycles=self.write_latency,
            callback=callback,
        )):
            self.enqueue_stalls += 1 # DEBUGGAR: THE OTHER UNIT IS STALLING, NOT SRAM BANK
            return None
        self._op_counter += 1
        return self._op_counter

    def tick(self) -> List[Tuple[int, Optional[bytes]]]:
        """
        Advance one cycle. Return list of completed operations as tuples
        (op_id, result) where result is bytes for reads or None for writes.
        Callbacks are invoked before returning.
        """
        completed: List[Tuple[int, Optional[bytes]]] = []

        # account busy cycle if there are pending ops at start of tick
        if self._pending:
            self.cycles_busy += 1

        # Decrement remaining_cycles for all pending ops (FIFO/order preserved)
        for op in self._pending.items:
            op.remaining_cycles -= 1

        # Collect completed ops (those with remaining_cycles <= 0), in order
        to_remove = []
        for op in self._pending.items:
            if op.remaining_cycles <= 0:
                if op.is_write:
                    # perform write: replace the slot contents with provided bytes
                    self.mem[op.addr] = op.data or b""
                    result = None
                else:
                    # perform read: return up to requested length of slot contents
                    slot_data = self.mem[op.addr]
                    result = bytes(slot_data[: op.length])
                completed.append((op.op_id, result))
                # invoke callback (swallow exceptions to avoid breaking sim)
                # DEBUGGAR: it NEEDS to have a callback
                if op.callback:
                    try:
                        op.callback(result)
                    except Exception:
                        pass
                to_remove.append(op)

        # remove completed ops from pending queue
        for op in to_remove:
            self._pending._items.remove(op)

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
    def __init__(self, bank_count: int = 4, bank_size: int = 1024, read_latency: int = 2, write_latency: int = 2):
        """
        bank_count: number of banks (B)
        bank_size: number of slots per bank (S)
        read_latency / write_latency: per-bank latencies (cycles)
        """
        super().__init__()
        self.bank_count = int(bank_count)
        self.bank_size = int(bank_size)  # interpreted as slots per bank
        self.read_latency = int(read_latency)
        self.write_latency = int(write_latency)

        # list of SRAMBank objects accessible as .banks (tests access this)
        self.banks: List[SRAMBank] = [
            SRAMBank(slots=self.bank_size, read_latency=self.read_latency, write_latency=self.write_latency)
            for _ in range(self.bank_count)
        ]

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

    def tick(self) -> List[Tuple[int, int, Optional[bytes]]]:
        """
        Return aggregated list of completed operations as
        tuples (bank_idx, op_id, result).
        """
        all_completed: List[Tuple[int, int, Optional[bytes]]] = []
        for b_idx, b in enumerate(self.banks):
            completed = b.tick()
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