from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Tuple
from base.clocked_object import Clocked

"""
- SRAMOperation: internal representation of a pending read/write
- SRAMBank: a single bank backed by a bytearray and with latency
- SRAMBanks: collection of banks with simple enqueue_read/enqueue_write APIs.
    Operations are processed on calls to tick()
"""

@dataclass
class SRAMOperation:
        op_id: int
        is_write: bool
        addr: int
        data: Optional[bytes]  # for write: bytes to write; for read: None
        length: int  # for read: number of bytes; for write: len(data)
        remaining_cycles: int
        callback: Optional[Callable[[Optional[bytes]], None]] = None


class SRAMBank(Clocked):
        """
        Single SRAM bank

        - size: number of bytes in bank
        - read_latency / write_latency: latency in cycles for operations
        - tick(): advance 1 cycle and complete ready operations
        """

        def __init__(self, size: int, read_latency: int = 1, write_latency: int = 1):
                super().__init__()

                self.size = int(size)
                self.mem = bytearray(self.size)
                self.read_latency = int(read_latency)
                self.write_latency = int(write_latency)

                self._op_counter = 0
                self._pending: List[SRAMOperation] = []

                # Stall accounting
                self.cycles_busy: int = 0         # number of cycles the bank had any pending ops
                self.enqueue_stalls: int = 0      # number of enqueue calls that found the bank busy

        def _check_bounds(self, addr: int, length: int):
                if addr < 0 or length < 0 or addr + length > self.size:
                        raise IndexError(f"Access out of bounds: addr={addr} len={length} size={self.size}")

        def enqueue_read(self, addr: int, length: int,
                                         callback: Optional[Callable[[bytes], None]] = None) -> int:
                """Enqueue a read. Returns operation id."""
                self._check_bounds(addr, length)
                # count an enqueue stall if there are already pending operations
                if self._pending:
                        self.enqueue_stalls += 1
                self._op_counter += 1
                op = SRAMOperation(
                        op_id=self._op_counter,
                        is_write=False,
                        addr=int(addr),
                        data=None,
                        length=int(length),
                        remaining_cycles=self.read_latency,
                        callback=callback,
                )
                self._pending.append(op)
                return op.op_id

        def enqueue_write(self, addr: int, data: bytes,
                                            callback: Optional[Callable[[None], None]] = None) -> int:
                """Enqueue a write. Returns operation id."""
                length = len(data)
                self._check_bounds(addr, length)
                # count an enqueue stall if there are already pending operations
                if self._pending:
                        self.enqueue_stalls += 1
                self._op_counter += 1
                op = SRAMOperation(
                        op_id=self._op_counter,
                        is_write=True,
                        addr=int(addr),
                        data=bytes(data),
                        length=length,
                        remaining_cycles=self.write_latency,
                        callback=callback,
                )
                self._pending.append(op)
                return op.op_id

        def tick(self) -> List[Tuple[int, Optional[bytes]]]:
                """
                Returns list of completed operations as tuples
                (op_id, read_data_or_None). For writes the returned data is None.
                Callbacks are invoked (if provided).
                """
                completed: List[Tuple[int, Optional[bytes]]] = []

                # account busy cycle if there are pending ops at start of tick
                if self._pending:
                        self.cycles_busy += 1

                for op in list(self._pending):
                        op.remaining_cycles -= 1
                        if op.remaining_cycles <= 0:
                                if op.is_write:
                                        # perform write
                                        self.mem[op.addr:op.addr + op.length] = op.data  # type: ignore
                                        result = None
                                else:
                                        # perform read
                                        result = bytes(self.mem[op.addr:op.addr + op.length])
                                completed.append((op.op_id, result))
                                if op.callback:
                                        try:
                                                op.callback(result)
                                        except Exception:
                                                # callbacks should not break simulation; swallow exceptions
                                                pass
                                self._pending.remove(op)
                return completed


class SRAMBanks(Clocked):
        """
        Collection of banks. Exposes a simple address mapping:
            linear address -> bank_id = (addr // bank_stride) % bank_count
            bank_offset = addr % bank_stride

        You can override mapping by providing a custom mapper function.

        Public API:
            enqueue_read(addr, length, callback=None) -> (bank_id, op_id)
            enqueue_write(addr, data, callback=None) -> (bank_id, op_id)
            tick() -> list of (bank_id, op_id, result)
        """

        def __init__(self,
                    bank_count: int,
                    bank_size: int,
                    read_latency: int = 1,
                    write_latency: int = 1,
                    mapper: Optional[Callable[[int], Tuple[int, int]]] = None):
                super().__init__()

                self.bank_count = int(bank_count)
                self.bank_size = int(bank_size)
                self.banks: List[SRAMBank] = [
                        SRAMBank(size=bank_size, read_latency=read_latency, write_latency=write_latency)
                        for _ in range(self.bank_count)
                ]

                # default mapper: bank = (addr // bank_size) % bank_count, offset = addr % bank_size
                if mapper is None:
                        def mapper_fn(addr: int) -> Tuple[int, int]:
                                bank = (addr // self.bank_size) % self.bank_count
                                offset = addr % self.bank_size
                                return bank, offset
                        self.mapper = mapper_fn
                else:
                        self.mapper = mapper

        def enqueue_read(self, addr: int, length: int,
                                         callback: Optional[Callable[[bytes], None]] = None) -> Tuple[int, int]:
                bank_id, offset = self.mapper(int(addr))
                # If the read crosses bank boundary, split across banks
                if offset + length <= self.bank_size:
                        op_id = self.banks[bank_id].enqueue_read(offset, length, callback)
                        return bank_id, op_id

                # split into multiple ops across successive banks (simple linear mapping)
                remaining = length
                cur_addr = addr
                first_op_id = None
                callbacks = []

                # create a composite callback that gathers pieces and calls the user's callback
                pieces: Dict[int, bytes] = {}

                def make_piece_callback(piece_idx: int, piece_len: int, user_cb):
                        def piece_cb(data: Optional[bytes]):
                                pieces[piece_idx] = data or b''
                                if user_cb:
                                        # when all pieces arrive, assemble and call user's callback
                                        if sum(len(v) for v in pieces.values()) == length:
                                                assembled = b''.join(pieces[i] for i in sorted(pieces.keys()))
                                                user_cb(assembled)
                        return piece_cb

                piece_idx = 0
                while remaining > 0:
                        bank_id, offset = self.mapper(cur_addr)
                        take = min(remaining, self.bank_size - offset)
                        cb = make_piece_callback(piece_idx, take, callback)
                        op_id = self.banks[bank_id].enqueue_read(offset, take, cb)
                        if first_op_id is None:
                                first_op_id = op_id
                        callbacks.append((bank_id, op_id))
                        remaining -= take
                        cur_addr += take
                        piece_idx += 1

                return (-1 if first_op_id is None else callbacks[0][0], first_op_id or -1)

        def enqueue_write(self, addr: int, data: bytes,
                                            callback: Optional[Callable[[None], None]] = None) -> Tuple[int, int]:
                bank_id, offset = self.mapper(int(addr))
                length = len(data)
                if offset + length <= self.bank_size:
                        op_id = self.banks[bank_id].enqueue_write(offset, data, callback)
                        return bank_id, op_id

                # split write across banks
                remaining = length
                cur_addr = addr
                src_idx = 0
                first_op_id = None

                # create a coordinating callback that invokes user's callback when all parts complete
                pending_parts = {"count": 0}
                total_parts = 0

                # precompute parts
                parts = []
                while remaining > 0:
                        b_id, off = self.mapper(cur_addr)
                        take = min(remaining, self.bank_size - off)
                        parts.append((b_id, off, data[src_idx:src_idx+take]))
                        remaining -= take
                        cur_addr += take
                        src_idx += take
                        total_parts += 1

                pending_parts["count"] = total_parts

                def make_part_callback():
                        def part_cb(_):
                                pending_parts["count"] -= 1
                                if pending_parts["count"] == 0 and callback:
                                        try:
                                                callback(None)
                                        except Exception:
                                                pass
                        return part_cb

                part_cb = make_part_callback()
                for (b_id, off, chunk) in parts:
                        op_id = self.banks[b_id].enqueue_write(off, chunk, part_cb)
                        if first_op_id is None:
                                first_op_id = op_id

                return (-1 if first_op_id is None else parts[0][0], first_op_id or -1)

        def tick(self) -> List[Tuple[int, int, Optional[bytes]]]:
                completed = []
                for i, bank in enumerate(self.banks):
                        for op_id, res in bank.tick():
                                completed.append((i, op_id, res))
                return completed

        def get_stats(self) -> Dict[str, object]:
                """Return aggregate stall statistics and per-bank details."""
                total_cycles_busy = sum(b.cycles_busy for b in self.banks)
                total_enqueue_stalls = sum(b.enqueue_stalls for b in self.banks)
                per_bank = [
                        {"bank": i, "cycles_busy": b.cycles_busy, "enqueue_stalls": b.enqueue_stalls, "queue_len": len(b._pending)}
                        for i, b in enumerate(self.banks)
                ]
                return {
                        "total_cycles_busy": total_cycles_busy,
                        "total_enqueue_stalls": total_enqueue_stalls,
                        "per_bank": per_bank,
                }