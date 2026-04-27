from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
from collections import deque
import heapq

from base.clocked_object import Clocked
from base.queue import SimQueue

import math

from memory.dram import DRAM


@dataclass(slots=True)
class DRAMOperation:
    tx_id: int
    row: int
    subidx: int
    dram_addr: int
    length: int
    is_write: bool
    data: Optional[bytes]
    remaining_cycles: int
    due_cycle: int = 0


@dataclass(slots=True)
class BackendTransaction:
    tx_id: int
    base_sp: int
    base_dram: int
    rows: int
    cols: int
    cur_row: int = 0
    subreqs_per_row: int = 0
    # per-row buffers: list of list of sub-chunks (bytes), size = rows x subreqs_per_row
    row_bufs: List[List[Optional[bytes]]] = field(default_factory=list)
    callback: Optional[Callable[[int], None]] = None
    is_store: bool = False  # store = SP->DRAM (writeback), load = DRAM->SP
    issued_subreqs: List[set] = field(default_factory=list)
    total_subreqs: int = 0
    completed_subreqs: int = 0


@dataclass
class SharedDRAMBurstChannel:
    """One shared DRAM-facing burst launch slot across multiple backends."""

    last_issue_tick: int = -1

    def can_issue(self, tick: int) -> bool:
        return int(tick) != int(self.last_issue_tick)

    def reserve(self, tick: int) -> None:
        self.last_issue_tick = int(tick)


class Backend(Clocked):
    """
    Responsibilities:
      - Accept high-level Load/Store transactions (row-major only for now).
      - Split each row into DRAM sub-requests (burst-sized).
      - Simulate a limited DRAM request queue and DRAM latency.
      - Assemble DRAM responses into SRAM-write vectors and invoke `send_sram_write`.
      - Accept SRAM-read responses (for stores) and split them into DRAM write requests.

    Key constructor params:
      dram_latency: cycles to return a DRAM response for each burst
      dram_q_depth: maximum outstanding DRAM bursts
      dram_burst_bytes: bytes per DRAM request (hardware width)
      elem_bytes: bytes per element (default 2 for 16-bit)
      send_sram_write: callback(spad_addr: int, row_bytes: bytes, row_idx:int, tx_id:int) -> bool
                       should return True if accepted, False if backend must stall that write.
    """

    def __init__(
        self,
        dram_latency: int = 6,
        dram_q_depth: int = 32,
        dram_burst_bytes: int = 8,
        elem_bytes: int = 2,
        delay_cycles: int = 1,
        shared_burst_channel: Optional[SharedDRAMBurstChannel] = None,
        send_sram_write: Optional[Callable[[int, bytes, int, int], bool]] = None,
        send_sram_read: Optional[Callable[[int, int, int], bytes]] = None,
    ):
        super().__init__()
        self.dram_latency = int(dram_latency)
        self.dram_q_depth = int(dram_q_depth)
        self.dram_burst_bytes = int(dram_burst_bytes)
        self.elem_bytes = int(elem_bytes)
        self.send_sram_write = send_sram_write
        self.send_sram_read = send_sram_read
        self.dram: Optional[DRAM] = None
        self.shared_burst_channel = shared_burst_channel

        # outstanding DRAM bursts being serviced by the (simulated) DRAM
        self._dram_pending: SimQueue[DRAMOperation] = SimQueue(max_size=self.dram_q_depth)
        self._dram_ready_heap: List[Tuple[int, int, int, int, DRAMOperation]] = []

        # queue of transactions waiting to be started
        self._tx_queue: SimQueue[BackendTransaction] = SimQueue(max_size=self.dram_q_depth)
        self._next_tx_id = 1

        # active transactions (tx_id -> transaction)
        self._active_txs: Dict[int, BackendTransaction] = {}

        # callback to send a completed SRAM write vector to Body
        self.send_sram_write = send_sram_write

        # statistics
        self.total_dram_bursts_issued = 0
        self.total_dram_bursts_completed = 0
        self.total_backend_stalls = 0  # increments when dram_q full and we cannot issue
        self.total_tx_completed = 0
        self.delay_cycles = int(delay_cycles)
        self.last_op_tick = -1
        self._tick = -1
        self._pending_sram_reads = deque()
        self._in_tick = False

    def _dram_bus_available(self) -> bool:
        # Model a split-transaction DRAM-facing burst channel:
        # - multiple bursts may remain in flight at once
        # - but only one new burst may launch every delay_cycles cycles
        # - outstanding concurrency is still bounded by dram_q_depth
        if self._dram_pending.is_full() or self.is_busy():
            return False
        if self.shared_burst_channel is not None and not self.shared_burst_channel.can_issue(self._tick):
            return False
        return True

    def _enqueue_dram_burst(self, req: DRAMOperation) -> bool:
        if not self._dram_bus_available():
            self.total_backend_stalls += 1
            return False
        if not self._dram_pending.enqueue(req):
            self.total_backend_stalls += 1
            return False
        reference_cycle = self._tick if self._in_tick else (self._tick + 1)
        req.due_cycle = reference_cycle + self.dram_latency - 1
        self.last_op_tick = self._tick
        if self.shared_burst_channel is not None:
            self.shared_burst_channel.reserve(self._tick)
        heapq.heappush(
            self._dram_ready_heap,
            (req.due_cycle, req.tx_id, req.row, req.subidx, req),
        )
        self.total_dram_bursts_issued += 1
        return True

    def attach_scratchpad(self, scratchpad: Any, tile_id: Optional[int] = None) -> Any:
        if hasattr(scratchpad, "attach_backend"):
            scratchpad.attach_backend(self, tile_id=tile_id)
            return scratchpad
        self.send_sram_write = scratchpad._accept_backend_write
        self.send_sram_read = scratchpad.backend_read_row
        if getattr(scratchpad, "backend", None) is not self:
            scratchpad.backend = self
        return scratchpad

    def attach_dram(self, dram: DRAM) -> DRAM:
        self.dram = dram
        return dram

    def is_busy(self, now: Optional[int] = None) -> bool:
        if self.delay_cycles <= 0:
            return False
        cur = self._tick if now is None else int(now)
        return (cur - self.last_op_tick) < self.delay_cycles

    # Public API for scheduler/driver
    def driver_to_backend_start_load(
        self, base_sp_addr: int, base_dram_addr: int, rows: int, cols: int, callback: Optional[Callable[[int], None]] = None
    ) -> int:
        """
        Start a LOAD transaction: DRAM -> Scratchpad.
        Returns tx_id.
        """
        tx_id = self._next_tx_id
        self._next_tx_id += 1

        subreqs = math.ceil((cols * self.elem_bytes) / self.dram_burst_bytes) if cols > 0 else 0
        tx = BackendTransaction(
            tx_id=tx_id,
            base_sp=base_sp_addr,
            base_dram=base_dram_addr,
            rows=rows,
            cols=cols,
            cur_row=0,
            subreqs_per_row=subreqs,
            row_bufs=[[None] * subreqs for _ in range(rows)],
            callback=callback,
            is_store=False,
            issued_subreqs=[set() for _ in range(rows)],
            total_subreqs=rows * subreqs,
        )
        if not self._tx_queue.enqueue(tx):
            self.total_backend_stalls += 1
            return -1
        return tx_id

    def driver_to_backend_start_store(
        self, base_sp_addr: int, base_dram_addr: int, rows: int, cols: int, callback: Optional[Callable[[int], None]] = None
    ) -> int:
        """
        Start a STORE transaction: Scratchpad -> DRAM (writeback).
        Backend expects to receive SRAM read responses via accept_sram_read_response().
        Returns tx_id.
        """
        tx_id = self._next_tx_id
        self._next_tx_id += 1

        subreqs = math.ceil((cols * self.elem_bytes) / self.dram_burst_bytes) if cols > 0 else 0
        tx = BackendTransaction(
            tx_id=tx_id,
            base_sp=base_sp_addr,
            base_dram=base_dram_addr,
            rows=rows,
            cols=cols,
            cur_row=0,
            subreqs_per_row=subreqs,
            row_bufs=[[None] * subreqs for _ in range(rows)],
            callback=callback,
            is_store=True,
            issued_subreqs=[set() for _ in range(rows)],
            total_subreqs=rows * subreqs,
        )
        if not self._tx_queue.enqueue(tx):
            self.total_backend_stalls += 1
            return -1
        return tx_id

    # Called by Body/other unit when it produces a SRAM-read response for a store transaction.
    def body_to_backend_sram_read_response(self, tx_id: int, row_idx: int, row_bytes: bytes) -> None:
        """
        Accepts a row-worth of bytes from the Scratchpad (for STORE).
        Splits into DRAM write subrequests and enqueue them (subject to DRAM queue depth).
        """
        self._accept_sram_read_response(tx_id, row_idx, row_bytes)

    def _accept_sram_read_response(self, tx_id: int, row_idx: int, row_bytes: bytes) -> None:
        tx = self._active_txs.get(tx_id)
        if not tx:
            # If tx not active yet, try enqueueing into tx_queue (unlikely)
            return
        if not tx.is_store:
            return

        # split row_bytes into sub-chunks of dram_burst_bytes
        subreqs = []
        for subidx in range(tx.subreqs_per_row):
            off = subidx * self.dram_burst_bytes
            chunk = row_bytes[off : off + self.dram_burst_bytes]
            dram_addr = tx.base_dram + (row_idx * tx.cols * self.elem_bytes) + off
            subreqs.append(DRAMOperation(
                            tx_id=tx.tx_id, 
                            row=row_idx, 
                            subidx=subidx, 
                            dram_addr=dram_addr, 
                            length=len(chunk), 
                            is_write=True, 
                            data=chunk, 
                            remaining_cycles=self.dram_latency))

        # Try to launch one burst immediately if the DRAM command channel can
        # accept it this cycle; keep the remaining chunks parked for later retry.
        row_buf = tx.row_bufs[row_idx]
        issued_now = False
        for req in subreqs:
            if not issued_now and self._enqueue_dram_burst(req):
                issued_now = True
                continue
            if row_buf[req.subidx] is None:
                # Cannot launch now; keep the chunk so a later tick can issue it.
                row_buf[req.subidx] = req.data
                self.total_backend_stalls += 1
            else:
                self.total_backend_stalls += 1

    # Internal helpers
    def backend_to_dram_issue_row_load_subreqs(self, tx: BackendTransaction) -> None:
        """
        Enqueue DRAM read subrequests for tx.cur_row if dram queue has capacity.
        """
        r = tx.cur_row
        # nothing to issue if 0 width
        if tx.subreqs_per_row == 0:
            # immediate "empty" row -> send empty sram write
            self.backend_to_body_complete_row_load(tx, r, b"")
            return

        # Issue only one new burst per call, but allow many outstanding bursts
        # to remain in flight concurrently.
        for s in range(tx.subreqs_per_row):
            if s in tx.issued_subreqs[r]:
                continue
            req = DRAMOperation(
                tx_id=tx.tx_id,
                row=r,
                subidx=s,
                dram_addr=tx.base_dram + (r * tx.cols * self.elem_bytes) + s * self.dram_burst_bytes,
                length=min(self.dram_burst_bytes, tx.cols * self.elem_bytes - s * self.dram_burst_bytes),
                is_write=False,
                data=None,
                remaining_cycles=self.dram_latency,
            )
            if self._enqueue_dram_burst(req):
                tx.issued_subreqs[r].add(s)
            else:
                pass
            return

    def _retry_one_deferred_store_burst(self) -> None:
        for tx in list(self._active_txs.values()):
            if not tx.is_store:
                continue
            for row_idx, row_buf in enumerate(tx.row_bufs):
                for subidx, chunk in enumerate(row_buf):
                    if chunk is None:
                        continue
                    req = DRAMOperation(
                        tx_id=tx.tx_id,
                        row=row_idx,
                        subidx=subidx,
                        dram_addr=tx.base_dram + (row_idx * tx.cols * self.elem_bytes) + subidx * self.dram_burst_bytes,
                        length=len(chunk),
                        is_write=True,
                        data=chunk,
                        remaining_cycles=self.dram_latency,
                    )
                    if self._enqueue_dram_burst(req):
                        row_buf[subidx] = None
                    else:
                        pass
                    return

    def _retry_one_deferred_load_burst(self) -> None:
        for tx in list(self._active_txs.values()):
            if tx.is_store or tx.cur_row >= tx.rows:
                continue
            issued_before = len(tx.issued_subreqs[tx.cur_row])
            self.backend_to_dram_issue_row_load_subreqs(tx)
            if len(tx.issued_subreqs[tx.cur_row]) > issued_before:
                return

    def backend_to_body_complete_row_load(self, tx: BackendTransaction, row: int, row_bytes: bytes) -> None:
        """
        Attempt to send an assembled row to the Body via callback.
        If Body stalls, buffer the row for retry.
        """
        sp_addr = tx.base_sp + row  # slot-oriented addressing
        accepted = True

        if self.send_sram_write:
            try:
                accepted = bool(self.send_sram_write(sp_addr, row_bytes, row, tx.tx_id))
            except Exception:
                accepted = False

        if not accepted:
            # Body is stalled; buffer row for retry
            tx.row_bufs[row] = [row_bytes]
            self.total_backend_stalls += 1
            return

        # Row successfully handed off
        tx.cur_row += 1
        # Clear per-row buffers
        if row < len(tx.row_bufs):
            tx.row_bufs[row] = [None] * tx.subreqs_per_row

        # Issue next row or complete transaction
        if tx.cur_row < tx.rows:
            self.backend_to_dram_issue_row_load_subreqs(tx)
        else:
            # transaction complete
            self.total_tx_completed += 1
            if tx.callback:
                try:
                    tx.callback(tx.tx_id)
                except Exception:
                    pass
            self._active_txs.pop(tx.tx_id, None)

    # Simulated DRAM response handler (internal)
    def dram_to_backend_on_response(self, req: DRAMOperation) -> None:
        """
        Handle DRAM response for a burst.
        For loads: assemble row, handoff to Body if complete.
        For stores: check for transaction completion.
        """
        if not req.is_write:
            if self.dram is not None:
                payload = self.dram.read(req.dram_addr, req.length)
            else:
                # Fallback payload for older tests that do not attach a DRAM.
                payload = bytes([req.tx_id & 0xFF, req.row & 0xFF, req.subidx & 0xFF]) * ((req.length + 2) // 3)
                payload = payload[:req.length]
            tx = self._active_txs.get(req.tx_id)
            if tx:
                tx.row_bufs[req.row][req.subidx] = payload
                # If row is complete, assemble and handoff
                if all(x is not None for x in tx.row_bufs[req.row]):
                    assembled = b"".join(tx.row_bufs[req.row])
                    expected = tx.cols * self.elem_bytes
                    assembled = assembled[:expected]
                    self.backend_to_body_complete_row_load(tx, req.row, assembled)
                else:
                    # Try to issue more subreqs if queue space is available
                    self.backend_to_dram_issue_row_load_subreqs(tx)
        else:
            if self.dram is not None and req.data is not None:
                self.dram.write(req.dram_addr, req.data)
            # DRAM write completed
            # Check for store transaction completion
            tx = self._active_txs.get(req.tx_id)
            if tx and tx.is_store:
                tx.completed_subreqs += 1
                all_buffered_chunks_drained = all(
                    all(chunk is None for chunk in row_buf)
                    for row_buf in tx.row_bufs
                )
                if (
                    tx.completed_subreqs >= tx.total_subreqs
                    and all_buffered_chunks_drained
                    and tx.tx_id in self._active_txs
                ):
                    self.total_tx_completed += 1
                    if tx.callback:
                        try:
                            tx.callback(tx.tx_id)
                        except Exception:
                            pass
                    self._active_txs.pop(tx.tx_id, None)

    # ------ Clock tick ------
    def tick(self, time=None) -> None:
        """
        Advance one cycle:
          - Move queued transactions to active
          - Issue new row subreqs for active transactions
          - Progress DRAM pending bursts
          - Retry stalled row handoffs
        """
        if time is None:
            self._tick += 1
        else:
            cycle = int(time)
            # Snap near-integer times to the nearest integer to avoid float drift.
            if isinstance(time, float):
                rounded = int(round(time))
                if abs(time - rounded) < 1e-6:
                    cycle = rounded
            if cycle < self._tick:
                # Allow time reset between independent simulations.
                self._tick = cycle - 1
            if cycle <= self._tick:
                return
            self._tick = cycle
        self._in_tick = True
        try:
            if self._pending_sram_reads:
                tx_id, row_idx, row_bytes = self._pending_sram_reads.popleft()
                self._accept_sram_read_response(tx_id, row_idx, row_bytes)
            # 1) Activate queued transactions
            while not self._tx_queue.is_empty():
                tx = self._tx_queue.dequeue()
                self._active_txs[tx.tx_id] = tx
                if not tx.is_store:
                    self.backend_to_dram_issue_row_load_subreqs(tx)
                else:
                    # For store: request all rows from scratchpad if not present
                    if self.send_sram_read:
                        for row_idx in range(tx.rows):
                            if all(x is None for x in tx.row_bufs[row_idx]):
                                row_bytes = self.send_sram_read(tx.base_sp + row_idx, row_idx, tx.tx_id)
                                self.body_to_backend_sram_read_response(tx.tx_id, row_idx, row_bytes)

            # 2) Progress only the bursts whose due cycle has arrived.
            while self._dram_ready_heap and self._dram_ready_heap[0][0] <= self._tick:
                _due, _tx_id, _row, _subidx, req = heapq.heappop(self._dram_ready_heap)
                if not self._dram_pending.remove(req):
                    continue
                self.dram_to_backend_on_response(req)
                self.total_dram_bursts_completed += 1

            # 3) Retry any deferred DRAM load/store bursts once the serialized bus is free.
            self._retry_one_deferred_load_burst()
            self._retry_one_deferred_store_burst()

            # 4) Retry handing off stalled assembled rows.
            for tx in list(self._active_txs.values()):
                if tx.cur_row < tx.rows:
                    buf_entry = tx.row_bufs[tx.cur_row]
                    if len(buf_entry) == 1 and isinstance(buf_entry[0], (bytes, bytearray)):
                        assembled = buf_entry[0]
                        self.backend_to_body_complete_row_load(tx, tx.cur_row, assembled)
        finally:
            self._in_tick = False

    def get_stats(self) -> Dict[str, Any]:
        """
        Return backend statistics.
        """
        return {
            "dram_q_depth": self.dram_q_depth,
            "dram_pending": len(self._dram_pending._raw_items),
            "outstanding_txs": len(self._active_txs),
            "queued_txs": len(self._tx_queue),
            "issued_bursts": self.total_dram_bursts_issued,
            "completed_bursts": self.total_dram_bursts_completed,
            "backend_stalls": self.total_backend_stalls,
            "tx_completed": self.total_tx_completed,
        }

    # Backwards-compatible stats API used by some tests.
    def backend_to_driver_get_stats(self) -> Dict[str, Any]:
        return self.get_stats()
