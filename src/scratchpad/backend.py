from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any
from collections import deque

from base.clocked_object import Clocked
from base.queue import SimQueue

import math


@dataclass
class DRAMOperation:
    tx_id: int
    row: int
    subidx: int
    dram_addr: int
    length: int
    is_write: bool
    data: Optional[bytes]
    remaining_cycles: int


@dataclass
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

        # outstanding DRAM bursts being serviced by the (simulated) DRAM
        self._dram_pending: SimQueue[DRAMOperation] = SimQueue(max_size=self.dram_q_depth)

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
        )
        if not self._tx_queue.enqueue(tx):
            self.total_backend_stalls += 1
            return -1
        return tx_id

    #TODO queue should be a different thing, separate queue class
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
        if self.is_busy():
            self.total_backend_stalls += 1
            self._pending_sram_reads.append((tx_id, row_idx, row_bytes))
            return
        self._accept_sram_read_response(tx_id, row_idx, row_bytes)

    def _accept_sram_read_response(self, tx_id: int, row_idx: int, row_bytes: bytes) -> None:
        self.last_op_tick = self._tick
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

        # try to push them into dram_pending, if cannot, keep them in per-tx buffer and count stalls
        pending = self._dram_pending
        row_buf = tx.row_bufs[row_idx]
        for req in subreqs:
            if not pending.enqueue(req):
                # cannot accept now; count stall and keep remaining subreqs in tx.row_bufs as pending write data
                self.total_backend_stalls += 1
                row_buf[req.subidx] = req.data  # store for later
                continue
            self.total_dram_bursts_issued += 1

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

        # Issue only subreqs that have not been issued yet
        for s in range(tx.subreqs_per_row):
            if s in tx.issued_subreqs[r]:
                continue
            # Try to enqueue every subrequest, count a stall for each failure
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
            if not self._dram_pending.enqueue(req):
                self.total_backend_stalls += 1
                # Do not return; keep trying to enqueue the rest
            else:
                tx.issued_subreqs[r].add(s)
                self.total_dram_bursts_issued += 1

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
            # Simulate deterministic payload for loads
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
            # DRAM write completed
            self.total_dram_bursts_completed += 1
            # Check for store transaction completion
            tx = self._active_txs.get(req.tx_id)
            if tx and tx.is_store:
                # Check if all bursts for all rows are done
                all_done = all(
                    all(chunk is None for chunk in row_buf)
                    for row_buf in tx.row_bufs
                )
                if all_done and not self._dram_pending and tx.tx_id in self._active_txs:
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
            if cycle <= self._tick:
                return
            self._tick = cycle
        if self._pending_sram_reads and not self.is_busy():
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

        # 2) Progress DRAM pending bursts
        n = len(self._dram_pending._raw_items)
        to_requeue = []
        for _ in range(n):
            req = self._dram_pending.dequeue()
            req.remaining_cycles -= 1
            if req.remaining_cycles <= 0:
                # DRAM response arrives this cycle
                self.dram_to_backend_on_response(req)
                self.total_dram_bursts_completed += 1 if req.is_write else 0
            else:
                to_requeue.append(req)
        for req in to_requeue:
            self._dram_pending.enqueue(req)

        # 3) Retry handing off stalled assembled rows
        for tx in list(self._active_txs.values()):
            # some rows may have been assembled and stored in row_bufs as a single-element list
            if tx.cur_row < tx.rows:
                buf_entry = tx.row_bufs[tx.cur_row]
                # case where we stored assembled row as single element for a stalled handoff
                if len(buf_entry) == 1 and isinstance(buf_entry[0], (bytes, bytearray)):
                    assembled = buf_entry[0]
                    # try to hand off again
                    self.backend_to_body_complete_row_load(tx, tx.cur_row, assembled)

    def backend_to_driver_get_stats(self) -> Dict[str, Any]:
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
