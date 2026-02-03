from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple, Any

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
class BACKENDTransaction:
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
        self._tx_queue: Deque[BACKENDTransaction] = deque()
        self._next_tx_id = 1

        # active transactions (tx_id -> transaction)
        self._active_txs: Dict[int, BACKENDTransaction] = {}

        # callback to send a completed SRAM write vector to Body
        self.send_sram_write = send_sram_write

        # statistics
        self.total_dram_bursts_issued = 0
        self.total_dram_bursts_completed = 0
        self.total_backend_stalls = 0  # increments when dram_q full and we cannot issue
        self.total_tx_completed = 0

    # Public API for scheduler/driver
    def start_load(
        self, base_sp_addr: int, base_dram_addr: int, rows: int, cols: int, callback: Optional[Callable[[int], None]] = None
    ) -> int:
        """
        Start a LOAD transaction: DRAM -> Scratchpad.
        Returns tx_id.
        """
        tx_id = self._next_tx_id
        self._next_tx_id += 1

        subreqs = math.ceil((cols * self.elem_bytes) / self.dram_burst_bytes) if cols > 0 else 0
        tx = BACKENDTransaction(
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
        self._tx_queue.append(tx)
        return tx_id

    #DEBUG queue should be a different thing, separate queue class
    def start_store(
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
        tx = BACKENDTransaction(
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
        self._tx_queue.append(tx)
        return tx_id

    # Called by Body/other unit when it produces a SRAM-read response for a store transaction.
    def accept_sram_read_response(self, tx_id: int, row_idx: int, row_bytes: bytes) -> None:
        """
        Accepts a row-worth of bytes from the Scratchpad (for STORE).
        Splits into DRAM write subrequests and enqueue them (subject to DRAM queue depth).
        """
        tx = self._active_txs.get(tx_id)
        if not tx:
            # If tx not active yet, try enqueueing into tx_queue (unlikely)
            return
        if not tx.is_store:
            return

        # split row_bytes into sub-chunks of dram_burst_bytes
        subreqs = []
        for s in range(tx.subreqs_per_row):
            off = s * self.dram_burst_bytes
            chunk = row_bytes[off : off + self.dram_burst_bytes]
            dram_addr = tx.base_dram + (row_idx * tx.cols * self.elem_bytes) + off
            subreqs.append(DRAMOperation(tx_id=tx.tx_id, row=row_idx, subidx=s, dram_addr=dram_addr, length=len(chunk), is_write=True, data=chunk, remaining_cycles=self.dram_latency))

        # try to push them into dram_pending, if cannot, keep them in per-tx buffer and count stalls
        for req in subreqs:
            if len(self._dram_pending._raw_items) >= self.dram_q_depth: #DEBUG to be handled by queues
                # cannot accept now; count stall and keep remaining subreqs in tx.row_bufs as pending write data
                self.total_backend_stalls += 1
                tx.row_bufs[row_idx][req.subidx] = req.data  # store for later
            else:
                if not self._dram_pending.enqueue(req):
                    self.total_backend_stalls += 1
                    # Optionally, buffer the request for retry if needed
                    # For loads: mark as pending in tx.issued_subreqs or tx.row_bufs
                    # For stores: keep in tx.row_bufs
                    return
                self.total_dram_bursts_issued += 1

    # Internal helpers
    def _issue_row_load_subreqs(self, tx: BACKENDTransaction) -> None:
        """
        Enqueue DRAM read subrequests for tx.cur_row if dram queue has capacity.
        """
        r = tx.cur_row
        # nothing to issue if 0 width
        if tx.subreqs_per_row == 0:
            # immediate "empty" row -> send empty sram write
            self._complete_row_load(tx, r, b"")
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

    def _complete_row_load(self, tx: BACKENDTransaction, row: int, row_bytes: bytes) -> None:
        # attempt to send to Body via callback
        sp_addr = tx.base_sp + row  # slot-oriented addressing: slot = base_sp + row
        accepted = True
        if self.send_sram_write:
            try:
                accepted = bool(self.send_sram_write(sp_addr, row_bytes, row, tx.tx_id))
            except Exception:
                accepted = False

        if not accepted:
            # Body is stalled; keep the row in buffer so we can retry later
            # store assembled row in row_bufs so we can issue later
            tx.row_bufs[row] = [row_bytes]  # mark as pending assembled row
            self.total_backend_stalls += 1
            return

        # successfully handed off row to Body
        tx.cur_row += 1
        # clear per-row buffers to free memory
        if row < len(tx.row_bufs):
            tx.row_bufs[row] = [None] * tx.subreqs_per_row

        # if there are more rows to issue, schedule their DRAM subreqs immediately (subject to queue)
        if tx.cur_row < tx.rows:
            self._issue_row_load_subreqs(tx)
        else:
            # transaction complete
            self.total_tx_completed += 1
            if tx.callback:
                try:
                    tx.callback(tx.tx_id)
                except Exception:
                    pass
            # remove from active list
            if tx.tx_id in self._active_txs:
                del self._active_txs[tx.tx_id]

    # Simulated DRAM response handler (internal)
    def _on_dram_response(self, req: DRAMOperation) -> None:
        # produce deterministic payload for loads (for emulator correctness tests user can override)
        if not req.is_write:
            # simulate real data: pattern bytes = tx_id,row,subidx repeated
            payload = bytes([req.tx_id & 0xFF, req.row & 0xFF, req.subidx & 0xFF]) * ((req.length + 2) // 3)
            payload = payload[: req.length]
            tx = self._active_txs.get(req.tx_id)
            if tx:
                tx.row_bufs[req.row][req.subidx] = payload
                # check if row is complete
                if all(x is not None for x in tx.row_bufs[req.row]):
                    # assemble row bytes
                    assembled = b"".join(tx.row_bufs[req.row])
                    # trim to expected row size
                    expected = tx.cols * self.elem_bytes
                    assembled = assembled[:expected]
                    # try to handoff
                    self._complete_row_load(tx, req.row, assembled)
                else:
                    # Try to issue more subreqs for this row if queue space is available
                    self._issue_row_load_subreqs(tx)
        else:
            # DRAM write completed; nothing else required for now
            self.total_dram_bursts_completed += 1
            # Check for store transaction completion
            tx = self._active_txs.get(req.tx_id)
            if tx and tx.is_store:
                # If all bursts for all rows are done, complete the transaction
                all_done = True
                for row_buf in tx.row_bufs:
                    if any(chunk is not None for chunk in row_buf):
                        all_done = False
                        break
                if all_done and not self._dram_pending and tx.tx_id in self._active_txs:
                    self.total_tx_completed += 1
                    if tx.callback:
                        try:
                            tx.callback(tx.tx_id)
                        except Exception:
                            pass
                    del self._active_txs[tx.tx_id]

    # ------ Clock tick ------
    def tick(self, time=None) -> None:
        """
        Advance one cycle
        Progress DRAM pending bursts, move tx from queue to active,
        issue new row subreqs for newly active transactions, and process DRAM completions.
        """
        # 1) Move queued transactions into active (start as many as capacity allows)
        while self._tx_queue:
            tx = self._tx_queue[0]
            self._tx_queue.popleft()
            self._active_txs[tx.tx_id] = tx
            if not tx.is_store:
                self._issue_row_load_subreqs(tx)
            else:
                # For store: request all rows from scratchpad if not already present
                if self.send_sram_read:
                    for row_idx in range(tx.rows):
                        # Only request if not already present
                        if all(x is None for x in tx.row_bufs[row_idx]):
                            row_bytes = self.send_sram_read(tx.base_sp + row_idx, row_idx, tx.tx_id)
                            self.accept_sram_read_response(tx.tx_id, row_idx, row_bytes)

        # 2) Progress dram pending bursts
        n = len(self._dram_pending._raw_items)
        to_requeue = []
        for _ in range(n):
            req = self._dram_pending.dequeue()
            req.remaining_cycles -= 1
            if req.remaining_cycles <= 0:
                # DRAM response arrives this cycle
                self._on_dram_response(req)
                self.total_dram_bursts_completed += 1 if req.is_write else 0
            else:
                to_requeue.append(req)
        for req in to_requeue:
            self._dram_pending.enqueue(req)

        # 3) Retry handing off any assembled rows that were previously stalled:
        for tx in list(self._active_txs.values()):
            # some rows may have been assembled and stored in row_bufs as a single-element list
            if tx.cur_row < tx.rows:
                buf_entry = tx.row_bufs[tx.cur_row]
                # case where we stored assembled row as single element for a stalled handoff
                if len(buf_entry) == 1 and isinstance(buf_entry[0], (bytes, bytearray)):
                    assembled = buf_entry[0]
                    # try to hand off again
                    self._complete_row_load(tx, tx.cur_row, assembled)

    def get_stats(self) -> Dict[str, Any]:
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
