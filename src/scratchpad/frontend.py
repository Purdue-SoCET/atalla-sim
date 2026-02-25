from typing import Optional, Callable, List, Any
from base.queue import SimQueue

class Frontend:
    def __init__(self, tile_id: int, spad: "Scratchpad", queue_size: int = 1):
        self.tile_id = tile_id
        self.spad = spad
        self.writeq = SimQueue(max_size=queue_size)  # (ready_cycle, base_sp_addr, row_bytes, row_idx, callback)
        self.readq = SimQueue(max_size=queue_size)   # (ready_cycle, base_sp_addr, row_idx, callback)
        self.pending_reads = {}            # tx_id -> callback
        self.write_stalled = False
        self.read_stalled = False

    def write(self, base_sp_addr: int, row_bytes: bytes, row_idx: int, callback: Optional[Callable]=None):
        now = getattr(self.spad, 'now', 0)
        latency = 2 + self.spad.tile_write_xbars[self.tile_id].delay
        ready_cycle = now + latency
        req = (ready_cycle, base_sp_addr, row_bytes, row_idx, callback)
        if not self.writeq.enqueue(req):
            self.write_stalled = True
            return False
        self.write_stalled = False
        return True

    def read(self, base_sp_addr: int, row_idx: int, callback: Callable[[List[bytes]], None]):
        now = getattr(self.spad, 'now', 0)
        latency = 2 + self.spad.tile_read_xbars[self.tile_id].delay
        ready_cycle = now + latency
        req = (ready_cycle, base_sp_addr, row_idx, callback)
        if not self.readq.enqueue(req):
            self.read_stalled = True
            return False
        self.read_stalled = False
        return True

    def tick(self, now):
        # Write path: only if backend is not using the crossbar
        if not self.spad.backend_write_inflight[self.tile_id]:
            head = self.writeq.peek()
            while head:
                ready_cycle, *_ = head
                if ready_cycle > now:
                    break
                _, base_sp_addr, row_bytes, row_idx, cb = self.writeq.dequeue()
                self.spad._accept_backend_write(base_sp_addr, row_bytes, row_idx, tx_id=0, tile_id=self.tile_id, frontend_cb=cb)
                head = self.writeq.peek()
        # Read path: only if backend is not using the crossbar
        if not self.spad.backend_read_inflight[self.tile_id]:
            head = self.readq.peek()
            while head:
                ready_cycle, *_ = head
                if ready_cycle > now:
                    break
                _, base_sp_addr, row_idx, cb = self.readq.dequeue()
                self.spad._accept_backend_read(base_sp_addr, row_idx, tx_id=0, tile_id=self.tile_id, frontend_cb=cb)
                head = self.readq.peek()
