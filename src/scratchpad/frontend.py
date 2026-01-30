from typing import Optional, Callable, List, Any
import collections

class ScratchpadFrontend:
    def __init__(self, tile_id: int, spad: "Scratchpad"):
        self.tile_id = tile_id
        self.spad = spad
        self.writeq = collections.deque()  # (ready_cycle, base_sp_addr, row_bytes, row_idx, callback)
        self.readq = collections.deque()   # (ready_cycle, base_sp_addr, row_idx, callback)
        self.pending_reads = {}            # tx_id -> callback

    def write(self, base_sp_addr: int, row_bytes: bytes, row_idx: int, callback: Optional[Callable]=None):
        now = getattr(self.spad, 'now', 0)
        latency = 2 + self.spad.tile_write_xbars[self.tile_id].delay
        ready_cycle = now + latency
        self.writeq.append((ready_cycle, base_sp_addr, row_bytes, row_idx, callback))

    def read(self, base_sp_addr: int, row_idx: int, callback: Callable[[List[bytes]], None]):
        now = getattr(self.spad, 'now', 0)
        latency = 2 + self.spad.tile_read_xbars[self.tile_id].delay
        ready_cycle = now + latency
        self.readq.append((ready_cycle, base_sp_addr, row_idx, callback))

    def tick(self, now):
        # Write path: only if backend is not using the crossbar
        if not self.spad.backend_write_inflight[self.tile_id]:
            while self.writeq and self.writeq[0][0] <= now:
                _, base_sp_addr, row_bytes, row_idx, cb = self.writeq.popleft()
                self.spad._accept_backend_write(base_sp_addr, row_bytes, row_idx, tx_id=0, tile_id=self.tile_id, frontend_cb=cb)
        # Read path: only if backend is not using the crossbar
        if not self.spad.backend_read_inflight[self.tile_id]:
            while self.readq and self.readq[0][0] <= now:
                _, base_sp_addr, row_idx, cb = self.readq.popleft()
                self.spad._accept_backend_read(base_sp_addr, row_idx, tx_id=0, tile_id=self.tile_id, frontend_cb=cb)