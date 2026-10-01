"""A pad's frontend: the VLSU side of its request port.

frontend.sv passes vec_req straight to the pad's head and returns tail's
responses; it holds no queue of its own. So this is a thin adapter over the
scratchpad's per-pad port, and `can_accept` is the RTL's !fe_vec_stall.
"""

from typing import Callable, List, Optional


class Frontend:
    def __init__(self, tile_id: int, spad: "Scratchpad"):
        self.tile_id = int(tile_id)
        self.spad = spad

    @property
    def stalls(self) -> int:
        return self.spad.tiles[self.tile_id].stalls

    def can_accept(self, now: Optional[int] = None) -> bool:
        return self.spad.can_accept(self.tile_id, now)

    def write(self, base_sp_addr: int, row_bytes: bytes, row_idx: int = 0,
              callback: Optional[Callable[[], None]] = None,
              now: Optional[int] = None) -> bool:
        return self.spad.submit_write(self.tile_id, base_sp_addr, row_bytes,
                                      callback=callback, now=now)

    def read(self, base_sp_addr: int, row_idx: int,
             callback: Callable[[List[bytes]], None],
             now: Optional[int] = None) -> bool:
        return self.spad.submit_read(self.tile_id, base_sp_addr, callback, now=now)
