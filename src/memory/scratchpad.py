from typing import Callable, Optional, List, Any

from base.clocked_object import Clocked

from base.sched import SimClock, WakeGroup

from memory.sc_sram_banks import SRAMBanks, _xor_bank
from memory.crossbar import Xbar
from memory import backend
from memory.frontend import Frontend

class Scratchpad(Clocked):
    """
    Top-level software-managed Scratchpad composed of:
            - two tiles of SRAM banks (each tile: NUM_BANKS x BANK_SIZE slots)
            - per-tile crossbars for read/write (to perform swizzle / deswizzle)
            - ability to attach one Backend instance per tile/slot

    Notes / simplifications:
      - Backend/frontend write path: incoming logical row (sequence of elements) is
        split into per-lane element bytes, routed through the write Xbar using the
        XOR swizzle (bank = _xor_bank(slot, lane, num_banks)) and then each resulting
        lane is written into its bank's slot via the SRAMBanks bank interface.
      - We expose a simple frontend_write / attach_backend hook. Full cycle-accurate
        frontend read path (coordinating per-lane SRAMBanks enqueue_read and
        collecting completions) can be added similarly when needed.
    """

    def __init__(
        self,
        num_banks: int = 32,
        bank_size: int = 32,
        read_latency: int = 2,
        write_latency: int = 2,
        xbar_delay: int = 3,
        elem_bytes: int = 2,
        frontend_queue_size: int = 2,
    ):
        super().__init__()
        self.num_banks = int(num_banks)
        self.bank_size = int(bank_size)    # slots per bank (rows)
        self.elem_bytes = int(elem_bytes)
        self.now = 0
        self.backend_write_inflight = [0, 0]
        self.backend_read_inflight = [0, 0]

        # Shared view of the current cycle. Banks read this instead of their
        # own last-tick counter, which goes stale once they are allowed to
        # sleep through cycles where nothing is due.
        self.clock = SimClock()

        # two tiles (tile 0 and tile 1). each tile is a SRAMBanks instance
        self.tiles: List[SRAMBanks] = [
            SRAMBanks(bank_count=self.num_banks, bank_size=self.bank_size, read_latency=read_latency,
                      write_latency=write_latency, clock=self.clock)
            for _ in range(2)
        ]

        # per-tile crossbars (separate read/write per tile)
        self.tile_write_xbars: List[Xbar] = [
            Xbar(delay=xbar_delay, num_banks=self.num_banks) for _ in range(2)
        ]
        self.tile_read_xbars: List[Xbar] = [
            Xbar(delay=xbar_delay, num_banks=self.num_banks) for _ in range(2)
        ]

        # Optional per-tile backend references (set via attach_backend / attach_backends).
        self.backends: List[Optional[backend.Backend]] = [None, None]
        # Backward-compatible alias for the first attached backend.
        self.backend: Optional[backend.Backend] = None

        self.frontends = [
            Frontend(0, self, queue_size=frontend_queue_size),
            Frontend(1, self, queue_size=frontend_queue_size)
        ]

        # Wake groups, ticked in the order data flows: a frontend hands work to
        # a crossbar, which hands it to a bank. Because each group runs after
        # the one that feeds it, a producer can wake a consumer for the current
        # cycle and the consumer still runs on time.
        self._fe_group = WakeGroup("spad.frontends")
        self._xbar_group = WakeGroup("spad.xbars")
        self._bank_group = WakeGroup("spad.banks")
        for fe in self.frontends:
            self._fe_group.add(fe)
        for xb in self.tile_write_xbars + self.tile_read_xbars:
            self._xbar_group.add(xb)
        for tile in self.tiles:
            for b in getattr(tile, "banks", []):
                self._bank_group.add(b)
        self.wake_groups = [self._fe_group, self._xbar_group, self._bank_group]

    def _write_path_can_accept(self, tile_id: int) -> bool:
        tile_id = self._normalize_tile_id(tile_id)
        xbar = self.tile_write_xbars[tile_id]
        return int(self.backend_write_inflight[tile_id]) < int(xbar.max_size) and xbar.can_accept()

    def _read_path_can_accept(self, tile_id: int) -> bool:
        tile_id = self._normalize_tile_id(tile_id)
        xbar = self.tile_read_xbars[tile_id]
        return int(self.backend_read_inflight[tile_id]) < int(xbar.max_size) and xbar.can_accept()

    def _tile_and_slot(self, sp_addr: int) -> tuple[int, int]:
        """
        Simple linear split of address space into two equal tiles.
        sp_addr is a slot index (as produced by Backend: base_sp + row)
        """
        tile_sz = self.bank_size
        tile = 0 if sp_addr < tile_sz else 1
        slot = sp_addr % tile_sz
        return tile, slot

    def _normalize_tile_id(self, tile_id: int) -> int:
        tile_id = int(tile_id)
        if tile_id < 0 or tile_id >= len(self.tiles):
            raise ValueError(f"tile_id out of range: {tile_id}")
        return tile_id

    def _refresh_backend_alias(self) -> None:
        self.backend = self.backends[0]
        if self.backend is None:
            self.backend = next((be for be in self.backends if be is not None), None)

    def _accept_backend_write(self, sp_addr: int, row_bytes: bytes, row_idx: int, tx_id: int, tile_id: int = None, frontend_cb=None) -> bool:
        """
        Backend -> Scratchpad write path:
          - split row_bytes into elem_bytes lanes (pad with zeros to full NUM_BANKS)
          - compute shift_mask mapping lane -> bank using XOR swizzle
          - submit through tile's write Xbar; when xbar completes, write outputs into per-bank slot
        Returns True if the write was accepted (xbar submission succeeded).
        """
        if tile_id is None:
            tile_id, slot = self._tile_and_slot(sp_addr)
        else:
            tile_id = self._normalize_tile_id(tile_id)
            slot = sp_addr % self.bank_size
        tile = self.tiles[tile_id]
        xbar = self.tile_write_xbars[tile_id]

        # build lane inputs: one element per lane (NUM_BANKS). pad with zero bytes for lanes past cols.
        lanes: List[bytes] = []
        total_lanes = self.num_banks
        # compute number of cols from row_bytes length
        cols = (len(row_bytes) + self.elem_bytes - 1) // self.elem_bytes if self.elem_bytes else 0

        for lane in range(total_lanes):
            off = lane * self.elem_bytes
            if off < len(row_bytes):
                lanes.append(row_bytes[off : off + self.elem_bytes])
            else:
                lanes.append(b"\x00" * self.elem_bytes)

        # shift mask: lane i -> target bank = _xor_bank(slot, i, num_banks)
        shift_mask: List[Optional[int]] = [None] * total_lanes
        for i in range(total_lanes):
            shift_mask[i] = _xor_bank(slot, i, self.num_banks)

        # callback invoked when xbar routes lanes to bank-indexed output slots
        def _xbar_cb(routed_out: List[Any]) -> None:
            bank_writes = []
            for bank_idx, val in enumerate(routed_out):
                if not val:
                    continue
                if not tile.banks[bank_idx].can_accept_enqueue():
                    return False
                bank_writes.append((bank_idx, bytes(val)))

            for bank_idx, lane_bytes in bank_writes:
                tile.banks[bank_idx].enqueue_write(slot, lane_bytes)

            self.backend_write_inflight[tile_id] = max(0, int(self.backend_write_inflight[tile_id]) - 1)
            if frontend_cb:
                frontend_cb()
            return True

        # submit to xbar (operation queued). We don't block on xbar completion here.
        try:
            op_id = xbar.enqueue(shift_mask, lanes, callback=_xbar_cb)
        except Exception:
            return False
        if op_id == -1:
            return False
        self.backend_write_inflight[tile_id] = int(self.backend_write_inflight[tile_id]) + 1
        return True
    
    def attach_backends(self, backend_objs):
        if len(backend_objs) != len(self.tiles):
            raise ValueError(f"expected {len(self.tiles)} backends, got {len(backend_objs)}")
        attached = []
        for tile_id, backend_obj in enumerate(backend_objs):
            attached.append(self.attach_backend(backend_obj, tile_id=tile_id))
        return attached

    def attach_backend(self, backend_obj, tile_id: int = None):
        if isinstance(backend_obj, (list, tuple)):
            return self.attach_backends(list(backend_obj))

        if tile_id is None:
            for candidate, attached in enumerate(self.backends):
                if attached is None:
                    tile_id = candidate
                    break
            else:
                raise ValueError("all scratchpad backend slots are already occupied")

        tile_id = self._normalize_tile_id(tile_id)
        self.backends[tile_id] = backend_obj

        def _send_sram_write(sp_addr: int, row_bytes: bytes, row_idx: int, tx_id: int, _tile_id: int = tile_id) -> bool:
            return self._accept_backend_write(sp_addr, row_bytes, row_idx, tx_id, tile_id=_tile_id)

        def _send_sram_read(sp_addr: int, row_idx: int, tx_id: int, _tile_id: int = tile_id) -> bytes:
            return self._backend_read_row_for_tile(sp_addr, row_idx, tx_id, tile_id=_tile_id)

        backend_obj.send_sram_write = _send_sram_write
        backend_obj.send_sram_read = _send_sram_read
        self._refresh_backend_alias()
        return backend_obj

    def _backend_read_row_for_tile(self, sp_addr: int, row_idx: int, tx_id: int, tile_id: int = None) -> bytes:
        if tile_id is None:
            tile_id, slot = self._tile_and_slot(sp_addr)
        else:
            tile_id = self._normalize_tile_id(tile_id)
            slot = sp_addr % self.bank_size
        tile = self.tiles[tile_id]
        lanes: List[bytes] = []
        for lane in range(self.num_banks):
            bank = _xor_bank(slot, lane, self.num_banks)
            blob = tile.banks[bank].mem[slot]
            lane_bytes = bytes(blob) if blob is not None else b""
            if len(lane_bytes) < self.elem_bytes:
                lane_bytes = lane_bytes + (b"\x00" * (self.elem_bytes - len(lane_bytes)))
            lanes.append(lane_bytes[: self.elem_bytes])
        return b"".join(lanes)

    def backend_read_row(self, sp_addr: int, row_idx: int, tx_id: int) -> bytes:
        return self._backend_read_row_for_tile(sp_addr, row_idx, tx_id)

    # minimal frontend helpers (write uses same swizzle path)
    def frontend_write(self, base_sp_addr: int, row_bytes: bytes, row_idx: int, tile_id: int = None) -> bool:
        """
        Frontend initiates a write into scratchpad (row-major).
        This enqueues the write in the appropriate frontend for arbitration.
        """
        if tile_id is None:
            tile_id, _ = self._tile_and_slot(base_sp_addr)
        return self.frontends[tile_id].write(base_sp_addr, row_bytes, row_idx)

    def _accept_backend_read(self, sp_addr: int, row_idx: int, tx_id: int, tile_id: int = None, frontend_cb=None) -> bool:
        """
        Backend/Frontend -> Scratchpad read path:
          - gather per-bank slot data for the row
          - deswizzle using XOR mapping
          - call frontend_cb with the list of lane bytes
        """
        if tile_id is None:
            tile_id, slot = self._tile_and_slot(sp_addr)
        else:
            tile_id = self._normalize_tile_id(tile_id)
            slot = sp_addr % self.bank_size
        tile = self.tiles[tile_id]
        xbar = self.tile_read_xbars[tile_id]

        # Gather per-bank data (bank order)
        per_bank = []
        for bank in range(self.num_banks):
            val = tile.banks[bank].mem[slot]
            per_bank.append(val if val is not None else b"\x00" * self.elem_bytes)

        # Callback after crossbar delay
        def _xbar_cb(routed_out: List[Any]) -> None:
            self.backend_read_inflight[tile_id] = max(0, int(self.backend_read_inflight[tile_id]) - 1)
            if frontend_cb:
                # For each lane, get the value from the bank where it was stored
                unswizzled = [routed_out[_xor_bank(slot, lane, self.num_banks)] for lane in range(self.num_banks)]
                frontend_cb(unswizzled)
        try:
            op_id = xbar.enqueue(list(range(self.num_banks)), per_bank, callback=_xbar_cb)
        except Exception:
            return False
        if op_id == -1:
            return False
        self.backend_read_inflight[tile_id] = int(self.backend_read_inflight[tile_id]) + 1
        return True

    # tick() to advance internal xbars and banks; call this from simulator each cycle
    def tick(self, time=None) -> None:
        now = time if time is not None else getattr(self, 'now', 0)
        self.now = now
        self.clock.advance_to(int(now))
        # These loops used to be wrapped in blanket `except Exception: pass`,
        # which silently discarded any failure inside the memory pipeline.
        # Nothing was actually being swallowed, and hiding errors here makes
        # the scheduling changes undebuggable, so the handlers are gone.
        self._fe_group.tick(now)
        self._xbar_group.tick(now)
        self._bank_group.tick(now)

    def get_stats(self) -> dict:
        stats = {
            "tiles": [],
            "backend_slots": [
                {"tile": tid, "attached": be is not None}
                for tid, be in enumerate(self.backends)
            ],
            "frontend_stalls": [
                {"tile": tid, "write_stalled": fe.write_stalled, "read_stalled": fe.read_stalled}
                for tid, fe in enumerate(self.frontends)
            ]
        }
        for tid, tile in enumerate(self.tiles):
            per = {"tile": tid, "banks": []}
            for i, b in enumerate(getattr(tile, "banks", [])):
                per["banks"].append({"bank": i, "cycles_busy": getattr(b, "cycles_busy", 0), "enqueue_stalls": getattr(b, "enqueue_stalls", 0), "queue_len": len(getattr(b, "_pending", []))})
            stats["tiles"].append(per)
        return stats
