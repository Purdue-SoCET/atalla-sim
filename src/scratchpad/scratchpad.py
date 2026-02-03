from typing import Callable, Optional, List, Any

from base.clocked_object import Clocked

from scratchpad.sc_sram_banks import SRAMBanks, _xor_bank
from scratchpad.crossbar import Xbar
from scratchpad import backend as backend_mod
from scratchpad.frontend import ScratchpadFrontend

class Scratchpad(Clocked):
    """
    Top-level software-managed Scratchpad composed of:
      - two tiles of SRAM banks (each tile: NUM_BANKS x BANK_SIZE slots)
      - per-tile crossbars for read/write (to perform swizzle / deswizzle)
      - ability to attach a Backend instance (Backend will call send_sram_write)

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
        self.backend_write_inflight = [False, False]
        self.backend_read_inflight = [False, False]

        # two tiles (tile 0 and tile 1). each tile is a SRAMBanks instance
        self.tiles: List[SRAMBanks] = [
            SRAMBanks(bank_count=self.num_banks, bank_size=self.bank_size, read_latency=read_latency, write_latency=write_latency)
            for _ in range(2)
        ]

        # per-tile crossbars (separate read/write per tile)
        self.tile_write_xbars: List[Xbar] = [
            Xbar(delay=xbar_delay, num_banks=self.num_banks) for _ in range(2)
        ]
        self.tile_read_xbars: List[Xbar] = [
            Xbar(delay=xbar_delay, num_banks=self.num_banks) for _ in range(2)
        ]

        # optional backend reference (set via attach_backend)
        self.backend: Optional[backend_mod.Backend] = None

        self.frontends = [
            ScratchpadFrontend(0, self, queue_size=frontend_queue_size),
            ScratchpadFrontend(1, self, queue_size=frontend_queue_size)
        ]

    def _tile_and_slot(self, sp_addr: int) -> tuple[int, int]:
        """
        Simple linear split of address space into two equal tiles.
        sp_addr is a slot index (as produced by Backend: base_sp + row)
        """
        tile_sz = self.bank_size
        tile = 0 if sp_addr < tile_sz else 1
        slot = sp_addr % tile_sz
        return tile, slot

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
            slot = sp_addr % self.bank_size
        tile = self.tiles[tile_id]
        xbar = self.tile_write_xbars[tile_id]
        self.backend_write_inflight[tile_id] = True

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
            self.backend_write_inflight[tile_id] = False
            for bank_idx, val in enumerate(routed_out):
                if not val:
                    continue
                try:
                    tile.banks[bank_idx].enqueue_write(slot, bytes(val))
                except Exception as e:
                    linear_addr = slot * self.num_banks + bank_idx
                    try:
                        tile.enqueue_write(linear_addr, bytes(val))
                    except Exception as e2:
                        pass
            if frontend_cb:
                frontend_cb()

        # submit to xbar (operation queued). We don't block on xbar completion here.
        try:
            xbar.enqueue(shift_mask, lanes, callback=_xbar_cb)
        except Exception:
            return False
        return True
    
    def attach_backend(self, backend):
        self.backend = backend
        backend.send_sram_write = self._accept_backend_write

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
            slot = sp_addr % self.bank_size
        tile = self.tiles[tile_id]
        xbar = self.tile_read_xbars[tile_id]
        self.backend_read_inflight[tile_id] = True

        # Gather per-bank data (bank order)
        per_bank = []
        for bank in range(self.num_banks):
            val = tile.banks[bank].mem[slot]
            per_bank.append(val if val is not None else b"\x00" * self.elem_bytes)

        # Callback after crossbar delay
        def _xbar_cb(routed_out: List[Any]) -> None:
            self.backend_read_inflight[tile_id] = False
            if frontend_cb:
                # For each lane, get the value from the bank where it was stored
                unswizzled = [routed_out[_xor_bank(slot, lane, self.num_banks)] for lane in range(self.num_banks)]
                frontend_cb(unswizzled)
        try:
            xbar.enqueue(list(range(self.num_banks)), per_bank, callback=_xbar_cb)
        except Exception:
            return False
        return True

    # tick() to advance internal xbars and banks; call this from simulator each cycle
    def tick(self, time=None) -> None:
        now = time if time is not None else getattr(self, 'now', 0)
        try:
            for tid, fe in enumerate(self.frontends):
                fe.tick(now)
            for xb in self.tile_write_xbars + self.tile_read_xbars:
                xb.tick()
            for tile in self.tiles:
                for b in getattr(tile, "banks", []):
                    try:
                        b.tick()
                    except Exception as e:
                        pass
        except Exception as e:
            pass

    def get_stats(self) -> dict:
        stats = {
            "tiles": [],
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