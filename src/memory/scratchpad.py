"""The scratchpad: independent pads of SRAM banks, timed like the RTL.

rtl/modules/memory/scratchpad/ (atalla, transpose_integration b1ba35ff). Per
pad the RTL is

    frontend --+
               +-- head --> wxbar --> scpad_cntrl --> 32 x sram_bank --> rxbar --> tail
    backend  --+   (1/cycle,   (FIFO)   (read FIFO,      (read 2,          (FIFO)
                    backend             write FIFO,       write 2)
                    first)              32 deep each)

and there is no crossbar: wxbar and rxbar are FIFO pass-throughs, so lane i
of a row lives in bank i at slot `row`, and every bank of a pad takes part in
every row access. This model keeps that behaviour at the level the rest of
the simulator sees -- when a request is accepted, when it reaches the banks,
when its data comes back -- without modelling each FIFO:

  * A pad accepts one request per cycle, from its frontend or its backend
    (head). It stalls both when either controller queue is full.
  * A request reaches the banks no sooner than INGRESS_CYCLES (2) after it is
    accepted.
  * Reads and writes are separate channels (scpad_cntrl's two FIFOs), each a
    row-wide sram_bank: one enable when the channel is not busy, then busy
    until done -- done_delay(2) = 3 cycles -- so each direction does at most
    one row every 3 cycles, and the two run side by side.
  * The banks are read and written on the enable's edge; a read and a write
    of one row on the same edge read the old row.
  * Read data reaches the requester EGRESS_CYCLES (2) after done: 5 cycles
    after the enable, 7 after acceptance for an uncontended read.

Measured against a Questa run of the RTL scratchpad -- single reads and
writes, back-to-back streams of each, interleaved traffic, read-after-write,
and backend DMA loads and stores -- this reproduces every accept, bank enable
and response cycle; tests/memory/test_scratchpad_rtl_timing.py replays them.
What it approximates: when the frontend and the backend both offer a request
in one cycle, the RTL grants the backend; here the first caller of the cycle
wins (the frontend's bridge runs earlier in the cycle). Queue capacity counts
requests in flight to the controller as well as those queued in it.

Callers pass `now`, the cycle they are in: the VLSU bridges run before the
scratchpad ticks, so self.now is still the previous cycle when they call.
"""

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, List, Optional, Tuple

from base.clocked_object import Clocked
from memory import backend
from memory.frontend import Frontend
from memory.sram_bank import SramChannel

#: head + wxbar: accepted to bank enable, at the earliest.
INGRESS_CYCLES = 2
#: rxbar + tail: read done to data at the requester.
EGRESS_CYCLES = 2
#: scpad_cntrl.sv FIFO_DEPTH = NUM_COLS.
QUEUE_DEPTH = 32


def bank_of_lane(lane: int) -> int:
    """No crossbar, no swizzle: lane i of a row is in bank i."""
    return int(lane)


class _Bank:
    """One bank's storage: `mem[slot]` holds that lane's element bytes."""

    def __init__(self, slots: int):
        self.mem: List[bytes] = [b"" for _ in range(int(slots))]


@dataclass
class _Request:
    write: bool
    slot: int
    lanes: Optional[List[bytes]]
    ready: int
    callback: Optional[Callable] = None


@dataclass
class _Pad:
    banks: List[_Bank]
    read: SramChannel
    write: SramChannel
    depth: int
    rq: Deque[_Request] = field(default_factory=deque)
    wq: Deque[_Request] = field(default_factory=deque)
    read_free: int = 0
    write_free: int = 0
    accepted_at: int = -1
    #: (due cycle, seq, callback, payload) -- reads in flight to the requester
    out: List[Tuple[int, int, Callable, Any]] = field(default_factory=list)
    reads: int = 0
    writes: int = 0
    stalls: int = 0

    def idle(self) -> bool:
        return not (self.rq or self.wq or self.out)


class Scratchpad(Clocked):

    def __init__(
        self,
        num_banks: int = 32,
        bank_size: int = 32,
        read_latency: int = 2,
        write_latency: int = 2,
        elem_bytes: int = 2,
        queue_depth: int = QUEUE_DEPTH,
        num_tiles: int = 2,
    ):
        super().__init__()
        self.num_banks = int(num_banks)
        self.bank_size = int(bank_size)    # slots per bank (rows per pad)
        self.elem_bytes = int(elem_bytes)
        self.read_latency = int(read_latency)
        self.write_latency = int(write_latency)
        #: Independent pads, each with its own frontend and backend, so one pad
        #: per VLSU gives every load/store unit a private path to memory.
        self.num_tiles = int(num_tiles)
        if self.num_tiles < 1:
            raise ValueError("num_tiles must be >= 1, got %s" % num_tiles)
        self.now = 0
        self._seq = 0
        #: Optional callable taking {"kind": "read"|"write", "tile", "slot",
        #: "cycle"} whenever a pad enables its banks.
        self.trace_hook: Optional[Callable[[dict], None]] = None

        self.tiles: List[_Pad] = [
            _Pad(banks=[_Bank(self.bank_size) for _ in range(self.num_banks)],
                 read=SramChannel(read_latency), write=SramChannel(write_latency),
                 depth=max(1, int(queue_depth)))
            for _ in range(self.num_tiles)
        ]
        self.backends: List[Optional[backend.Backend]] = [None] * self.num_tiles
        #: Backward-compatible alias for the first attached backend.
        self.backend: Optional[backend.Backend] = None
        self.frontends = [Frontend(tile_id, self) for tile_id in range(self.num_tiles)]

    # -- geometry ----------------------------------------------------------------
    @property
    def tile_bytes(self) -> int:
        """Capacity of one pad."""
        return self.num_banks * self.bank_size * self.elem_bytes

    @property
    def total_bytes(self) -> int:
        """Capacity of the whole scratchpad -- the chip's local memory."""
        return self.tile_bytes * self.num_tiles

    def _normalize_tile_id(self, tile_id: int) -> int:
        tile_id = int(tile_id)
        if tile_id < 0 or tile_id >= len(self.tiles):
            raise ValueError(f"tile_id out of range: {tile_id}")
        return tile_id

    def _tile_and_slot(self, sp_addr: int, tile_id: Optional[int]) -> Tuple[int, int]:
        """A named pad addresses from zero. Without one, the address space is
        split linearly into num_tiles pads -- the backend/DMA fallback."""
        if tile_id is not None:
            return self._normalize_tile_id(tile_id), int(sp_addr) % self.bank_size
        tile, slot = divmod(int(sp_addr), self.bank_size)
        if tile < 0 or tile >= self.num_tiles:
            raise ValueError("scratchpad address %s falls outside %d pads of %d slots"
                             % (sp_addr, self.num_tiles, self.bank_size))
        return tile, slot

    def _lanes(self, row_bytes: bytes) -> List[bytes]:
        """Split a row into per-lane element bytes. Only the lanes the row
        covers are written, as the RTL's valid_mask does."""
        eb = self.elem_bytes
        n = min(self.num_banks, (len(row_bytes) + eb - 1) // eb) if eb else 0
        return [bytes(row_bytes[i * eb:(i + 1) * eb]).ljust(eb, b"\x00") for i in range(n)]

    def _cycle(self, now: Optional[int]) -> int:
        return int(self.now) if now is None else int(now)

    # -- the request port (head) ---------------------------------------------------
    def can_accept(self, tile_id: int, now: Optional[int] = None) -> bool:
        """head's grant: one request per cycle, none while either controller
        queue is full (w_stall stalls both directions)."""
        pad = self.tiles[self._normalize_tile_id(tile_id)]
        return (pad.accepted_at != self._cycle(now)
                and len(pad.rq) < pad.depth and len(pad.wq) < pad.depth)

    def submit_write(self, tile_id: int, slot: int, row_bytes: bytes,
                     callback: Optional[Callable[[], None]] = None,
                     now: Optional[int] = None) -> bool:
        """Accept a row write this cycle, or refuse it (the caller retries).
        `callback` fires when the banks' write done is visible."""
        return self._submit(tile_id, True, slot, self._lanes(row_bytes), callback, now)

    def submit_read(self, tile_id: int, slot: int,
                    callback: Callable[[List[bytes]], None],
                    now: Optional[int] = None) -> bool:
        """Accept a row read this cycle, or refuse it. `callback(lanes)` gets
        one bytes object per bank when the data reaches the requester."""
        return self._submit(tile_id, False, slot, None, callback, now)

    def _submit(self, tile_id, write, slot, lanes, callback, now) -> bool:
        tile_id = self._normalize_tile_id(tile_id)
        pad = self.tiles[tile_id]
        cycle = self._cycle(now)
        if not self.can_accept(tile_id, cycle):
            pad.stalls += 1
            return False
        req = _Request(write=write, slot=int(slot) % self.bank_size, lanes=lanes,
                       ready=cycle + INGRESS_CYCLES, callback=callback)
        (pad.wq if write else pad.rq).append(req)
        pad.accepted_at = cycle
        self.request_wake(cycle)
        return True

    # -- legacy entry points ---------------------------------------------------------
    def frontend_write(self, base_sp_addr: int, row_bytes: bytes, row_idx: int,
                       tile_id: Optional[int] = None, now: Optional[int] = None) -> bool:
        tile_id, slot = self._tile_and_slot(base_sp_addr, tile_id)
        return self.submit_write(tile_id, slot, row_bytes, now=now)

    def read_row_now(self, sp_addr: int, tile_id: Optional[int] = None) -> bytes:
        """The row as stored, with no timing -- for checks and debug only."""
        tile_id, slot = self._tile_and_slot(sp_addr, tile_id)
        eb = self.elem_bytes
        return b"".join(bytes(b.mem[slot] or b"").ljust(eb, b"\x00")[:eb]
                        for b in self.tiles[tile_id].banks)

    def backend_read_row(self, sp_addr: int, row_idx: int, tx_id: int) -> bytes:
        return self.read_row_now(sp_addr)

    # -- backends ------------------------------------------------------------------
    def attach_backends(self, backend_objs):
        if len(backend_objs) != len(self.tiles):
            raise ValueError(f"expected {len(self.tiles)} backends, got {len(backend_objs)}")
        return [self.attach_backend(b, tile_id=t) for t, b in enumerate(backend_objs)]

    def attach_backend(self, backend_obj, tile_id: Optional[int] = None):
        """Wire a backend's SRAM side to one pad. Its writes (DMA loads) and
        reads (DMA stores) go through the pad's request port like the
        frontend's, at the backend's own cycle."""
        if isinstance(backend_obj, (list, tuple)):
            return self.attach_backends(list(backend_obj))
        if tile_id is None:
            free = [t for t, b in enumerate(self.backends) if b is None]
            if not free:
                raise ValueError("all scratchpad backend slots are already occupied")
            tile_id = free[0]
        tile_id = self._normalize_tile_id(tile_id)
        self.backends[tile_id] = backend_obj

        def _send_sram_write(sp_addr, row_bytes, row_idx, tx_id, _t=tile_id, _b=backend_obj):
            return self.submit_write(_t, sp_addr, row_bytes, now=_b._tick)

        def _request_sram_read(sp_addr, row_idx, tx_id, _t=tile_id, _b=backend_obj):
            def _done(lanes, _tx=tx_id, _row=row_idx):
                _b.queue_sram_read_response(_tx, _row, b"".join(lanes))
            return self.submit_read(_t, sp_addr, _done, now=_b._tick)

        backend_obj.send_sram_write = _send_sram_write
        backend_obj.request_sram_read = _request_sram_read
        backend_obj.send_sram_read = None
        self.backend = next((b for b in self.backends if b is not None), None)
        return backend_obj

    def write_path_idle(self, tile_id: Optional[int] = None) -> bool:
        """Every accepted write has reached the banks (on one pad, or all)."""
        pads = self.tiles if tile_id is None else [self.tiles[self._normalize_tile_id(tile_id)]]
        return all(not pad.wq for pad in pads)

    # -- one cycle -----------------------------------------------------------------
    def next_wake(self, now: int) -> Optional[int]:
        if all(pad.idle() for pad in self.tiles):
            return None
        return now + 1

    def tick(self, time=None) -> None:
        now = int(time) if time is not None else int(self.now) + 1
        self.now = now
        for pad in self.tiles:
            if pad.idle():
                continue
            self._step_pad(pad, now)

    def _step_pad(self, pad: _Pad, now: int) -> None:
        eb = self.elem_bytes
        hook = self.trace_hook
        # Read before write: on one edge a read sees the row as it was.
        if pad.rq and pad.rq[0].ready <= now and now >= pad.read_free:
            req = pad.rq.popleft()
            lanes = [bytes(b.mem[req.slot] or b"") for b in pad.banks]
            pad.read_free = pad.read.next_enable(now)
            due = pad.read_free + EGRESS_CYCLES
            self._seq += 1
            heapq.heappush(pad.out, (due, self._seq, req.callback, lanes))
            pad.reads += 1
            if hook is not None:
                hook({"kind": "read", "tile": self.tiles.index(pad),
                      "slot": req.slot, "cycle": now})
        if pad.wq and pad.wq[0].ready <= now and now >= pad.write_free:
            req = pad.wq.popleft()
            for lane, data in enumerate(req.lanes or []):
                pad.banks[bank_of_lane(lane)].mem[req.slot] = data[:eb]
            pad.write_free = pad.write.next_enable(now)
            if req.callback is not None:
                self._seq += 1
                heapq.heappush(pad.out, (pad.write_free, self._seq, req.callback, None))
            pad.writes += 1
            if hook is not None:
                hook({"kind": "write", "tile": self.tiles.index(pad),
                      "slot": req.slot, "cycle": now})
        while pad.out and pad.out[0][0] <= now:
            _due, _seq, callback, payload = heapq.heappop(pad.out)
            if callback is None:
                continue
            if payload is None:
                callback()
            else:
                callback(payload)

    def get_stats(self) -> dict:
        return {
            "tiles": [{"tile": t, "reads": p.reads, "writes": p.writes,
                       "stalls": p.stalls, "read_queue": len(p.rq),
                       "write_queue": len(p.wq)}
                      for t, p in enumerate(self.tiles)],
            "backend_slots": [{"tile": t, "attached": b is not None}
                              for t, b in enumerate(self.backends)],
        }
