"""Writeback: the arbiter and the per-bank WB buffer (plan section 7).

Two levels, as in the diagram:

  1. Arbiter -> WB buffer. Each cycle up to `grants_per_cycle` source heads
     (FU group outputs, the RTree, each VLSU, the GSAU, the transpose unit)
     move into the buffer, in `priority` order. A vector result needs room
     in its bank's FIFO. Mask results (compares) go out on the mask file's
     own write ports instead: one per mask bank (Q21), and none on a bank
     the scalar side writes this cycle.
  2. WB buffer -> banks. One FIFO per bank, `depth` deep. Each cycle each
     bank writes up to `write_ports` entries that have waited `t_wbbuf`
     cycles, oldest first.

A source that is not granted keeps its head; an FU group then freezes.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Set


@dataclass
class WBEntry:
    """A result offered to writeback."""
    source: str
    reg: int
    value: Any                          # a vector (list) or a 32-bit mask
    enable: Optional[List[bool]] = None  # per-element write enable; None = all
    is_mask: bool = False
    ready: int = 0                      # cycle it may drain (set in the buffer)


class WBBufferV2:

    def __init__(self, num_banks: int, depth: int = 4, write_ports: int = 1, t_wbbuf: int = 1):
        self.num_banks = int(num_banks)
        self.depth = int(depth)
        self.write_ports = int(write_ports)
        self.t_wbbuf = int(t_wbbuf)
        self.fifos: List[Deque[WBEntry]] = [deque() for _ in range(self.num_banks)]
        self.occupancy_hist: List[Dict[int, int]] = [dict() for _ in range(self.num_banks)]

    def room(self, bank: int) -> bool:
        return len(self.fifos[bank]) < self.depth

    def push(self, entry: WBEntry, bank: int, now: int) -> None:
        if not self.room(bank):
            raise AssertionError("WB buffer bank %d overflow" % bank)
        entry.ready = int(now) + self.t_wbbuf
        self.fifos[bank].append(entry)

    def due(self, now: int) -> List[WBEntry]:
        """The entries every bank writes this cycle (not yet removed)."""
        out = []
        for f in self.fifos:
            for e in list(f)[: self.write_ports]:
                if e.ready <= now:
                    out.append(e)
                else:
                    break
        return out

    def pop_due(self, now: int) -> List[WBEntry]:
        out = []
        for f in self.fifos:
            n = 0
            while f and n < self.write_ports and f[0].ready <= now:
                out.append(f.popleft())
                n += 1
        return out

    def sample(self) -> None:
        for b, f in enumerate(self.fifos):
            h = self.occupancy_hist[b]
            h[len(f)] = h.get(len(f), 0) + 1

    @property
    def empty(self) -> bool:
        return not any(self.fifos)


class WBArbiter:

    def __init__(self, priority: Sequence[str], grants_per_cycle: int = 4, mask_banks: int = 2):
        self.order = {s: i for i, s in enumerate(priority)}
        self.grants_per_cycle = int(grants_per_cycle)
        self.mask_banks = int(mask_banks)
        self.stats = dict(grants={}, denied={}, mask_grants=0, cycles_full=0)

    def arbitrate(self, offers: Iterable[WBEntry], bank_of: Callable[[int], int],
                  occupancy: Callable[[int], int], depth: int,
                  blocked_mask_banks: Set[int] = frozenset()) -> List[WBEntry]:
        """The offers granted this cycle, in priority order. Each source
        offers at most its head. `occupancy(bank)` is the bank FIFO's
        registered fill; two grants to one bank in a cycle each need room."""
        granted, used_mask, n = [], set(), 0
        added: Dict[int, int] = {}
        for e in sorted(offers, key=lambda o: self.order.get(o.source, len(self.order))):
            ok = False
            if e.is_mask:
                mb = int(e.reg) % self.mask_banks
                if mb not in used_mask and mb not in blocked_mask_banks:
                    used_mask.add(mb)
                    ok = True
                    self.stats["mask_grants"] += 1
            elif n < self.grants_per_cycle:
                b = bank_of(e.reg)
                if occupancy(b) + added.get(b, 0) < depth:
                    added[b] = added.get(b, 0) + 1
                    n += 1
                    ok = True
            if ok:
                granted.append(e)
                self.stats["grants"][e.source] = self.stats["grants"].get(e.source, 0) + 1
            else:
                self.stats["denied"][e.source] = self.stats["denied"].get(e.source, 0) + 1
        if n >= self.grants_per_cycle:
            self.stats["cycles_full"] += 1
        return granted
