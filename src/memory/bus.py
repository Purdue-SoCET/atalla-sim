"""A cache's port onto the shared DRAM: AXI-style bursts on the channel the
scratchpad backends use.

In the RTL the instruction cache, the data cache and the four scratchpad
backends all reach one memory (sim_ram_rr.sv, in the testbench), which serves
one requester at a time, round robin. Here the six masters share one
SharedDRAMBurstChannel: one burst launches per cycle in total, of
`burst_bytes`, and returns `dram_latency` cycles later. The masters take
turns in a round-robin ticker, so a cache line fill waits behind the
scratchpads' DMA bursts, and they behind it.

The cache keeps its own 64-bit beat interface (icache.sv's iwait / iload,
lockup_free_cache.sv's ram_mem_complete): this port turns a line read into
bursts and says when each beat has arrived; written beats collect in a
small buffer and leave as bursts. The data itself moves through the cache's
own memory object -- this only times it.

    cache side (the CORE phase)    read(), beat_ready(), finish_read(),
                                   can_write(), write()
    bus side (the BACKEND phase)   tick(): launch at most one burst, if the
                                   shared channel is free this cycle
"""

from collections import deque
from typing import Deque, List, Optional, Tuple

from base.clocked_object import Clocked


class BusMaster(Clocked):
    def __init__(self, name: str, channel, *, dram_latency: int = 6,
                 burst_bytes: int = 32, beat_bytes: int = 8,
                 max_outstanding: int = 16, write_buffer_bytes: int = 64):
        super().__init__()
        self.name = name
        self.channel = channel
        self.dram_latency = int(dram_latency)
        self.burst_bytes = int(burst_bytes)
        self.beat_bytes = int(beat_bytes)
        self.max_outstanding = int(max_outstanding)
        self.write_buffer_bytes = int(write_buffer_bytes)
        self._tick = -1
        #: Bursts waiting to launch: (kind, txn, addr, nbytes, first beat).
        self._launch_q: Deque[Tuple[str, int, int, int, int]] = deque()
        self._due: Deque[int] = deque()          # in-flight bursts' return cycles
        self._txn = 0
        self._read_txn: Optional[int] = None
        self._read_arrival: List[Optional[int]] = []
        self._wbuf: Optional[Tuple[int, int]] = None   # (addr, bytes) being gathered
        self._wq_bytes = 0                         # written, not yet launched
        self._read_started: Optional[int] = None
        self.stats = dict(read_bursts=0, write_bursts=0, read_bytes=0, write_bytes=0,
                          reads=0, read_cycles=0, contention_cycles=0,
                          write_stall_cycles=0, busy_cycles=0)

    @property
    def now(self) -> int:
        """The cycle the cache side is in: the bus ticks after it."""
        return self._tick + 1

    # -- cache side ---------------------------------------------------------------------
    def read(self, base: int, nbytes: int) -> None:
        """Start reading `nbytes` from `base`. A read still in flight is
        abandoned; its bursts still use the bus."""
        self._txn += 1
        self._read_txn = self._txn
        self._read_arrival = [None] * (nbytes // self.beat_bytes)
        self._read_started = self.now
        for off in range(0, nbytes, self.burst_bytes):
            n = min(self.burst_bytes, nbytes - off)
            self._launch_q.append(("r", self._txn, base + off, n, off // self.beat_bytes))
        self.stats["reads"] += 1

    def beat_ready(self, beat: int) -> bool:
        if self._read_txn is None or beat >= len(self._read_arrival):
            return False
        t = self._read_arrival[beat]
        return t is not None and t <= self.now

    def finish_read(self) -> None:
        if self._read_started is not None:
            self.stats["read_cycles"] += self.now - self._read_started
        self._read_txn = None
        self._read_started = None

    def can_write(self) -> bool:
        if self._wq_bytes + self.beat_bytes > self.write_buffer_bytes:
            self.stats["write_stall_cycles"] += 1
            return False
        return True

    def write(self, addr: int, nbytes: int) -> None:
        """One written beat; contiguous beats gather into a burst."""
        self._wq_bytes += nbytes
        if self._wbuf is not None and self._wbuf[0] + self._wbuf[1] == addr:
            self._wbuf = (self._wbuf[0], self._wbuf[1] + nbytes)
        else:
            self._flush_wbuf()
            self._wbuf = (addr, nbytes)
        a, n = self._wbuf
        if n >= self.burst_bytes or (a + n) % self.burst_bytes == 0:
            self._flush_wbuf()

    def _flush_wbuf(self) -> None:
        if self._wbuf is not None:
            a, n = self._wbuf
            self._launch_q.append(("w", 0, a, n, 0))
            self._wbuf = None

    @property
    def idle(self) -> bool:
        return (not self._launch_q and self._wbuf is None
                and all(d <= self.now for d in self._due))

    # -- bus side -----------------------------------------------------------------------
    def tick(self, time: Optional[float] = None) -> None:
        cycle = self._consume_tick(time, attr_name="_tick")
        if cycle is None:
            return
        while self._due and self._due[0] <= cycle:
            self._due.popleft()
        if self._launch_q or self._due:
            self.stats["busy_cycles"] += 1
        if not self._launch_q or len(self._due) >= self.max_outstanding:
            return
        if not self.channel.can_issue(cycle):
            self.stats["contention_cycles"] += 1
            return
        kind, txn, addr, n, first = self._launch_q.popleft()
        self.channel.reserve(cycle)
        due = cycle + self.dram_latency
        self._due.append(due)
        if kind == "r":
            self.stats["read_bursts"] += 1
            self.stats["read_bytes"] += n
            if txn == self._read_txn:
                for b in range(first, first + n // self.beat_bytes):
                    self._read_arrival[b] = due
        else:
            self.stats["write_bursts"] += 1
            self.stats["write_bytes"] += n
            self._wq_bytes -= n
