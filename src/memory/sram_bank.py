"""The SRAM bank every Atalla memory is built from: sram_bank.sv.

rtl/modules/common/memory/sram_bank.sv (atalla, transpose_integration
b1ba35ff). The scratchpad pads, the transpose unit, the systolic-array
buffers and the dcache all instantiate it, each with its own latencies:

    scratchpad (body.sv)              read 2, write 2
    transpose unit                    read 2, write 4   (defaults, not overridden)
    dcache                            read 2, write 4
    systolic-array in/out/skew        read 1, write 1

What the bank does, per direction (read: ren/rdone, write: wen/wdone):

  * The array is read and written on the clock edge after the enable,
    whatever the latency: `if (ren) rdata <= mem[raddr]`,
    `if (wen) mem[waddr] <= wdata`. A read and a write of one address on the
    same edge read the old value.
  * The latency only times the done pulse. An enable taken while idle makes
    the channel busy and raises done `done_delay(latency)` cycles later:
    one cycle for a latency of 0 or 1, latency + 1 for anything longer.
  * While busy the channel ignores further enables -- it does not queue
    them, and it is not pipelined. A channel takes at most one access every
    done_delay(latency) cycles; the caller must wait for !busy (the RTL
    scratchpad's controller and the transpose FSM both do).
  * Reads and writes are independent: each has its own busy and done.

SramChannel is one direction, cycle by cycle, for models that tick every
cycle (the transpose FSM). Event-driven models use its deadline form,
`next_enable()`, which is the same rule.
"""

from typing import Any, List, Optional


def done_delay(latency: int) -> int:
    """Cycles from an enable to its done pulse being visible."""
    return 1 if int(latency) <= 1 else int(latency) + 1


class SramChannel:
    """One direction of sram_bank.sv: busy, a countdown, a one-cycle done."""

    def __init__(self, latency: int):
        self.latency = max(0, int(latency))
        self.busy = False
        self.cnt = 0
        self.done = False

    def edge(self, enable: bool) -> bool:
        """One rising edge. Returns whether an enable was taken (the channel
        was idle); an enable while busy is ignored, as in the RTL."""
        taken = bool(enable) and not self.busy
        done = False
        if taken:
            self.busy = True
            self.cnt = self.latency - 1 if self.latency > 1 else 0
            if self.latency <= 1:
                done, self.busy = True, False
        elif self.busy:
            if self.cnt == 0:
                done, self.busy = True, False
            else:
                self.cnt -= 1
        self.done = done
        return taken

    def next_enable(self, enabled_at: int) -> int:
        """The first cycle the channel can take another enable, given one
        taken at `enabled_at`: the cycle its done is visible."""
        return int(enabled_at) + done_delay(self.latency)


class SramBank:
    """Storage plus a read and a write channel, one clock edge per edge()."""

    def __init__(self, height: int, read_latency: int = 2, write_latency: int = 4,
                 init: Any = None):
        self.height = int(height)
        self.mem: List[Any] = [init] * self.height
        self.rdata: Any = init
        self.read = SramChannel(read_latency)
        self.write = SramChannel(write_latency)

    @property
    def busy(self) -> bool:
        return self.read.busy or self.write.busy

    @property
    def rdone(self) -> bool:
        return self.read.done

    @property
    def wdone(self) -> bool:
        return self.write.done

    def edge(self, ren: bool = False, raddr: int = 0,
             wen: bool = False, waddr: int = 0, wdata: Any = None) -> None:
        """One rising edge. The data path ignores busy, as the RTL's does: a
        read enable always loads rdata, a write enable always writes."""
        if ren:
            self.rdata = self.mem[int(raddr)]
        if wen:
            self.mem[int(waddr)] = wdata
        self.read.edge(ren)
        self.write.edge(wen)
