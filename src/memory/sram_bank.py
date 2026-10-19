"""The SRAM bank every Atalla memory is built from: sram_bank.sv.

rtl/modules/common/memory/sram_bank.sv (atalla, transpose_integration
b1ba35ff). The scratchpad pads, the transpose unit, the systolic-array
buffers and the dcache all instantiate it, each with its own latencies.
The defaults here are the ones PCACTI recommends at 800 MHz in FreePDK45
(the VTG column, each memory in its current shape):

    scratchpad (body.sv, 8T)          read 2, write 1
    transpose unit                    read 1, write 1
    dcache (one 16 x 2144 b macro)    read 1, write 4
    systolic-array in/out/skew        read 1, write 1

What the bank does, per direction (read: ren/rdone, write: wen/wdone):

  * The array is read and written on the clock edge after the enable,
    whatever the latency: `if (ren) rdata <= mem[raddr]`,
    `if (wen) mem[waddr] <= wdata`. A read and a write of one address on the
    same edge read the old value.
  * The latency only times the done pulse. An enable taken while idle makes
    the channel busy and raises done `done_delay(latency)` cycles later:
    `latency` cycles, and one for a latency of 0 or 1.
  * While busy the channel ignores further enables -- it does not queue
    them, and it is not pipelined. A channel takes at most one access every
    done_delay(latency) cycles; the caller must wait for !busy (the RTL
    scratchpad's controller and the transpose FSM both do).
  * Reads and writes are independent: each has its own busy and done.

Where the RTL is wrong this does what was meant: sram_bank.sv loads its
counter with LATENCY - 1 and raises done one edge after it reaches zero, so
a latency of N >= 2 takes N + 1 cycles (a setting of 2 behaves as 3). Here
it takes N. To reproduce a trace of today's RTL, pass N + 1 for a setting of
N >= 2 (the RTL replays in tests/ do).

SramChannel is one direction, cycle by cycle, for models that tick every
cycle (the transpose FSM). Event-driven models use its deadline form,
`next_enable()`, which is the same rule.
"""



def done_delay(latency: int) -> int:
    """Cycles from an enable to its done pulse being visible."""
    return max(1, int(latency))


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
            self.cnt = self.latency - 2 if self.latency > 1 else 0
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


