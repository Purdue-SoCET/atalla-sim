"""Instruction cache: rtl/modules/scheduler/icache.sv.

8 KB, direct-mapped: 128 lines of 64 bytes.

    address = [31:13] tag | [12:6] index | [5:0] offset

A packet is 20 bytes, so one can straddle two lines. A packet whose last byte
falls past the end of its line is "split", and needs both lines to hit.

Misses fill one line at a time over a 64-bit memory interface, eight beats a
line. The cache supports early restart: a line still filling counts as a hit
once the beats covering the packet have arrived. The RTL tests that with
`fill_count > last_chunk` on a 3-bit counter, and two consequences fall out
without any special-casing here, so they are worth knowing:

  * a split packet's first line uses last_chunk = 7, and a 3-bit count never
    exceeds 7, so it cannot early-restart -- it waits for the whole fill;
  * a split packet whose lines both miss fills them one after the other,
    with a return to IDLE between.

Only tags and valid bits are modelled, not the data array. Instruction memory
is read-only, so a line's contents are always the program image's bytes, and
the fetched packet is taken straight from the image once the cache says hit.
What the model has to get right is *when* that happens.

The memory behind the cache is outside the scheduler core. With a `port`
(memory/bus.py) a fill goes out as bursts on the DRAM channel the data
cache and the scratchpad backends share, and iwait holds while the next
beat's burst hasn't returned. Without one, the timing is a parameter:
`first_beat_wait` cycles of iwait before a fill's first beat and
`beat_wait` before each beat after. Both default to 0 -- a beat every
cycle, an 8-cycle fill.
"""

from base.rtl_module import RTLModule

LINES = 128
LINE_BYTES = 64
BEATS = LINE_BYTES // 8          # 64-bit fill interface
PACKET_BYTES = 20

IDLE, FILL = 0, 1


def split_address(addr: int):
    """(tag, index, offset) -- icache.sv: {tag_a, idx_a, off_a} = addr_a."""
    return (addr >> 13) & 0x7FFFF, (addr >> 6) & 0x7F, addr & 0x3F


class ICache(RTLModule):
    REGS = dict(state=IDLE, fill_count=0, active_idx=0, active_tag=0,
                wait_left=0,
                valid=[False] * LINES, tags=[0] * LINES)
    INS = dict(imemaddr=0, imemREN=False, halt=False)
    OUTS = dict(ihit=False)

    def __init__(self, name: str = "icache", *, first_beat_wait: int = 0,
                 beat_wait: int = 0, port=None):
        super().__init__(name)
        #: memory/bus.py BusMaster, or None for the parameterised timing.
        self.port = port
        self._port_act = None
        self.iwait_cycles = 0
        self.first_beat_wait = max(0, int(first_beat_wait))
        self.beat_wait = max(0, int(beat_wait))
        self.fills = 0
        self.fill_cycles = 0

    # -- the hit logic -----------------------------------------------------
    def _line_hit(self, idx: int, tag: int, last_chunk: int) -> bool:
        if self.valid[idx] and self.tags[idx] == tag:
            return True
        # Early restart: the line being filled, once its needed beats are in.
        return (self.state == FILL and self.active_idx == idx
                and self.active_tag == tag and self.fill_count > last_chunk)

    def lookup(self, addr: int):
        """Everything the RTL computes combinationally from one address."""
        tag_a, idx_a, off_a = split_address(addr)
        tag_b, idx_b, _ = split_address(addr + LINE_BYTES)
        end_byte = off_a + PACKET_BYTES - 1                 # 7-bit in the RTL
        split = bool(end_byte & 0x40)
        last_a = 7 if split else (end_byte >> 3) & 7
        last_b = (end_byte >> 3) & 7
        hit_a = self._line_hit(idx_a, tag_a, last_a)
        hit_b = self._line_hit(idx_b, tag_b, last_b)
        full_hit = (hit_a and hit_b) if split else hit_a
        return dict(tag_a=tag_a, idx_a=idx_a, tag_b=tag_b, idx_b=idx_b,
                    split=split, hit_a=hit_a, hit_b=hit_b, full_hit=full_hit)

    # -- one cycle -----------------------------------------------------------
    def eval_data(self) -> None:
        for reg in self.REGS:
            setattr(self, reg + "_n", getattr(self, reg))
        self._port_act = None

        look = self.lookup(self.in_imemaddr)
        self.out_ihit = bool(self.in_imemREN and look["full_hit"])

        if self.state == IDLE:
            if self.in_imemREN and not self.in_halt:
                if not look["hit_a"]:
                    self._start_fill(look["idx_a"], look["tag_a"])
                elif look["split"] and not look["hit_b"]:
                    self._start_fill(look["idx_b"], look["tag_b"])
            return

        # FILL
        self.fill_cycles += 1
        if self.in_halt:
            self.state_n, self.fill_count_n = IDLE, 0
            return
        if self.port is not None:
            if not self.port.beat_ready(self.fill_count):
                self.iwait_cycles += 1
                return
        else:
            if self.wait_left > 0:
                self.wait_left_n = self.wait_left - 1
                self.iwait_cycles += 1
                return
            self.wait_left_n = self.beat_wait
        if self.fill_count == BEATS - 1:
            if self.port is not None:
                self._port_act = self.port.finish_read
            valid = list(self.valid)
            tags = list(self.tags)
            valid[self.active_idx] = True
            tags[self.active_idx] = self.active_tag
            self.valid_n, self.tags_n = valid, tags
            self.state_n, self.fill_count_n = IDLE, 0
        else:
            self.fill_count_n = self.fill_count + 1

    def _start_fill(self, idx: int, tag: int) -> None:
        valid = list(self.valid)
        valid[idx] = False                  # n_cache[idx].valid = 0
        self.valid_n = valid
        self.state_n, self.fill_count_n = FILL, 0
        self.active_idx_n, self.active_tag_n = idx, tag
        self.wait_left_n = self.first_beat_wait
        self.fills += 1
        if self.port is not None:
            base = ((tag << 7) | idx) << 6
            self._port_act = lambda: self.port.read(base, LINE_BYTES)

    def commit(self) -> None:
        if self._port_act is not None:
            self._port_act()
            self._port_act = None
        super().commit()

    def warm(self, addresses) -> None:
        """Mark the lines holding these packets valid, as if already fetched.
        A test convenience; the RTL has no such port."""
        valid, tags = list(self.valid), list(self.tags)
        for addr in addresses:
            for a in (addr, addr + PACKET_BYTES - 1):
                tag, idx, _ = split_address(a)
                valid[idx], tags[idx] = True, tag
        self.valid = self.valid_n = valid
        self.tags = self.tags_n = tags
