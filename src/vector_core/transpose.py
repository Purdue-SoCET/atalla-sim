"""Transpose unit: a VEC_LEN x VEC_LEN tile transpose, cycle-matched to the RTL.

Source: atalla rtl/modules/vector/transpose_unit.sv (branch
transpose_integration, b1ba35ff), with the sram_bank and clos it instantiates.
Checked cycle by cycle against a Questa trace of tb/unit/vector/
transpose_unit_tb.sv -- every state, counter, handshake and output vector over
the whole testbench; see tests/vector_core/test_transpose_rtl_trace.py.

Storage
    Rows arrive one at a time. Row r puts its element i in bank (i + r) % N at
    address r. Column c is read back from bank b at address (b - c) % N and
    routed to output position (b - c) % N. Neither direction touches a bank
    twice, so a row goes in, and a column comes out, without a bank conflict.
    Both rotations are permutations across banks, which is what
    memory.crossbar.Xbar models, so the Clos network is an Xbar.

The FSM, one push and one column at a time, as the RTL has it

    IDLE --push--> WAIT_CLOS_WRITE --> BUSY_WRITE --wdone--> IDLE
    IDLE --pop---> POPPING --> WAIT_SRAM --rdone--> WAIT_CLOS_READ --> DONE
                      ^                                                 |
                      +------------- next column, once taken -----------+

    WAIT_CLOS_*   clos_latency cycles (lat_count 0..2): the vector crosses the
                  network, a 3-stage pipe -- input, center and output modules
                  with two latches between them. It leaves the third stage on
                  the edge that ends the wait, which is the write enable's edge
                  for a row and the edge into DONE for a column.
    BUSY_WRITE /  wait for the bank's done flag. sram_bank raises it one cycle
    WAIT_SRAM     after the enable for a latency of 0 or 1, and latency + 1
                  cycles after for anything longer.
    DONE          holds valid_out until the consumer takes the column. After
                  the last column the unit returns to IDLE; one pop request
                  drains the whole tile.

    push    = 1 + clos_latency + done(write_latency)   = 1 + 3 + 5 =  9 cycles
    column  = 1 + done(read_latency) + clos_latency + 1 = 1 + 3 + 3 + 1 = 8
    drain   = 1 (IDLE sees the pop) + 32 columns          = 257 cycles

The bank latencies are sram_bank's defaults (read 2, write 4): the unit
instantiates its banks without overriding them. Whether that is intended is
an open question for the RTL; the sim matches it.

`count` is one counter shared by the two directions, as in the RTL: a pop
rewinds it, so the next tile starts at row 0 while rows M..N-1 still hold the
previous tile's data. Banks start at 0.0.

API (the timing holds when the caller follows the core's order each cycle:
requests before tick(), writebacks taken after it)
    ready_in             a push or pop is accepted this cycle
    push(vec)            feed one row in
    pop(dsts=None)       start a drain of all VEC_LEN columns
    can_pop_writeback()  a transposed column is valid (the FSM is in DONE)
    pop_writeback()      take it: {"col", "data", and "dst" if pop() named one}
    tick(time)

The RTL reads vec_in for the whole push and expects the producer to hold it.
This model latches the row when it accepts it, as an input register would.
"""

from typing import Dict, List, Optional, Sequence

from base.clocked_object import Clocked
from memory.crossbar import Xbar
from memory.sram_bank import SramChannel, done_delay

Time = float

#: Cycles the FSM spends in WAIT_CLOS_WRITE / WAIT_CLOS_READ.
CLOS_LATENCY = 3
#: sram_bank parameter defaults, which transpose_unit.sv does not override.
SRAM_READ_LATENCY = 2
SRAM_WRITE_LATENCY = 4
#: Banks, and therefore the tile edge and the vector length.
TRANSPOSE_VEC_LEN = 32

IDLE = "IDLE"
WAIT_CLOS_WRITE = "WAIT_CLOS_WRITE"
BUSY_WRITE = "BUSY_WRITE"
POPPING = "POPPING"
WAIT_SRAM = "WAIT_SRAM"
WAIT_CLOS_READ = "WAIT_CLOS_READ"
DONE = "DONE"


def bank_done_delay(latency: int) -> int:
    """Cycles from a bank enable to its done flag being visible (sram_bank)."""
    return done_delay(latency)


class TransposeUnit(Clocked):
    """VEC_LEN SRAM banks behind a Clos network, driven by the RTL's FSM."""

    def __init__(
        self,
        vec_len: int = TRANSPOSE_VEC_LEN,
        clos_latency: int = CLOS_LATENCY,
        sram_read_latency: int = SRAM_READ_LATENCY,
        sram_write_latency: int = SRAM_WRITE_LATENCY,
    ):
        super().__init__()
        self.vec_len = max(1, int(vec_len))
        self.clos_latency = int(clos_latency)
        if self.clos_latency < 1:
            raise ValueError("clos_latency must be >= 1")
        self.sram_read_latency = max(0, int(sram_read_latency))
        self.sram_write_latency = max(0, int(sram_write_latency))
        self._tick = -1

        #: The Clos network: clos_latency stages, one vector in flight.
        self.clos = Xbar(delay=self.clos_latency, num_banks=self.vec_len,
                         max_size=self.clos_latency)
        # banks[bank][address]: VEC_LEN banks of VEC_LEN entries.
        self.banks = [[0.0] * self.vec_len for _ in range(self.vec_len)]
        # The 32 banks run in lockstep, so one channel pair stands in for all.
        self._rd = SramChannel(self.sram_read_latency)
        self._wr = SramChannel(self.sram_write_latency)

        # RTL registers
        self.state = IDLE
        #: Row being written, or column being popped -- one counter for both.
        self.count = 0
        self.lat_count = 0

        # Requests and handshakes for the current cycle
        self._push_req: Optional[List[float]] = None
        self._pop_req = False
        self._pop_dsts: Optional[List[int]] = None
        self._taken = False

        self._row: Optional[List[float]] = None     # latched push data
        self._routed: Optional[List[float]] = None  # what left the network
        self._out: Optional[Dict] = None            # the column held in DONE

        self.pushes = 0
        self.columns = 0

    # -- costs ---------------------------------------------------------------
    @property
    def push_cycles(self) -> int:
        """Accept to ready again, for one row."""
        return 1 + self.clos_latency + bank_done_delay(self.sram_write_latency)

    @property
    def column_cycles(self) -> int:
        """POPPING to the end of DONE, with a consumer that is always ready."""
        return 1 + bank_done_delay(self.sram_read_latency) + self.clos_latency + 1

    @property
    def drain_cycles(self) -> int:
        """IDLE seeing the pop to the end of the last column's DONE."""
        return 1 + self.vec_len * self.column_cycles

    # -- handshake -------------------------------------------------------------
    @property
    def ready_in(self) -> bool:
        """The RTL's ready_in (state == IDLE), less a request already made this
        cycle -- one request per cycle, as the single input port allows."""
        return self.state == IDLE and self._push_req is None and not self._pop_req

    @property
    def valid_out(self) -> bool:
        return self.state == DONE and not self._taken

    @property
    def busy(self) -> bool:
        return self.state != IDLE

    def push(self, vec: Sequence[float]) -> bool:
        """Feed one row in. False when the unit cannot take it this cycle."""
        if not self.ready_in:
            return False
        data = list(vec)
        if len(data) != self.vec_len:
            raise ValueError("transpose push expects %d elements, got %d"
                             % (self.vec_len, len(data)))
        self._push_req = data
        self.request_wake(self._tick + 1)
        return True

    def pop(self, dsts: Optional[Sequence[int]] = None) -> bool:
        """Start a drain of the whole tile. `dsts` optionally names the
        destination register of each column; it rides along to writeback."""
        if not self.ready_in:
            return False
        if dsts is not None:
            dsts = [int(d) for d in dsts]
            if len(dsts) != self.vec_len:
                raise ValueError("transpose pop expects %d destinations, got %d"
                                 % (self.vec_len, len(dsts)))
        self._pop_req = True
        self._pop_dsts = dsts
        self.request_wake(self._tick + 1)
        return True

    # -- writeback port --------------------------------------------------------
    def can_pop_writeback(self) -> bool:
        return self.valid_out

    def peek_writeback(self) -> Optional[Dict]:
        return self._out if self.valid_out else None

    def pop_writeback(self) -> Optional[Dict]:
        """Take the column. The FSM moves on at the next clock edge."""
        if not self.valid_out:
            return None
        self._taken = True
        return self._out

    def has_pending(self) -> bool:
        return self.state != IDLE or self._push_req is not None or self._pop_req

    # -- one clock edge ----------------------------------------------------------
    def tick(self, time: Optional[Time] = None) -> None:
        cycle = self._consume_tick(time, attr_name="_tick")
        if cycle is None:
            return
        n, last = self.vec_len, self.clos_latency - 1
        state, count, lat = self.state, self.count, self.lat_count
        ren = state == POPPING
        wen = state == WAIT_CLOS_WRITE and lat == last
        n_state, n_count, n_lat = state, count, lat
        row_lands, col_lands = None, None

        if state == IDLE:
            if self._push_req is not None:
                self._row = self._push_req
                self._send(shift=[(i + count) % n for i in range(n)], vals=self._row)
                n_lat, n_state = 0, WAIT_CLOS_WRITE
            elif self._pop_req:
                n_count, n_state = 0, POPPING
        elif state == WAIT_CLOS_WRITE:
            if lat == last:
                row_lands = count              # wen: written on this edge
                n_state = BUSY_WRITE
            else:
                n_lat = lat + 1
        elif state == BUSY_WRITE:
            if self._wr.done:
                n_count = 0 if count == n - 1 else count + 1
                n_state = IDLE
                self.pushes += 1
        elif state == POPPING:
            n_state = WAIT_SRAM
        elif state == WAIT_SRAM:
            if self._rd.done:
                vals = [self.banks[b][(b - count) % n] for b in range(n)]
                self._send(shift=[(b - count) % n for b in range(n)], vals=vals)
                n_lat, n_state = 0, WAIT_CLOS_READ
        elif state == WAIT_CLOS_READ:
            if lat == last:
                col_lands = count              # held in DONE from this edge
                n_state = DONE
            else:
                n_lat = lat + 1
        elif state == DONE:
            if self._taken:
                self.columns += 1
                self._out, self._taken = None, False
                if count == n - 1:
                    n_count, n_state = 0, IDLE
                    self._pop_dsts = None
                else:
                    n_count, n_state = count + 1, POPPING

        self._rd.edge(ren)
        self._wr.edge(wen)
        self._push_req, self._pop_req = None, False
        self.state, self.count, self.lat_count = n_state, n_count, n_lat
        # The network moves on the same edge; whatever leaves its last stage
        # now is what the write enable or DONE takes.
        self.clos.tick(cycle)
        if row_lands is not None:
            self._write_row(row_lands)
        if col_lands is not None:
            self._out = self._column(col_lands)
            self._taken = False

    def _send(self, shift: List[int], vals: List[float]) -> None:
        self._routed = None
        op = self.clos.enqueue(shift, list(vals), callback=self._arrived)
        assert op != -1, "transpose clos refused a vector"

    def _arrived(self, out) -> bool:
        self._routed = list(out)
        return True

    def _take_routed(self) -> List[float]:
        assert self._routed is not None, \
            "the Clos output was sampled before it left the network"
        out, self._routed = self._routed, None
        return out

    def _write_row(self, addr: int) -> None:
        """wen: every bank takes its element of the rotated row at `addr`."""
        for bank, value in enumerate(self._take_routed()):
            self.banks[bank][addr] = value
        self._row = None

    def _column(self, col: int) -> Dict:
        entry = {"col": col, "data": self._take_routed()}
        if self._pop_dsts is not None:
            entry["dst"] = self._pop_dsts[col]
        return entry

    def next_wake(self, now: int) -> Optional[int]:
        """Idle with nothing requested and an empty network: nothing to do."""
        if self.has_pending():
            return now + 1
        return self.clos.next_wake(now)
