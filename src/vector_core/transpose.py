"""Transpose unit: a VEC_LEN x VEC_LEN tile transpose.

Vectors arrive one row at a time, so the unit never sees a whole matrix before
it has to start storing one. It skews each row on the way in: row r puts its
element i in bank (i + r) % VEC_LEN, every element at address r. Reading
column c walks the opposite rotation -- bank b is read at address (b - c) %
VEC_LEN and routed to output position (b - c) % VEC_LEN -- which delivers
column c of the pushed matrix. Neither direction touches a bank twice, so a
whole row goes in, and a whole column comes out, without a bank conflict.

Both rotations are permutations of a vector across banks, which is what
memory.crossbar.Xbar already models, so the Clos network is an Xbar: the
per-input destination index is its shift mask and the network depth is its
delay. It also supplies the output-side backpressure -- a column whose
consumer has no room is held in the network's tail, which is the FSM's DONE
state.

Cost is 4 cycles per vector each way: 3 through the network plus 1 SRAM
access. Pushing an M-row matrix costs 4M cycles. One pop request drains the
whole tile -- VEC_LEN columns, 4 * VEC_LEN cycles -- without a per-column
request.

API
    push(vec)            feed one row in; False when the unit is not idle
    pop(dsts=None)       start a drain of all VEC_LEN columns
    can_pop_writeback()  a transposed column is waiting
    pop_writeback()      take it: {"col", "data", and "dst" if pop() named one}
    tick(time)

`count` is one counter shared by the two directions,
as in the RTL, so a pop resets the write position -- push M rows, pop, and the
next tile starts at row 0 while rows M..VEC_LEN-1 still hold the previous
tile's data. Banks reset to 0.0, so an M-row matrix pops back with zeros in
the rows that were never pushed.
"""

from typing import Dict, List, Optional, Sequence

from base.clocked_object import Clocked
from base.queue import SimQueue
from memory.crossbar import Xbar

Time = float

#: Depth of the Clos network, in pipeline stages.
CLOS_LATENCY = 3
#: One cycle to reach a bank, one to come back.
SRAM_WRITE_LATENCY = 1
SRAM_READ_LATENCY = 1
#: Banks, and therefore the tile edge and the vector length.
TRANSPOSE_VEC_LEN = 32

IDLE = "IDLE"
WAIT_CLOS_WRITE = "WAIT_CLOS_WRITE"
BUSY_WRITE = "BUSY_WRITE"
POPPING = "POPPING"
WAIT_CLOS_READ = "WAIT_CLOS_READ"
DONE = "DONE"


class TransposeUnit(Clocked):
    """A Clos crossbar plus VEC_LEN SRAM banks, driven by a push/pop FSM.

    A push costs clos_latency + sram_write_latency cycles, a column of a pop
    costs sram_read_latency + clos_latency: the bank access and the network
    traversal, and nothing else.
    """

    def __init__(
        self,
        vec_len: int = TRANSPOSE_VEC_LEN,
        clos_latency: int = CLOS_LATENCY,
        sram_read_latency: int = SRAM_READ_LATENCY,
        sram_write_latency: int = SRAM_WRITE_LATENCY,
        out_depth: Optional[int] = None,
    ):
        super().__init__()
        self.vec_len = max(1, int(vec_len))
        self.clos_latency = max(1, int(clos_latency))
        self.sram_read_latency = max(1, int(sram_read_latency))
        self.sram_write_latency = max(1, int(sram_write_latency))
        self._tick = -1

        #: The Clos network. One vector is in flight at a time, so the extra
        #: slot is only ever the tail a full consumer is holding up.
        self.clos = Xbar(delay=self.clos_latency, num_banks=self.vec_len,
                         max_size=self.clos_latency + 1)
        # banks[bank][address]: VEC_LEN banks of VEC_LEN entries.
        self.banks = [[0.0] * self.vec_len for _ in range(self.vec_len)]
        # A full drain fits by default, so a pop never stalls on its own output.
        self.outputs = SimQueue(max(1, int(out_depth or self.vec_len)))

        self.state = IDLE
        #: Row being written, or column being popped -- one counter for both,
        #: so starting a pop also rewinds the write position.
        self.count = 0
        self.timer = 0
        self._row_addr = 0
        self._rotated = None
        self._push_req = None
        self._pop_req = False
        self._pop_dsts = None

    # -- handshake ---------------------------------------------------------
    @property
    def ready_in(self) -> bool:
        """A push or a pop is accepted only while the FSM is idle."""
        return self.state == IDLE and self._push_req is None and not self._pop_req

    @property
    def valid_out(self) -> bool:
        """Holding a transposed column that the consumer has not taken."""
        return self.state == DONE

    @property
    def busy(self) -> bool:
        return self.state != IDLE

    def push(self, vec: Sequence[float]) -> bool:
        """Feed one row in. False when the unit cannot take it this cycle."""
        if not self.ready_in:
            return False
        data = list(vec)
        if len(data) != self.vec_len:
            raise ValueError(
                "transpose push expects %d elements, got %d"
                % (self.vec_len, len(data))
            )
        self._push_req = data
        self.request_wake(self._tick + 1)
        return True

    def pop(self, dsts: Optional[Sequence[int]] = None) -> bool:
        """Start a drain of the whole tile.

        `dsts` optionally names the destination register for each of the
        VEC_LEN columns; it is carried through to the writeback entries.
        """
        if not self.ready_in:
            return False
        if dsts is not None:
            dsts = [int(d) for d in dsts]
            if len(dsts) != self.vec_len:
                raise ValueError(
                    "transpose pop expects %d destinations, got %d"
                    % (self.vec_len, len(dsts))
                )
        self._pop_req = True
        self._pop_dsts = dsts
        self.request_wake(self._tick + 1)
        return True

    # -- writeback port ----------------------------------------------------
    def can_pop_writeback(self) -> bool:
        return not self.outputs.is_empty()

    def pop_writeback(self) -> Optional[Dict]:
        return self.outputs.dequeue()

    def peek_writeback(self) -> Optional[Dict]:
        return self.outputs.peek()

    def has_pending(self) -> bool:
        return (self.state != IDLE or self._push_req is not None
                or self._pop_req or not self.outputs.is_empty())

    # -- the two rotations -------------------------------------------------
    def _send_row(self, vec: List[float]) -> None:
        """Row `count` goes in rotated by `count`, so it lands on a diagonal."""
        n, row = self.vec_len, self.count
        self._row_addr = row
        shift = [(i + row) % n for i in range(n)]
        op = self.clos.enqueue(shift, list(vec), callback=self._row_routed)
        assert op != -1, "transpose clos refused a row"

    def _send_column(self) -> None:
        """Column `count` is the same rotation backwards: bank b holds that
        column's row (b - count) % n, at that same address."""
        n, col = self.vec_len, self.count
        shift = [(b - col) % n for b in range(n)]
        raw = [self.banks[b][(b - col) % n] for b in range(n)]
        op = self.clos.enqueue(shift, raw, callback=self._column_routed)
        assert op != -1, "transpose clos refused a column"

    def _row_routed(self, out) -> bool:
        """The rotated row has left the network; the banks take it."""
        self._rotated = list(out)
        if self.sram_write_latency <= 1:
            self._commit_row()
        else:
            self.state = BUSY_WRITE
            self.timer = self.sram_write_latency - 1
        return True

    def _commit_row(self) -> None:
        addr = self._row_addr
        for bank, value in enumerate(self._rotated):
            self.banks[bank][addr] = value
        self._rotated = None
        self.count = (self.count + 1) % self.vec_len
        self.state = IDLE

    def _column_routed(self, out) -> bool:
        """The transposed column has left the network. Returning False leaves
        it in the network's tail, which is what DONE means."""
        if self.outputs.is_full():
            self.state = DONE
            return False
        col = self.count
        entry = {"col": col, "data": list(out)}
        if self._pop_dsts is not None:
            entry["dst"] = self._pop_dsts[col]
        self.outputs.enqueue(entry)

        if col == self.vec_len - 1:
            self.count = 0
            self._pop_dsts = None
            self.state = IDLE
        else:
            self.count = col + 1
            self.state = POPPING
            self.timer = self.sram_read_latency
        return True

    # -- FSM ---------------------------------------------------------------
    def tick(self, time: Optional[Time] = None) -> None:
        cycle = self._consume_tick(time, attr_name="_tick")
        if cycle is None:
            return
        if self.state == IDLE:
            self._start()
        self._advance()
        # Last, so anything handed over this cycle enters the network now and
        # anything leaving it settles the FSM before the next tick.
        self.clos.tick(cycle)

    def _start(self) -> None:
        """Take one request. A push wins a tie, as in the RTL."""
        if self._push_req is not None:
            row, self._push_req = self._push_req, None
            self.state = WAIT_CLOS_WRITE
            self._send_row(row)
        elif self._pop_req:
            self._pop_req = False
            self.count = 0
            self.state = POPPING
            self.timer = self.sram_read_latency

    def _advance(self) -> None:
        """Only the two SRAM states are timed here; the network times itself."""
        if self.state not in (POPPING, BUSY_WRITE):
            return
        self.timer -= 1
        if self.timer > 0:
            return
        if self.state == POPPING:
            self.state = WAIT_CLOS_READ
            self._send_column()
        else:
            self._commit_row()

    def next_wake(self, now: int) -> Optional[int]:
        """Idle with nothing requested and an empty network means nothing can
        happen. Otherwise defer to whichever of the two is ready first."""
        if self.state != IDLE or self._push_req is not None or self._pop_req:
            return now + 1
        return self.clos.next_wake(now)
