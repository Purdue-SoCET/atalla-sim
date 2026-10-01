"""The transpose unit: skewed SRAM storage plus a Clos rotation.

Covers that a tile pushed row by row pops back column by column, that one
pop request drains the whole tile, and the RTL's costs: 9 cycles a row in, 8
a column out. test_transpose_rtl_trace.py checks the same model against the
RTL itself, every cycle of its testbench.
"""
import pytest

from memory.crossbar import Xbar
from vector_core.transpose import (
    BUSY_WRITE, CLOS_LATENCY, DONE, IDLE, POPPING, SRAM_READ_LATENCY,
    SRAM_WRITE_LATENCY, TRANSPOSE_VEC_LEN, TransposeUnit, WAIT_CLOS_READ,
    WAIT_CLOS_WRITE, WAIT_SRAM, bank_done_delay,
)

N = TRANSPOSE_VEC_LEN


def tile(rows, cols=N):
    return [[float(r * cols + c) for c in range(cols)] for r in range(rows)]


def transposed(matrix, cols):
    return [[row[c] for row in matrix] for c in range(cols)]


class Driver:
    """Drives the unit the way a scheduler would: retry every cycle."""

    def __init__(self, unit):
        self.unit = unit
        self.cycle = 0
        self.columns = []

    def step(self):
        self.unit.tick(float(self.cycle))
        self.cycle += 1
        while self.unit.can_pop_writeback():
            self.columns.append(self.unit.pop_writeback())

    def push(self, vec, limit=64):
        """Cycles from the cycle the request is taken until the unit is idle."""
        for _ in range(limit):
            if self.unit.push(vec):
                break
            self.step()
        else:
            raise AssertionError("push never accepted")
        start = self.cycle
        self.step()                              # the cycle the push is taken
        while self.unit.state != IDLE:
            self.step()
        return self.cycle - start

    def drain(self, dsts=None, limit=4096):
        start = self.cycle
        assert self.unit.pop(dsts)
        for _ in range(limit):
            self.step()
            if len(self.columns) == self.unit.vec_len:
                return self.cycle - start
        raise AssertionError("drain never finished")


def test_pushed_rows_pop_back_as_columns():
    unit = TransposeUnit()
    drv = Driver(unit)
    rows = tile(N)

    for row in rows:
        drv.push(row)
    drv.drain()

    assert [e["col"] for e in drv.columns] == list(range(N))
    assert [e["data"] for e in drv.columns] == transposed(rows, N)


def test_a_row_costs_9_cycles_and_a_column_8():
    """The RTL's numbers, with its bank latencies (read 2, write 4):
    push   = accept + 3 in the network + 5 until the bank's write done
    column = POPPING + 3 until read done + 3 in the network + DONE.
    A column is taken in the cycle it reaches DONE, so a full drain hands the
    last one over 32 * 8 cycles after the pop is taken."""
    unit = TransposeUnit()
    assert (SRAM_READ_LATENCY, SRAM_WRITE_LATENCY, CLOS_LATENCY) == (2, 4, 3)
    assert unit.push_cycles == 1 + 3 + 5 == 9
    assert unit.column_cycles == 1 + 3 + 3 + 1 == 8
    drv = Driver(unit)

    for row in tile(N):
        assert drv.push(row) == 9

    assert drv.drain() == N * 8


def test_latencies_are_parameters():
    unit = TransposeUnit(vec_len=4, clos_latency=4, sram_read_latency=1,
                         sram_write_latency=6)
    assert unit.push_cycles == 1 + 4 + 7
    assert unit.column_cycles == 1 + 1 + 4 + 1
    drv = Driver(unit)
    assert drv.push([1.0, 2.0, 3.0, 4.0]) == 12
    for row in tile(3, 4):
        drv.push(row)
    assert drv.drain() == 4 * 7


@pytest.mark.parametrize("latency,delay", [(0, 1), (1, 1), (2, 3), (4, 5)])
def test_bank_done_follows_sram_bank(latency, delay):
    """sram_bank raises done one cycle after the enable for a latency of 0 or
    1, and latency + 1 cycles after for anything longer -- so 1 -> 2 costs
    two extra cycles, not one."""
    assert bank_done_delay(latency) == delay


def test_one_pop_request_drains_the_whole_tile():
    """The doc's rule: push N vectors, send a single pop, get N back."""
    unit = TransposeUnit(vec_len=4)
    drv = Driver(unit)
    rows = tile(4, 4)
    for row in rows:
        drv.push(row)

    assert unit.pop()
    requests = 1
    while len(drv.columns) < 4:
        if unit.pop():                      # must be refused while draining
            requests += 1
        drv.step()

    assert requests == 1
    assert [e["data"] for e in drv.columns] == transposed(rows, 4)


def test_short_matrix_pops_zeros_for_rows_never_pushed():
    """Mx32 tiles are the reason the skew exists; unwritten rows read as 0."""
    unit = TransposeUnit(vec_len=4)
    drv = Driver(unit)
    rows = tile(2, 4)
    for row in rows:
        drv.push(row)
    drv.drain()

    expected = transposed(rows + [[0.0] * 4] * 2, 4)
    assert [e["data"] for e in drv.columns] == expected


def test_a_pop_rewinds_the_write_position():
    """count is one counter for both directions, so the next tile starts at
    row 0 while the untouched rows still hold the old tile."""
    unit = TransposeUnit(vec_len=4)
    drv = Driver(unit)
    old = tile(4, 4)
    for row in old:
        drv.push(row)
    drv.drain()

    new_row = [100.0, 101.0, 102.0, 103.0]
    drv.push(new_row)
    drv.columns.clear()
    drv.drain()

    assert [e["data"] for e in drv.columns] == transposed([new_row] + old[1:], 4)


def test_a_busy_unit_refuses_a_push():
    unit = TransposeUnit(vec_len=4)
    drv = Driver(unit)
    assert unit.push([1.0] * 4)
    drv.step()
    assert unit.state != IDLE
    assert not unit.ready_in
    assert not unit.push([2.0] * 4)
    assert not unit.pop()


def test_the_clos_network_is_the_repo_crossbar():
    """Both rotations are permutations across banks, so the network is an Xbar
    and every vector, either direction, is one request through it."""
    unit = TransposeUnit(vec_len=4)
    assert isinstance(unit.clos, Xbar)
    # A 3-stage pipe: input, center and output modules, two latches between.
    assert unit.clos.delay == CLOS_LATENCY == 3
    assert unit.clos.num_banks == 4

    drv = Driver(unit)
    for row in tile(4, 4):
        drv.push(row)
    drv.drain()

    assert unit.clos.total_submitted == 8, "4 rows in, 4 columns out"
    assert unit.clos.total_completed == 8
    assert unit.clos.inflight() == 0


def test_output_backpressure_parks_the_fsm_in_done():
    """There is no output queue: DONE holds valid_out, and the column, until
    the consumer takes it. Every cycle it waits is a cycle on the drain."""
    unit = TransposeUnit(vec_len=4)
    drv = Driver(unit)
    for row in tile(4, 4):
        drv.push(row)

    assert unit.pop()
    for _ in range(20):                     # tick without taking anything
        unit.tick(float(drv.cycle))
        drv.cycle += 1
    assert unit.state == DONE and unit.valid_out and unit.count == 0

    start = drv.cycle
    taken = unit.pop_writeback()
    assert taken["col"] == 0 and not unit.valid_out
    while len(drv.columns) < 3:
        drv.step()
    assert [e["col"] for e in drv.columns] == [1, 2, 3]
    assert drv.cycle - start == 3 * unit.column_cycles


def test_pop_destinations_ride_along():
    unit = TransposeUnit(vec_len=4)
    drv = Driver(unit)
    for row in tile(4, 4):
        drv.push(row)
    drv.drain(dsts=[7, 8, 9, 10])

    assert [e["dst"] for e in drv.columns] == [7, 8, 9, 10]


def _walk(unit, cycles, start=0):
    seen = []
    for cycle in range(start, start + cycles):
        unit.tick(float(cycle))
        seen.append(unit.state)
    return seen


def test_the_fsm_walks_the_states_the_doc_names():
    unit = TransposeUnit(vec_len=4)
    assert unit.state == IDLE
    assert unit.push([1.0] * 4)

    # 3 cycles in the network, the write enable on the last, then BUSY_WRITE
    # until the bank says done (5 cycles for a latency of 4).
    assert _walk(unit, 9) == \
        [WAIT_CLOS_WRITE] * 3 + [BUSY_WRITE] * 5 + [IDLE]

    assert unit.pop()
    # POPPING enables the read, WAIT_SRAM waits for done (3 cycles for a
    # latency of 2), the column crosses the network, and DONE holds it.
    assert _walk(unit, 9, start=9) == \
        [POPPING] + [WAIT_SRAM] * 3 + [WAIT_CLOS_READ] * 3 + [DONE] * 2


def test_busy_write_lasts_until_the_bank_is_done():
    unit = TransposeUnit(vec_len=4, clos_latency=2, sram_write_latency=1)
    assert unit.push([1.0] * 4)
    assert _walk(unit, 4) == [
        WAIT_CLOS_WRITE, WAIT_CLOS_WRITE,   # 2 cycles in the network
        BUSY_WRITE,                         # done one cycle after the enable
        IDLE,
    ]


def test_an_idle_unit_sleeps():
    unit = TransposeUnit(vec_len=4)
    assert unit.next_wake(5) is None
    assert unit.push([1.0] * 4)
    assert unit.next_wake(5) == 6
    unit.tick(0.0)
    assert unit.next_wake(5) == 6, "busy units must keep ticking"


def test_push_length_is_checked():
    unit = TransposeUnit(vec_len=4)
    with pytest.raises(ValueError, match="expects 4 elements"):
        unit.push([1.0, 2.0])
    with pytest.raises(ValueError, match="expects 4 destinations"):
        unit.pop([0, 1])


# --- as a functional unit of the vector core -------------------------------

def test_transpose_has_its_own_vliw_slot():
    """A peer of the GSAU and the VLSUs: its own slot group, one per packet."""
    from vector_core.vector_core import PACKET_UNITS, VectorCore

    vc = VectorCore(veggie_size=32 * 16, lane_count=4, dtype="fp16")
    assert PACKET_UNITS == ("gsau", "vlsu", "transpose", "datapath")
    assert vc.transpose_slots == 1

    packet = vc._empty_packet()
    assert vc._packet_append_inst(
        packet, {"unit": "transpose", "kind": "push", "src": 0})
    assert not vc._packet_append_inst(
        packet, {"unit": "transpose", "kind": "pop", "dst": 128}), \
        "one transpose per packet, like the GSAU"

    # and it packs alongside a full complement of every other unit
    assert vc._packet_append_inst(packet, {"unit": "gsau"})
    for vls in range(vc.vlsu_slots):
        assert vc._packet_append_inst(packet, {"unit": "vlsu", "vls": vls})
    for _ in range(vc.datapath_slots):
        assert vc._packet_append_inst(packet, {"unit": "datapath"})
    assert len(packet["transpose"]) == 1
    assert vc.scheduler_backlog() == 0, "nothing enqueued yet"


def test_a_tile_transposes_through_the_core():
    from vector_core.vector_core import VectorCore

    vc = VectorCore(veggie_size=32 * 16, lane_count=4, dtype="fp16")
    n = vc.vector_len
    rows = [[float(r * n + c) for c in range(n)] for r in range(n)]

    for r, row in enumerate(rows):
        vc.write_vreg(r, row, dtype="fp16")
        assert vc.enqueue_scheduler_instruction(
            {"unit": "transpose", "kind": "push", "src": r})
    # One instruction drains the tile into 32 consecutive registers.
    assert vc.enqueue_scheduler_instruction(
        {"unit": "transpose", "kind": "pop", "dst": 128})

    for cycle in range(1, 1200):
        vc.tick(float(cycle))

    for c in range(n):
        assert vc.read_vreg(128 + c) == [rows[r][c] for r in range(n)], \
            "column %d" % c


def test_a_pop_can_name_its_destinations_individually():
    from vector_core.vector_core import VectorCore

    vc = VectorCore(veggie_size=4 * 16, lane_count=2, dtype="fp16")
    n = vc.vector_len
    rows = [[float(r * n + c) for c in range(n)] for r in range(n)]
    for r, row in enumerate(rows):
        vc.write_vreg(r, row, dtype="fp16")
        assert vc.enqueue_scheduler_instruction(
            {"unit": "transpose", "kind": "push", "src": r})

    dsts = [20, 10, 30, 12]
    assert vc.enqueue_scheduler_instruction(
        {"unit": "transpose", "kind": "pop", "dst": dsts})
    for cycle in range(1, 300):
        vc.tick(float(cycle))

    for c, reg in enumerate(dsts):
        assert vc.read_vreg(reg) == [rows[r][c] for r in range(n)]


def test_the_core_rejects_a_malformed_transpose():
    from vector_core.vector_core import VectorCore

    vc = VectorCore(veggie_size=4 * 16, lane_count=2, dtype="fp16")
    with pytest.raises(ValueError, match="kind must be"):
        vc._issue_transpose({"unit": "transpose", "kind": "rotate"})
    with pytest.raises(ValueError, match="push requires src"):
        vc._issue_transpose({"unit": "transpose", "kind": "push"})
    with pytest.raises(ValueError, match="pop requires dst"):
        vc._issue_transpose({"unit": "transpose", "kind": "pop"})
    with pytest.raises(ValueError, match="needs 4 destinations"):
        vc._issue_transpose({"unit": "transpose", "kind": "pop", "dst": [1, 2]})
