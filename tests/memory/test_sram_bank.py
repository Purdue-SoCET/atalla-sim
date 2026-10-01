"""The shared SRAM bank model: sram_bank.sv's handshake.

The transpose unit's replay (tests/vector_core/test_transpose_rtl_trace.py)
checks the same channels cycle by cycle against Questa with read 2 / write 4;
these pin the rules directly.
"""
import pytest

from memory.sram_bank import SramBank, SramChannel, done_delay


@pytest.mark.parametrize("latency,delay", [(0, 1), (1, 1), (2, 3), (4, 5)])
def test_done_is_latency_plus_one_except_for_short_latencies(latency, delay):
    assert done_delay(latency) == delay


def _edges(channel, enables):
    """done after each edge, for a list of enable values."""
    out = []
    for en in enables:
        channel.edge(en)
        out.append(int(channel.done))
    return out


def test_a_read_raises_done_once_three_cycles_later():
    ch = SramChannel(2)
    assert _edges(ch, [1, 0, 0, 0, 0]) == [0, 0, 1, 0, 0]


def test_an_enable_while_busy_is_ignored_not_queued():
    """The channel is not pipelined: a second enable before done is dropped,
    and only one done comes back."""
    ch = SramChannel(2)
    assert _edges(ch, [1, 1, 0, 0, 0, 0]) == [0, 0, 1, 0, 0, 0]
    assert ch.edge(True), "free again once done is visible"


def test_back_to_back_accesses_go_every_done_delay_cycles():
    ch = SramChannel(4)
    taken, cycle, free = [], 0, 0
    for cycle in range(20):
        if ch.edge(True):
            taken.append(cycle)
    assert taken == [0, 5, 10, 15]
    assert [ch.next_enable(c) for c in taken[:-1]] == taken[1:], \
        "the deadline form gives the same cadence"


def test_reads_and_writes_are_independent():
    bank = SramBank(height=4, read_latency=2, write_latency=2, init=0)
    bank.edge(ren=True, raddr=0, wen=True, waddr=1, wdata=7)
    assert bank.read.busy and bank.write.busy
    bank.edge()
    bank.edge()
    assert bank.rdone and bank.wdone


def test_the_array_is_read_and_written_on_the_enable_edge():
    """Latency times only done: the data moves on the first edge, and a read
    and a write of one address on that edge read the old value."""
    bank = SramBank(height=4, read_latency=4, write_latency=4, init=0)
    bank.edge(wen=True, waddr=2, wdata=11)
    assert bank.mem[2] == 11, "written on the enable's edge, long before done"
    bank = SramBank(height=4, read_latency=2, write_latency=2, init=0)
    bank.mem[3] = 5
    bank.edge(ren=True, raddr=3, wen=True, waddr=3, wdata=9)
    assert bank.rdata == 5 and bank.mem[3] == 9
