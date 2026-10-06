"""memory/bus.py: a cache's AXI-style port on the shared DRAM channel."""
from memory.backend import SharedDRAMBurstChannel
from memory.bus import BusMaster


def port(channel=None, **kw):
    return BusMaster("p", channel or SharedDRAMBurstChannel(), dram_latency=6,
                     burst_bytes=32, **kw)


def run(ports, cycles, start=0):
    for c in range(start, start + cycles):
        for p in ports:
            p.tick(float(c))


def test_a_line_read_is_two_bursts_back_after_the_latency():
    p = port()
    p._tick = -1
    p.read(0x100, 64)                 # in cycle 0, before the bus ticks
    run([p], 1)                       # cycle 0: burst 0 launches
    run([p], 1, 1)                    # cycle 1: burst 1 launches
    assert p.stats["read_bursts"] == 2
    # beats 0-3 arrive in cycle 6, beats 4-7 in cycle 7
    run([p], 5, 2)                    # through cycle 6's tick; now = 7
    assert all(p.beat_ready(b) for b in range(8))
    p2 = port()
    p2.read(0, 64)
    run([p2], 6)                      # now = 6
    assert [p2.beat_ready(b) for b in range(8)] == [True] * 4 + [False] * 4


def test_two_masters_share_one_launch_a_cycle():
    ch = SharedDRAMBurstChannel()
    a, b = port(ch), port(ch)
    a.read(0, 64)
    b.read(0x1000, 64)
    run([a, b], 4)
    # four bursts, one a cycle: the second master waits while the first launches
    assert a.stats["read_bursts"] + b.stats["read_bursts"] == 4
    assert b.stats["contention_cycles"] == 2


def test_written_beats_gather_into_bursts():
    p = port()
    for k in range(8):
        assert p.can_write()
        p.write(0x200 + 8 * k, 8)
    assert not p.idle
    run([p], 10)
    assert p.stats["write_bursts"] == 2 and p.stats["write_bytes"] == 64
    assert p.idle


def test_the_write_buffer_holds_one_line():
    p = port(write_buffer_bytes=64)
    for k in range(8):
        p.write(8 * k, 8)
    assert not p.can_write()
    run([p], 1)
    assert p.can_write()
