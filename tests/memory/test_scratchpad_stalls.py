"""Scratchpad backpressure, as the RTL's head and scpad_cntrl apply it."""
from memory.scratchpad import Scratchpad


def _row(v, n=8):
    return b"".join(int(v + i).to_bytes(2, "little") for i in range(n))


def test_a_full_queue_stalls_both_directions():
    """w_stall is rd_fifo_full || wr_fifo_full: with the read queue full a
    write is refused too, and goes up once the head read reaches the banks."""
    spad = Scratchpad(num_banks=8, bank_size=8, elem_bytes=2, queue_depth=2, num_tiles=1)
    got = []
    assert spad.submit_read(0, 0, got.append, now=0)
    spad.tick(0)
    assert spad.submit_read(0, 1, got.append, now=1)
    spad.tick(1)
    assert not spad.can_accept(0, now=2)
    assert not spad.submit_write(0, 2, _row(1), now=2), "read queue full stalls writes"
    spad.tick(2)                                  # the first read is enabled
    assert spad.submit_write(0, 2, _row(1), now=3)
    assert spad.tiles[0].stalls == 1


def test_refused_requests_are_not_lost():
    """A caller that retries every cycle gets every row in, in order."""
    spad = Scratchpad(num_banks=8, bank_size=16, elem_bytes=2, queue_depth=1, num_tiles=1)
    rows = [_row(10 * k) for k in range(6)]
    k, cycle = 0, 0
    while k < len(rows) or not spad.write_path_idle():
        if k < len(rows) and spad.submit_write(0, k, rows[k], now=cycle):
            k += 1
        spad.tick(cycle)
        cycle += 1
    assert [spad.read_row_now(s, tile_id=0) for s in range(6)] == rows
    # one write enable every 3 cycles, each 2 after its acceptance
    assert cycle == 2 + 3 * 5 + 1
