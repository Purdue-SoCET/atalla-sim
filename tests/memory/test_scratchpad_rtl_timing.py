"""The scratchpad against the RTL: every bank enable and every response.

data/scratchpad_rtl_trace.txt.xz is a Questa run of data/scratchpad_meas_tb.sv
on the atalla scratchpad (transpose_integration, b1ba35ff), pad 0: a single
write and read, 16 back-to-back writes and 16 back-to-back reads, alternating
writes and reads, a read right after a write to the same row, and backend
DMA loads and stores (4 rows each). One line per cycle; data/README.md says
how to regenerate it.

The replay feeds the model the requests the RTL accepted, on the cycles it
accepted them, and checks the cycles the model enables its banks and hands
data back against the RTL's -- read enables, write enables, frontend
responses and backend responses, every one.
"""
import lzma
import os

from memory.scratchpad import Scratchpad

TRACE = os.path.join(os.path.dirname(__file__), "data", "scratchpad_rtl_trace.txt.xz")


def load_trace():
    rows = []
    with lzma.open(TRACE, "rt") as f:
        for line in f:
            parts = line.split()
            d = {k: int(v) for k, v in (p.split("=") for p in parts[2:])}
            d["cycle"], d["phase"] = int(parts[0]), parts[1]
            rows.append(d)
    return rows


def _replay(trace):
    spad = Scratchpad(num_banks=32, bank_size=64, read_latency=2,
                      write_latency=2, elem_bytes=2, num_tiles=1)
    seen = {"read": [], "write": [], "fe_res": [], "be_res": []}
    spad.trace_hook = lambda e: seen[e["kind"]].append(e["cycle"])
    be_slot = 0

    def on_fe(_lanes):
        seen["fe_res"].append(spad.now)

    def on_be(_lanes):
        seen["be_res"].append(spad.now)

    for rtl in trace:
        c = rtl["cycle"]
        if rtl["fe_acc"]:
            if rtl["fe_w"]:
                assert spad.submit_write(0, rtl["fe_row"], b"\x01\x00" * 32, now=c)
            else:
                assert spad.submit_read(0, rtl["fe_row"], on_fe, now=c)
        if rtl["be_req"]:
            if rtl["be_w"]:
                assert spad.submit_write(0, be_slot, b"\x02\x00" * 32, now=c)
            else:
                assert spad.submit_read(0, be_slot, on_be, now=c)
            be_slot += 1
        spad.tick(c)
    return seen


def test_every_enable_and_response_matches_the_rtl():
    trace = load_trace()
    want = {
        "read": [r["cycle"] for r in trace if r["rd_en"]],
        "write": [r["cycle"] for r in trace if r["wr_en"]],
        "fe_res": [r["cycle"] for r in trace if r["res"]],
        "be_res": [r["cycle"] for r in trace if r["be_res"]],
    }
    assert len(want["read"]) == 30 and len(want["write"]) == 30, \
        "the whole measurement, not an excerpt"
    got = _replay(trace)
    for kind in want:
        assert got[kind] == want[kind], "%s: model %s, RTL %s" % (kind, got[kind], want[kind])


def test_an_uncontended_read_takes_7_cycles():
    """Accepted at a, banks enabled at a + 2, done at a + 5, data at a + 7."""
    trace = load_trace()
    acc = next(r["cycle"] for r in trace if r["phase"] == "rd1" and r["fe_acc"])
    res = next(r["cycle"] for r in trace if r["phase"] == "rd1" and r["res"])
    assert res - acc == 7


def test_each_direction_takes_one_row_every_3_cycles():
    trace = load_trace()
    for phase, key in (("rd_stream", "rd_en"), ("wr_stream", "wr_en")):
        cycles = [r["cycle"] for r in trace if r["phase"] == phase and r[key]]
        assert len(cycles) == 16
        assert {b - a for a, b in zip(cycles, cycles[1:])} == {3}
