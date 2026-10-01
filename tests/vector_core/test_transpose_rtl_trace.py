"""The transpose unit against the RTL, cycle by cycle.

data/transpose_unit_rtl_trace.txt.xz is a Questa run of the atalla
testbench tb/unit/vector/transpose_unit_tb.sv on transpose_unit.sv
(transpose_integration, b1ba35ff): every cycle after reset, every 1..32-row
tile, with and without output backpressure. data/README.md says how to
regenerate it.

The replay drives the model with the trace's own inputs -- the push and pop
requests, ready_out, and the pushed data -- and, before every clock edge,
checks what the RTL had in that cycle: state, count, lat_count, ready_in,
valid_out, the bank enables and done flags, and every transposed column.
"""
import lzma
import os

from vector_core.transpose import (
    CLOS_LATENCY, DONE, IDLE, POPPING, WAIT_CLOS_WRITE, TransposeUnit)

TRACE = os.path.join(os.path.dirname(__file__), "data",
                     "transpose_unit_rtl_trace.txt.xz")
N = 32


def _vector(hexstr):
    """%h of a packed [31:0][15:0] array: element 31 comes first."""
    return [int(hexstr[len(hexstr) - 4 * (i + 1):len(hexstr) - 4 * i], 16)
            for i in range(N)]


def load_trace():
    cycles = []
    with lzma.open(TRACE, "rt") as f:
        for line in f:
            if line.startswith("  vec_in="):
                cycles[-1]["vec_in"] = _vector(line.split("=", 1)[1].strip())
            elif line.startswith("  vec_out="):
                cycles[-1]["vec_out"] = _vector(line.split("=", 1)[1].strip())
            else:
                parts = line.split()
                c = {k: int(v) for k, v in (p.split("=") for p in parts[2:])}
                c["cycle"], c["state"] = int(parts[0]), parts[1]
                cycles.append(c)
    return cycles


def observed(unit):
    """The model's view of the signals the trace logs, before the edge."""
    return {
        "state": unit.state, "cnt": unit.count, "lat": unit.lat_count,
        "rdy_in": int(unit.state == IDLE), "vld_out": int(unit.state == DONE),
        "ren": int(unit.state == POPPING),
        "wen": int(unit.state == WAIT_CLOS_WRITE
                   and unit.lat_count == CLOS_LATENCY - 1),
        "rdone": int(unit._rd.done), "wdone": int(unit._wr.done),
    }


def test_every_cycle_of_the_rtl_testbench_matches():
    trace = load_trace()
    assert len(trace) > 40_000, "the whole testbench, not an excerpt"
    unit = TransposeUnit()
    columns = 0
    for rtl in trace:
        cyc = rtl["cycle"]
        got = observed(unit)
        want = {k: (rtl["state"] if k == "state" else rtl[k]) for k in got}
        assert got == want, "cycle %d: model %s, RTL %s" % (cyc, got, want)

        if unit.state == IDLE and rtl["push"]:
            assert unit.push([float(v) for v in rtl["vec_in"]])
        elif unit.state == IDLE and rtl["pop"]:
            assert unit.pop()
        if unit.state == DONE:
            col = unit.peek_writeback()
            assert [int(v) for v in col["data"]] == rtl["vec_out"], \
                "cycle %d: column %d differs" % (cyc, col["col"])
            if rtl["rdy_out"]:
                unit.pop_writeback()
                columns += 1
        unit.tick(float(cyc))

    assert columns == 4032, "64 tests' drains, plus the testbench's repeats"
