#!/usr/bin/env python3
"""Reproduce the DRAM subsystem report's measurements on the model.

The Spring 2026 DRAM subsystem report measured the AXI bus and the Ramulator
wrapper in RTL simulation. This runs the same traffic through memory/axi.py
and memory/ramulator.py and prints the report's numbers beside the model's.

    tools/memory_bench.py table1      # the AXI bus, zero-latency DRAM
    tools/memory_bench.py table3      # 1024x1024 int16 load, single-beat ARs
    tools/memory_bench.py table4      # the same, 2-beat bursts
    tools/memory_bench.py all

table3 and table4 follow tb/unit/memory/ramulator_wrapper/test_sdma.sv
(memory_FA26): the wrapper driven directly with 256-bit (32-byte) beats,
one AR offered a cycle, the matrix read row-major from 0x0 and then
tile-major (32x32 tiles, one 64-byte tile row a burst) from 0x200000, on
one Ramulator instance. The DDR4 configurations are the branch's
configs/ddr4_*.yaml; the HBM3 rows are left out (their HBM3_8Gb preset is
Ramulator 2.0's, which 2.1 replaced).

table1 has no bench in the repository: the report measured the earlier,
two-scratchpad interconnect. This drives today's bus (SP0, SP1, D$, I$
reading; SP0, SP1, D$ writing) with axi_tb.sv's burst lengths
(0 : 1-7 : 8-15 weighted 3 : 4 : 3) as fast as each master can.
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from memory.axi import AXIBus, AXIParams, FixedLatencyDRAM   # noqa: E402

MAT, TILE, ELEM, BEAT = 1024, 32, 2, 32
ROW_BYTES = MAT * ELEM
MATRIX_BYTES = MAT * ROW_BYTES
TILE_BASE = 0x200000

#: DDR4 configs of tables 3 and 4: name -> RamulatorDRAM("rtl") overrides.
DDR4_CONFIGS = {
    "DDR4 1ch 128b": dict(channels=1, channel_width=128),
    "DDR4 1ch 256b": dict(channels=1, channel_width=256),
    "DDR4 1ch 512b": dict(channels=1, channel_width=512),
    "DDR4 2ch 128b": dict(channels=2, channel_width=128),
    "DDR4 4ch 128b": dict(channels=4, channel_width=128),
}

#: The report's rows: (cycles, cyc/row, cyc/8-beat, cyc/inter, B/cycle, stall %).
#: B/cycle as printed; tools/memory_bench.py recomputes it from cycles.
REPORT = {
    "table3": {
        ("DDR4 1ch 128b", "row"): (207824, 202, 3, 3, 10.90, 68.0),
        ("DDR4 1ch 128b", "tile"): (138126, 134, 2, 2, 15.18, 52.2),
        ("DDR4 1ch 256b", "row"): (242003, 236, 3, 3, 8.66, 72.5),
        ("DDR4 1ch 256b", "tile"): (190603, 186, 2, 2, 11.00, 65.2),
        ("DDR4 1ch 512b", "row"): (259211, 253, 3, 3, 8.90, 74.3),
        ("DDR4 1ch 512b", "tile"): (259510, 253, 3, 3, 8.80, 74.2),
        ("DDR4 2ch 128b", "row"): (104145, 101, 1, 1, 20.13, 36.1),
        ("DDR4 2ch 128b", "tile"): (90679, 88, 1, 1, 23.12, 26.9),
        ("DDR4 4ch 128b", "row"): (65924, 64, 1, 1, 31.81, 0.0),
        ("DDR4 4ch 128b", "tile"): (66299, 64, 1, 1, 31.63, 0.0),
    },
    "table4": {
        ("DDR4 1ch 128b", "row"): (70802, 69, 1, 1, 29.61, 51.9),
        ("DDR4 1ch 128b", "tile"): (69578, 67, 1, 1, 30.14, 51.4),
        ("DDR4 1ch 256b", "row"): (104017, 101, 1, 1, 20.16, 67.5),
        ("DDR4 1ch 256b", "tile"): (87037, 84, 1, 1, 24.90, 61.4),
        ("DDR4 1ch 512b", "row"): (121273, 118, 1, 1, 17.29, 72.1),
        ("DDR4 1ch 512b", "tile"): (121262, 118, 1, 1, 17.29, 72.1),
        ("DDR4 2ch 128b", "row"): (65737, 64, 1, 1, 31.90, 25.2),
        ("DDR4 2ch 128b", "tile"): (65647, 64, 1, 1, 31.94, 25.1),
        ("DDR4 4ch 128b", "row"): (65562, 64, 1, 1, 31.98, 25.0),
        ("DDR4 4ch 128b", "tile"): (65686, 64, 1, 1, 31.92, 25.2),
    },
}
#: Table 1, the customized interconnect: (cycles, B/cycle, utilization %).
REPORT_TABLE1 = {"read": (10010, 3.280, 41.00), "write": (8000, 2.225, 27.81)}


def sdma_addresses(pattern: str, rows: int = MAT):
    """Beat addresses in test_sdma.sv's order (sdma_gen_row_major / _tile_major)."""
    if pattern == "row":
        return [r * ROW_BYTES + b * BEAT for r in range(rows) for b in range(ROW_BYTES // BEAT)]
    per_tile_row = TILE * ELEM // BEAT
    out = []
    for tr in range(rows // TILE):
        for tc in range(MAT // TILE):
            base = TILE_BASE + tr * TILE * ROW_BYTES + tc * TILE * ELEM
            for r in range(TILE):
                out.extend(base + r * ROW_BYTES + b * BEAT for b in range(per_tile_row))
    return out


def sdma_load(bus, dram, port, addrs, burst: bool, start: int):
    """Offer one AR a cycle (2-beat bursts or single beats) until every beat
    is back; test_sdma.sv's sdma_drain. Returns (cycles, metrics, end cycle)."""
    group = TILE * ELEM // BEAT                  # 2 beats: one 64-byte tile row
    ars = ([(addrs[i], group) for i in range(0, len(addrs), group)] if burst
           else [(a, 1) for a in addrs])
    arrivals = []
    nxt, stalls, c = 0, 0, start
    on_beat = lambda i, cyc: arrivals.append(cyc)
    while len(arrivals) < len(addrs):
        if nxt < len(ars):
            if port.can_read():
                port.read(c, ars[nxt][0], ars[nxt][1], on_beat)
                nxt += 1
            else:
                stalls += 1
        bus.tick(float(c))
        dram.tick(float(c))
        c += 1
    arrivals.sort()
    cycles = arrivals[-1] - start
    gaps = [b - a for a, b in zip([start] + arrivals, arrivals)]
    pairs = [arrivals[i + 1] - arrivals[i] for i in range(0, len(arrivals) - 1, group)]
    return cycles, dict(cyc_row=cycles // (len(addrs) * BEAT // ROW_BYTES), cyc_group=sum(pairs) // len(pairs),
                        cyc_inter=sum(gaps) // len(gaps),
                        bpc=len(addrs) * BEAT / cycles, stall=100.0 * stalls / cycles), c


def run_sdma(table: str, configs=None, rows: int = MAT):
    from memory.ramulator import RamulatorDRAM
    burst = table == "table4"
    results = {}
    for name, over in DDR4_CONFIGS.items():
        if configs and name not in configs:
            continue
        dram = RamulatorDRAM("rtl", **over)
        bus = AXIBus(dram, AXIParams(ar_depth=2, beat_bytes=BEAT))
        port = bus.port("sp0")
        c = 0
        for pattern in ("row", "tile"):
            t = time.time()
            cycles, m, c = sdma_load(bus, dram, port, sdma_addresses(pattern, rows), burst, c)
            results[(name, pattern)] = (cycles, m, time.time() - t)
    return results


def print_sdma(table: str, results):
    title = {"table3": "Table 3: single-beat ARs", "table4": "Table 4: 2-beat bursts"}[table]
    print(title)
    print("%-14s %-5s %9s %9s %7s %7s %6s %6s %7s %7s %6s %6s" % (
        "config", "", "cycles", "report", "diff", "B/cyc", "report", "c/row", "report",
        "stall%", "report", "c/int"))
    for (name, pattern), (cycles, m, _secs) in results.items():
        rep = REPORT[table].get((name, pattern))
        rc = rep[0] if rep else 0
        print("%-14s %-5s %9d %9d %+6.1f%% %7.2f %6.2f %6d %7d %7.1f %6.1f %6d" % (
            name, pattern, cycles, rc, 100.0 * (cycles - rc) / rc if rc else 0,
            m["bpc"], MATRIX_BYTES / rc if rc else 0, m["cyc_row"], rep[1] if rep else 0,
            m["stall"], rep[5] if rep else 0, m["cyc_inter"]))
    print("report B/cycle recomputed from its cycles (2,097,152 B / cycles)")


def axi_random(kind: str, per_master: int = 120, seed: int = 1, latency: int = 0):
    """Table 1: masters issue random-length bursts as fast as the bus takes them."""
    rng = random.Random(seed)
    dram = FixedLatencyDRAM(latency)
    bus = AXIBus(dram)
    names = ("sp0", "sp1", "dcache", "icache") if kind == "read" else ("sp0", "sp1", "dcache")

    def length():
        k = rng.choices((0, 1, 2), weights=(3, 4, 3))[0]
        return 1 + (0 if k == 0 else rng.randint(1, 7) if k == 1 else rng.randint(8, 15))

    todo = {n: [length() for _ in range(per_master)] for n in names}
    beats = sum(map(sum, todo.values()))
    left = {"n": beats if kind == "read" else len(names) * per_master}

    def got(*_):
        left["n"] -= 1

    c = 0
    while left["n"]:
        for n in names:
            p = bus.port(n)
            if todo[n]:
                b = todo[n][0]
                addr = rng.randrange(0, 1 << 20, 64)
                if kind == "read" and p.can_read():
                    p.read(c, addr, b, got)
                    todo[n].pop(0)
                elif kind == "write" and p.can_write(b):
                    p.write(c, addr, b, on_done=got)
                    todo[n].pop(0)
        bus.tick(float(c))
        dram.tick(float(c))
        c += 1
    return c, beats


def print_table1(per_master: int, seed: int):
    print("Table 1: AXI bus, zero-latency DRAM, random burst lengths")
    print("%-6s %8s %8s %8s %8s %8s %8s" % ("", "cycles", "beats", "B/cyc", "util%",
                                             "report", "util%"))
    for kind in ("read", "write"):
        cycles, beats = axi_random(kind, per_master, seed)
        rep = REPORT_TABLE1[kind]
        print("%-6s %8d %8d %8.3f %8.2f %8.3f %8.2f" % (
            kind, cycles, beats, 8.0 * beats / cycles, 100.0 * beats / cycles, rep[1], rep[2]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("table", choices=["table1", "table3", "table4", "all"])
    ap.add_argument("--config", action="append", choices=list(DDR4_CONFIGS),
                    help="tables 3 and 4: only this DDR4 config (repeatable)")
    ap.add_argument("--per-master", type=int, default=120,
                    help="table 1: bursts per master")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    tables = ["table1", "table3", "table4"] if args.table == "all" else [args.table]
    for t in tables:
        if t == "table1":
            print_table1(args.per_master, args.seed)
        else:
            print_sdma(t, run_sdma(t, args.config))
        print()


if __name__ == "__main__":
    main()
