"""A log of every scratchpad DMA's start and end (opt-in: run --sdma-log).

    atalla-sim run prog.in --sdma-log -           # to the terminal
    atalla-sim run prog.in --sdma-log sdma.log    # to a file

One line when an SDMA is handed to its scratchpad and one when the
scratchpad reports it done, in cycle order:

    cycle    41 start #0 scpad.ld sp1 pc 0x10 (asm line 7): dram 0x2000 -> row 0, 8x8 (128 B, dram stride 16 B), x7
    cycle    77 end   #0 scpad.ld sp1 pc 0x10: 36 cycles

#n numbers the SDMAs in issue order and ties an end to its start. A load
(scpad.ld) copies DRAM to the scratchpad, a store (scpad.st) the scratchpad
to DRAM; rows x cols are BF16 elements; x<r> is the rs1 the SDMA holds until
it ends. Cycles are the core's cycle count. With nothing asked for, no
observer is attached and the SDMA path pays nothing.
"""

from typing import Callable, Dict, List, Optional


class SdmaLog:
    """Records SDMA starts and ends of an AtallaPlatform
    (build_atalla_platform(sdma_log=True) attaches one as platform.sdma_log)."""

    def __init__(self, platform, describe_pc: Optional[Callable[[int], str]] = None):
        self.p = platform
        self.describe_pc = describe_pc
        #: One dict per SDMA: the fields of the start line, then "start",
        #: "end" (None while in flight) and "id".
        self.records: List[Dict] = []
        self.lines: List[str] = []
        platform.core.vector.on_sdma = self._event

    def _event(self, event: str, info: Dict) -> None:
        cycle = int(self.p.core.cycles)
        if event == "start":
            info["id"], info["start"], info["end"] = len(self.records), cycle, None
            self.records.append(info)
            where = ""
            if self.describe_pc is not None:
                d = self.describe_pc(info["pc"])
                where = " (%s)" % d if d else ""
            if info["op"] == "scpad.ld":
                path = "dram %#x -> row %d" % (info["dram_addr"], info["spad_row"])
            else:
                path = "row %d -> dram %#x" % (info["spad_row"], info["dram_addr"])
            self.lines.append("cycle %5d start #%d %s sp%d pc %#x%s: %s, %dx%d (%d B, dram stride %d B), x%d"
                              % (cycle, info["id"], info["op"], info["sid"], info["pc"], where,
                                 path, info["rows"], info["cols"], info["bytes"],
                                 info["dram_stride"], info["rs1"]))
        else:
            info["end"] = cycle
            self.lines.append("cycle %5d end   #%d %s sp%d pc %#x: %d cycles"
                              % (cycle, info["id"], info["op"], info["sid"], info["pc"],
                                 cycle - info["start"]))

    def text(self) -> str:
        busy = [r for r in self.records if r["end"] is None]
        tail = ["%d SDMA(s) still in flight: %s" % (len(busy), ", ".join("#%d" % r["id"] for r in busy))] \
            if busy else []
        return "\n".join(self.lines + tail) + ("\n" if self.lines or tail else "")

    def write(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.text())
