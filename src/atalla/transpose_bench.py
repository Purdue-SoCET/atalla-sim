"""Two testbenches for the transpose unit: how long one tile takes.

The unit transposes a rows x VEC_LEN tile (rows <= VEC_LEN). It reads rows
from the vector register file and writes columns back to it, so the two
benches differ in where the tile starts and ends:

    vrf    VRF -> transpose -> VRF
           rows already in registers; done when the last column is in the VRF.

    spad   scratchpad -> VLSU -> VRF -> transpose -> VRF -> VLSU -> scratchpad
           rows already in a pad; done when the last column has committed to
           the pad's banks.

Both run on the full platform from build_tpu_platform, and the bench plays
the scheduler: it issues an instruction only once what it depends on has
written back -- a push after its row's load, a store after its column's
writeback. The core issues packets in order, as the RTL does.

Cycle accounting: the first instruction is enqueued in cycle 0, and the
result's `cycles` is the number of cycles ticked up to and including the one
in which the last event was observed. Every phase is reported as the
(first, last) cycle in which one of its events happened, so overlapping
phases show up as overlapping intervals.

Data is checked, not just timed: every column that comes back must be the
transposed row data, with zeros for the rows that were never pushed.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from atalla.sysarr_tpu_system import build_tpu_platform
from memory.sc_sram_banks import _xor_bank
from vector_core.transpose import BUSY_WRITE, IDLE, WAIT_CLOS_WRITE

#: Registers the rows are loaded into and the columns are popped into.
ROW_REG_BASE = 0
COL_REG_BASE = 128
#: Scratchpad slots the tile is read from and the transpose is written to.
IN_SLOT_BASE = 0
OUT_SLOT_BASE = 1024

DEFAULT_LIMIT = 100_000


@dataclass
class TransposeBenchResult:
    bench: str
    rows: int
    cols: int
    cycles: int
    #: phase name -> (cycle its first event happened, cycle its last ended)
    phases: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    #: event name -> list of cycles, one per row or column
    events: Dict[str, List[int]] = field(default_factory=dict)
    config: Dict[str, object] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["%s bench: %d x %d tile -> %d cycles"
                 % (self.bench, self.rows, self.cols, self.cycles)]
        for name, (lo, hi) in self.phases.items():
            lines.append("  %-10s cycles %5d .. %5d  (%d)" % (name, lo, hi, hi - lo + 1))
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, object]:
        return {"bench": self.bench, "rows": self.rows, "cols": self.cols,
                "cycles": self.cycles,
                "phases": {k: list(v) for k, v in self.phases.items()},
                "events": self.events, "config": self.config}


def tile_values(rows: int, cols: int) -> List[List[int]]:
    """Distinct, non-zero, fp16-exact values, so a dropped or misplaced
    element shows up and a zero column can never look already stored."""
    return [[r * cols + c + 1 for c in range(cols)] for r in range(rows)]


def expected_columns(rows: int, cols: int) -> List[List[int]]:
    data = tile_values(rows, cols)
    return [[data[r][c] if r < rows else 0 for r in range(cols)]
            for c in range(cols)]


def _read_slot(spad, pad: int, slot: int, n: int) -> List[int]:
    out = []
    for lane in range(n):
        blob = spad.tiles[pad].banks[_xor_bank(slot, lane, spad.num_banks)].mem[slot]
        blob = bytes(blob or b"\x00\x00").ljust(2, b"\x00")
        out.append(int.from_bytes(blob[:2], "little", signed=False))
    return out


def _write_slot(spad, pad: int, slot: int, values) -> None:
    for lane, v in enumerate(values):
        spad.tiles[pad].banks[_xor_bank(slot, lane, spad.num_banks)].mem[slot] = \
            int(v).to_bytes(2, "little", signed=False)


class _Bench:
    """Shared machinery: the platform, a cycle counter, event bookkeeping."""

    def __init__(self, rows: int, limit: int, platform_kwargs: Dict):
        self.platform = build_tpu_platform(**platform_kwargs)
        self.vc = self.platform.vc
        self.n = self.vc.vector_len
        if not 1 <= rows <= self.n:
            raise ValueError("rows must be in 1..%d, got %d" % (self.n, rows))
        self.rows = rows
        self.limit = limit
        self.cycle = 0
        self.events: Dict[str, List[int]] = {}
        #: (instruction, event) pairs not yet accepted by the core
        self._pending: List[Tuple[Dict, Optional[str]]] = []
        self._unit_state = IDLE

    def mark(self, name: str) -> None:
        self.events.setdefault(name, []).append(self.cycle)

    def issue(self, inst: Dict, event: Optional[str] = None) -> None:
        """Queue an instruction; it is handed to the core in order, retrying
        each cycle while the core's packet queue is full."""
        self._pending.append((inst, event))

    def _drain_pending(self) -> None:
        while self._pending:
            inst, event = self._pending[0]
            if not self.vc.enqueue_scheduler_instruction(dict(inst)):
                return
            self._pending.pop(0)
            if event:
                self.mark(event)

    def step(self) -> Optional[Dict]:
        """One cycle: hand over what is queued, tick, report the writeback."""
        self._drain_pending()
        self.platform.tick(float(self.cycle))
        self._watch_unit()
        wb = self.vc.last_wb if self.vc.wb_valid else None
        return wb

    def _watch_unit(self) -> None:
        """Timestamp what the transpose unit itself does, as opposed to when
        the bench handed it an instruction: a row taken, a row stored in the
        banks, the drain starting."""
        prev, now = self._unit_state, self.vc.transpose.state
        writing = (WAIT_CLOS_WRITE, BUSY_WRITE)
        if prev == IDLE and now in writing:
            self.mark("unit_push")
        elif prev in writing and now not in writing:
            self.mark("unit_row_stored")
            if now != IDLE:          # straight into the next request
                self.mark("unit_pop_start")
        elif prev == IDLE and now != IDLE:
            self.mark("unit_pop_start")
        self._unit_state = now

    def advance(self) -> None:
        self.cycle += 1
        if self.cycle > self.limit:
            raise RuntimeError("transpose bench did not finish in %d cycles"
                               % self.limit)

    def check_column(self, col: int, data) -> None:
        want = expected_columns(self.rows, self.n)[col]
        got = [int(round(float(x))) for x in list(data)[:self.n]]
        if got != want:
            raise AssertionError("column %d came back wrong: %s, expected %s"
                                 % (col, got, want))

    def result(self, bench: str, phases: Dict[str, Tuple[str, str]],
               config: Dict) -> TransposeBenchResult:
        """`phases` maps a phase to its (start event, end event): the phase
        runs from the first start event to the last end event."""
        spans = {}
        for phase, (start, end) in phases.items():
            if self.events.get(start) and self.events.get(end):
                spans[phase] = (min(self.events[start]), max(self.events[end]))
        last = max(c for cs in self.events.values() for c in cs)
        return TransposeBenchResult(bench=bench, rows=self.rows, cols=self.n,
                                    cycles=last + 1, phases=spans,
                                    events={k: list(v) for k, v in self.events.items()},
                                    config=config)


def run_vrf_bench(rows: int = 32, *, limit: int = DEFAULT_LIMIT,
                  **platform_kwargs) -> TransposeBenchResult:
    """VRF -> transpose -> VRF. Rows sit in registers before cycle 0; done
    when the last transposed column has been written into the VRF."""
    b = _Bench(rows, limit, platform_kwargs)
    data = tile_values(rows, b.n)
    for r in range(rows):
        b.vc.write_vreg(ROW_REG_BASE + r, [float(v) for v in data[r]])

    for r in range(rows):
        b.issue({"unit": "transpose", "kind": "push", "src": ROW_REG_BASE + r},
                event="push_issue")
    b.issue({"unit": "transpose", "kind": "pop", "dst": COL_REG_BASE},
            event="pop_issue")

    cols_done = 0
    while cols_done < b.n:
        wb = b.step()
        if wb is not None and wb.get("source") == "transpose":
            col = int(wb["meta"]["col"])
            b.check_column(col, b.vc.read_vreg(COL_REG_BASE + col))
            b.mark("col_wb")
            cols_done += 1
        b.advance()

    return b.result("vrf", {"push": ("unit_push", "unit_row_stored"),
                            "drain": ("unit_pop_start", "col_wb")},
                    {"rows": rows, **platform_kwargs})


def run_spad_bench(rows: int = 32, *, load_pad: int = 0, store_pad: int = 0,
                   overlap: bool = True, load_window: Optional[int] = None,
                   limit: int = DEFAULT_LIMIT,
                   **platform_kwargs) -> TransposeBenchResult:
    """Scratchpad -> VLSU -> VRF -> transpose -> VRF -> VLSU -> scratchpad.

    The tile sits in `load_pad` before cycle 0; done when the last column has
    committed to `store_pad`'s banks. Each pad has its own VLSU, so loading
    and storing through different pads uses two VLSUs.

    overlap=True   push each row as soon as it is loaded, store each column
                   as soon as it is written back -- what a good schedule does.
    overlap=False  one phase at a time: all loads, then all pushes and the
                   pop, then all stores. The difference is what overlap buys.

    load_window    at most this many loads issued but not yet written back.
                   None issues every load in cycle 0. Packets issue in order,
                   so a long run of queued loads makes the first push wait
                   behind them; a window lets it in sooner.
    """
    b = _Bench(rows, limit, platform_kwargs)
    spad, n = b.platform.spad, b.n
    for pad in (load_pad, store_pad):
        if not 0 <= pad < len(spad.tiles):
            raise ValueError("pad must be in 0..%d" % (len(spad.tiles) - 1))
    data = tile_values(rows, n)
    for r in range(rows):
        _write_slot(spad, load_pad, IN_SLOT_BASE + r, data[r])
    want = expected_columns(rows, n)

    if load_window is not None and load_window < 1:
        raise ValueError("load_window must be >= 1")
    loads_issued = 0
    loaded: List[int] = []       # rows whose load has written back, in order
    pushed = 0
    columns: List[int] = []      # columns written back, in order
    stores_issued = 0
    in_flight: Dict[int, int] = {}   # column -> issue cycle, until committed
    committed = 0

    def issue_loads() -> None:
        nonlocal loads_issued
        window = rows if load_window is None else load_window
        while loads_issued < rows and loads_issued - len(loaded) < window:
            r = loads_issued
            b.issue({"unit": "vlsu", "kind": "load", "vls": load_pad,
                     "dst": ROW_REG_BASE + r, "addr": IN_SLOT_BASE + r,
                     "dtype": b.vc.dtype_default}, event="load_issue")
            loads_issued += 1

    def push_ready_rows() -> None:
        nonlocal pushed
        ready = loaded if overlap else (loaded if len(loaded) == rows else [])
        while pushed < len(ready):
            r = ready[pushed]
            b.issue({"unit": "transpose", "kind": "push",
                     "src": ROW_REG_BASE + r}, event="push_issue")
            pushed += 1
            if pushed == rows:
                b.issue({"unit": "transpose", "kind": "pop",
                         "dst": COL_REG_BASE}, event="pop_issue")

    def store_ready_columns() -> None:
        nonlocal stores_issued
        ready = columns if overlap else (columns if len(columns) == n else [])
        while stores_issued < len(ready):
            c = ready[stores_issued]
            b.issue({"unit": "vlsu", "kind": "store", "vls": store_pad,
                     "src": COL_REG_BASE + c, "addr": OUT_SLOT_BASE + c,
                     "dtype": b.vc.dtype_default}, event="store_issue")
            in_flight[c] = b.cycle
            stores_issued += 1

    issue_loads()
    while committed < n:
        wb = b.step()
        if wb is not None:
            src = wb.get("source")
            if src == "vlsu":
                r = int(wb["dst"]) - ROW_REG_BASE
                b.mark("load_wb")
                loaded.append(r)
            elif src == "transpose":
                c = int(wb["meta"]["col"])
                b.check_column(c, b.vc.read_vreg(COL_REG_BASE + c))
                b.mark("col_wb")
                columns.append(c)
        # A store has no writeback; it is done when its row is in the banks.
        for c in list(in_flight):
            if _read_slot(spad, store_pad, OUT_SLOT_BASE + c, n) == want[c]:
                b.mark("store_commit")
                del in_flight[c]
                committed += 1
        issue_loads()
        push_ready_rows()
        store_ready_columns()
        b.advance()

    return b.result(
        "spad",
        {"load": ("load_issue", "load_wb"),
         "push": ("unit_push", "unit_row_stored"),
         "drain": ("unit_pop_start", "col_wb"),
         "store": ("store_issue", "store_commit")},
        {"rows": rows, "load_pad": load_pad, "store_pad": store_pad,
         "overlap": overlap, "load_window": load_window, **platform_kwargs})
