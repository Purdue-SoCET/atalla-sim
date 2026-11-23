"""The vloop sequencer: hardware software pipelining of CISC vector loops.

A vloop instruction (relu.vl, ...) names a loop over n scratchpad rows:
load row i from the input pad, run a fixed body of micro-ops on it, store the
result to row i of the output pad. The sequencer overlaps the rows by modulo
scheduling: a new row starts every II cycles, and each micro-op runs at a fixed
offset from its row's start. One table per loop says which micro-ops, on
which units, at which offsets; one control (this class) runs every table.

The table
---------
schedule() builds a LoopTable from the body's micro-ops, the II and the unit
latencies: each op as early as its operands allow (offset = producer's offset +
producer's latency, so a consumer takes a result the cycle it comes out --
chaining), then later if a resource it needs is already used in that phase of
the II window (resources(): its FU row; a pad's request port, and its read
channel for 2 cycles per row read). Loads and stores share the pad's one
request per cycle when the input and output pads are the same, so the store
then avoids the load's phase. A body that doesn't fit the II asked for gets
the next II that does.
phase = offset % II says in which cycle of the window an op fires; stage =
offset // II in which window of its row. S = last stage + 1 rows are in flight.

The control
-----------
    IDLE   ready for a vloop. start() takes one (dispatch, in its EX cycle).
    ARM    wait until the input and output pads are quiet: no SDMA running on
           them, nothing left in their VLSUs from ordinary packets.
    RUN    each cycle, the ops whose phase is this cycle's and whose stage holds
           a row fire, for row (window - stage). `valid` has a bit per stage:
           every II cycles it shifts up and a 1 comes in while rows are left
           to load. Prologue, kernel and epilogue are just how full it is.
    DRAIN  wait for the last stores to leave the output pad's VLSU; then IDLE.

Stall: if an op due this cycle can't go -- an operand isn't there yet (a late
load) or its VLSU won't take a request -- nothing fires and the schedule
holds for the cycle. Ops already in an FU row finish into their buffer,
which is safe because the schedule's reuse of that buffer slot holds too.

Values in flight live in buffers keyed by (value, row): written when the
producer's result comes out, freed when the last reader has read it. They
stand for the per-slot circular buffers; peak_buffer records how many
entries each value needed, which is how deep its buffer must be.

FU rows
-------
The sequencer computes nothing itself: its ALU/MUL/... micro-ops run on the
vector core's FU rows, attached with connect_fu_rows({kind: row}). A row is
expected to take a whole vector per cycle, be fully pipelined, and offer

    row.latency                     cycles from fire to result
    row.fire(cycle, op, src0, src1, mask, tag)
                                    start op ("gt", "mul", ...) on whole vectors;
                                    mask (an int) zeroes masked-off elements;
                                    src1 may be None
    row.take(cycle)                 [(tag, result vector)] due by `cycle`
    row.busy                        anything still in the pipeline
    row.ready()                     (optional) whether it can take an op this
                                    cycle; an op due on a row that can't stalls

On the M x N vector core these are vector_core/v2/kernel_port.FUGroupPort.

A table is scheduled with the connected rows' latencies (FU_ROW_LATENCY until
then). With no row of a kind connected, a loop that needs one raises.

Variants
--------
A loop can have several bodies that compute the same result with different
costs (variants_of; "keep" is LOOPS' body). The one in use is chosen per
sequencer, use_variant(loop, name) or VLoopSequencer(variants=...). The
built-in trade is buffer area against scratchpad reads: with_reloads()
makes a body that loads a value again for its late readers instead of
holding it, so its buffer is shallower and the input pad is read more
(gelu.vl's "reload" and "reload_all"). buffer_bits() and stats["loads"]
give both sides.

The sequencer holds its pads from start() until DRAIN ends (held_pads();
decode 2 blocks VLSU ops and SDMAs on them), and tells decode 2 which FU
rows it will fire in the next cycle (fires_next()), so ordinary lane ops
don't take them.
"""

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import struct

ISSUE_SLOTS = 4
LOAD, STORE = "load", "store"
#: Cycles from firing to result, per FU row kind (ALU and MUL as specified;
#: EXP, RECIP, SQRT the lane model's), until FU rows are connected and their
#: own latencies are used.
FU_ROW_LATENCY = {"alu": 4, "mul": 4, "exp": 14, "recip": 11, "sqrt": 8}
#: Cycles from a load's request to its data: this model's VLSU adds a cycle
#: on the way to the pad and one on the way back around the pad's 6. The
#: same whether the sequencer runs after the VLSUs in a cycle (lane side) or
#: before them (M x N side). Measured; see tests/scheduler/test_vloop.py.
LOAD_LATENCY = 8
DEFAULT_LATENCY = dict(FU_ROW_LATENCY, load=LOAD_LATENCY, store=0)
#: Cycles a pad's read channel is busy per row read: a pad reads at most one
#: row every 2 cycles (memory/scratchpad.py, the SRAM's read latency). Its
#: write channel takes a row every cycle, and it accepts one request a cycle.
PAD_READ_CYCLES = 2
PAD_WRITE_CYCLES = 1

IDLE, ARM, RUN, DRAIN = "IDLE", "ARM", "RUN", "DRAIN"


def bf16_from_bits(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (int(b) & 0xFFFF) << 16))[0]


def bf16_to_bits(x: float) -> int:
    """x must already be a BF16 value (FU rows round their results)."""
    return struct.unpack("<I", struct.pack("<f", float(x)))[0] >> 16


def pack_mask(vec: Sequence[float]) -> int:
    """A compare's 1.0/0.0 elements as a mask, element i in bit i."""
    return sum(1 << i for i, x in enumerate(vec) if x)


@dataclass(frozen=True)
class MicroOp:
    """One op of a loop body.

    unit   "load", "store", or the FU row kind it runs on
    op     the FU op ("gt", "mul", ...)
    srcs   values it reads: a value an earlier op produces, "v0" (zeros), or
           "c:<float>" (a constant broadcast to every element)
    mask   a mask value; masked-off elements come out 0 (zeroing)
    out    the value it produces
    to_mask  pack the 1.0/0.0 result into a mask
    jit    value(s) this op's result must meet: it fires as late as that
           allows (its result lands when the latest of them is ready)
           instead of as early as possible. For reloads (with_reloads).
    """
    name: str
    unit: str
    op: str = ""
    srcs: Tuple[str, ...] = ()
    mask: Optional[str] = None
    out: Optional[str] = None
    to_mask: bool = False
    jit: Union[str, Tuple[str, ...], None] = None

    def jit_values(self) -> Tuple[str, ...]:
        if not self.jit:
            return ()
        return (self.jit,) if isinstance(self.jit, str) else tuple(self.jit)

    def reads(self) -> List[str]:
        named = [s for s in self.srcs if s != "v0" and not s.startswith("c:")]
        return named + ([self.mask] if self.mask else [])


@dataclass(frozen=True)
class LoopTable:
    name: str
    ii: int
    ops: Tuple[MicroOp, ...]
    offsets: Tuple[int, ...]
    #: The II asked for; `ii` is larger when the body didn't fit in it.
    ii_wanted: int = 0
    #: Per op, the offset its result is ready (offset + latency); None for
    #: an op that produces nothing.
    ready: Tuple[Optional[int], ...] = ()

    @property
    def stages(self) -> int:
        return max(self.offsets) // self.ii + 1

    def phase(self, i: int) -> int:
        return self.offsets[i] % self.ii

    def stage(self, i: int) -> int:
        return self.offsets[i] // self.ii

    def buffer_depths(self) -> Dict[str, int]:
        """Entries each value's buffer needs in steady state: written when
        ready, freed when its last reader fires, a new row every II cycles
        -- (last read - ready) // II + 1. What peak_buffer measures once a
        run has rows enough to fill the pipeline; independent of n."""
        last: Dict[str, int] = {}
        for op, off in zip(self.ops, self.offsets):
            for v in op.reads():
                last[v] = max(last.get(v, off), off)
        return {op.out: (last[op.out] - r) // self.ii + 1
                for op, r in zip(self.ops, self.ready) if op.out and op.out in last}

    def readers(self) -> Dict[str, int]:
        """How many reads each value gets per row."""
        out: Dict[str, int] = {}
        for op in self.ops:
            for v in op.reads():
                out[v] = out.get(v, 0) + 1
        return out


def resources(op: MicroOp, share_pad: bool,
              pad_read_cycles: int = PAD_READ_CYCLES) -> Tuple[Tuple[str, int], ...]:
    """(resource, cycles it is busy) for each resource an op takes when it
    fires. A load takes its pad's request port for a cycle and its read
    channel for pad_read_cycles; a store the request port and the write
    channel. One pad for input and output shares the request port (the read
    and write channels stay separate). An FU op takes its row for a cycle
    (fully pipelined)."""
    if op.unit == LOAD:
        return (("pad" if share_pad else "pad_in", 1), ("pad_in:read", pad_read_cycles))
    if op.unit == STORE:
        return (("pad" if share_pad else "pad_out", 1), ("pad_out:write", PAD_WRITE_CYCLES))
    return ((op.unit, 1),)


class _NoFit(Exception):
    pass


def _candidates(op: MicroOp, lo: int, ii: int, need, latency: Dict[str, int],
                produced: Dict[str, int]) -> List[int]:
    """Offsets to try for `op`, best first. Normally lo, lo+1, ... (as early
    as possible). With `jit`, the offset that meets it, then the nearest
    ones either side (earlier first on a tie: it costs a buffer cycle, not
    latency), never before lo. Offsets where a resource busy for several
    cycles starts on a multiple of that come first, so such windows tile the
    II without gaps (three 2-cycle pad reads fit II = 6)."""
    if op.jit:
        want = max([lo] + [produced[v] - latency[op.unit] for v in op.jit_values()])
        cands = sorted(range(max(lo, want - ii + 1), want + ii),
                       key=lambda c: (abs(c - want), c))
    else:
        cands = list(range(lo, lo + ii))
    multi = [c for _, c in need if c > 1 and ii % c == 0]
    return sorted(cands, key=lambda c: any((c % ii) % m for m in multi))    # stable


def _schedule_at(ops: Sequence[MicroOp], ii: int, latency: Dict[str, int], share_pad: bool,
                 pad_read_cycles: int) -> Tuple[int, ...]:
    produced: Dict[str, int] = {}
    used = set()                                     # (resource, phase)
    per_phase = [0] * ii                             # ops per phase: ISSUE_SLOTS at most
    offsets = []
    for op in ops:
        lo = max([produced[v] for v in op.reads()] or [0])  # earliest its operands allow
        need = resources(op, share_pad, pad_read_cycles)
        for t in _candidates(op, lo, ii, need, latency, produced):
            busy = {(res, (t + j) % ii) for res, cycles in need for j in range(cycles)}
            if (len(busy) == sum(c for _, c in need) and not busy & used
                    and per_phase[t % ii] < ISSUE_SLOTS):
                break
        else:
            raise _NoFit(op.name)
        used |= busy
        per_phase[t % ii] += 1
        offsets.append(t)
        if op.out:
            produced[op.out] = t + latency[op.unit]
    return tuple(offsets)


def schedule(name: str, ops: Sequence[MicroOp], ii: int, latency: Dict[str, int],
             share_pad: bool, pad_read_cycles: int = PAD_READ_CYCLES) -> LoopTable:
    """Modulo-schedule a body, ops given in dependence order (every value is
    produced before it is read). Runs once per loop and variant.

    Each op goes as early as its operands allow (or as late as its `jit`
    allows), then later while a resource it needs (resources()) is taken in
    that phase or its phase already has ISSUE_SLOTS ops. `ii` is the II
    wanted; if the body doesn't fit, the next larger II that fits is used
    (the table's ii_wanted keeps the one asked for) -- e.g. a body with more
    pad reads per row than a pad can do in `ii` cycles."""
    worst = ii + sum(c for op in ops for _, c in resources(op, share_pad, pad_read_cycles))
    for cand in range(ii, worst + 1):
        try:
            offsets = _schedule_at(ops, cand, latency, share_pad, pad_read_cycles)
        except _NoFit:
            continue
        ready = tuple(off + latency[op.unit] if op.out else None
                      for op, off in zip(ops, offsets))
        return LoopTable(name, cand, tuple(ops), offsets, ii_wanted=ii, ready=ready)
    raise ValueError("%s: no II from %d to %d fits" % (name, ii, worst))


#: relu: v > 0 as a mask, then v * 1.0 with that mask zeroing the rest.
RELU_OPS = (
    MicroOp("LD", LOAD, out="v"),
    MicroOp("CMP", "alu", "gt", srcs=("v", "v0"), out="m", to_mask=True),
    MicroOp("MUL", "mul", "mul", srcs=("v", "c:1.0"), mask="m", out="r"),
    MicroOp("ST", STORE, srcs=("r",)),
)

#: gelu (tanh form): x / (1 + e^t), t = x*(-c1 - c2*x^2), c1 = 2*sqrt(2/pi),
#: c2 = 0.044715*c1; constants rounded to BF16. Four MULs on the one MUL group
#: make II = 4. Known, accepted: gelu.vl(-inf) is NaN (t = +inf, s = 0,
#: y = -inf * 0); true GELU gives 0. The functional sim's vector_loop.gelu
#: defines the same sequence.
GELU_OPS = (
    MicroOp("LD",    LOAD,    out="x"),
    MicroOp("SQ",    "mul", "mul", srcs=("x", "x"),             out="a"),  # x^2
    MicroOp("MULC",  "mul", "mul", srcs=("a", "c:-0.0712890625"), out="b"),  # -c2*x^2
    MicroOp("ADDC",  "alu", "add", srcs=("b", "c:-1.59375"),    out="c"),  # -c1 - c2*x^2
    MicroOp("MULX",  "mul", "mul", srcs=("c", "x"),             out="t"),  # t = -2u
    MicroOp("EXP",   "exp", "exp", srcs=("t",),                 out="e"),
    MicroOp("ADD1",  "alu", "add", srcs=("e", "c:1.0"),         out="d"),
    MicroOp("RECIP", "recip", "recip", srcs=("d",),             out="s"),
    MicroOp("MULY",  "mul", "mul", srcs=("x", "s"),             out="y"),
    MicroOp("ST",    STORE,   srcs=("y",)),
)


def with_reloads(body: Sequence[MicroOp], value: str, readers: Sequence[str]
                 ) -> Tuple[MicroOp, ...]:
    """`body` with `value` (a loaded value) loaded again from the input pad
    for each op named in `readers`, instead of held in its buffer until
    then: buffer area traded for scratchpad reads. Each reload is placed
    just before its reader and fires just in time (jit) to meet the reader's
    other operands, so its copy is buffered for about one entry. The copy
    is named "<value>@<reader>", the load "LD_<value>@<reader>"."""
    loads = {op.out for op in body if op.unit == LOAD}
    if value not in loads:
        raise ValueError("%r isn't loaded, so it can't be reloaded" % value)
    names = {op.name for op in body}
    missing = [r for r in readers if r not in names]
    if missing:
        raise ValueError("no ops named %s" % missing)
    out: List[MicroOp] = []
    for op in body:
        if op.name in readers:
            if value not in op.srcs:
                raise ValueError("%s doesn't read %r" % (op.name, value))
            copy = "%s@%s" % (value, op.name)
            others = tuple(v for v in op.reads() if v != value)
            out.append(MicroOp("LD_" + copy, LOAD, out=copy, jit=others or None))
            op = replace(op, srcs=tuple(copy if v == value else v for v in op.srcs))
        out.append(op)
    return tuple(out)


#: funct -> (name, body, II). isa.VLOOP_FUNCTS gives the same names. The
#: body here is the loop's default variant, "keep".
LOOPS = {0: ("relu.vl", RELU_OPS, 2), 1: ("gelu.vl", GELU_OPS, 4)}

DEFAULT_VARIANT = "keep"
#: Other bodies a loop can run, by funct, chosen per sequencer with
#: use_variant(); every variant must compute the same result. gelu: x is
#: read three times (SQ at the start, MULX, MULY at the end), so keeping it
#: needs the deepest buffer of any value.
#:   reload      x loaded again for MULY: 2 loads a row
#:   reload_all  x loaded again for MULX and MULY: 3 loads a row
EXTRA_VARIANTS: Dict[int, Dict[str, Tuple[MicroOp, ...]]] = {
    1: {"reload": with_reloads(GELU_OPS, "x", ("MULY",)),
        "reload_all": with_reloads(GELU_OPS, "x", ("MULX", "MULY"))},
}


def variants_of(funct: int) -> Dict[str, Tuple[MicroOp, ...]]:
    """A loop's bodies by variant name, "keep" (LOOPS' body) first."""
    if funct not in LOOPS:
        raise ValueError("no vloop table for funct %d" % funct)
    return {DEFAULT_VARIANT: tuple(LOOPS[funct][1]), **EXTRA_VARIANTS.get(funct, {})}


def funct_of(loop) -> int:
    """A loop's funct from its funct or its name ("gelu.vl")."""
    if isinstance(loop, int):
        return loop
    for f, (name, _, _) in LOOPS.items():
        if name == loop:
            return f
    raise ValueError("no vloop named %r" % loop)


def table_name(name: str, variant: str, same_pad: bool) -> str:
    """How tables_run and the perf report name a table: "gelu.vl",
    "gelu.vl:reload", "relu.vl/same_pad"."""
    return (name + ("" if variant == DEFAULT_VARIANT else ":" + variant)
            + ("/same_pad" if same_pad else ""))


@dataclass
class _Loop:
    table: LoopTable
    in_row: int
    out_row: int
    n: int
    sid_in: int
    sid_out: int


class VLoopSequencer:
    """See the module docstring. `vls_units` are the vector core's VLSUs,
    one per scratchpad; `sdma_busy(pad)` says whether an SDMA is running
    on a pad."""

    def __init__(self, vls_units: Sequence, sdma_busy: Callable[[int], bool],
                 vector_len: int = 32, latency: Optional[Dict[str, int]] = None,
                 fu_rows: Optional[Dict[str, object]] = None,
                 load_latency: Optional[int] = None,
                 variants: Optional[Dict[object, str]] = None):
        self.vls = list(vls_units)
        self.sdma_busy = sdma_busy
        self.n_elems = int(vector_len)
        self.latency = dict(DEFAULT_LATENCY, **(latency or {}))
        if load_latency is not None:
            self.latency[LOAD] = int(load_latency)
        self.rows: Dict[str, object] = {}
        self._tables: Dict[Tuple[int, bool, str], LoopTable] = {}
        self.mode = IDLE
        self.loop: Optional[_Loop] = None
        self.sc = 0                 # schedule cycle: advances when nothing stalls
        self.valid = 0              # bit s: stage s holds a row
        self.buf: Dict[Tuple[str, int], object] = {}
        self._refs: Dict[Tuple[str, int], int] = {}
        self._load_fired: Dict[Tuple[str, int], int] = {}     # (value, row) -> cycle
        self.stats = dict(loops=0, rows=0, arm_cycles=0, run_cycles=0, drain_cycles=0,
                          stall_cycles=0, loads=0, stores=0, fu_ops=0,
                          load_latency_min=None, load_latency_max=None)
        self.stats["fired"] = {}            # op name -> count
        self.stats["max_fired_per_cycle"] = 0
        self.peak_buffer: Dict[str, int] = {}
        #: Every table a loop has run with, by table_name() (the perf
        #: monitor reports their II, offsets and stages).
        self.tables_run: Dict[str, LoopTable] = {}
        #: (cycle, op name, row) for every op fired, oldest first.
        self.trace: List[Tuple[int, str, int]] = []
        self.keep_trace = False
        #: funct -> the variant its loops run (variants_of); "keep" if absent.
        self.variants: Dict[int, str] = {}
        for loop, variant in (variants or {}).items():
            self.use_variant(loop, variant)
        if fu_rows:
            self.connect_fu_rows(fu_rows)

    def use_variant(self, loop, variant: str) -> None:
        """Run loop `loop` (funct or name) with body `variant` from now on."""
        if self.mode != IDLE:
            raise RuntimeError("variant changed while a loop runs")
        funct = funct_of(loop)
        if variant not in variants_of(funct):
            raise ValueError("%s has no variant %r (has %s)"
                             % (LOOPS[funct][0], variant, sorted(variants_of(funct))))
        self.variants[funct] = variant

    def variant(self, funct: int) -> str:
        return self.variants.get(funct, DEFAULT_VARIANT)

    def connect_fu_rows(self, rows: Dict[str, object]) -> None:
        """Attach the vector core's FU rows, by kind ("alu", "mul", ...); the
        tables are rescheduled with their latencies."""
        if self.mode != IDLE:
            raise RuntimeError("FU rows connected while a loop runs")
        self.rows.update(rows)
        for kind, row in rows.items():
            self.latency[kind] = int(row.latency)
        self._tables.clear()

    # -- tables ---------------------------------------------------------------------
    def table_for(self, funct: int, same_pad: bool) -> LoopTable:
        """The table for loop `funct` in its current variant."""
        key = (funct, same_pad, self.variant(funct))
        if key not in self._tables:
            body = variants_of(funct)[key[2]]
            name, _, ii = LOOPS[funct]
            self._tables[key] = schedule(name, body, ii, self.latency, share_pad=same_pad)
        return self._tables[key]

    def buffer_bits(self) -> Dict[str, int]:
        """Bits each value's buffer needs for the tables run so far, at their
        steady-state depth (LoopTable.buffer_depths, the deepest over the
        tables): a vector entry is n_elems BF16 values, a mask entry one bit
        per element. The area side of a variant's trade; independent of how
        many rows ran (peak_buffer is the depth a run actually reached)."""
        out: Dict[str, int] = {}
        for t in self.tables_run.values():
            masks = {op.out for op in t.ops if op.to_mask}
            for v, depth in t.buffer_depths().items():
                out[v] = max(out.get(v, 0), depth * self.n_elems * (1 if v in masks else 16))
        return out

    # -- what decode 2 sees ------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self.mode == IDLE

    @property
    def idle(self) -> bool:
        return self.mode == IDLE and not any(r.busy for r in self.rows.values())

    def held_pads(self) -> List[bool]:
        held = [False] * 4
        if self.loop is not None:
            held[self.loop.sid_in] = held[self.loop.sid_out] = True
        return held

    def fires_next(self, unit: str) -> bool:
        """Whether an op on `unit` is due next cycle, if nothing stalls."""
        if self.mode != RUN:
            return False
        t, lp = self.loop.table, self.loop
        sc = self.sc + 1
        for i, op in enumerate(t.ops):
            if op.unit == unit and sc % t.ii == t.phase(i):
                row = sc // t.ii - t.stage(i)
                if 0 <= row < lp.n:
                    return True
        return False

    # -- start ------------------------------------------------------------------------------
    def start(self, funct: int, in_row: int, out_row: int, n: int, sid_in: int,
              sid_out: int) -> None:
        if self.mode != IDLE:
            raise RuntimeError("vloop started while the sequencer is busy")
        same = sid_in == sid_out
        table = self.table_for(funct, same)
        self.tables_run[table_name(table.name, self.variant(funct), same)] = table
        self.loop = _Loop(table, int(in_row), int(out_row), int(n), int(sid_in), int(sid_out))
        self.mode = ARM
        self.stats["loops"] += 1

    # -- the clock ----------------------------------------------------------------------------
    def tick(self, cycle: int) -> None:
        """One cycle: take load data and FU results that are due, then act."""
        self._collect(cycle)
        if self.mode == ARM:
            self.stats["arm_cycles"] += 1
            lp = self.loop
            if all(self._quiet(p) and not self.sdma_busy(p) for p in (lp.sid_in, lp.sid_out)):
                self.sc, self.valid = 0, (1 if lp.n > 0 else 0)
                self.mode = RUN if lp.n > 0 else DRAIN
        elif self.mode == RUN:
            self.stats["run_cycles"] += 1
            self._step(cycle)
        elif self.mode == DRAIN:
            self.stats["drain_cycles"] += 1
            lp = self.loop
            if self._quiet(lp.sid_in) and self._quiet(lp.sid_out) and not self.buf:
                self.mode, self.loop = IDLE, None

    def _quiet(self, pad: int) -> bool:
        v = self.vls[pad]
        return (v.issue_q.is_empty() and v.req_q.is_empty() and v.rsp_q.is_empty()
                and v.wb_q.is_empty() and v.outstanding_loads() == 0)

    def _put(self, name: str, row: int, value) -> None:
        key = (name, row)
        self.buf[key] = value
        self._refs[key] = self.loop.table.readers().get(name, 0)
        live = sum(1 for k in self.buf if k[0] == name)
        self.peak_buffer[name] = max(self.peak_buffer.get(name, 0), live)

    def _collect(self, cycle: int) -> None:
        if self.loop is None:
            return
        v = self.vls[self.loop.sid_in]
        while True:
            wb = v.wb_q.peek()
            if wb is None or not (wb.get("meta") or {}).get("vloop"):
                break
            v.wb_q.dequeue()
            meta = wb["meta"]
            row = int(meta["row"])
            lat = cycle - self._load_fired.pop((meta["value"], row))
            s = self.stats
            s["load_latency_min"] = lat if s["load_latency_min"] is None else min(s["load_latency_min"], lat)
            s["load_latency_max"] = lat if s["load_latency_max"] is None else max(s["load_latency_max"], lat)
            data = [bf16_from_bits(b) for b in wb["data"]][: self.n_elems]
            self._put(meta["value"], row, data + [0.0] * (self.n_elems - len(data)))
        for r in self.rows.values():
            for (name, row, to_mask), value in r.take(cycle):
                self._put(name, row, pack_mask(value) if to_mask else value)

    def _operand(self, src: str, row: int):
        if src == "v0":
            return [0.0] * self.n_elems
        if src.startswith("c:"):
            return [float(src[2:])] * self.n_elems
        return self.buf[(src, row)]

    def _can_fire(self, op: MicroOp, row: int) -> bool:
        if any((v, row) not in self.buf for v in op.reads()):
            return False
        lp = self.loop
        if op.unit == LOAD:
            return self.vls[lp.sid_in].can_accept_issue()
        if op.unit == STORE:
            return self.vls[lp.sid_out].can_accept_issue()
        row = self.rows.get(op.unit)
        return row is None or not hasattr(row, "ready") or row.ready()

    def _step(self, cycle: int) -> None:
        lp, t = self.loop, self.loop.table
        phase, window = self.sc % t.ii, self.sc // t.ii
        due = [(i, window - t.stage(i)) for i in range(len(t.ops))
               if t.phase(i) == phase and (self.valid >> t.stage(i)) & 1]
        if not all(self._can_fire(t.ops[i], row) for i, row in due):
            self.stats["stall_cycles"] += 1
            return
        self.stats["max_fired_per_cycle"] = max(self.stats["max_fired_per_cycle"], len(due))
        for i, row in due:
            self._fire(cycle, t.ops[i], row)
        self.sc += 1
        if self.sc % t.ii == 0:
            more = 1 if self.sc // t.ii < lp.n else 0
            self.valid = ((self.valid << 1) | more) & ((1 << t.stages) - 1)
            if self.valid == 0:
                self.mode = DRAIN

    def _fire(self, cycle: int, op: MicroOp, row: int) -> None:
        lp = self.loop
        self.stats["fired"][op.name] = self.stats["fired"].get(op.name, 0) + 1
        if self.keep_trace:
            self.trace.append((cycle, op.name, row))
        if op.unit == LOAD:
            self._load_fired[(op.out, row)] = cycle
            assert self.vls[lp.sid_in].enqueue_issue({
                "kind": "load", "scratchpad": 0, "vd": -1, "addr": lp.in_row + row,
                "vl": self.n_elems, "dtype": "bf16",
                "meta": {"vloop": True, "value": op.out, "row": row}})
            self.stats["loads"] += 1
        elif op.unit == STORE:
            data = [bf16_to_bits(x) for x in self._operand(op.srcs[0], row)]
            assert self.vls[lp.sid_out].enqueue_issue({
                "kind": "store", "scratchpad": 0, "data": data, "addr": lp.out_row + row,
                "vl": self.n_elems, "meta": {"vloop": True}})
            self.stats["stores"] += 1
            self.stats["rows"] += 1
        else:
            if op.unit not in self.rows:
                raise RuntimeError(
                    "%s needs a %s FU row, and none is connected (connect_fu_rows)"
                    % (lp.table.name, op.unit))
            src0 = self._operand(op.srcs[0], row)
            src1 = self._operand(op.srcs[1], row) if len(op.srcs) > 1 else None
            mask = self.buf[(op.mask, row)] if op.mask else None
            self.rows[op.unit].fire(cycle, op.op, src0, src1, mask,
                                    tag=(op.out, row, op.to_mask))
            self.stats["fu_ops"] += 1
        for v in op.reads():
            key = (v, row)
            self._refs[key] -= 1
            if self._refs[key] == 0:
                del self.buf[key], self._refs[key]
