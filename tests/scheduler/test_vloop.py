"""The vloop sequencer (vector_core/vloop.py): every CISC vector loop
(relu.vl, gelu.vl, ...) over n scratchpad rows, alone and on the
scheduler-driven platform.

Most tests here run for every loop in vloop.LOOPS, in every variant
(vloop.variants_of: "keep", and e.g. gelu.vl's "reload" / "reload_all"),
as a Case. Adding a loop means:
  1. its body in vloop.LOOPS and its funct in isa.VLOOP_FUNCTS,
  2. its element function in the functional sim (vector_loop.VLOOP_FUNCTIONS),
  3. a Spec in SPECS below: the bit-exact element result, input values to
     draw from, the math function it computes and how close it must be.
test_every_loop_has_a_spec fails until all three agree. Adding a variant
(vloop.EXTRA_VARIANTS) needs nothing here: every variant must store the
same bits, so the same tests check it.

Two platforms: the lane core ("v1"), whose FU rows here are StubRow test
doubles for ALU and MUL only, and the M x N core ("v2"), whose FU groups are
the rows. A loop runs on v1 only if it needs nothing but ALU and MUL.

Loop-specific tests (exact tables, buffer depths, the activity view) follow
the generic ones.
"""
import math
import random
import struct
from collections import deque
from dataclasses import dataclass
from typing import Callable, Tuple

import pytest

from atalla.atalla_platform import build_atalla_platform
from base.dtype import DType, cast_scalar, cast_vector
from scheduler.isa import VLOOP_FUNCTS
from vector_core.v2.fu_group import _OPS
from vector_core.vector_lanes import _op_lut
from vector_core.vloop import (DEFAULT_LATENCY, DEFAULT_VARIANT, ISSUE_SLOTS, LOAD,
                               LOAD_LATENCY, LOOPS, STORE, MicroOp, VLoopSequencer,
                               bf16_from_bits, bf16_to_bits, resources, schedule,
                               table_name, variants_of, with_reloads)
from test_execute import HALT, OP, base, i, program
from test_vector import DRAM_IN, DRAM_OUT, meta, sdma, tile_data, vmem, vv


def bf(x):
    return cast_scalar(x, DType.BF16)


# -- what each loop must compute ---------------------------------------------------------------
def relu_ref(x):
    """+0.0 for negatives, -0.0, -inf and NaN: zeroing writes +0."""
    return x if x > 0 else 0.0


#: gelu.vl's constants: -c1, -c2 of tanh GELU in BF16 (c1 = 2*sqrt(2/pi),
#: c2 = 0.044715*c1).
GELU_C1, GELU_C2 = -1.59375, -0.0712890625


def gelu_ref(x):
    """gelu.vl's micro-op chain, BF16 after every step. -inf gives NaN (accepted)."""
    a = bf(x * x)
    b = bf(a * GELU_C2)
    c = bf(b + GELU_C1)
    t = bf(c * x)
    e = bf(_OPS["exp"](t, 0))
    d = bf(e + 1.0)
    s = bf(_OPS["recip"](d, 0))
    return bf(x * s)


def tanh_gelu(x):
    k = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + math.tanh(k * (x + 0.044715 * x ** 3)))


@dataclass(frozen=True)
class Spec:
    """ref: the exact BF16 result per element (what both sims must store).
    edges: inputs every test row is seeded with besides random BF16 values in
    [-6, 6]. true / rel_tol / abs_tol: the math function the loop computes,
    and how far ref may be from it (rel_tol where |true| > 1/64, abs_tol
    below)."""
    ref: Callable[[float], float]
    edges: Tuple[float, ...]
    true: Callable[[float], float]
    rel_tol: float = 0.0
    abs_tol: float = 0.0


INF = float("inf")
SPECS = {
    "relu.vl": Spec(relu_ref,
                    edges=(-3.0, -1.5, -0.5, -0.0, 0.0, 0.25, 1.0, 2.5, 4.0, INF, -INF),
                    true=lambda x: max(x, 0.0)),
    # Measured over every BF16 value in [-8, 8]: 2.7% relative (BF16 rounding
    # of t = -2u, amplified by exp), 6.6e-4 absolute near 0. -inf is left out
    # of the edges: gelu.vl(-inf) is NaN, see test_gelu_of_minus_inf_is_nan.
    "gelu.vl": Spec(gelu_ref,
                    edges=(0.0, -0.0, 1e-3, -1e-3, 0.5, -0.5, 5.5, -5.5, 12.0, -12.0,
                           100.0, -100.0, INF),
                    true=tanh_gelu, rel_tol=2 ** -5, abs_tol=2 ** -10),
}

FUNCTS = sorted(LOOPS)
NAME = {f: LOOPS[f][0] for f in FUNCTS}
#: The FU row kinds the v1 platform has (StubRows).
V1_KINDS = {"alu", "mul"}


@dataclass(frozen=True)
class Case:
    """One loop in one variant: what a generic test runs."""
    funct: int
    variant: str = DEFAULT_VARIANT

    @property
    def name(self):
        return LOOPS[self.funct][0]

    @property
    def ii(self):
        return LOOPS[self.funct][2]

    @property
    def body(self):
        return variants_of(self.funct)[self.variant]

    @property
    def spec(self):
        return SPECS[self.name]

    @property
    def kinds(self):
        """The FU row kinds the body uses, in body order."""
        out = []
        for op in self.body:
            if op.unit not in (LOAD, STORE) and op.unit not in out:
                out.append(op.unit)
        return out

    @property
    def arches(self):
        return ["v1", "v2"] if set(self.kinds) <= V1_KINDS else ["v2"]

    @property
    def loads(self):
        return sum(1 for op in self.body if op.unit == LOAD)

    def __str__(self):
        return table_name(self.name, self.variant, False)


def as_case(c):
    return c if isinstance(c, Case) else Case(c)


RELU, GELU = Case(0), Case(1)
CASES = [Case(f, v) for f in FUNCTS for v in variants_of(f)]
#: One case per loop, for what a variant can't change.
LOOP_CASES = [pytest.param(Case(f), id=NAME[f]) for f in FUNCTS]
VARIANT_CASES = [pytest.param(c, id=str(c)) for c in CASES]
ARCH_CASES = [pytest.param(c, a, id="%s-%s" % (c, a)) for c in CASES for a in c.arches]
#: The variants other than "keep", each against its loop's "keep".
OTHER_VARIANTS = [pytest.param(c, id=str(c)) for c in CASES if c.variant != DEFAULT_VARIANT]


# -- a stand-in FU row for the lane core (test double) --------------------------------------
class StubRow:
    """Fully pipelined, a whole vector per cycle, zeroing mask, BF16 results."""
    OPS = _op_lut()

    def __init__(self, latency):
        self.latency = latency
        self.pipe = deque()
        self.fired = set()

    def fire(self, cycle, op, src0, src1, mask, tag):
        assert cycle not in self.fired, "two ops in one cycle on one row"
        self.fired.add(cycle)
        b = src1 if src1 is not None else [0.0] * len(src0)
        out = cast_vector([self.OPS[op](x, y) for x, y in zip(src0, b)], DType.BF16)
        if mask is not None:
            out = [v if (mask >> k) & 1 else 0.0 for k, v in enumerate(out)]
        self.pipe.append((cycle + self.latency, tag, list(out)))

    def take(self, cycle):
        out = []
        while self.pipe and self.pipe[0][0] <= cycle:
            _, tag, v = self.pipe.popleft()
            out.append((tag, v))
        return out

    @property
    def busy(self):
        return bool(self.pipe)


# -- programs and platforms --------------------------------------------------------------------
def vl(funct, rs_in, rs_out, rs_n, sid_in, sid_out):
    return (OP["vloop"] | rs_in << 7 | rs_out << 15 | rs_n << 23 | sid_in << 31
            | sid_out << 33 | funct << 35)


def vloop_program(funct, n, sid_in=0, sid_out=1, in_row=0, out_row=0, rows=None, after=()):
    """DRAM_IN -> pad sid_in rows in_row.., the loop over n rows into pad
    sid_out from out_row, then the output rows to DRAM_OUT."""
    rows = rows or max(n, 1)
    pk = [*base(1, DRAM_IN), *base(3, meta(sid_in, rows)), i("addi.s", 2, 0, in_row * 64),
          (sdma("scpad.ld", 2, 1, 3),),
          (i("addi.s", 5, 0, in_row),), (i("addi.s", 6, 0, out_row),),
          (i("addi.s", 7, 0, n),),
          (vl(funct, 5, 6, 7, sid_in, sid_out),),
          *after,
          i("addi.s", 2, 0, out_row * 64), *base(1, DRAM_OUT), *base(3, meta(sid_out, rows)),
          (sdma("scpad.st", 2, 1, 3),), HALT]
    return program(*pk)


def relu_program(n, **kw):
    return vloop_program(0, n, **kw)


def platform(prog, data, arch="v1", case=None):
    """v1: the lane core with StubRows for ALU and MUL. v2: the M x N core.
    `case` sets its loop's variant."""
    variants = {case.funct: case.variant} if case is not None else None
    if arch == "v2":
        return build_atalla_platform(prog, data, warm_icache=True, vector_arch="fu_groups",
                                     vloop_variants=variants)
    plat = build_atalla_platform(prog, data, warm_icache=True, vloop_variants=variants)
    plat.core.vector.vloop.connect_fu_rows({"alu": StubRow(4), "mul": StubRow(4)})
    return plat


def values(case, seed):
    """Inputs for a loop: a fifth from its spec's edges, the rest random BF16 in [-6, 6]."""
    rng = random.Random(seed)
    edges = as_case(case).spec.edges
    return lambda r, c: rng.choice(edges) if rng.random() < 0.2 else bf(rng.uniform(-6, 6))


def make_grid(case, rows, seed):
    vals = values(case, seed)
    return [[vals(r, c) for c in range(32)] for r in range(rows)]


def run(case, n, arch="v2", seed=0, after=(), **kw):
    """The loop over n rows of random inputs; the platform and the input grid."""
    case = as_case(case)
    rows = max(n, 1)
    grid = make_grid(case, rows, seed)
    data = tile_data(DRAM_IN, rows, lambda r, c: grid[r][c])
    plat = platform(vloop_program(case.funct, n, rows=rows, after=after, **kw), data, arch,
                    case)
    plat.core.vector.vloop.keep_trace = True
    plat.run_until_done(limit=400_000)
    return plat, grid


def dram_rows(plat, addr, rows):
    out = []
    for r in range(rows):
        row = []
        for k in range(16):
            w = int.from_bytes(plat.dram.read(addr + r * 64 + 4 * k, 4), "little")
            row += [bf16_from_bits(w & 0xFFFF), bf16_from_bits(w >> 16)]
        out.append(row)
    return out


def bits(rows):
    """BF16 bit patterns: tells -0.0 from +0.0, and NaN equals itself."""
    return [[bf16_to_bits(v) for v in r] for r in rows]


def expected(case, grid):
    ref = as_case(case).spec.ref
    return bits([[ref(x) for x in row] for row in grid])


def table(case, same_pad, latency=DEFAULT_LATENCY):
    case = as_case(case)
    return schedule(case.name, case.body, case.ii, latency, share_pad=same_pad)


# == every loop, every variant =======================================================================
# -- the registry --------------------------------------------------------------------------------
def test_every_loop_has_a_spec():
    """isa.VLOOP_FUNCTS, vloop.LOOPS and SPECS name the same loops."""
    assert {f: n for f, n in VLOOP_FUNCTS.items()} == NAME
    assert set(SPECS) == set(NAME.values())


@pytest.mark.golden
def test_every_loop_is_in_the_functional_sim():
    from scheduler import golden  # noqa: F401  (puts the functional sim on sys.path)
    from src.components.vector_loop import VLOOP_FUNCTIONS
    from src.misc.opcode_table import VLOOP_FUNCTS as GOLDEN_FUNCTS
    assert dict(GOLDEN_FUNCTS) == dict(VLOOP_FUNCTS)
    assert set(VLOOP_FUNCTIONS) == set(NAME.values())


@pytest.mark.parametrize("case", VARIANT_CASES)
def test_every_variant_stores_what_its_loop_stores(case):
    """The same values reach the store: a variant may change how often a
    value is loaded, never what is computed from it."""
    def stored_from(body):
        loaded = {op.out for op in body if op.unit == LOAD}
        made = {op.out: op for op in body if op.out}

        def expr(v):
            if v in loaded or v.split("@")[0] in loaded:
                return v.split("@")[0]
            op = made[v]
            return (op.unit, op.op, tuple(expr(s) if s in made else s for s in op.srcs),
                    expr(op.mask) if op.mask else None)
        return [expr(op.srcs[0]) for op in body if op.unit == STORE]
    assert stored_from(case.body) == stored_from(Case(case.funct).body)


# -- the table ------------------------------------------------------------------------------------
@pytest.mark.parametrize("same_pad", [True, False], ids=["one_pad", "two_pads"])
@pytest.mark.parametrize("case", VARIANT_CASES)
def test_the_table_keeps_its_rules(case, same_pad):
    """No resource taken twice in a phase (an FU row, a pad's request port,
    its read channel for 2 cycles a read), at most ISSUE_SLOTS ops per phase,
    every op after its operands are ready, and the II asked for unless the
    body can't fit in it."""
    t = table(case, same_pad)
    assert len(t.offsets) == len(t.ops) and t.ii >= t.ii_wanted == case.ii
    seen, per_phase, ready = set(), [0] * t.ii, {}
    for k, op in enumerate(t.ops):
        for res, cycles in resources(op, same_pad):
            for j in range(cycles):
                slot = (res, (t.offsets[k] + j) % t.ii)
                assert slot not in seen, (op.name, slot)
                seen.add(slot)
        per_phase[t.phase(k)] += 1
        for v in op.reads():
            assert t.offsets[k] >= ready[v], "%s reads %s before it is ready" % (op.name, v)
        if op.out:
            ready[op.out] = t.offsets[k] + DEFAULT_LATENCY[op.unit]
    assert max(per_phase) <= ISSUE_SLOTS
    assert t.stages == max(t.offsets) // t.ii + 1


# -- results -----------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 5, 9, 32])
@pytest.mark.parametrize("case,arch", ARCH_CASES)
def test_the_loop_computes_every_row(case, arch, n):
    plat, grid = run(case, n, arch, seed=n)
    assert bits(dram_rows(plat, DRAM_OUT, n)) == expected(case, grid)
    seq = plat.core.vector.vloop
    assert seq.stats["rows"] == n and seq.stats["loops"] == 1 and seq.idle
    assert seq.stats["fired"] == {op.name: n for op in case.body}
    assert seq.stats["loads"] == case.loads * n
    assert seq.stats["max_fired_per_cycle"] <= ISSUE_SLOTS
    t = seq.table_for(case.funct, False)
    if n > t.stages:
        # a full pipeline reaches the depths the table says the buffers need
        assert seq.peak_buffer == t.buffer_depths()


@pytest.mark.parametrize("kw", [
    dict(sid_in=0, sid_out=0),                                 # in place
    dict(sid_in=1, sid_out=0, in_row=2, out_row=9),            # other way, row offsets
], ids=["in_place", "offsets"])
@pytest.mark.parametrize("case,arch", ARCH_CASES)
def test_the_loop_on_one_pad_and_with_row_offsets(case, arch, kw):
    """In place, a reload of row r must come before row r's store."""
    plat, grid = run(case, 8, arch, seed=3, **kw)
    assert bits(dram_rows(plat, DRAM_OUT, 8)) == expected(case, grid)


@pytest.mark.parametrize("case,arch", ARCH_CASES)
def test_zero_rows_does_nothing_and_halts(case, arch):
    plat, _ = run(case, 0, arch)
    seq = plat.core.vector.vloop
    assert seq.stats["loops"] == 1 and seq.stats["loads"] == 0 and seq.idle


@pytest.mark.parametrize("case", LOOP_CASES)
def test_the_loop_is_close_to_its_math_function(case):
    """The spec's exact result against the function it stands for."""
    spec = case.spec
    for v in range(-512, 513):
        x = bf(v / 64)
        got, true = spec.ref(x), spec.true(x)
        tol = spec.rel_tol * abs(true) if abs(true) > 2 ** -6 else spec.abs_tol
        assert abs(got - true) <= tol, "x=%r: %r vs %r" % (x, got, true)


# -- timing on the M x N core ------------------------------------------------------------------
@pytest.mark.parametrize("same_pad", [True, False], ids=["one_pad", "two_pads"])
@pytest.mark.parametrize("case", VARIANT_CASES)
def test_the_schedule_runs_as_the_table_says(case, same_pad):
    """Row r's op k fires at start + II*r + offset[k], with the core's own
    latencies (a group's L + 1); no stalls; the load latency is the table's."""
    n = 6
    kw = dict(sid_in=0, sid_out=0) if same_pad else {}
    plat, _ = run(case, n, "v2", **kw)
    seq = plat.core.vector.vloop
    t = seq.table_for(case.funct, same_pad)
    assert [op.name for op in t.ops] == [op.name for op in case.body]
    fired = {(name, row): c for c, name, row in seq.trace}
    s = fired[(t.ops[0].name, 0)]
    for r in range(n):
        assert [fired[(op.name, r)] - s for op in t.ops] == \
            [t.ii * r + off for off in t.offsets], r
    assert seq.stats["stall_cycles"] == 0
    assert seq.stats["load_latency_min"] == seq.stats["load_latency_max"] == seq.latency["load"]


#: A normal op per FU kind, to share a group with a loop: (op, its result from vs1).
NORMAL_OP = {"alu": ("add.vv", lambda x: bf(x + x)), "mul": ("mul.vv", lambda x: bf(x * x))}
SHARE_CASES = [pytest.param(c, k, id="%s-%s" % (c, k))
               for c in CASES for k in c.kinds if k in NORMAL_OP]


@pytest.mark.parametrize("case,kind", SHARE_CASES)
def test_a_vector_op_shares_a_group_with_a_running_loop(case, kind):
    """A slot op on a group the loop fires into: when both want it in one
    cycle the loop wins and the slot op stalls (reason "kernel"); both
    results are right."""
    m, ref = NORMAL_OP[kind]
    # Each to its own register, so no WAW spaces them out: they issue on
    # consecutive cycles and must meet the loop's phase on the group.
    dsts = range(16, 32)
    after = [(vmem("vreg.ld", 1, 0, 0, sid=2),)] + [(vv(m, d, 1, 1),) for d in dsts]
    rows = 24
    plat, grid = run(case, rows, "v2", seed=5, after=after)
    assert bits(dram_rows(plat, DRAM_OUT, rows)) == expected(case, grid)
    vc = plat.vc
    want = [ref(x) for x in vc.read_vreg(1)]
    assert all(vc.read_vreg(d) == want for d in dsts)
    assert sum(vc.stats["stall"]["kernel"]) >= 1, "no slot op met the loop at the %s" % kind


# -- missing FU rows -------------------------------------------------------------------------
@pytest.mark.parametrize("case", LOOP_CASES)
def test_without_fu_rows_the_loop_says_what_is_missing(case):
    data = tile_data(DRAM_IN, 1, lambda r, c: 1.0)
    plat = build_atalla_platform(vloop_program(case.funct, 1), data, warm_icache=True)
    with pytest.raises(RuntimeError, match="needs a %s FU row" % case.kinds[0]):
        plat.run_until_done(limit=200_000)


@pytest.mark.parametrize("case", [p for p in LOOP_CASES
                                  if not set(p.values[0].kinds) <= V1_KINDS])
def test_on_the_lane_core_a_loop_needing_more_rows_says_which(case):
    missing = next(k for k in case.kinds if k not in V1_KINDS)
    data = tile_data(DRAM_IN, 1, lambda r, c: 1.0)
    plat = platform(vloop_program(case.funct, 1), data, "v1")
    with pytest.raises(RuntimeError, match="needs a %s FU row" % missing):
        plat.run_until_done(limit=200_000)


# -- against the functional sim -------------------------------------------------------------
@pytest.mark.golden
@pytest.mark.parametrize("n,kw", [
    (3, {}),                                                  # pad 0 -> pad 1
    (6, dict(sid_in=1, sid_out=0, in_row=2, out_row=9)),      # other way, row offsets
    (8, dict(sid_in=0, sid_out=0, in_row=4, out_row=4)),      # in place
    (32, {}),                                                 # a whole tile
], ids=["pad0_to_pad1", "offsets", "in_place", "n32"])
@pytest.mark.parametrize("case,arch", ARCH_CASES)
def test_the_loop_matches_the_functional_sim(case, arch, n, kw):
    """The functional sim runs the loop a row at a time (vector_loop.py); the
    final state -- scalar and vector registers, every DRAM halfword -- must
    be the same, whichever variant ran."""
    from scheduler import golden
    grid = make_grid(case, n, 40 + n)
    data = tile_data(DRAM_IN, n, lambda r, c: grid[r][c])
    prog = vloop_program(case.funct, n, rows=n, **kw)
    plat = platform(prog, data, arch, case)
    plat.run_until_done(limit=400_000)
    assert bits(dram_rows(plat, DRAM_OUT, n)) == expected(case, grid)
    g = golden.run_golden_state(prog, data)
    assert golden.differences(plat, g) == []


# -- variants: buffer area against loads ------------------------------------------------------
@pytest.mark.parametrize("case", OTHER_VARIANTS)
def test_a_reload_variant_trades_buffer_bits_for_loads(case):
    """Against its loop's "keep": more loads per row, fewer buffer bits, and
    no faster (an II raised by the pad's reads shows as a lower ceiling)."""
    n = 16
    keep, var = (run(c, n, "v2")[0] for c in (Case(case.funct), case))
    rk, rv = keep.perf.report()["vloop"], var.perf.report()["vloop"]
    assert rv["loads_per_row"] == case.loads > rk["loads_per_row"] == 1
    assert rv["buffer_bits_total"] < rk["buffer_bits_total"]
    assert rv["rows_per_cycle_ceiling"] <= rk["rows_per_cycle_ceiling"]
    assert rv["stall_cycles"] == rk["stall_cycles"] == 0


def test_with_reloads_places_each_reload_just_in_time():
    body = with_reloads(GELU.body, "x", ("MULY",))
    names = [op.name for op in body]
    assert names[names.index("MULY") - 1] == "LD_x@MULY"
    ld = body[names.index("LD_x@MULY")]
    assert ld.unit == LOAD and ld.out == "x@MULY" and ld.jit_values() == ("s",)
    assert body[names.index("MULY")].srcs == ("x@MULY", "s")
    t = schedule("g", body, 4, DEFAULT_LATENCY, share_pad=False)
    # its data lands within a pad-read window (2 cycles) of s, and no later
    # than MULY needs it: MULY fires as early as without the reload
    s_ready = t.offsets[names.index("RECIP")] + DEFAULT_LATENCY["recip"]
    lands = t.offsets[names.index("LD_x@MULY")] + DEFAULT_LATENCY[LOAD]
    muly = t.offsets[names.index("MULY")]
    assert abs(lands - s_ready) <= 2 and lands <= muly
    assert muly == table(GELU, False).offsets[-2]


def test_with_reloads_refuses_what_it_cannot_do():
    with pytest.raises(ValueError, match="isn't loaded"):
        with_reloads(GELU.body, "s", ("MULY",))
    with pytest.raises(ValueError, match="no ops named"):
        with_reloads(GELU.body, "x", ("NOPE",))
    with pytest.raises(ValueError, match="doesn't read"):
        with_reloads(GELU.body, "x", ("EXP",))


def test_a_variant_is_chosen_by_funct_or_name_and_checked():
    seq = VLoopSequencer([], lambda pad: False, variants={"gelu.vl": "reload"})
    assert seq.variant(1) == "reload" and seq.variant(0) == DEFAULT_VARIANT
    assert [op.name for op in seq.table_for(1, False).ops] == \
        [op.name for op in Case(1, "reload").body]
    seq.use_variant(1, "keep")
    assert len(seq.table_for(1, False).ops) == len(GELU.body)
    with pytest.raises(ValueError, match="has no variant"):
        seq.use_variant("relu.vl", "reload")
    with pytest.raises(ValueError, match="no vloop named"):
        seq.use_variant("tanh.vl", "keep")


# -- the perf monitor ---------------------------------------------------------------------------
def run_observed(case, n, **kw):
    """The loop over n rows on the M x N core, with the activity probe on."""
    case = as_case(case)
    data = tile_data(DRAM_IN, n, lambda r, c: float(c - 16))
    plat = build_atalla_platform(vloop_program(case.funct, n, rows=n, **kw), data,
                                 warm_icache=True, vector_arch="fu_groups", activity=True,
                                 vloop_variants={case.funct: case.variant})
    plat.run_until_done(limit=400_000)
    return plat


@pytest.mark.parametrize("case", VARIANT_CASES)
def test_the_perf_monitor_reports_the_loop(case):
    n = 8
    body = case.body
    plat = run_observed(case, n)
    r = plat.perf.report()
    seq = plat.core.vector.vloop
    ii = seq.table_for(case.funct, False).ii          # >= case.ii (pad reads)
    assert sum(r["cycle_breakdown"].values()) == r["cycles"]
    assert r["instruction_mix"]["vloop"] == 1
    vl_ = r["vloop"]
    assert vl_["loops"] == 1 and vl_["rows"] == n
    assert vl_["cycles"]["total"] == sum(seq.stats[k] for k in
                                         ("arm_cycles", "run_cycles", "drain_cycles"))
    assert vl_["rows_per_cycle_ceiling"] == 1 / ii and 0 < vl_["rows_per_run_cycle"] <= 1 / ii
    t = vl_["tables"][str(case)]
    assert t["ii"] == ii and t["ii_wanted"] == case.ii
    assert list(t["offsets"]) == [op.name for op in body]
    fu_ops = sum(1 for op in body if op.unit not in (LOAD, STORE))
    assert vl_["fu_ops"] == fu_ops * n and vl_["element_ops"] == fu_ops * n * 32
    assert vl_["load_latency"]["min"] == seq.latency["load"]
    assert set(vl_["peak_buffer"]) == {op.out for op in body if op.out}
    assert vl_["loads_per_row"] == case.loads
    assert vl_["buffer_bits_total"] == sum(vl_["buffer_bits"].values()) > 0
    busy = r["unit_busy"]
    assert busy["vloop"]["cycles"] == vl_["cycles"]["total"]
    for k in case.kinds:
        assert busy["vloop:" + k]["cycles"] > 0, k
    # the loop's ops are not lane work or FLOPs
    assert "lanes" not in busy and r["work"]["lane_flops"] == 0
    assert "buffers" in plat.perf.summary()


def test_a_program_without_a_loop_reports_no_vloop_section():
    """So reports of programs without one stay as they were."""
    plat = build_atalla_platform(program((i("addi.s", 5, 0, 1),), HALT), {},
                                 vector_arch="fu_groups")
    plat.run_until_done(limit=10_000)
    r = plat.perf.report()
    assert "vloop" not in r and "vloop" not in r["unit_busy"]


# == relu.vl ======================================================================================
def test_relu_table_same_pad():
    """LD 0; CMP when the data is back (8); MUL when the mask is (12); the
    store's operand is ready at 16, the load's phase on the shared pad, so 17."""
    t = table(0, True)
    assert (t.name, t.ii, t.offsets, t.stages) == ("relu.vl", 2, (0, 8, 12, 17), 9)
    assert [t.phase(k) for k in range(4)] == [0, 0, 0, 1]


def test_relu_table_separate_pads():
    """Input and output on different pads: the store goes as soon as r is."""
    t = table(0, False)
    assert (t.offsets, t.stages) == ((0, 8, 12, 16), 9)


def test_relu_on_the_lane_core_runs_as_the_table_says():
    """No stalls: row r's LD at s + 2r, CMP +8, MUL +12, ST +17 (one pad).
    The measured load latency is the table's 8, and the buffers it needed."""
    n = 6
    plat, _ = run(0, n, "v1", sid_in=0, sid_out=0)
    seq = plat.core.vector.vloop
    fired = {(name, row): c for c, name, row in seq.trace}
    s = fired[("LD", 0)]
    for r in range(n):
        assert [fired[(op, r)] - s for op in ("LD", "CMP", "MUL", "ST")] == \
            [2 * r, 2 * r + 8, 2 * r + 12, 2 * r + 17], r
    assert seq.stats["stall_cycles"] == 0
    assert seq.stats["load_latency_min"] == seq.stats["load_latency_max"] == LOAD_LATENCY
    assert seq.peak_buffer == {"v": 3, "m": 1, "r": 1}


def test_relu_mxn_table_follows_the_groups_latencies():
    """A kernel port's latency is the group's L + 1 (it takes the result from
    the output register): ALU and MUL L = 4 give CMP -> MUL 5 and MUL -> ST 5."""
    plat, _ = run(0, 1, "v2", sid_in=0, sid_out=0)
    seq = plat.core.vector.vloop
    load = seq.latency["load"]
    assert seq.table_for(0, True).offsets == \
        (0, load, load + 5, load + 10 + (1 if (load + 10) % 2 == 0 else 0))


def test_the_tables_take_the_connected_rows_latencies():
    plat = build_atalla_platform(relu_program(1), {}, warm_icache=True)
    seq = plat.core.vector.vloop
    assert seq.table_for(0, True).offsets == (0, 8, 12, 17)
    seq.connect_fu_rows({"alu": StubRow(3), "mul": StubRow(5)})
    assert seq.table_for(0, True).offsets == (0, 8, 11, 17)     # r ready at 16 -> 17
    assert seq.table_for(0, False).offsets == (0, 8, 11, 16)


def test_a_vreg_ld_on_the_output_pad_waits_for_the_loop():
    """The loop holds pad 1; a vreg.ld of a row it writes waits until the
    loop is done, then reads the result."""
    after = [(i("addi.s", 9, 0, 0),), (vmem("vreg.ld", 20, 9, 9, sid=1),)]
    plat, grid = run(0, 4, "v1", seed=7, after=after)
    assert plat.vc.read_vreg(20) == [relu_ref(x) for x in grid[0]]


def test_scalar_packets_keep_issuing_while_the_loop_runs():
    after = [(i("addi.s", 10 + k, 0, k),) for k in range(6)]
    plat, _ = run(0, 16, "v1", after=after)
    seq = plat.core.vector.vloop
    last_store = max(c for c, name, _ in seq.trace if name == "ST")
    issued = {p.pc: c for c, p in plat.core.issued}
    vloop_pc = next(p.pc for _, p in plat.core.issued
                    if any(o.valid and o.mnemonic == "relu.vl" for o in p.packet.vector))
    scalar_issues = [issued[vloop_pc + 20 * (k + 1)] for k in range(6)]
    assert max(scalar_issues) < last_store, "the scalar packets waited for the loop"
    assert [plat.core.scalar_reg(10 + k) for k in range(6)] == list(range(6))


def test_the_activity_view_draws_the_mxn_core_and_the_sequencer():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import activity_view
    plat = run_observed(0, 8)
    a, busy = plat.activity, plat.perf.busy
    ids_ = [b.id for b in a.blocks]
    labels = {b.id: b.label for b in a.blocks}
    groups = [g for g in ids_ if g.startswith("fu")]
    assert [labels[g] for g in groups] == ["ALU", "MUL", "EXP", "SQRT", "RECIP", "RTree"]
    k = a.counts("vloop")
    assert k["active"] > 0 and k["stalled"] > 0          # ARM waits for the scpad.ld
    assert k["active"] + k["stalled"] == busy["vloop"]
    assert a.counts(groups[0])["active"] > 0 and a.counts(groups[1])["active"] > 0
    page = activity_view.render(a.trace())
    assert '"vloop"' in page and '"fu0"' in page


def test_on_the_lane_core_ops_held_for_the_loop_are_named():
    """v1 holds a lane op out of a cycle the sequencer fires into its unit:
    decode 2 names that stall "alu:vloop", and the monitor counts it."""
    after = [(vmem("vreg.ld", 1, 0, 0, sid=2),)] + [(vv("add.vv", 9, 1, 1),)] * 6
    plat, _ = run(0, 24, "v1", after=after)
    r = plat.perf.report()
    assert r["cycle_breakdown"].get("stall:alu:vloop", 0) > 0
    assert r["vloop"]["ops_stalled_by_loop"]["decode2_cycles"] == \
        sum(v for k, v in r["cycle_breakdown"].items() if k.endswith(":vloop"))


# == gelu.vl ======================================================================================
def test_gelu_table():
    """Four MULs on one MUL group: II = 4, one MUL per phase. The pads never
    clash (LD in phase 0, ST in phase 3), so one pad or two give one table."""
    for same in (True, False):
        t = table(1, same)
        assert (t.name, t.ii, t.offsets, t.stages) == \
            ("gelu.vl", 4, (0, 8, 13, 17, 22, 26, 40, 44, 55, 59), 15)
        assert sorted(t.phase(k) for k, op in enumerate(t.ops) if op.unit == "mul") == \
            [0, 1, 2, 3]


@pytest.mark.parametrize("variant,ii,run_cycles,peak_x,bits_total", [
    ("keep",       4, 128, 14, 11264),
    ("reload",     4, 128,  4,  7168),    # x until MULX; its copy 2 entries
    ("reload_all", 6, 156,  1,  5632),    # 3 reads x 2 cycles: II 6
])
def test_gelu_variants_cycles_and_buffers(variant, ii, run_cycles, peak_x, bits_total):
    """The trade on the M x N core, 16 rows, one pad or two (the same): the
    numbers the sweep reports. reload costs no cycles; reload_all saves
    another 1.5 Kbit but its pad reads make II 6, 1.5x the cycles a row."""
    for kw in ({}, dict(sid_in=0, sid_out=0)):
        plat, _ = run(Case(1, variant), 16, "v2", **kw)
        seq = plat.core.vector.vloop
        assert seq.table_for(1, bool(kw)).ii == ii
        assert seq.stats["run_cycles"] == run_cycles and seq.stats["stall_cycles"] == 0
        assert seq.peak_buffer["x"] == peak_x
        assert sum(seq.buffer_bits().values()) == bits_total


def test_a_pad_reads_one_row_every_two_cycles():
    """Why reload_all can't keep II 4: its three reads need six cycles of
    the input pad's read channel. Wanting II 4 gives II 6."""
    t = table(Case(1, "reload_all"), False)
    assert (t.ii_wanted, t.ii) == (4, 6)
    reads = sorted(t.phase(k) for k, op in enumerate(t.ops) if op.unit == LOAD)
    assert reads == [0, 2, 4]


def test_gelu_of_minus_inf_is_nan():
    """Accepted behaviour (documented in vloop.GELU_OPS and the functional
    sim's vector_loop): t = +inf, so s = 0 and y = -inf * 0 = NaN, where true
    GELU gives 0. +inf stays +inf; a large finite negative gives -0.0."""
    assert math.isnan(gelu_ref(-INF))
    assert gelu_ref(INF) == INF
    assert struct.pack("<f", gelu_ref(-100.0)) == struct.pack("<f", -0.0)
    data = tile_data(DRAM_IN, 1, lambda r, c: [-INF, INF, -100.0, 1.0][c % 4])
    plat = platform(vloop_program(1, 1), data, "v2")
    plat.run_until_done(limit=200_000)
    got = dram_rows(plat, DRAM_OUT, 1)[0]
    assert math.isnan(got[0]) and got[1] == INF
    assert bits([got[2:4]]) == bits([[-0.0, gelu_ref(1.0)]])
