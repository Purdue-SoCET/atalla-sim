"""Stage 5: vector and DMA dispatch, on a platform driven by the scheduler.

Programs run through atalla/atalla_platform.py: the scheduler issues into the
vector core's units, the scratchpad pads and their DMA backends, and DRAM.
The golden tests run the same program on the functional sim and compare
every scalar, vector and mask register and the DRAM words at the end.

Test programs stay inside what the functional sim supports: scratchpads 0
and 1, rows 0-31, full 32-element rows, and they never write m0 (the
functional sim lets a program write it; the RTL holds it at all ones).

The functional sim runs with its reductions in the lane model's order
(golden.lanes_reduce: per lane, then a tree across lanes, rounded once to
BF16), which is what the lane datapath computes; its own sequential fp32
sum can differ in the last BF16 bit.
"""
import random
import struct

import pytest

from scheduler import golden
from scheduler.core import SchedulerCore
from atalla.atalla_platform import build_atalla_platform
from scheduler.semantics import u32
from scheduler.vector import bf16_round

from test_execute import HALT, OP, base, i, mi, program, r


# -- encoders --------------------------------------------------------------------
def vv(m, vd, vs1, vs2, mask=0):
    return OP[m] | vd << 7 | vs1 << 15 | vs2 << 23 | mask << 31


def vs(m, vd, vs1, rs1, mask=0):
    return OP[m] | vd << 7 | vs1 << 15 | rs1 << 23 | mask << 31


def vi(m, vd, vs1, imm, mask=0):
    return OP[m] | vd << 7 | vs1 << 15 | (imm & 0xFF) << 23 | mask << 31


def mvv(m, md, vs1, vs2, mask=0):
    return OP[m] | md << 7 | vs1 << 15 | vs2 << 23 | mask << 31


def mvs(m, md, vs1, rs1, mask=0):
    return OP[m] | md << 7 | vs1 << 15 | rs1 << 23 | mask << 31


def vmem(m, v, rs1, rs2, sid=0, cols=31):
    return OP[m] | v << 7 | rs1 << 15 | rs2 << 23 | cols << 31 | sid << 36


def sdma(m, rs1, rs2, rs3):
    return OP[m] | rs1 << 7 | rs2 << 15 | rs3 << 23


def vts(rd, vs1, idx):
    return OP["vmov.vts"] | rd << 7 | vs1 << 15 | idx << 23


def mts(rd, ms):
    return OP["mv.mts"] | rd << 7 | ms << 15


def bf(x):
    return struct.unpack("<I", struct.pack("<f", x))[0] >> 16


def fbits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def tile_data(base_addr, rows, values):
    """DRAM words holding `rows` x 32 BF16 values, row-major."""
    data = {}
    for rr in range(rows):
        for k in range(16):
            a, b = values(rr, 2 * k), values(rr, 2 * k + 1)
            data[base_addr + rr * 64 + 4 * k] = bf(a) | bf(b) << 16
    return data


def meta(sid, rows, cols=32, full=0):
    return (sid << 30) | ((rows - 1) << 25) | ((cols - 1) << 20) | full


DRAM_IN = 0x8000
DRAM_OUT = 0xA000
LOAD_ROWS = 8


def prologue(rows=LOAD_ROWS, first_vreg=8):
    """Load `rows` x 32 from DRAM_IN into pad 0 and into v8.. by rows."""
    pk = [*base(1, DRAM_IN), *base(3, meta(0, rows)), i("addi.s", 2, 0, 0),
          (sdma("scpad.ld", 2, 1, 3),)]
    for rr in range(rows):
        pk.append((i("addi.s", 4, 0, rr),))
        pk.append((vmem("vreg.ld", first_vreg + rr, 2, 4),))
    return pk


def epilogue(regs, sid=1):
    """Store vregs `regs` to pad `sid` rows 0.., then the pad to DRAM_OUT."""
    pk = [i("addi.s", 2, 0, 0)]
    for k, v in enumerate(regs):
        pk.append((i("addi.s", 4, 0, k),))
        pk.append((vmem("vreg.st", v, 2, 4, sid=sid),))
    pk += [*base(1, DRAM_OUT), *base(3, meta(sid, len(regs))),
           (sdma("scpad.st", 2, 1, 3),)]
    return pk


def values(rr, col):
    return float((rr * 32 + col) % 13 - 6) * 0.5


def run_platform(prog, data, warm_icache=True):
    """Warm icache by default: these tests time the vector side, not fetch."""
    plat = build_atalla_platform(prog, data, warm_icache=warm_icache)
    plat.run_until_done(limit=200_000)
    return plat


def compare_to_golden(prog, data):
    golden.require(pytest)
    g = golden.run_golden_state(prog, data, reductions="lanes")
    plat = run_platform(prog, data)
    c = plat.core
    for rr in range(1, 256):
        assert c.scalar_reg(rr) == u32(g["sregs"].get(rr, 0)), "x%d" % rr
    for v in range(256):
        assert plat.vc.read_vreg(v) == [bf16_round(x) for x in g["vregs"][v]], "v%d" % v
    for m in range(1, 16):
        assert c.decode2.mrf.read(m) == u32(g["mregs"].get(m, 0)), "m%d" % m
    for a, w in g["mem"].items():
        got = int.from_bytes(plat.dram.read(a, 4), "little")
        assert got == u32(w), hex(a)
    return plat


# -- the register files ---------------------------------------------------------------

def test_v0_reads_zero_and_m0_reads_every_lane():
    c = SchedulerCore({})
    assert c.decode2.vrf.read(0) == [0.0] * 32
    assert c.decode2.mrf.read(0) == 0xFFFFFFFF


def issue_of(prog, pc, data=None):
    plat = run_platform(prog, data or tile_data(DRAM_IN, LOAD_ROWS, values))
    return {p.pc: cy for cy, p in plat.core.issued}[pc]


def test_vector_bank_conflict_costs_two_cycles():
    """add.vv v6, v16, v20 reads two registers of bank 0: reggie serves them
    one per cycle, so the packet issues in 4 instead of 2."""
    assert issue_of(program(vv("add.vv", 6, 16, 17), HALT), 0) == 2
    assert issue_of(program(vv("add.vv", 6, 16, 20), HALT), 0) == 4


def test_mask_bank_conflict_costs_two_cycles():
    """Two lane ops masked by m2 and m4 read mask bank 0 twice."""
    pk = (vv("add.vv", 6, 16, 17, 2), vv("add.vv", 7, 18, 19, 3))
    assert issue_of(program(pk, HALT), 0) == 2
    pk = (vv("add.vv", 6, 16, 17, 2), vv("add.vv", 7, 18, 19, 4))
    assert issue_of(program(pk, HALT), 0) == 4


# -- dispatch, hazards and writeback ------------------------------------------------------

def test_sdma_holds_its_rs1_until_the_scratchpad_is_done():
    """vreg.ld reads x2, which the SDMA holds: it issues only after the
    SDMA's completion clears x2's busy bit, never while the pad is busy."""
    prog = program(*prologue(rows=1), HALT)
    plat = run_platform(prog, tile_data(DRAM_IN, 1, values))
    c = plat.core
    t = {p.pc: cy for cy, p in c.issued}
    sdma_pc, ld_pc = 20 * 6, 20 * 8
    assert t[ld_pc] - t[sdma_pc] > plat.backends[0].dram_latency
    assert c.vector.stats["sdma_loads"] == 1


def test_dependent_vector_op_waits_for_the_writeback():
    pk = prologue()
    plat = run_platform(program(*pk, vv("add.vv", 6, 8, 9), vv("mul.vv", 7, 6, 6), HALT),
                        tile_data(DRAM_IN, LOAD_ROWS, values))
    t = {p.pc: cy for cy, p in plat.core.issued}
    add_pc, mul_pc = 20 * len(pk), 20 * (len(pk) + 1)
    assert t[mul_pc] > t[add_pc] + 2
    assert plat.vc.read_vreg(7) == [bf16_round(x * x) for x in plat.vc.read_vreg(6)]


def test_two_lane_ops_and_a_vlsu_op_issue_in_one_packet():
    pk = prologue()
    plat = run_platform(program(*pk, (vv("add.vv", 5, 8, 9), vv("sub.vv", 6, 10, 11),
                                      vmem("vreg.ld", 7, 2, 4)), HALT),
                        tile_data(DRAM_IN, LOAD_ROWS, values))
    assert plat.core.vector.stats["lane_ops"] == 2


def test_halt_waits_for_the_vector_side():
    prog = program(*prologue(rows=2), *epilogue([8, 9]), HALT)
    plat = run_platform(prog, tile_data(DRAM_IN, 2, values))
    out = plat.dram.read(DRAM_OUT, 128)
    inn = plat.dram.read(DRAM_IN, 128)
    assert out == inn
    assert plat.core.vector.idle


# -- against the functional sim --------------------------------------------------------------

def test_golden_arithmetic_masks_and_moves():
    data = tile_data(DRAM_IN, LOAD_ROWS, values)
    pk = prologue() + [
        (vv("add.vv", 20, 8, 9), vv("mul.vv", 21, 10, 11)),
        (vv("sub.vv", 22, 12, 13),),
        *base(5, fbits(1.5)),
        (vs("mul.vs", 23, 14, 5),),
        (vs("add.vs", 24, 15, 5),),
        (mvv("mgt.mvv", 1, 8, 9),),
        (mvs("mlt.mvs", 2, 10, 5),),
        (vv("add.vv", 25, 20, 21, 1),),             # masked by m1
        (vv("mul.vv", 26, 22, 23, 2),),             # masked by m2
        (vi("expi.vi", 27, 9, 0),),
        (vi("rsum.vi", 28, 10, 0x40),),             # broadcast
        (vi("rmax.vi", 29, 11, 0x20 | 7),),         # one element
        (vi("rmin.vi", 30, 12, 3),),                # over vs1
        (vts(10, 21, 5),),
        (mts(11, 1),),
    ]
    prog = program(*pk, *epilogue([20, 21, 22, 25, 26, 27]), HALT)
    compare_to_golden(prog, data)


LANE_VV = ["add.vv", "sub.vv", "mul.vv"]
LANE_VS = ["add.vs", "sub.vs", "mul.vs"]
COMPARES_MVV = ["mgt.mvv", "mlt.mvv", "meq.mvv", "mneq.mvv"]


def random_vector_program(seed, n=40):
    """Lane ops over v8..v15 (loaded) and v16..v31, masks m1..m7 from
    compares, then everything stored back out through pad 1."""
    rng = random.Random(seed)
    pk = prologue()
    pk += [*base(5, fbits(rng.choice([0.5, 1.25, -2.0]))),
           *base(6, fbits(rng.choice([0.75, 3.0])))]
    srcs, dsts, masks = list(range(8, 16)), list(range(16, 32)), [0]
    reduced = set()
    for _ in range(n):
        slots, wv, wm, lanes = [], set(), set(), 0
        for _ in range(rng.randint(1, 2)):
            kind = rng.choice("vvsmrx")
            d = rng.choice(dsts)
            a, b = rng.choice(srcs + dsts), rng.choice(srcs + dsts)
            mask = rng.choice(masks)
            if d in wv or a in wv or b in wv or mask in wm:
                continue
            if kind == "v":
                slots.append(vv(rng.choice(LANE_VV), d, a, b, mask))
            elif kind == "s":
                slots.append(vs(rng.choice(LANE_VS), d, a, rng.choice((5, 6)), mask))
            elif kind == "m":
                md = rng.randrange(1, 8)
                if md in wm or md == mask:
                    continue
                slots.append(mvv(rng.choice(COMPARES_MVV), md, a, b))
                wm.add(md)
                masks.append(md)
                continue
            elif kind == "r":
                imm = rng.choice([0x40, 0x20 | rng.randrange(32), rng.randrange(32)])
                slots.append(vi(rng.choice(["rsum.vi", "rmin.vi", "rmax.vi"]), d, a, imm))
                reduced.add(d)
                wv.add(d)
                continue
            else:
                slots.append(vi("expi.vi", d, a, 0, mask))
            reduced.discard(d)
            wv.add(d)
        if slots:
            pk.append(tuple(slots))
    pk += epilogue([v for v in dsts if v not in reduced][:8])
    pk.append(HALT)
    return program(*pk)


@pytest.mark.parametrize("seed", range(6))
def test_golden_random_vector_programs(seed):
    def vals(rr, col, _r=random.Random(seed)):
        return _r.choice([-3.0, -1.5, -0.5, 0.0, 0.25, 1.0, 2.5, 4.0])
    compare_to_golden(random_vector_program(seed), tile_data(DRAM_IN, LOAD_ROWS, vals))


def test_gemm_writes_back_and_lw_vi_reserves_nothing():
    """32 weight loads and two gemm.vv through the GSAU and the systolic
    array complete and write back; lw.vi reserves no register. gemm values
    are checked against the functional sim by the kernel tests."""
    pk = prologue() + [(vi("lw.vi", 0, 8 + k % 8, 0),) for k in range(32)]
    plat = build_atalla_platform(program(*pk, vv("gemm.vv", 20, 9, 0),
                                            vv("gemm.vv", 21, 10, 0), HALT),
                                    tile_data(DRAM_IN, LOAD_ROWS, values))
    plat.run_until_done(limit=50_000)
    c = plat.core
    assert c.vector.stats["gsau_ops"] == 34
    assert c.decode2.scoreboard.idle and plat.vc.gsau.writebacks.is_empty()


def test_transpose_pushes_rows_and_pops_columns():
    """tpus.vi pushes vs1 into the transpose unit as a row; tpop.vi takes the
    next transposed column into its register (named in the vs1 field, as the
    ISA sheet's `vs1 <= transpose_unit` has it). 32 rows in, 32 columns out:
    column j of the tile lands in the j-th tpop's register."""
    pk = prologue(rows=32)
    pk += [(vi("tpus.vi", 0, 8 + r, 0),) for r in range(32)]
    pk += [(vi("tpop.vi", 0, 64 + j, 0),) for j in range(32)]
    plat = run_platform(program(*pk, HALT), tile_data(DRAM_IN, 32, values))
    vc = plat.vc
    rows = [vc.read_vreg(8 + r) for r in range(32)]
    for j in range(32):
        assert vc.read_vreg(64 + j) == [rows[r][j] for r in range(32)], "column %d" % j
    st = plat.core.vector.stats
    assert (st["transpose_pushes"], st["transpose_pops"]) == (32, 32)
    t = {p.pc: cy for cy, p in plat.core.issued}
    first_push = min(cy for pc, cy in t.items() if pc >= 20 * (len(pk) - 64))
    # 32 pushes at 9 cycles and a 257-cycle drain, at least
    assert plat.cycle - first_push >= 32 * 9 + 32 * 8
