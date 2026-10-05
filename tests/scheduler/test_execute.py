"""Stage 4: scalar execute, writeback, and the non-blocking load/store unit.

Timing, with a warm icache: a packet at pc P reaches decode 2 in cycle
2 + P/20 when nothing stalls. A packet issued in cycle t is in the D2/EX
latch -- its EX cycle -- in t+1. A result offered in cycle v is in the EX/WB
latch in v+1 (that is the cycle `scalar_writes` records), which writes the
register and clears its busy bit on that edge; a dependent packet issues in
v+2.
"""
import random

import pytest

from scheduler import golden
from scheduler.control import decode_scalar
from scheduler.core import SchedulerCore
from scheduler.dcache import DCacheConfig
from scheduler.isa import INST_W, OPCODES
from scheduler.semantics import (
    branch_increment, branch_offset, control_outcome, scalar_value, u32)

OP = {m: op for op, (m, _) in OPCODES.items()}


# -- encoders --------------------------------------------------------------------
def r(m, rd=0, rs1=0, rs2=0):
    return OP[m] | rd << 7 | rs1 << 15 | rs2 << 23


def i(m, rd=0, rs1=0, imm=0):
    return OP[m] | rd << 7 | rs1 << 15 | (imm & 0xFFF) << 23


def mi(m, rd=0, imm=0):
    return OP[m] | rd << 7 | (imm & 0x1FFFFFF) << 15


def lw(rd, rs1, imm=0):
    return i("lw.s", rd, rs1, imm)


def sw(data, rs1, imm=0):
    return i("sw.s", data, rs1, imm)              # the data register is in [14:7]


def br(m, rs1, rs2, off, incr=0):
    w = (off // 4) & 0x3FF
    return OP[m] | incr << 7 | (w >> 9) << 14 | rs1 << 15 | rs2 << 23 | (w & 0x1FF) << 31


HALT = r("halt.s")
NOP = r("nop.s")


def packet(*slots):
    slots = list(slots) + [NOP] * (4 - len(slots))
    return sum(w << (INST_W * (3 - k)) for k, w in enumerate(slots))


def program(*packets):
    return {20 * k: packet(*p) if isinstance(p, tuple) else packet(p)
            for k, p in enumerate(packets)}


def core_for(prog, data=None, warm=True, **kw):
    c = SchedulerCore(prog, data=data, **kw)
    if warm:
        c.warm_icache()
    return c


def in_program_order(core):
    acc = core.lsu.accepted
    return acc == sorted(acc) and len(acc) == core.lsu._next_id


def issue_cycles(core):
    return {p.pc: c for c, p in core.issued}


def writes_to(core, reg):
    return [(c, v) for c, rd, v in core.scalar_writes if rd == reg]


# -- semantics ------------------------------------------------------------------------

def test_branch_fields():
    w = br("bne.s", 3, 4, -40, incr=5)
    assert branch_offset(w) == -40 and branch_increment(w) == 5
    w = br("beq.s", 3, 4, 2044, incr=127)
    assert branch_offset(w) == 2044 and branch_increment(w) == 127


@pytest.mark.parametrize("m,a,b,want", [
    ("add.s", 5, 7, 12), ("sub.s", 5, 7, u32(-2)), ("sll.s", 1, 31, 1 << 31),
    ("srl.s", 0x80000000, 4, 0x08000000), ("sra.s", 0x80000000, 4, 0xF8000000),
    ("slt.s", u32(-1), 0, 1), ("sltu.s", u32(-1), 0, 0),
    ("mul.s", 0x10000, 0x10000, 0), ("div.s", u32(-7), 2, u32(-3)),
    ("mod.s", 7, 3, 1), ("xor.s", 0xF0, 0xFF, 0x0F),
])
def test_scalar_values(m, a, b, want):
    assert scalar_value(decode_scalar(r(m, 1, 2, 3)), a, b) == want


def test_branch_compares_signed_and_increments_unsigned():
    op = decode_scalar(br("blt.s", 1, 2, 40, incr=100))
    out = control_outcome(op, 0x100, u32(-1), 0)
    assert out.taken and out.target == 0x128
    assert (out.rd, out.value) == (1, u32(-1 + 100))


def test_jal_offset_is_bytes():
    op = decode_scalar(mi("jal", 9, 60))
    out = control_outcome(op, 0x40, 0, 0)
    assert (out.target, out.rd, out.value) == (0x40 + 60, 9, 0x40 + 20)


# -- EX1 ------------------------------------------------------------------------------

def test_alu_result_and_dependent_issue():
    """addi x1 issues in 2, EX in 3, EX/WB in 4. The dependent addi x2, x1
    (decode 2 in 3) issues in 5, the cycle after x1's busy bit clears."""
    c = core_for(program(i("addi.s", 1, 0, 5), i("addi.s", 2, 1, 1), HALT))
    c.run_until_done()
    assert writes_to(c, 1) == [(4, 5)]
    assert issue_cycles(c) == {0: 2, 20: 5, 40: 6}
    assert writes_to(c, 2) == [(7, 6)]
    assert c.scalar_reg(2) == 6


def test_independent_alu_packets_issue_every_cycle():
    c = core_for(program(*[i("addi.s", k + 1, 0, k) for k in range(6)], HALT))
    c.run_until_done()
    assert [issue_cycles(c)[20 * k] for k in range(6)] == list(range(2, 8))


def test_taken_branch_redirects_and_skips_the_fall_through():
    """beq x0, x0 at pc 0 is taken to 40 and, with a cold BTB, mispredicted:
    the redirect in its EX cycle (3) flushes the packet at 20."""
    c = core_for(program(br("beq.s", 0, 0, 40), i("addi.s", 5, 0, 99),
                         i("addi.s", 6, 0, 7), HALT))
    c.run_until_done()
    assert c.scalar_reg(5) == 0 and c.scalar_reg(6) == 7
    assert 20 not in issue_cycles(c)
    assert c.flushes == 1


def test_wrong_path_load_never_reaches_the_cache():
    """The load after a mispredicted taken branch is flushed before EX: no
    cache request, so no miss to clean up."""
    c = core_for(program(*base(1, 0x1000), br("beq.s", 0, 0, 40), lw(5, 1, 0),
                         i("addi.s", 6, 0, 1), HALT), data=DATA)
    c.run_until_done()
    assert c.scalar_reg(5) == 0 and c.scalar_reg(6) == 1
    assert c.lsu._next_id == 0 and c.dcache.stats["requests"] == 0


def test_counted_loop():
    """x10 counts 0..3 under bne's increment: the body runs four times."""
    prog = program(
        i("addi.s", 11, 0, 3),
        i("addi.s", 12, 12, 2),                       # body, pc 20
        br("bne.s", 10, 11, -20, incr=1),             # pc 40, back to 20
        HALT)
    c = core_for(prog)
    c.run_until_done()
    assert c.scalar_reg(12) == 8 and c.scalar_reg(10) == 4


# -- EX2-EX4 ----------------------------------------------------------------------------

def test_mul_latency_and_initiation_interval():
    """mul.s: EX in 3, result offered in 5 (done), EX/WB in 6. EX4's ready
    is low from its EX cycle through done, so the second mul (in decode 2
    from 3) issues in 6 and is in EX in 7: one mul every 4 cycles."""
    c = core_for(program(r("mul.s", 1, 2, 3), r("mul.s", 2, 6, 7), HALT))
    c.run_until_done()
    assert writes_to(c, 1) == [(6, 0)]
    assert issue_cycles(c)[20] == 6


def test_bf16_add_initiation_interval_is_three():
    """EX in 3, done in 4, ready again in 5: the second add issues in 5."""
    c = core_for(program(r("add.bf", 1, 2, 3), r("add.bf", 2, 6, 7), HALT))
    c.run_until_done()
    assert issue_cycles(c)[20] == 5
    assert writes_to(c, 1)[0][0] == 5


def test_divide_takes_66():
    c = core_for(program(i("addi.s", 2, 0, 100), i("divi.s", 1, 2, 7), HALT))
    c.run_until_done()
    t = issue_cycles(c)[20]                       # EX in t+1, result t+67
    assert writes_to(c, 1) == [(t + 68, 14)]


def test_writeback_bank_conflict_delays_the_lower_priority():
    """mul x1 (EX4) and add x5 (EX1) both offer in cycle 5: same bank (1).
    EX1 outranks EX4, so the mul's write waits a cycle."""
    c = core_for(program(r("mul.s", 1, 2, 3), (), i("addi.s", 5, 0, 3), HALT))
    c.run_until_done()
    assert writes_to(c, 5) == [(6, 3)]
    assert writes_to(c, 1) == [(7, 0)]


def test_scalar_writes_order_by_priority_without_conflict():
    c = core_for(program(r("mul.s", 1, 2, 3), (), i("addi.s", 6, 0, 3), HALT))
    c.run_until_done()
    assert writes_to(c, 6) == [(6, 3)] and writes_to(c, 1) == [(6, 0)]


# -- EX5 and the data cache ----------------------------------------------------------------

DATA = {0x1000 + 4 * k: 0x100 + k for k in range(64)}
DATA.update({0x2000 + 4 * k: 0x200 + k for k in range(64)})
DATA.update({0x3000 + 4 * k: 0x300 + k for k in range(64)})


def base(reg, addr):
    """lui + addi for a 32-bit address."""
    return (mi("lui.s", reg, addr >> 7), i("addi.s", reg, reg, addr & 0x7F))


def test_load_hit_latency():
    """A load in EX in cycle e goes straight to the cache (the queue is empty):
    the hit answers in e+4, so x5 is in the EX/WB latch in e+5."""
    c = core_for(program(base(1, 0x1000)[0], base(1, 0x1000)[1], lw(5, 1, 8), HALT),
                 data=DATA)
    c.dcache.warm([0x1000])
    c.run_until_done()
    e = issue_cycles(c)[40] + 1
    assert writes_to(c, 5) == [(e + 5, 0x102)]


def test_load_miss_latency():
    """A miss evaluated in e+4 fills 28 cycles later; the load writes back
    from the fill cycle."""
    c = core_for(program(base(1, 0x1000)[0], base(1, 0x1000)[1], lw(5, 1, 8), HALT),
                 data=DATA)
    c.run_until_done()
    e = issue_cycles(c)[40] + 1
    assert writes_to(c, 5) == [(e + 4 + 28 + 1, 0x102)]


def test_misses_do_not_block_issue():
    """Two loads that miss to different lines, then independent ALU work: the
    ALU packets issue back to back while both misses are outstanding. The
    second miss's lookup overlaps the first fill, so its fill starts the
    cycle after the first's and completes exactly 28 later; a blocking LSU
    would only start the second lookup after the first fill."""
    prog = program(base(1, 0x1000)[0], base(1, 0x1000)[1],
                   base(2, 0x2000)[0], base(2, 0x2000)[1],
                   lw(5, 1, 0), lw(6, 2, 0),
                   *[i("addi.s", 20 + k, 0, k) for k in range(6)], HALT)
    c = core_for(prog, data=DATA)
    c.run_until_done()
    t = issue_cycles(c)
    alu = [t[20 * (6 + k)] for k in range(6)]
    assert alu == list(range(alu[0], alu[0] + 6))
    assert alu[-1] < writes_to(c, 5)[0][0]           # all before the first fill
    assert c.dcache.stats["max_mshrs"] == 2
    first, second = writes_to(c, 5)[0][0], writes_to(c, 6)[0][0]
    assert second - first == 28
    assert (c.scalar_reg(5), c.scalar_reg(6)) == (0x100, 0x200)


def test_hit_under_miss():
    prog = program(base(1, 0x1000)[0], base(1, 0x1000)[1],
                   base(2, 0x2000)[0], base(2, 0x2000)[1],
                   lw(5, 1, 0), lw(6, 2, 4), HALT)
    c = core_for(prog, data=DATA)
    c.dcache.warm([0x2000])
    c.run_until_done()
    assert writes_to(c, 6)[0][0] < writes_to(c, 5)[0][0]
    assert (c.scalar_reg(5), c.scalar_reg(6)) == (0x100, 0x201)


def test_store_miss_then_load_sees_the_store():
    """The store misses and is acknowledged at once; the load to the same
    word joins the store's MSHR and is answered after it, with its data."""
    prog = program(base(1, 0x3000)[0], base(1, 0x3000)[1], i("addi.s", 7, 0, 77),
                   sw(7, 1, 12), lw(8, 1, 12), lw(9, 1, 16), HALT)
    c = core_for(prog, data=DATA)
    c.run_until_done()
    assert (c.scalar_reg(8), c.scalar_reg(9)) == (77, 0x304)
    assert c.dcache.stats["fills"] == 1 and c.dcache.stats["secondary"] == 2
    assert c.memory.read(0x300C) == 77               # written back at halt


def test_full_mshr_retries_in_program_order():
    """With one MSHR, the second miss is retried until the first fill frees
    it; nothing younger overtakes it."""
    prog = program(base(1, 0x1000)[0], base(1, 0x1000)[1],
                   base(2, 0x2000)[0], base(2, 0x2000)[1],
                   lw(5, 1, 0), lw(6, 2, 0), lw(7, 1, 4), HALT)
    c = core_for(prog, data=DATA, dcache_config=DCacheConfig(mshrs=1))
    c.run_until_done()
    assert c.lsu.stats["retries"] > 0
    assert (c.scalar_reg(5), c.scalar_reg(6), c.scalar_reg(7)) == (0x100, 0x200, 0x101)


def test_retry_keeps_a_store_ahead_of_a_younger_load():
    """One MSHR, held by a miss to 0x1000. The store to 0x2008 is retried;
    the load from 0x2008 behind it must still see the store, so the retried
    store goes back to the front of the queue, not the end."""
    prog = program(base(1, 0x1000)[0], base(1, 0x1000)[1],
                   base(2, 0x2000)[0], base(2, 0x2000)[1], i("addi.s", 7, 0, 77),
                   lw(5, 1, 0), sw(7, 2, 8), lw(8, 2, 8), lw(9, 3, 0), HALT)
    c = core_for(prog, data=DATA, dcache_config=DCacheConfig(mshrs=1))
    c.dcache.warm([0])                                   # x3 = 0: lw x9 hits
    c.run_until_done()
    assert c.lsu.stats["retries"] > 0
    assert in_program_order(c)
    assert c.scalar_reg(8) == 77 and c.memory.read(0x2008) == 77


def test_lsu_queue_full_stalls_issue():
    """With a 1-deep queue: the first load goes straight to the cache, the
    second waits in the queue, and the third cannot issue until the second
    leaves it -- decode 2 sees EX5 not ready."""
    prog = program(base(1, 0x1000)[0], base(1, 0x1000)[1],
                   lw(5, 1, 0), lw(6, 1, 4), lw(7, 1, 8), HALT)
    c = core_for(prog, data=DATA, lsu_depth=1)
    c.dcache.warm([0x1000])
    c.run_until_done()
    t = issue_cycles(c)
    assert t[60] - t[40] == 1 and t[80] - t[60] > 1
    assert (c.scalar_reg(5), c.scalar_reg(6), c.scalar_reg(7)) == (0x100, 0x101, 0x102)


def test_halt_waits_for_stores_and_writes_dirty_lines_back():
    prog = program(base(1, 0x1000)[0], base(1, 0x1000)[1], i("addi.s", 7, 0, 5),
                   sw(7, 1, 0), sw(7, 1, 64), HALT)
    c = core_for(prog, data=DATA)
    c.run_until_done()
    assert c.halted_at is not None and c.done_at >= c.halted_at
    assert c.memory.read(0x1000) == 5 and c.memory.read(0x1040) == 5
    assert c.lsu.idle and not c.dcache.mshrs


# -- against the functional sim ----------------------------------------------------------

def run_both(prog, data=None):
    golden.require(pytest)
    regs, mem = golden.run_golden(prog, data)
    c = SchedulerCore(prog, data=data)
    c.run_until_done(limit=200_000)
    assert in_program_order(c)
    for r in range(1, 256):
        assert c.scalar_reg(r) == u32(regs.get(r, 0)), "x%d" % r
    for a in set(mem) | set(c.memory.words):
        assert c.memory.read(a) == u32(mem.get(a, 0)), hex(a)
    return c


def test_golden_loop_with_memory():
    """Sum a 16-word array into x20 and store running sums back."""
    prog = program(
        *base(1, 0x1000),                                    # pc 0, 20
        i("addi.s", 11, 0, 15),                              # 40
        (lw(5, 1, 0), i("addi.s", 21, 21, 1)),               # 60: loop
        r("add.s", 20, 20, 5),                               # 80
        (sw(20, 1, 0), i("addi.s", 1, 1, 4)),                # 100
        br("bne.s", 10, 11, -60, incr=1),                    # 120 -> 60
        HALT)
    c = run_both(prog, DATA)
    assert c.scalar_reg(20) == sum(0x100 + k for k in range(16))


#: Register-amount shifts are left out: the functional sim's numpy code
#: raises on a shift amount of 2**31 or more, which random values hit.
R_OPS = ["add.s", "sub.s", "or.s", "and.s", "xor.s", "slt.s", "sltu.s", "mul.s"]
I_OPS = ["addi.s", "subi.s", "ori.s", "andi.s", "xori.s", "slli.s", "srli.s",
         "srai.s", "slti.s", "sltui.s", "muli.s"]


def random_program(seed, n=60):
    """Legal packets: at most one op per EX unit, at most 4 register reads,
    no op reading or writing a register another op in its packet writes.
    Memory ops use x1/x2 as bases into DATA's three pages."""
    rng = random.Random(seed)
    regs = list(range(3, 16))
    pkts = [*base(1, 0x1000), *base(2, 0x3000)]
    for _ in range(n):
        slots, written, reads, units = [], set(), 0, set()
        for _ in range(rng.randint(1, 4)):
            kind = rng.choice("rrriimls")
            rd, a, b = rng.choice(regs), rng.choice(regs), rng.choice(regs)
            if rd in written or a in written or b in written:
                continue
            if kind == "r":
                m = rng.choice(R_OPS)
                unit = 4 if m == "mul.s" else 1
                w, nreads = r(m, rd, a, b), 2
            elif kind == "i":
                m = rng.choice(I_OPS)
                unit = 4 if m == "muli.s" else 1
                imm = rng.randrange(32) if m[:-3] in ("sll", "srl", "sra") else rng.randrange(-2048, 2048)
                w, nreads = i(m, rd, a, imm), 1
            elif kind == "m":
                unit, nreads = 5, 1
                w = lw(rd, rng.choice((1, 2)), 4 * rng.randrange(48))
            elif kind == "l":
                unit, nreads, w = 1, 0, mi("lui.s", rd, rng.randrange(1 << 25))
            else:
                unit, nreads = 5, 2
                w = sw(a, rng.choice((1, 2)), 4 * rng.randrange(48))
                rd = None
            if unit in units or reads + nreads > 4:
                continue
            units.add(unit)
            reads += nreads
            slots.append(w)
            if rd is not None:
                written.add(rd)
        if slots:
            pkts.append(tuple(slots))
    pkts.append(HALT)
    return program(*pkts)


@pytest.mark.parametrize("seed", range(8))
def test_golden_random_programs(seed):
    run_both(random_program(seed), DATA)


@pytest.mark.parametrize("seed", range(4))
def test_golden_random_programs_small_cache(seed):
    """A 512 B, 2-way cache with 2 MSHRs: evictions, retries and write-backs
    in the middle of a program."""
    golden.require(pytest)
    prog = random_program(100 + seed, n=80)
    regs, mem = golden.run_golden(prog, DATA)
    c = SchedulerCore(prog, data=DATA, dcache_config=DCacheConfig(
        size_bytes=512, ways=2, line_bytes=32, mshrs=2, targets_per_mshr=2))
    c.run_until_done(limit=200_000)
    assert in_program_order(c)
    for rr in range(1, 256):
        assert c.scalar_reg(rr) == u32(regs.get(rr, 0)), "x%d" % rr
    for a in set(mem) | set(c.memory.words):
        assert c.memory.read(a) == u32(mem.get(a, 0)), hex(a)
