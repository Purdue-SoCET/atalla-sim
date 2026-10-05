"""The scalar core's lockup-free data cache (src/scheduler/dcache.py).

Cycle expectations, default config (read 2 / write 4, 8 beats, memory free):
a request taken in cycle a reads its set at the edge ending a; the read's
done is visible in a+3 and the lookup evaluates in a+4. A load hit answers in
a+4. A store hit writes at the end of a+4 and answers when the write's done
is visible, a+9. A miss evaluated in t starts its fill in t+1:

    t+1        read the set           t+2..t+4   wait (done in t+4)
    t+5..t+7   pick a way             t+8        invalidate it
    t+9..t+13  wait (done in t+13)    t+14..t+21 8 beats from memory
    [8 cycles writing back a dirty victim]
    t+22       write the line         t+23..t+27 wait
    t+28       answer the loads
"""
import random

import pytest

from scheduler.dcache import (
    FILL, HIT, RETRY, STORE_ACK, DCache, DCacheConfig, DCacheReq, WordMemory,
    plru_touch, plru_victim)


def mem_of(addr):
    return (addr * 7 + 3) & 0xFFFFFFFF


class Driver:
    """A non-blocking requester: presents queued requests in order, holds each
    until taken, puts a retried one back at the front, never waits for
    loads."""

    def __init__(self, cache):
        self.cache = cache
        self.queue = []
        self.cycle = 0
        self.taken = []           # (cycle, req)
        self.resp = {}            # id -> (cycle, DCacheResp), last seen
        self.log = []             # (cycle, DCacheResp)

    def push(self, *reqs):
        self.queue.extend(reqs)

    def step(self, halt=False):
        req = self.queue[0] if self.queue else None
        taken, resps = self.cache.step(req, halt=halt)
        if taken:
            self.taken.append((self.cycle, self.queue.pop(0)))
        for r in resps:
            self.log.append((self.cycle, r))
            self.resp[r.id] = (self.cycle, r)
            if r.kind == RETRY:
                self.queue.insert(0, DCacheReq(r.id, r.addr, r.store,
                                               self._data_of(r.id)))
        self.cycle += 1
        return resps

    def _data_of(self, rid):
        for _, req in reversed(self.taken):
            if req.id == rid:
                return req.data
        raise KeyError(rid)

    def run(self, cycles=None, until=None, halt=False, limit=100000):
        n = 0
        while True:
            if cycles is not None and n >= cycles:
                return
            if until is not None and until():
                return
            assert n < limit, "did not finish"
            self.step(halt=halt)
            n += 1

    def drain(self):
        self.run(until=lambda: not self.queue and self.cache.idle())

    def at(self, rid):
        return self.resp[rid][0]

    def got(self, rid):
        return self.resp[rid][1]


def make(config=DCacheConfig(), init=mem_of):
    return DCache(config=config, memory=WordMemory(init))


# ---------------------------------------------------------------------------
# configuration and replacement
# ---------------------------------------------------------------------------

def test_default_geometry_is_the_rtls():
    c = DCacheConfig()
    assert (c.sets, c.ways, c.line_words, c.beats, c.mshrs) == (16, 4, 16, 8, 8)
    assert (c.set_of(0x454), c.tag_of(0x454), c.word_of(0x454)) == (1, 1, 5)


@pytest.mark.parametrize("bad", [dict(ways=3), dict(line_bytes=12),
                                 dict(size_bytes=3000), dict(mshrs=0)])
def test_rejects_bad_geometry(bad):
    with pytest.raises(ValueError):
        DCacheConfig(**bad)


def test_plru_cycles_through_every_way():
    for ways in (1, 2, 4, 8):
        tree, seen = [0] * (ways - 1), []
        for _ in range(ways):
            v = plru_victim(tree, ways)
            seen.append(v)
            tree = plru_touch(tree, ways, v)
        assert sorted(seen) == list(range(ways))


def test_plru_never_picks_the_last_used():
    tree = [0, 0, 0]
    for way in (2, 0, 3, 1, 1, 2):
        tree = plru_touch(tree, 4, way)
        assert plru_victim(tree, 4) != way


# ---------------------------------------------------------------------------
# single requests: the RTL's latencies
# ---------------------------------------------------------------------------

def test_load_hit_in_4():
    c = make()
    c.warm([0x48])
    d = Driver(c)
    d.push(DCacheReq("ld", 0x48))
    d.drain()
    assert d.taken[0][0] == 0
    assert d.at("ld") == 4
    assert d.got("ld") == DCacheResp_("ld", 0x48, False, HIT, mem_of(0x48))


def DCacheResp_(*a):
    from scheduler.dcache import DCacheResp
    return DCacheResp(*a)


def test_store_hit_in_9_and_visible_after():
    c = make()
    c.warm([0x50])
    d = Driver(c)
    d.push(DCacheReq("st", 0x50, True, 0xAAAA0001), DCacheReq("ld", 0x50))
    d.drain()
    assert (d.at("st"), d.got("st").kind) == (9, HIT)
    # the load is taken in 10, once the lookup is idle and the SRAM free
    assert d.taken[1][0] == 10
    assert (d.at("ld"), d.got("ld").data) == (14, 0xAAAA0001)


def test_load_miss_filled_28_after_the_miss():
    c = make()
    d = Driver(c)
    d.push(DCacheReq("ld", 0x48))
    d.drain()
    assert d.at("ld") == 4 + 28
    assert d.got("ld") == DCacheResp_("ld", 0x48, False, FILL, mem_of(0x48))
    d.push(DCacheReq("ld2", 0x7C))           # same line: now a hit
    d.drain()
    assert d.got("ld2").kind == HIT and d.got("ld2").data == mem_of(0x7C)


def test_store_miss_is_answered_at_once_and_allocates():
    c = make()
    d = Driver(c)
    d.push(DCacheReq("st", 0x454, True, 0xBBBB0002))
    d.run(cycles=5)
    assert (d.at("st"), d.got("st").kind) == (4, STORE_ACK)
    d.drain()
    d.push(DCacheReq("ld", 0x454), DCacheReq("ld2", 0x450))
    d.drain()
    assert d.got("ld").kind == HIT and d.got("ld").data == 0xBBBB0002
    assert d.got("ld2").data == mem_of(0x450)
    assert c.stats["fills"] == 1


def test_memory_port_during_a_fill():
    c = make()
    d = Driver(c)
    d.push(DCacheReq("ld", 0x48))
    reads = []
    while d.cycle < 40:
        d.step()
        if c.out_mem:
            reads.append((d.cycle - 1, c.out_mem))
    assert reads == [(t, ("read", 0x40, t - 18)) for t in range(18, 26)]


def test_dirty_victim_costs_8_more_and_reaches_memory():
    cfg = DCacheConfig()
    c = make(cfg)
    d = Driver(c)
    stride = cfg.sets * cfg.line_bytes          # same set, next tag
    lines = [0x40 + k * stride for k in range(cfg.ways)]
    c.warm(lines)
    d.push(*[DCacheReq(("st", k), a, True, 0x1000 + k) for k, a in enumerate(lines)])
    d.drain()
    victim = plru_victim(c.lru[1], cfg.ways)
    victim_addr = lines[victim]
    start = d.cycle
    d.push(DCacheReq("ld", 0x40 + cfg.ways * stride))
    d.drain()
    t = start + 4                                # the miss is evaluated here
    assert d.at("ld") == t + 28 + 8
    assert c.memory.read(victim_addr) == 0x1000 + victim
    assert c.stats["writebacks"] == 1 and c.stats["evictions"] == 1


def test_memory_latency_parameters():
    cfg = DCacheConfig(first_beat_wait=5)
    c = make(cfg)
    d = Driver(c)
    d.push(DCacheReq("ld", 0x48))
    d.drain()
    assert d.at("ld") == 4 + 28 + 5


# ---------------------------------------------------------------------------
# lockup-free behaviour
# ---------------------------------------------------------------------------

def test_hit_under_miss():
    c = make()
    c.warm([0x840])
    d = Driver(c)
    d.push(DCacheReq("miss", 0x2080), DCacheReq("hit", 0x844))
    d.drain()
    assert d.got("hit").kind == HIT and d.got("hit").data == mem_of(0x844)
    assert d.at("hit") < d.at("miss")
    # In 5 the hit and the fill's set read both want the SRAM; the lookup has
    # priority, so the hit is taken in 5 and answers in 9, and the fill reads
    # in 8, when the hit's read is done: 3 cycles late.
    assert d.taken[1][0] == 5 and d.at("hit") == 9
    assert d.at("miss") == 4 + 28 + 3


def test_two_misses_overlap_and_fill_in_order():
    c = make()
    d = Driver(c)
    d.push(DCacheReq("a", 0x3100), DCacheReq("b", 0x3140))
    d.drain()
    assert c.stats["misses"] == 2 and c.stats["max_mshrs"] == 2
    assert d.at("a") < d.at("b")
    assert d.got("a").data == mem_of(0x3100) and d.got("b").data == mem_of(0x3140)


def test_secondary_misses_merge_in_order():
    c = make()
    d = Driver(c)
    d.push(DCacheReq("l0", 0x4180), DCacheReq("s", 0x4184, True, 0xCCCC0003),
           DCacheReq("l1", 0x4184), DCacheReq("s2", 0x4184, True, 0xDDDD0004))
    d.drain()
    assert c.stats["fills"] == 1 and c.stats["secondary"] == 3
    assert d.got("s").kind == STORE_ACK and d.got("s2").kind == STORE_ACK
    assert d.at("l0") == d.at("l1")              # one answer cycle
    assert d.got("l0").data == mem_of(0x4180)
    assert d.got("l1").data == 0xCCCC0003        # after s, before s2
    d.push(DCacheReq("l2", 0x4184))
    d.drain()
    assert d.got("l2").data == 0xDDDD0004


def test_full_mshr_answers_retry_then_accepts():
    c = make(DCacheConfig(mshrs=2))
    d = Driver(c)
    d.push(*[DCacheReq(k, 0x1000 * (k + 1)) for k in range(3)])
    d.drain()
    kinds = [r.kind for _, r in d.log if r.id == 2]
    assert kinds[0] == RETRY and kinds[-1] == FILL
    assert all(d.got(k).data == mem_of(0x1000 * (k + 1)) for k in range(3))
    assert c.stats["max_mshrs"] == 2


def test_full_targets_answer_retry():
    c = make(DCacheConfig(targets_per_mshr=2))
    d = Driver(c)
    d.push(*[DCacheReq(k, 0x200 + 4 * k) for k in range(3)])
    d.drain()
    assert [r.kind for _, r in d.log if r.id == 2][0] == RETRY
    assert all(d.got(k).kind in (FILL, HIT) for k in range(3))


def test_lookups_cannot_starve_a_fill():
    c = make()
    c.warm([0x840])
    d = Driver(c)
    d.push(DCacheReq("miss", 0x2080))
    d.push(*[DCacheReq(("h", k), 0x844) for k in range(200)])
    d.run(until=lambda: "miss" in d.resp)
    # Each SRAM access of the fill waits at most for one lookup's read.
    assert d.at("miss") < 4 + 28 + 5 * 4
    d.drain()


def test_store_to_a_line_being_evicted_is_not_lost():
    """Fill a set with dirty lines, miss in it, and store to every line of the
    set while the fill runs. Whichever line is evicted, no store is lost."""
    cfg = DCacheConfig()
    stride = cfg.sets * cfg.line_bytes
    c = make(cfg)
    lines = [0x40 + k * stride for k in range(cfg.ways)]
    c.warm(lines)
    d = Driver(c)
    d.push(DCacheReq("miss", 0x40 + cfg.ways * stride))
    for rnd in range(12):
        d.push(*[DCacheReq(("st", rnd, k), a + 8, True, rnd * 16 + k)
                 for k, a in enumerate(lines)])
    d.drain()
    d.run(until=lambda: c.out_flushed, halt=True)
    for k, a in enumerate(lines):
        assert c.memory.read(a + 8) == 11 * 16 + k


# ---------------------------------------------------------------------------
# halt
# ---------------------------------------------------------------------------

def test_halt_writes_back_dirty_lines_and_holds_flushed():
    c = make()
    d = Driver(c)
    d.push(DCacheReq("s1", 0x48, True, 1), DCacheReq("s2", 0x2080, True, 2),
           DCacheReq("l", 0x3100))
    d.run(until=lambda: c.out_flushed, halt=True)
    assert c.memory.read(0x48) == 1 and c.memory.read(0x2080) == 2
    assert c.stats["flush_writebacks"] == 2
    assert not c.out_req_ready
    d.run(cycles=3, halt=True)
    assert c.out_flushed
    d.step(halt=False)
    d.push(DCacheReq("after", 0x48))
    d.drain()
    assert d.got("after").kind == HIT and d.got("after").data == 1


def test_halt_waits_for_outstanding_misses():
    c = make()
    d = Driver(c)
    d.push(DCacheReq("st", 0x48, True, 9))
    d.run(until=lambda: c.out_flushed, halt=True)
    assert "st" in d.resp and c.memory.read(0x48) == 9


# ---------------------------------------------------------------------------
# random traffic against a reference memory
# ---------------------------------------------------------------------------

def check_against_reference(cfg, seed, n=600, addresses=None):
    rng = random.Random(seed)
    c = make(cfg)
    d = Driver(c)
    stride = cfg.sets * cfg.line_bytes
    if addresses is None:
        addresses = [s * cfg.line_bytes + t * stride + 4 * w
                     for s in range(3) for t in range(cfg.ways + 2)
                     for w in (0, 1, cfg.line_words - 1)]
    for i in range(n):
        a = rng.choice(addresses)
        if rng.random() < 0.4:
            d.push(DCacheReq(i, a, True, rng.getrandbits(32)))
        else:
            d.push(DCacheReq(i, a))
        if rng.random() < 0.1:
            d.run(cycles=rng.randrange(1, 40))
    d.drain()
    d.run(until=lambda: c.out_flushed, halt=True)

    # Lookups are serial, so requests take effect in the order taken,
    # except a take answered RETRY, which had no effect.
    retried_takes = set()
    for cyc, r in d.log:
        if r.kind == RETRY:
            # the take it answers is the last one of this id before `cyc`
            k = max(j for j, (tc, q) in enumerate(d.taken) if q.id == r.id and tc < cyc)
            retried_takes.add(k)
    shadow = {}
    for j, (_, req) in enumerate(d.taken):
        if j in retried_takes:
            continue
        if req.store:
            shadow[req.addr] = req.data
        else:
            want = shadow.get(req.addr, mem_of(req.addr))
            got = d.got(req.id)
            assert got.kind in (HIT, FILL), (req, got)
            assert got.data == want, (req, got, hex(want))
    for a in addresses:
        assert c.memory.read(a) == shadow.get(a, mem_of(a)), hex(a)
    assert all(d.got(i).kind != RETRY for i in range(n))
    return c


@pytest.mark.parametrize("seed", range(6))
def test_random_default_config(seed):
    c = check_against_reference(DCacheConfig(), seed)
    assert c.stats["secondary"] > 0 and c.stats["evictions"] > 0


@pytest.mark.parametrize("seed", range(6))
def test_random_small_config_with_slow_memory(seed):
    cfg = DCacheConfig(size_bytes=512, ways=2, line_bytes=32, mshrs=3,
                       targets_per_mshr=3, first_beat_wait=3, write_wait=1,
                       sram_read_latency=1, sram_write_latency=2)
    c = check_against_reference(cfg, 100 + seed)
    assert c.stats["retries"] > 0


def test_random_direct_mapped():
    check_against_reference(DCacheConfig(size_bytes=1024, ways=1, line_bytes=16), 7)


def test_random_traffic_takes_the_present_block_path():
    """A lookup that read the set before a fill's line write and misses after
    it opens a new entry for a block that is now present; that fill needs no
    memory. Check it happens in random traffic and stays correct."""
    total = 0
    for seed in range(6):
        total += check_against_reference(DCacheConfig(), 200 + seed).stats["present_fills"]
    assert total > 0
