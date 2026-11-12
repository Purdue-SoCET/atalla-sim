"""M2 (banked register file) and M3 (writeback arbiter and per-bank buffer)."""
import pytest

from vector_core.v2.vrf_banks import BankedVRF
from vector_core.v2.writeback import WBArbiter, WBBufferV2, WBEntry
from vector_core.v2.config import DEFAULT_WB_PRIORITY


# -- M2 ----------------------------------------------------------------------------------
def test_mod_bank_map():
    vrf = BankedVRF(8, 32)
    assert vrf.num_regs == 256
    assert vrf.bank_addr(0) == (0, 0)
    assert vrf.bank_addr(10) == (2, 1)        # the plan's example: v10 = bank 2, address 1
    assert vrf.bank_addr(255) == (7, 31)
    with pytest.raises(ValueError):
        vrf.bank_addr(256)


def test_write_rounds_to_bf16_and_enable_keeps_old_elements():
    vrf = BankedVRF(8, 32)
    vrf.write(5, [1.0] * 32)
    vrf.write(5, [1.00390625 + 1e-6] * 32, enable=[i < 4 for i in range(32)])
    got = vrf.read(5)
    assert got[:4] == [1.0078125] * 4 or got[:4] == [1.00390625] * 4   # BF16-rounded
    assert got[4:] == [1.0] * 28
    vrf.write(5, [None] * 32)                 # None is never written
    assert vrf.read(5)[4:] == [1.0] * 28


def test_untimed_api_matches_v1_core():
    from vector_core.vector_core import VectorCore
    v1 = VectorCore(veggie_size=512, lane_count=16, dtype="bf16")
    vrf = BankedVRF(8, 32)
    data = [0.1 * i for i in range(32)]
    v1.write_vreg(17, data)
    vrf.write(17, data)
    assert vrf.read(17) == v1.read_vreg(17)


# -- M3 ----------------------------------------------------------------------------------
def offer(src, reg, mask=False):
    return WBEntry(src, reg, 0xFFFF if mask else [1.0] * 32, is_mask=mask)


def test_grants_per_cycle_and_priority():
    arb = WBArbiter(DEFAULT_WB_PRIORITY, grants_per_cycle=2)
    offers = [offer("fu:mul", 1), offer("vlsu0", 2), offer("fu:alu", 3), offer("gsau", 4)]
    got = arb.arbitrate(offers, lambda r: r % 8, lambda b: 0, depth=4)
    assert [e.source for e in got] == ["vlsu0", "gsau"]
    assert arb.stats["denied"] == {"fu:alu": 1, "fu:mul": 1}


def test_full_bank_fifo_refuses():
    arb = WBArbiter(DEFAULT_WB_PRIORITY, grants_per_cycle=8)
    occ = {2: 4}
    got = arb.arbitrate([offer("fu:alu", 10), offer("fu:mul", 3)], lambda r: r % 8,
                        lambda b: occ.get(b, 0), depth=4)
    assert [e.source for e in got] == ["fu:mul"]          # bank 2 is full


def test_two_grants_to_one_bank_need_two_free_entries():
    arb = WBArbiter(DEFAULT_WB_PRIORITY, grants_per_cycle=8)
    got = arb.arbitrate([offer("vlsu0", 2), offer("vlsu1", 10)], lambda r: r % 8,
                        lambda b: 3, depth=4)
    assert [e.source for e in got] == ["vlsu0"]


def test_mask_writes_one_per_mask_bank_and_blocked_banks():
    arb = WBArbiter(DEFAULT_WB_PRIORITY, grants_per_cycle=1, mask_banks=2)
    offers = [offer("fu:alu", 2, mask=True), offer("vlsu0", 1)]
    got = arb.arbitrate(offers, lambda r: r % 8, lambda b: 0, depth=4)
    assert {e.source for e in got} == {"fu:alu", "vlsu0"}  # masks don't use vector grants
    got = arb.arbitrate([offer("fu:alu", 2, mask=True)], lambda r: r % 8, lambda b: 0,
                        depth=4, blocked_mask_banks={0})
    assert got == []                                       # mv.stm owns mask bank 0 (Q21)


def test_buffer_drains_after_t_wbbuf_one_write_per_bank():
    buf = WBBufferV2(num_banks=8, depth=4, write_ports=1, t_wbbuf=1)
    buf.push(offer("a", 2), bank=2, now=10)
    buf.push(offer("b", 10), bank=2, now=10)
    buf.push(offer("c", 3), bank=3, now=10)
    assert buf.pop_due(10) == []
    first = buf.pop_due(11)
    assert sorted(e.source for e in first) == ["a", "c"]   # one per bank, oldest first
    assert [e.source for e in buf.pop_due(12)] == ["b"]
    assert buf.empty
