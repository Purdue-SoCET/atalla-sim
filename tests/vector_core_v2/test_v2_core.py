"""M4: slots, crossbar arbitration, bank and mask read ports, the single-mode
scoreboard and writeback, on VectorCoreV2 alone (no scheduler).

Slot k (0-based) owns banks 2k (src0) and 2k+1 (src1); register r is in bank
r % 8. So slot 1 reads v2/v10/... as src0 and v3/v11/... as src1.
"""
import pytest

from vector_core.v2.config import VectorCoreConfig
from vector_core.v2.core import VectorCoreV2
from vector_core.v2.fu_group import elementwise
from vector_core.v2.slots import SlotOp
from vector_core.v2.vrf_banks import SlotContractError


def make(**kw):
    kw.setdefault("arch", "fu_groups")
    vc = VectorCoreV2(VectorCoreConfig(**kw))
    vc.fire_log = []
    return vc


def run(vc, cycles=60, start=0):
    for c in range(start, start + cycles):
        vc.tick(float(c))
        if vc.idle:
            return c
    return None


def fires(vc, name):
    return [c for c, _, n in vc.fire_log if n == name]


def elem(slot, op, src0, src1, vd, name, dest=None, mask=0):
    return SlotOp(slot=slot, dest=dest or ("fu:mul" if op == "mul" else
                                           "fu:exp" if op == "exp" else "fu:alu"),
                  kind="elem", op=op, src0=src0, src1=src1, vd=vd, mask_reg=mask, name=name)


@pytest.mark.parametrize("op,group,L", [("add", "fu:alu", 1), ("mul", "fu:mul", 1),
                                        ("exp", "fu:exp", 14)])
def test_dependent_delay_is_L_plus_3(op, group, L):
    vc = make()
    vc.write_vreg(2, [0.5] * 32)
    vc.write_vreg(3, [0.25] * 32)
    # producer in V2 (slot 1) writes v4, which is V3's (slot 2) src0 bank
    vc.issue_round([elem(1, op, 2, 3 if op != "exp" else None, 4, "p", group)])
    vc.issue_round([elem(2, "add", 4, 5, 12, "c")])
    run(vc)
    assert fires(vc, "c")[0] - fires(vc, "p")[0] == L + 3


def test_reduction_dependent_delay_is_9():
    vc = make()
    vc.write_vreg(2, [1.0] * 32)
    vc.issue_round([SlotOp(slot=1, dest="fu:alu", kind="reduce", op="sum", src0=2, vd=4,
                           mask_reg=0, imm=0x40, name="p")])
    vc.issue_round([elem(2, "add", 4, 5, 12, "c")])
    run(vc)
    assert fires(vc, "c")[0] - fires(vc, "p")[0] == 9
    assert vc.read_vreg(4) == [32.0] * 32


def test_values_end_to_end():
    vc = make()
    a = [0.125 * i for i in range(32)]
    b = [1.5 - 0.0625 * i for i in range(32)]
    vc.write_vreg(2, a)
    vc.write_vreg(3, b)
    vc.issue_round([elem(1, "mul", 2, 3, 4, "p")])
    run(vc)
    assert vc.read_vreg(4) == elementwise("mul", a, b, [True] * 32)


def test_four_slots_fire_together_when_masks_allow():
    vc = make()
    vc.issue_round([
        elem(0, "add", 0, 1, 16, "a", mask=0),                 # m0: mask bank 0
        elem(1, "mul", 2, 3, 17, "b", mask=1),                 # m1: mask bank 1
        SlotOp(slot=2, dest="vlsu0", kind="load", vd=20, vlsu={"addr": 0, "vl": 32},
               name="ld"),
        SlotOp(slot=3, dest="gsau", kind="gemm", src0=6, vd=21, name="gm"),
    ])
    vc.tick(0.0)
    assert sorted(n for c, _, n in vc.fire_log if c == 0) == ["a", "b", "gm", "ld"]


def test_two_unmasked_ops_share_mask_bank_0():
    vc = make()                                               # m0_uses_port=True (Q16a)
    vc.issue_round([elem(0, "add", 0, 1, 16, "a"), elem(1, "mul", 2, 3, 17, "b")])
    run(vc, 5)
    assert fires(vc, "a") == [0] and fires(vc, "b") == [1]
    assert vc.stats["stall"]["mask"][1] == 1


def test_m0_without_a_port():
    vc = make(m0_uses_port=False)
    vc.issue_round([elem(0, "add", 0, 1, 16, "a"), elem(1, "mul", 2, 3, 17, "b")])
    vc.tick(0.0)
    assert fires(vc, "a") == [0] and fires(vc, "b") == [0]


def test_two_slots_on_one_group_collide():
    vc = make(m0_uses_port=False)
    vc.issue_round([elem(0, "add", 0, 1, 16, "a"), elem(1, "sub", 2, 3, 17, "b")])
    run(vc, 5)
    assert fires(vc, "a") == [0] and fires(vc, "b") == [1]    # II = 1: one cycle later
    assert vc.stats["stall"]["xbar"][1] == 1


def test_strict_rejects_a_source_outside_the_pair():
    vc = make()
    with pytest.raises(SlotContractError):
        vc.issue_round([elem(0, "add", 4, 1, 16, "a")])      # v4 is bank 4, not slot 0's


def test_compat_charges_an_out_of_pair_read():
    vc = make(read_mode="compat", m0_uses_port=False)
    vc.write_vreg(2, [1.0] * 32)
    vc.write_vreg(3, [2.0] * 32)
    vc.issue_round([elem(0, "add", 2, 3, 16, "a")])           # both sources in slot 1's pair
    run(vc, 10)
    assert fires(vc, "a") == [1]                              # +1 cycle
    assert vc.stats["compat_cross_pair_reads"][0] == 1
    assert vc.read_vreg(16) == [3.0] * 32


def test_both_sources_in_one_bank_cost_a_cycle():
    vc = make(m0_uses_port=False)
    vc.issue_round([elem(0, "add", 0, 8, 16, "a")])           # v0 and v8 are both bank 0
    run(vc, 10)
    assert fires(vc, "a") == [1]


def test_compat_slots_contend_for_one_bank():
    vc = make(read_mode="compat", m0_uses_port=False, compat_read_penalty=0)
    vc.issue_round([elem(0, "add", 2, 1, 16, "a"), elem(1, "mul", 10, 3, 17, "b")])
    run(vc, 10)
    # v2 and v10 are both bank 2, owned by slot 1, which reads first
    assert fires(vc, "b") == [0] and fires(vc, "a") == [1]


def test_war_inside_a_round():
    vc = make(m0_uses_port=False)
    vc.write_vreg(2, [1.0] * 32)
    vc.write_vreg(9, [5.0] * 32)
    # round 1 makes v2 busy for a while (exp); round 2: slot 1 reads v2 and v11,
    # slot 0 overwrites v11 -- it must not fire until slot 1 has read v11
    vc.issue_round([elem(1, "exp", 2, None, 2 + 8, "e", "fu:exp")])     # writes v10
    vc.issue_round([elem(1, "add", 10, 11, 18, "r"),
                    elem(0, "mul", 0, 9, 11, "w")])
    run(vc, 80)
    assert fires(vc, "w")[0] >= fires(vc, "r")[0]
    assert vc.stats["stall"]["war"][0] > 0


def test_two_results_to_two_banks_write_in_the_same_cycle():
    vc = make(m0_uses_port=False)
    vc.issue_round([elem(0, "add", 0, 1, 16, "a", mask=1),     # v16: bank 0
                    elem(1, "mul", 2, 3, 17, "b", mask=0)])    # v17: bank 1
    run(vc, 20)
    assert fires(vc, "a") == fires(vc, "b") == [0]
    assert vc.stats["wb_writes"] == 2
    assert vc.readable_at[16] == vc.readable_at[17] == 0 + 1 + 3
    assert vc.idle


def test_watchdog_on_a_stuck_round():
    vc = make(watchdog_cycles=20)
    vc.busy[2] = 1                         # a write that will never land
    vc.issue_round([elem(1, "add", 2, 3, 4, "stuck")])
    with pytest.raises(RuntimeError, match="no progress"):
        run(vc, 100)
