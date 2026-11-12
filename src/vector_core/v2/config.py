"""Configuration for the M x N vector core (v2) and the factory that picks
between it and today's lane datapath (v1).

The design is docs/vector_core_v2_change_plan.md; every default below is a
decision recorded there (the Q number says which). VectorCoreConfig(arch=
"lanes") -- the default -- builds today's VectorCore unchanged.

    S slots (num_slots)        V1..VS, each wired to a pair of banks
    j FU groups (fu_groups)    full-width (32-element) units, one vector per cycle
    B = 2S banks               bank = reg % B, addr = reg // B (Q1)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class FUGroupSpec:
    """One full-width functional-unit group: its element ops, latency L and
    initiation interval. `rtree` marks the ALU group, whose reduction results
    pass through the reduction tree."""
    name: str
    ops: Tuple[str, ...]
    latency: int
    ii: int = 1
    rtree: bool = False


#: Q6: ALU 1, MUL 1, EXP 14, SQRT 8, RECIP 11, all II = 1. The ALU group runs
#: add/sub/compares and is the reduction tree's input.
DEFAULT_V2_GROUPS: Tuple[FUGroupSpec, ...] = (
    FUGroupSpec("alu", ("add", "sub", "max", "min", "and", "or", "xor",
                        "gt", "lt", "eq", "ne"), 1, rtree=True),
    FUGroupSpec("mul", ("mul",), 1),
    FUGroupSpec("exp", ("exp",), 14),
    FUGroupSpec("sqrt", ("sqrt",), 8),
    FUGroupSpec("recip", ("recip",), 11),
)

#: Q11 / section 7: fixed tie-break order of the writeback arbiter's sources.
DEFAULT_WB_PRIORITY: Tuple[str, ...] = (
    "vlsu0", "vlsu1", "vlsu2", "vlsu3", "gsau", "transpose",
    "fu:rtree", "fu:alu", "fu:mul", "fu:exp", "fu:sqrt", "fu:recip")


@dataclass(frozen=True)
class VectorCoreConfig:
    arch: str = "lanes"                  # "lanes" (v1, default) | "fu_groups" (v2)
    # v1 (today's arguments, unchanged)
    lane_count: int = 16
    fu_latencies: Optional[Dict[str, int]] = None
    # v2
    num_slots: int = 4
    vector_len: int = 32
    fu_groups: Tuple[FUGroupSpec, ...] = DEFAULT_V2_GROUPS
    regs_per_bank: int = 32              # 32 x 8 banks = 256 registers (Q1)
    bank_read_ports: int = 1             # 1R1W banks (Q2)
    bank_write_ports: int = 1
    t_read: int = 1                      # Q10: D = L + 3
    t_wbbuf: int = 1
    t_write_visible: int = 1
    rtree_levels: int = 5                # Q7: 5 levels x 1 cycle
    wb_grants_per_cycle: int = 4         # Q11; sweep 1/2/4/8
    wb_fifo_depth: int = 4
    wb_priority: Tuple[str, ...] = DEFAULT_WB_PRIORITY
    mask_banks: int = 2                  # Q16: shared mask file, 1 read port per bank
    m0_uses_port: bool = True            # Q16a
    read_mode: str = "strict"            # "strict" | "compat" (Q34)
    compat_read_penalty: int = 1
    strict: bool = True                  # raise SlotContractError in strict read mode
    round_queue_depth: int = 2           # rounds waiting in front of the slots
    transpose_queue: int = 2             # as scheduler/vector.py TRANSPOSE_QUEUE
    watchdog_cycles: int = 10_000

    def __post_init__(self):
        if self.arch not in ("lanes", "fu_groups"):
            raise ValueError("arch must be 'lanes' or 'fu_groups', got %r" % self.arch)
        if self.read_mode not in ("strict", "compat"):
            raise ValueError("read_mode must be 'strict' or 'compat', got %r" % self.read_mode)
        if self.num_slots < 1:
            raise ValueError("num_slots must be >= 1")
        if self.wb_grants_per_cycle < 1 or self.wb_fifo_depth < 1:
            raise ValueError("wb_grants_per_cycle and wb_fifo_depth must be >= 1")
        names = [g.name for g in self.fu_groups]
        if len(set(names)) != len(names):
            raise ValueError("FU group names must be unique: %s" % names)
        if sum(g.rtree for g in self.fu_groups) > 1:
            raise ValueError("at most one FU group carries the reduction tree")

    @property
    def num_banks(self) -> int:
        return 2 * self.num_slots

    @property
    def num_vregs(self) -> int:
        return self.num_banks * self.regs_per_bank

    def slot_banks(self, slot: int) -> Tuple[int, int]:
        """Slot k's bank pair (0-based slot): (even bank for src0, odd for src1)."""
        return 2 * slot, 2 * slot + 1

    def group_for_op(self, op: str) -> str:
        for g in self.fu_groups:
            if op in g.ops:
                return g.name
        raise ValueError("no FU group runs %r" % op)


def make_vector_core(cfg: Optional[VectorCoreConfig] = None, *, veggie_size: int = 32 * 16,
                     vls_count: int = 4, dtype: str = "bf16", lane_count: Optional[int] = None,
                     fu_latencies: Optional[Dict[str, int]] = None):
    """Today's VectorCore for arch "lanes", the M x N VectorCoreV2 for
    "fu_groups". `lane_count` and `fu_latencies` override the config's v1
    fields (they are what build_tpu_platform has always passed)."""
    cfg = cfg or VectorCoreConfig()
    if cfg.arch == "lanes":
        from vector_core.vector_core import VectorCore
        return VectorCore(veggie_size=int(veggie_size),
                          lane_count=int(lane_count if lane_count is not None else cfg.lane_count),
                          vls_count=int(vls_count),
                          fu_latencies=fu_latencies if fu_latencies is not None else cfg.fu_latencies,
                          dtype=str(dtype))
    from vector_core.v2.core import VectorCoreV2
    if str(dtype).lower() != "bf16":
        raise ValueError("the M x N vector core models BF16 only (Q14), not %r" % dtype)
    return VectorCoreV2(cfg, vls_count=int(vls_count))
