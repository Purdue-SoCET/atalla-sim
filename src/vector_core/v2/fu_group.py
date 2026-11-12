"""Full-width FU groups and the reduction tree (plan section 2).

A whole 32-element vector enters a group in one cycle. The value is computed
when the token enters, with the same element ops and BF16 rounding as the
v1 lanes (vector_lanes._op_lut, cast_scalar), so element values match v1 and
only the timing changes.

Timing, per FU group of latency L:

    cycle c      a slot fires; the token is accepted at the end of c
    c+1 .. c+L   the token moves through L stages
    c+L+1        it sits in the output register, offered to writeback

A token not granted by writeback holds the output register, and in "stall"
mode the whole group freezes: no stage moves and nothing new is accepted.

The ALU group feeds the reduction tree (RTree): a reduction's elements leave
the ALU after L_alu cycles and enter the tree, whose own output is a separate
writeback source, "fu:rtree" (Q7 as revised to follow the diagram).

The groups are driven by their owner (VectorCoreV2) in two steps per cycle,
so nothing depends on call order:

    ready(op, granted)    from registered state and this cycle's writeback
                          grant of the output register
    commit(granted, token)  advance one cycle, accepting `token` if given
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from base.dtype import DType, cast_scalar
from vector_core.vector_lanes import GlobalReductionUnit, _op_lut

_OPS = _op_lut()
_OPS["recip"] = lambda a, _b: (math.copysign(math.inf, a) if a == 0.0 else 1.0 / a)

#: Identity of each reduction, for masked-off elements (Q7: the tree shape
#: stays fixed).
REDUCE_IDENTITY = {"sum": 0.0, "min": math.inf, "max": -math.inf}
COMPARE_OPS = ("gt", "lt", "eq", "ne")


@dataclass
class Token:
    """One vector instruction inside an FU group.

    `kind` is "vreg" (writes vector register `dst` where `enable` is set),
    "mask" (writes mask register `dst`; `value` is the 32-bit mask) or
    "reduce" (a reduction, finished by the RTree into a full-vector write)."""
    kind: str
    dst: int
    value: Any
    enable: Optional[List[bool]] = None
    seq: int = 0
    slot: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


def elementwise(op: str, src0: Sequence[float], src1: Sequence[float],
                mask: Sequence[bool]) -> List[Optional[float]]:
    """Each active element's BF16 result, None for masked-off ones (their
    write enable is off, Q10b)."""
    fn = _OPS[op]
    return [cast_scalar(fn(a, b), DType.BF16) if m else None
            for a, b, m in zip(src0, src1, mask)]


def compare_mask(op: str, src0: Sequence[float], src1: Sequence[float], mask: int,
                 old: int) -> int:
    """A compare's 32-bit mask: the result bits where `mask` is set, the
    destination's old bits elsewhere (as v1's _collect_lanes)."""
    fn = _OPS[op]
    bits = sum(1 << i for i, (a, b) in enumerate(zip(src0, src1)) if fn(a, b))
    return (bits & mask) | (old & ~mask & 0xFFFFFFFF)


def reduce_tree32(values: Sequence[float], mask: Sequence[bool], op: str) -> float:
    """The RTree: adjacent pairs, level by level, masked-off elements as the
    identity. Sums are formed in double precision; the register file rounds
    once to BF16. With every element active this is v1's order exactly (each
    lane folds its 2 contiguous elements, then a pairwise tree across lanes)."""
    ident = REDUCE_IDENTITY[op]
    pair = {"sum": lambda a, b: a + b,
            "min": lambda a, b: a if a <= b else b,
            "max": lambda a, b: a if a >= b else b}[op]
    level = [float(v) if m else ident for v, m in zip(values, mask)]
    while len(level) > 1:
        nxt = [pair(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def reduction_vector(src0: Sequence[float], mask: Sequence[bool], op: str, out_mode: str,
                     index: int) -> List[float]:
    """A reduction's full output vector, placed as imm says (broadcast, one
    element, or one element over vs1's active elements -- v1's seed)."""
    n = len(src0)
    q = [cast_scalar(float(a), DType.BF16) for a in src0]
    result = reduce_tree32(q, mask, op)
    seed = [q[i] if mask[i] else 0.0 for i in range(n)] if out_mode == "partial_passthru" else None
    return GlobalReductionUnit(n).format_output_vector(result, out_mode, seed, index)


class FUGroup:
    """A pipelined full-width unit of latency L and initiation interval II."""

    def __init__(self, name: str, latency: int, ii: int = 1, stall_mode: str = "stall"):
        if latency < 1 or ii < 1:
            raise ValueError("latency and ii must be >= 1")
        self.name = name
        self.latency = int(latency)
        self.ii = int(ii)
        self.stall_mode = stall_mode
        self.stages: List[List[Any]] = []     # [token, cycles left], oldest first
        self.out: Optional[Token] = None
        self.last_accept: Optional[int] = None
        self.cycle = 0
        self.stats = dict(accepts=0, busy_cycles=0, frozen_cycles=0)

    # -- this cycle's view --------------------------------------------------------
    def frozen(self, granted: bool) -> bool:
        return self.out is not None and not granted

    def ready(self, granted: bool = False) -> bool:
        """Can a token enter at the end of this cycle?"""
        if self.frozen(granted):
            return False
        return self.last_accept is None or self.cycle - self.last_accept >= self.ii

    @property
    def idle(self) -> bool:
        return not self.stages and self.out is None

    # -- the edge -----------------------------------------------------------------
    def _advance(self, granted: bool) -> List[Token]:
        """Move every stage one cycle; return the tokens that finished."""
        if self.out is not None and granted:
            self.out = None
        if self.out is not None:                  # frozen: nothing moves
            self.stats["frozen_cycles"] += 1
            return []
        for st in self.stages:
            if st[1] > 0:
                st[1] -= 1
        done = []
        while self.stages and self.stages[0][1] == 0:
            done.append(self.stages.pop(0)[0])
        return done

    def commit(self, granted: bool = False, token: Optional[Token] = None) -> None:
        if not self.idle:
            self.stats["busy_cycles"] += 1
        for t in self._advance(granted):
            if self.out is not None:
                raise AssertionError("%s: two tokens left in one cycle" % self.name)
            self.out = t
        self._accept(token)
        self.cycle += 1

    def _accept(self, token: Optional[Token]) -> None:
        if token is None:
            return
        self.stages.append([token, self.latency])
        self.last_accept = self.cycle
        self.stats["accepts"] += 1


class ReductionTree(FUGroup):
    """The RTree: `levels` pipelined levels of one cycle each. Its tokens
    already carry the finished vector (computed at the ALU)."""

    def __init__(self, levels: int = 5):
        super().__init__("rtree", latency=levels)


class ALUGroup(FUGroup):
    """The ALU group and its reduction tree. Plain results go to `out`
    (writeback source fu:alu); a reduction's go through `rtree` (fu:rtree)."""

    def __init__(self, latency: int = 1, ii: int = 1, rtree_levels: int = 5):
        super().__init__("alu", latency, ii)
        self.rtree = ReductionTree(rtree_levels)
        self._held_reduce: Optional[Token] = None

    def ready_for(self, kind: str, alu_granted: bool, rtree_granted: bool) -> bool:
        """Every op needs the ALU stage, which a stuck ALU output freezes. A
        reduction also needs the tree to take it L_alu cycles later, so a
        frozen tree blocks reductions only (plan section 2)."""
        if not FUGroup.ready(self, alu_granted):
            return False
        if kind == "reduce":
            return self._held_reduce is None and not self.rtree.frozen(rtree_granted)
        return True

    @property
    def idle(self) -> bool:
        return (not self.stages and self.out is None and self._held_reduce is None
                and self.rtree.idle)

    def commit(self, granted: bool = False, token: Optional[Token] = None,
               rtree_granted: bool = False) -> None:
        if not self.idle:
            self.stats["busy_cycles"] += 1
        into_tree = self._held_reduce
        self._held_reduce = None
        # The ALU's stages move unless a plain result is stuck in `out`.
        finished = self._advance(granted)
        for t in finished:
            if t.kind == "reduce":
                if into_tree is not None:
                    raise AssertionError("alu: two reductions entered the tree in one cycle")
                into_tree = t
            else:
                if self.out is not None:
                    raise AssertionError("alu: two results left in one cycle")
                self.out = t
        tree_ready = self.rtree.ready(rtree_granted)
        if into_tree is not None and not tree_ready:
            self._held_reduce = into_tree          # wait at the tree's input
            into_tree = None
        self.rtree.commit(rtree_granted, into_tree)
        self._accept(token)
        self.cycle += 1
