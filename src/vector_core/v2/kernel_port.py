"""A kernel sequencer's port into an FU group of the M x N vector core.

The vloop sequencer (vector_core/vloop.py) runs a whole loop by itself and
keeps its values in private buffers, not in the banked register file. It
fires straight into the FU groups and takes the results back, bypassing the
slots, the crossbars and writeback:

    cycle c        fire(): a "kernel" token enters the group at the end of c.
                   It wins the group's input: a slot op aimed at the group in
                   c stalls (reason "kernel").
    c+L            the token reaches the group's output register
    c+L+1          take() removes it there -- it never goes to writeback --
                   and the sequencer can use it in the same cycle (chaining)

so a port's latency is L + 1. The value is computed when the token enters,
with the group's element ops and BF16 rounding (fu_group.elementwise), and
masking is zeroing: a masked-off element comes out +0.0. A compare's result
is its 1.0/0.0 elements; the sequencer packs those into a mask.

ready() is from registered state: the group can take a token this cycle, and
nothing else has been fired into it yet.
"""

from typing import Any, List, Optional, Sequence, Tuple

from vector_core.v2.fu_group import Token, elementwise, zeroed

KERNEL = "kernel"


class FUGroupPort:

    def __init__(self, vc, group: str):
        self.vc = vc
        self.name = group
        self.group = vc.groups[group]
        self.latency = self.group.latency + 1

    def ready(self) -> bool:
        if self.name in self.vc.kernel_tokens:
            return False
        g = self.group
        if g is self.vc.alu:
            return g.ready_for("plain", False, False)
        return g.ready(False)

    def fire(self, cycle: int, op: str, src0: Sequence[float],
             src1: Optional[Sequence[float]] = None, mask: Optional[int] = None,
             tag: Any = None) -> None:
        n = len(src0)
        b = list(src1) if src1 is not None else [0.0] * n
        value = zeroed(elementwise(op, src0, b, [True] * n))
        if mask is not None:
            value = [v if (mask >> i) & 1 else 0.0 for i, v in enumerate(value)]
        if not self.ready():
            raise AssertionError("%s: kernel token fired while the group can't take it"
                                 % self.name)
        self.vc.kernel_tokens[self.name] = Token(KERNEL, -1, value, meta={"tag": tag})

    def take(self, cycle: int) -> List[Tuple[Any, List[float]]]:
        g = self.group
        if g.out is not None and g.out.kind == KERNEL:
            t, g.out = g.out, None
            return [(t.meta["tag"], t.value)]
        return []

    @property
    def busy(self) -> bool:
        g = self.group
        return (self.name in self.vc.kernel_tokens
                or (g.out is not None and g.out.kind == KERNEL)
                or any(st[0].kind == KERNEL for st in g.stages))


def group_tokens(g) -> List[Token]:
    """Every token in an FU group: its stages and output register, and for
    the ALU group the reduction tree and the reduction waiting for it."""
    toks = [st[0] for st in g.stages] + ([g.out] if g.out is not None else [])
    held = getattr(g, "_held_reduce", None)
    if held is not None:
        toks.append(held)
    tree = getattr(g, "rtree", None)
    if tree is not None:
        toks += group_tokens(tree)
    return toks


def normal_ops_in_groups(vc) -> bool:
    """Whether any FU group holds a slot op's token (not a vloop's)."""
    return any(t.kind != KERNEL for g in vc.groups.values() for t in group_tokens(g))


def kernel_ports(vc) -> dict:
    """A port per FU group of `vc`, by group name."""
    return {name: FUGroupPort(vc, name) for name in vc.groups}
