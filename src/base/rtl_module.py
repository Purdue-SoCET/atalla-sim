"""Base class for pipeline models transcribed from RTL.

The RTL is combinational logic feeding registers that all update on one clock
edge. This simulator ticks components one at a time in list order, which cannot
reproduce that: a module ticked first would see either this cycle's or last
cycle's value from a module ticked later, depending on registration order. For
a pipeline with backpressure that is not a detail -- registration order would
decide the answer.

So these modules are not Clocked. They expose three phases that their parent --
the single Clocked object the platform registers -- drives each cycle:

    eval_ready()   backward-flowing readiness/backpressure. Depends only on
                   register state and external ready inputs, never on this
                   cycle's valid/data, so one ordered pass suffices and no
                   combinational loop can form.
    eval_data()    forward-flowing valid/data, and every next-state value,
                   written to a `<reg>_n` shadow attribute.
    commit()       copy each `<reg>_n` into `<reg>`. Nothing else.

The discipline that makes this correct: eval_ready and eval_data must never
mutate register state, and commit must never read anything but the shadows.
Initiation intervals and backpressure delays then fall out of the structure
instead of being hand-tuned.

Subclasses declare their state and ports as class-level dicts rather than
writing them out in __init__:

    REGS  {name: reset}   creates self.<name> and self.<name>_n
    INS   {name: reset}   creates self.in_<name>
    OUTS  {name: reset}   creates self.out_<name>
    CLEAR_ON_COMMIT       input attribute names reset to their declared value
                          each commit, for signals that are level-sensitive in
                          the RTL and must not persist across cycles

A reset value that is a list (or any mutable) is copied per instance and per
reset, so two instances never alias each other's state.
"""

import copy


def _fresh(value):
    return copy.deepcopy(value) if isinstance(value, (list, dict, set)) else value


class RTLModule:
    REGS = {}
    INS = {}
    OUTS = {}
    CLEAR_ON_COMMIT = ()

    def __init__(self, name: str = ""):
        self.name = name
        for reg, reset in self.REGS.items():
            setattr(self, reg, _fresh(reset))
            setattr(self, reg + "_n", _fresh(reset))
        for sig, reset in self.INS.items():
            setattr(self, "in_" + sig, _fresh(reset))
        for sig, reset in self.OUTS.items():
            setattr(self, "out_" + sig, _fresh(reset))

    def eval_ready(self) -> None:
        """Settle backward (readiness) wires. Must not mutate register state."""

    def eval_data(self) -> None:
        """Settle forward (valid/data) wires and every `<reg>_n`. No mutation."""

    def commit(self) -> None:
        """Latch every register from its shadow, then reset level-sensitive inputs."""
        for reg in self.REGS:
            setattr(self, reg, getattr(self, reg + "_n"))
        for sig in self.CLEAR_ON_COMMIT:
            key = sig[3:] if sig.startswith("in_") else sig
            setattr(self, sig, _fresh(self.INS[key]))

    def __repr__(self) -> str:
        return "<%s %s>" % (self.__class__.__name__, self.name or id(self))
