"""Access to the functional simulator, used as a golden reference.

The functional sim is a git submodule. It is a *reference*, never a runtime
dependency: the scheduler model decodes and executes on its own, and this
module exists only so tests can check that model against it. When the submodule
is absent -- a clone without the key, or CI -- HAVE_GOLDEN is False and those
tests skip, the same way src/native/kernels.py degrades when the .so has not
been built.

    ATALLA_NO_GOLDEN=1   force the fallback, to prove the skip path works
"""

import os
import sys
from pathlib import Path

#: Where the submodule is mounted, relative to the repo root.
SUBMODULE_PATH = Path(__file__).resolve().parents[2] / "third_party" / "atalla-functional-sim"

HAVE_GOLDEN = False
load_error = None
_disabled = os.environ.get("ATALLA_NO_GOLDEN", "") not in ("", "0")

if _disabled:
    load_error = "disabled by ATALLA_NO_GOLDEN"
elif not SUBMODULE_PATH.is_dir():
    load_error = "submodule not present at %s" % SUBMODULE_PATH
else:
    # The functional sim imports as `src.misc.opcode_table`, so its own root
    # has to be on the path. Prepend rather than append: it owns the name
    # `src`, which would otherwise collide with this repo's own src/.
    root = str(SUBMODULE_PATH)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from src.misc.opcode_table import OPCODES as GOLDEN_OPCODES  # noqa: F401
        from src.components.decode import (  # noqa: F401
            decode_instruction as golden_decode_instruction,
            decode_packet as golden_decode_packet,
        )
        HAVE_GOLDEN = True
    except Exception as exc:                      # pragma: no cover - env dependent
        load_error = "%s: %s" % (type(exc).__name__, exc)


def require(pytest_module) -> None:
    """Skip the calling test when the golden reference is unavailable."""
    if not HAVE_GOLDEN:
        pytest_module.skip("functional sim unavailable (%s)" % load_error)


#: Lanes feeding the RTL's reduction tree (vector_pkg.vh NUM_LANES).
RTL_LANES = 16

_IDENTITY = {"sum": 0.0, "min": float("inf"), "max": float("-inf")}


def _bf16(x):
    """The functional sim's bf16_round, on one float32."""
    import numpy as np
    u = np.array([x], dtype=np.float32).view(np.uint32)
    u = (u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return u.view(np.float32)[0]


def hardware_reduce(values, mask: int, op: str, lanes: int = RTL_LANES):
    """A reduction in the RTL's order rather than the functional sim's
    sequential fp32 one: each lane folds its contiguous slice, a masked-off
    element contributing the op's identity (alu_FU.sv); then a pairwise tree
    across the lanes (reduction_tree.sv). Every step is a BF16 op: done in
    fp32, rounded to BF16."""
    import numpy as np
    def pair(a, b):
        if op == "sum":
            with np.errstate(over="ignore", invalid="ignore"):
                return _bf16(np.float32(a) + np.float32(b))
        return min(a, b) if op == "min" else max(a, b)
    q = [_bf16(v) for v in values]
    w = len(q) // lanes
    level = []
    for lane in range(lanes):
        acc = np.float32(_IDENTITY[op])
        for i in range(lane * w, (lane + 1) * w):
            if (int(mask) >> i) & 1:
                acc = pair(acc, q[i])
        level.append(acc)
    while len(level) > 1:
        level = [pair(level[i], level[i + 1]) for i in range(0, len(level), 2)]
    return np.array(level[0], dtype=np.float32)


def run_golden_state(instr, data=None, workdir=None, packet_length: int = 4,
                     reductions: str = "functional"):
    """Run a program image on the functional sim; return its state at halt
    as plain values: {"sregs": {r: int}, "mregs": {r: int},
    "vregs": {r: [float]}, "mem": {addr: int}}.

    reductions="hardware" swaps the functional sim's rsum/rmin/rmax for
    hardware_reduce, the RTL's order. Only rsum can differ: a sum's value
    depends on its order, a minimum's does not."""
    if not HAVE_GOLDEN:
        raise RuntimeError("functional sim unavailable (%s)" % load_error)
    import tempfile
    from src.functional_sim import run
    from src.misc.memory import Memory
    from src.components.scalar_register_file import ScalarRegisterFile, mask_register_file
    from src.components.vector_register_file import VectorRegisterFile
    from src.components.execute import ExecuteUnit
    from src.components.scpad import Scratchpad

    mem = Memory()
    mem.instr_mem = dict(instr)
    mem.data_mem = dict(data or {})
    sregs, mregs, vregs = ScalarRegisterFile(), mask_register_file(), VectorRegisterFile()
    tmp = tempfile.TemporaryDirectory() if workdir is None else None
    out = Path(workdir if workdir is not None else tmp.name)
    names = ["mem", "sregs", "vregs", "mregs", "scpad0", "scpad1", "perf"]
    files = [str(out / ("%s.out" % n)) for n in names]
    from src.components.vector_lanes import VectorLanes
    saved = {}
    if reductions == "hardware":
        for name, op in (("reduce_sum", "sum"), ("reduce_min", "min"), ("reduce_max", "max")):
            saved[name] = VectorLanes.__dict__[name]
            setattr(VectorLanes, name,
                    lambda self, a, mask, _op=op: hardware_reduce(self._ensure_vec(a), mask, _op))
    elif reductions != "functional":
        raise ValueError("reductions must be 'functional' or 'hardware'")
    try:
        run(mem, sregs, mregs, vregs,
            Scratchpad(slots_per_bank=32), Scratchpad(slots_per_bank=32), ExecuteUnit(),
            0, packet_length, *files)
    finally:
        for name, fn in saved.items():
            setattr(VectorLanes, name, fn)
        if tmp is not None:
            tmp.cleanup()
    return {"sregs": {r: int(v) for r, v in sregs.regs.items()},
            "mregs": {r: int(v) for r, v in mregs.regs.items()},
            "vregs": {r: [float(x) for x in vregs.read(r)] for r in range(vregs.num_regs)},
            "mem": dict(mem.data_mem)}


def run_golden(instr, data=None, workdir=None, packet_length: int = 4):
    """Scalar registers and data memory at halt; see run_golden_state."""
    st = run_golden_state(instr, data, workdir, packet_length)
    return st["sregs"], st["mem"]
