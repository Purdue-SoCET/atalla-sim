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


def run_golden_state(instr, data=None, workdir=None, packet_length: int = 4):
    """Run a program image on the functional sim; return its state at halt
    as plain values: {"sregs": {r: int}, "mregs": {r: int},
    "vregs": {r: [float]}, "mem": {addr: int}}."""
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
    try:
        run(mem, sregs, mregs, vregs,
            Scratchpad(slots_per_bank=32), Scratchpad(slots_per_bank=32), ExecuteUnit(),
            0, packet_length, *files)
    finally:
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
