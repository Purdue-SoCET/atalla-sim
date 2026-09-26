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
