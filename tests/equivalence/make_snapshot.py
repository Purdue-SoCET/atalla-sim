#!/usr/bin/env python3
"""Write snapshot.json: what every scenario in scenarios.py produces today.

Run this ONLY on code whose behaviour you trust (for example, before a
cleanup, or after a change you meant to make), then commit snapshot.json:

    python tests/equivalence/make_snapshot.py               # every scenario
    python tests/equivalence/make_snapshot.py kernel_gemm   # just these

Kernel scenarios first build their program and store it in programs/
(the test then runs those stored programs, never the compiler). Scenarios
that can't run here (a missing submodule) are left out, and the
equivalence test skips them.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenarios  # noqa: E402

SNAPSHOT = Path(__file__).resolve().parent / "snapshot.json"


def main() -> None:
    names = sys.argv[1:] or list(scenarios.SCENARIOS)
    snap = json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else {}
    for name in names:
        try:
            if name.startswith("kernel_"):
                scenarios.freeze_kernel(name[len("kernel_"):])
            snap[name] = scenarios.run(name)
            print("ok      ", name)
        except (Exception, SystemExit) as e:          # e.g. a submodule not checked out
            print("skipped ", name, "--", str(e).splitlines()[0][:100] if str(e) else type(e))
    SNAPSHOT.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
    print("wrote", SNAPSHOT)


if __name__ == "__main__":
    main()
