"""The simulator still does exactly what snapshot.json recorded.

Every scenario in scenarios.py is run again and its numbers compared with
the snapshot: cycle counts, statistics and fingerprints of registers, DRAM
and outputs. Any difference fails, with the keys that changed. If a change
of behaviour is intended, regenerate the snapshot (make_snapshot.py).
"""
import json
from pathlib import Path

import pytest

import scenarios

SNAPSHOT = json.loads((Path(__file__).resolve().parent / "snapshot.json").read_text())


def _diff(old, new, path=""):
    """The keys whose values differ, as 'a.b.c: old -> new'."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for k in sorted(set(old) | set(new)):
            out += _diff(old.get(k), new.get(k), "%s.%s" % (path, k) if path else k)
        return out
    return [] if old == new else ["%s: %r -> %r" % (path, old, new)]


@pytest.mark.parametrize("name", sorted(scenarios.SCENARIOS))
def test_behaviour_is_unchanged(name):
    if name not in SNAPSHOT:
        pytest.skip("not in snapshot.json (couldn't run when it was made)")
    got = scenarios.run(name)
    changed = _diff(SNAPSHOT[name], got)
    assert not changed, "\n".join(changed[:20])


#: One of each kind of run, repeated with wake scheduling off: the scheduler
#: kernel path, the harness path, and the tiled co-simulation.
LEGACY_CHECK = ["kernel_gemm", "kernel_softmax", "tpu_experiment_default",
                "transpose_bench_spad", "tiled_cosim_64"]


def test_legacy_scheduling_gives_the_same_results():
    """ATALLA_SCHED=legacy ticks every component every cycle (base/sched.py).
    It must give exactly what the default, sleeping idle components, gives."""
    import os
    import subprocess
    import sys
    code = ("import json, sys, scenarios; "
            "print(json.dumps({n: scenarios.run(n) for n in sys.argv[1:]}))")
    env = dict(os.environ, ATALLA_SCHED="legacy")
    here = Path(__file__).resolve().parent
    proc = subprocess.run([sys.executable, "-c", code, *LEGACY_CHECK], cwd=here, env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    for name in LEGACY_CHECK:
        changed = _diff(SNAPSHOT[name], got[name])
        assert not changed, name + ":\n" + "\n".join(changed[:20])
