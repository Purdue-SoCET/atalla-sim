"""What this atalla-sim is: `atalla-sim --version`.

The simulator's version and commit, a fingerprint of the ISA it decodes
(the opcode table: two builds with the same fingerprint decode the same
programs), the RTL each part of the model follows, and the commits of the
compiler, functional sim and other submodules. Put it in bug reports and
next to results.
"""

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[2]

#: The RTL (atalla repository branches and files) each part of the model
#: follows. Parts not listed are atalla-sim's own models, which are the
#: reference for them.
RTL_REFERENCES = [
    ("scheduler (fetch, decode, issue, scalar execute, dependencies)",
     "atalla scheduler_integration_SP26_joshklug"),
    ("AXI bus and Ramulator wrapper", "atalla memory_FA26"),
    ("DDR controller timings (ramulator_config ddr_cntrl)", "atalla ddr_cntrl_final, dram_pkg.vh"),
    ("transpose unit timing", "transpose_unit.sv, matched cycle for cycle against a Questa trace"),
    ("vector lanes, VLSUs, GSAU, scratchpads, DMA, MEISSA",
     "atalla-sim's own models (the reference)"),
]


def _git(*args, strip: bool = True) -> Optional[str]:
    if shutil.which("git") is None:
        return None
    p = subprocess.run(["git", "-C", str(ROOT), *args], stdout=subprocess.PIPE,
                       stderr=subprocess.DEVNULL, universal_newlines=True)
    if p.returncode != 0:
        return None
    return p.stdout.strip() if strip else p.stdout


def package_version() -> str:
    try:
        from importlib.metadata import version
        return version("atalla-sim")
    except Exception:                                   # not installed: read pyproject
        for line in (ROOT / "pyproject.toml").read_text().splitlines():
            if line.startswith("version"):
                return line.split("=", 1)[1].strip().strip('"')
    return "?"


def isa_fingerprint() -> str:
    from scheduler.isa import OPCODES
    text = "\n".join("%d %s %s" % (op, name, fmt) for op, (name, fmt) in sorted(OPCODES.items()))
    return "%d opcodes, table %s" % (len(OPCODES), hashlib.sha1(text.encode()).hexdigest()[:10])


def source_state() -> str:
    commit = _git("rev-parse", "--short", "HEAD")
    if commit is None:
        return "not a git checkout"
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or "?"
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return "git %s on %s%s" % (commit, branch, ", with uncommitted changes" if dirty else "")


def submodules() -> Dict[str, str]:
    out = {}
    status = _git("submodule", "status", strip=False) or ""
    for line in status.splitlines():
        flag, sha, rest = line[0], line[1:41], line[42:]
        path, _, desc = rest.partition(" ")
        note = {"-": " (not checked out)", "+": " (not the pinned commit)"}.get(flag, "")
        out[path] = "%s %s%s" % (sha[:7], desc.strip("()") if desc else "", note)
    return out


def text() -> str:
    lines = ["atalla-sim %s (%s)" % (package_version(), source_state()),
             "ISA: %s (src/scheduler/isa.py)" % isa_fingerprint(),
             "models follow:"]
    lines += ["  %-62s %s" % row for row in RTL_REFERENCES]
    subs = submodules()
    if subs:
        lines.append("submodules:")
        lines += ["  %-36s %s" % kv for kv in sorted(subs.items())]
    return "\n".join(lines)
