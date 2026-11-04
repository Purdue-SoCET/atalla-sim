"""`atalla-sim doctor`: what this installation can do, and how to fix what it can't.

Each check says ok, missing or out of date, and for anything not ok the
command that fixes it. Only Python and numpy are required to simulate; the
rest add features (C programs, golden checks, speed, DDR4 timing, plots).
Exit status 1 when something required is missing.
"""

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[2]

OK, WARN, MISSING = "ok", "warn", "missing"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""
    required: bool = False


def _git(*args) -> Optional[str]:
    if shutil.which("git") is None:
        return None
    p = subprocess.run(["git", "-C", str(ROOT), *args], stdout=subprocess.PIPE,
                       stderr=subprocess.DEVNULL, universal_newlines=True)
    return p.stdout if p.returncode == 0 else None


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def check_python() -> Check:
    v = "%d.%d.%d" % sys.version_info[:3]
    return Check("python", OK, "%s (%s)" % (v, sys.executable), required=True)


def check_module(name: str, why: str, fix: str, required: bool = False) -> Check:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:                          # noqa: BLE001 - report anything
        return Check(name, MISSING, "%s: %s" % (why, exc), fix, required)
    return Check(name, OK, ("%s %s" % (getattr(mod, "__version__", ""), why)).strip(),
                 required=required)


#: What each submodule adds, and what a run loses without it.
SUBMODULES = {
    "third_party/aihw-ppci-compiler": ("compiler", "C programs (atalla-sim run prog.c)"),
    "third_party/atalla-functional-sim": ("functional-sim",
                                          "--golden checks and the assembly kernels"),
    "third_party/ramulator2": ("ramulator2-src", "DDR4 timing (--dram ramulator)"),
    "third_party/atalla-graph": ("atalla-graph-src", "the MLIR model frontend"),
}


def check_submodules() -> List[Check]:
    out = []
    status = _git("submodule", "status")
    states = {}
    if status is not None:
        for line in status.splitlines():
            flag, sha, path = line[0], line[1:41], line[42:].split(" ")[0]
            states[path] = (flag, sha)
    for path, (name, adds) in SUBMODULES.items():
        d = ROOT / path
        populated = d.is_dir() and any(d.iterdir())
        flag, sha = states.get(path, (None, ""))
        fix = "git submodule update --init %s" % path
        if not populated:
            if flag is None and status is not None:
                continue                              # not a submodule of this checkout
            out.append(Check(name, MISSING, "not checked out: no %s" % adds, fix))
        elif flag == "+":
            out.append(Check(name, WARN, "checked out at a different commit than the one "
                             "pinned (%s); results may differ from the documented ones"
                             % sha[:7], fix))
        elif flag == "U":
            out.append(Check(name, WARN, "merge conflict in the submodule", fix))
        elif flag is None:
            out.append(Check(name, OK, "%s (present, not a registered submodule)" % path))
        else:
            out.append(Check(name, OK, "%s %s" % (path, sha[:7])))
    return out


def check_compiler() -> Check:
    compiler = ROOT / "third_party" / "aihw-ppci-compiler"
    if not (compiler / "ppci").is_dir():
        return Check("ppci", MISSING, "the compiler isn't checked out",
                     "git submodule update --init third_party/aihw-ppci-compiler")
    p = subprocess.run([sys.executable, "-c", "import ppci; print(ppci.__version__)"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                       env=dict(os.environ, PYTHONPATH=str(compiler)))
    if p.returncode:
        last = (p.stderr.strip().splitlines() or ["?"])[-1]
        return Check("ppci", MISSING, "the compiler doesn't import: %s" % last,
                     "pip install -r %s/requirements.txt (if any), then rerun" % _rel(compiler))
    return Check("ppci", OK, "the C compiler imports (%s)" % p.stdout.strip())


def check_golden() -> Check:
    from scheduler import golden
    if golden.HAVE_GOLDEN:
        return Check("golden", OK, "the functional sim imports (--golden works)")
    return Check("golden", MISSING, "functional sim unavailable: %s" % golden.load_error,
                 "git submodule update --init third_party/atalla-functional-sim")


def check_native() -> Check:
    import native
    if native.HAVE_NATIVE:
        return Check("native", OK, "SIMD kernels loaded (%s)" % native.NATIVE_ISA)
    return Check("native", WARN, "not built: runs use the numpy fallback (same results, "
                 "about 5x slower): %s" % native.load_error, "make native")


def check_ramulator() -> Check:
    from memory import ramulator
    if ramulator.ramulator_available():
        return Check("ramulator", OK, "Ramulator 2 loaded (--dram ramulator works)")
    return Check("ramulator", WARN, "not built: DRAM is fixed-latency only (%s)"
                 % ramulator._load_error,
                 "make ramulator   (needs a C++20 g++ and CMake)")


def check_atalla_graph() -> Optional[Check]:
    src = ROOT / "third_party" / "atalla-graph"
    if not src.is_dir():
        return None
    for d in ("install/bin", "build/bin", "build-atalla-sim/bin"):
        if (src / d / "atalla-graph").is_file():
            return Check("atalla-graph", OK, "%s" % _rel(src / d / "atalla-graph"))
    on_path = shutil.which("atalla-graph")
    if on_path:
        return Check("atalla-graph", OK, on_path)
    return Check("atalla-graph", WARN, "the MLIR frontend isn't built",
                 "make atalla-graph   (downloads LLVM/MLIR: several GB)")


def run_checks() -> List[Check]:
    for p in (str(ROOT / "src"),):
        if p not in sys.path:
            sys.path.insert(0, p)
    checks = [check_python(),
              check_module("numpy", "", "pip install numpy", required=True)]
    checks += check_submodules()
    checks += [check_compiler(), check_golden(), check_native(), check_ramulator()]
    ag = check_atalla_graph()
    if ag is not None:
        checks.append(ag)
    checks += [check_module("pytest", "for the test suite", "pip install pytest"),
               check_module("matplotlib", "for the plotting tools", "pip install matplotlib")]
    for c in checks:
        if c.name in ("pytest", "matplotlib") and c.status == MISSING:
            c.status = WARN
    return checks


def render(checks: List[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines = ["atalla-sim doctor (%s)" % ROOT, ""]
    for c in checks:
        lines.append("  %-7s  %-*s  %s" % (c.status.upper() if c.status != OK else OK,
                                          width, c.name, c.detail))
        if c.fix and c.status != OK:
            lines.append("  %-7s  %-*s  fix: %s" % ("", width, "", c.fix))
    broken = [c for c in checks if c.required and c.status != OK]
    rest = [c for c in checks if not c.required and c.status != OK]
    lines.append("")
    if broken:
        lines.append("cannot simulate: %s missing." % ", ".join(c.name for c in broken))
    else:
        lines.append("ready to simulate%s." % (
            "; %d optional part%s missing or out of date (above)" % (len(rest), "s" * (len(rest) != 1))
            if rest else ", with every optional part"))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="atalla-sim doctor", description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    checks = run_checks()
    print(render(checks))
    return 1 if any(c.required and c.status != OK for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
