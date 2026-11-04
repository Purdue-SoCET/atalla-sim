"""The `atalla-sim` command: one entry point for the simulator's tools.

    atalla-sim run KERNEL [options]   run a program on the whole machine (tools/run_kernel.py)
    atalla-sim doctor                 check the installation: what works, what's missing, how to fix it
    atalla-sim --version              the version, ISA fingerprint, RTL followed, submodule commits

`pip install -e .` from the repository puts `atalla-sim` on PATH; without
installing, `python -m atalla.cli` from `src/` (or `PYTHONPATH=src`) does the
same. The tools under tools/ keep working on their own; each subcommand is
the same code.

Only the standard library is imported here, so an unsupported Python gets a
clear message instead of an import error deep inside the simulator.
"""

import sys

MIN_PYTHON = (3, 9)


def require_python() -> None:
    if sys.version_info < MIN_PYTHON:
        sys.exit("atalla-sim needs Python %d.%d or newer; %s is %s. Use a newer interpreter, "
                 "e.g. `python3.11 -m venv .venv && . .venv/bin/activate && pip install -e .`"
                 % (MIN_PYTHON + (sys.executable, sys.version.split()[0])))


def _repo_root():
    from pathlib import Path
    return Path(__file__).resolve().parents[2]


def _tool(name: str):
    """A module from tools/, which needs a source checkout (an editable
    install), not just the installed package."""
    root = _repo_root()
    tools = root / "tools"
    if not (tools / (name + ".py")).is_file():
        sys.exit("`atalla-sim` needs the source checkout's tools/ (install with "
                 "`pip install -e .` from the repository); not found at %s" % tools)
    for p in (str(root / "src"), str(tools)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import importlib
    return importlib.import_module(name)


def _run(argv):
    return _tool("run_kernel").main(argv)


def _version(argv):
    from atalla import version
    print(version.text())


def _doctor(argv):
    from atalla import doctor
    return doctor.main(argv)


#: name -> (handler, one-line help). Handlers take the remaining arguments
#: and return an exit status (None is 0).
COMMANDS = {
    "run": (_run, "run a program on the whole machine (a kernel name, .c, .in or .elf)"),
    "doctor": (_doctor, "check the installation and say how to fix what's missing"),
    "version": (_version, "the version, ISA fingerprint, RTL followed and submodule commits"),
}


def usage() -> str:
    width = max(len(n) for n in COMMANDS)
    lines = ["usage: atalla-sim COMMAND [ARGS...]", "", "commands:"]
    lines += ["  %-*s  %s" % (width, n, h) for n, (_, h) in COMMANDS.items()]
    lines += ["", "`atalla-sim COMMAND -h` describes a command's options."]
    return "\n".join(lines)


def main(argv=None) -> None:
    require_python()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        sys.exit(0 if argv else 2)
    name, rest = argv[0], argv[1:]
    if name == "--version":
        name = "version"
    if name not in COMMANDS:
        sys.exit("atalla-sim: unknown command %r\n\n%s" % (name, usage()))
    status = COMMANDS[name][0](rest)
    sys.exit(status or 0)


if __name__ == "__main__":
    main()
