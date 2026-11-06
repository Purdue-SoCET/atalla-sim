"""The `atalla-sim` command: one entry point for the simulator's tools.

    atalla-sim run KERNEL [options]   run a program on the whole machine (tools/run_kernel.py)
    atalla-sim bench                  cycles and speed on the reference kernels vs tests/perf_baseline.json
    atalla-sim gallery OUT            a static site of the reference kernels' traces, ready to host
    atalla-sim view TRACE.json        the activity viewer (also --join, --empty, --serve)
    atalla-sim sweep KERNEL --param KEY=V1,V2   one program over a grid of machine parameters
    atalla-sim new-component NAME     a new clocked component and its test, from the template
    atalla-sim compare-rtl A.trace B.trace   where an RTL trace and atalla-sim's part ways
    atalla-sim model PLAN.json        run a model's operation plan op by op (a stub ahead of atalla-graph)
    atalla-sim models verify basic    build a PyTorch model into one ELF and check it against PyTorch
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


def _bench(argv):
    return _tool("bench").main(argv)


def _gallery(argv):
    return _tool("build_gallery").main(argv)


def _view(argv):
    return _tool("activity_view").main(argv)


def _sweep(argv):
    return _tool("sweep").main(argv)


def _new_component(argv):
    from atalla import new_component
    return new_component.main(argv)


def _compare_rtl(argv):
    import argparse
    from atalla import trace
    ap = argparse.ArgumentParser(prog="atalla-sim compare-rtl",
                                 description="Compare two execution traces (an RTL simulation's "
                                             "and `atalla-sim run --trace`'s): where they part "
                                             "ways, and the events around it. The format is in "
                                             "src/atalla/trace.py.")
    ap.add_argument("a", help="a trace, e.g. the RTL's")
    ap.add_argument("b", help="another, e.g. atalla-sim's")
    ap.add_argument("--limit", type=int, default=10, help="differences listed (default 10)")
    ap.add_argument("--window", type=int, default=3, help="cycles of context (default 3)")
    ap.add_argument("--no-align", action="store_true",
                    help="compare cycles as written, instead of lining up the first issue")
    args = ap.parse_args(argv)
    try:
        text, status = trace.report(args.a, args.b, args.limit, args.window,
                                    align=not args.no_align)
    except (OSError, ValueError) as exc:
        sys.exit(str(exc))
    print(text)
    return status


def _model(argv):
    import argparse
    import json
    root = _repo_root()
    for p in (str(root / "src"),):
        if p not in sys.path:
            sys.path.insert(0, p)
    import numpy as np
    from atalla import config, model_runner
    ap = argparse.ArgumentParser(prog="atalla-sim model",
                                 description="Run a model's operation plan (atalla-graph's "
                                             "--plan-json) op by op on atalla-sim. A stub "
                                             "ahead of atalla-graph: see "
                                             "src/atalla/model_runner.py for what it covers.")
    ap.add_argument("plan", help="the plan JSON (schema atalla-graph-plan-v1)")
    ap.add_argument("--inputs", metavar="NPZ",
                    help="arg0, arg1, ... in the plan's input order (and const<id> for "
                         "constants the plan has no data for); random if not given")
    ap.add_argument("--seed", type=int, default=0, help="for random inputs")
    ap.add_argument("--reference", metavar="NPY", help="the framework's output, to compare with")
    ap.add_argument("--config", metavar="NAME|PATH", help="the machine (default atalla-0x01)")
    ap.add_argument("--report-json", metavar="PATH", help="write the report as JSON")
    ap.add_argument("--activity", metavar="PATH",
                    help="record which blocks work when in every accelerator operation, as "
                         "one workload: a .html page, or a .json for atalla-sim view")
    args = ap.parse_args(argv)
    try:
        plan = model_runner.Plan.load(args.plan)
        cfg = config.load(args.config)
        if args.inputs:
            inputs = dict(np.load(args.inputs))
        else:
            rng = np.random.default_rng(args.seed)
            inputs = {"arg%d" % k: rng.normal(0, 0.5, size=shape).astype(np.float32)
                      for k, (_, shape, _) in enumerate(plan.inputs)}
        ref = np.load(args.reference) if args.reference else None
        kw = config.platform_kwargs(cfg)
        if args.activity:
            kw["activity"] = True
        report = model_runner.run_model(args.plan, inputs, kw, ref, config.summary(cfg))
    except (OSError, ValueError, KeyError, config.ConfigError) as exc:
        sys.exit("atalla-sim model: %s" % exc)
    print(model_runner.summary(report))
    work = report.pop("activity", None)
    if args.activity:
        out = __import__("pathlib").Path(args.activity)
        if out.suffix == ".html":
            out.write_text(_tool("activity_view").render(work))
        else:
            out.write_text(json.dumps(work, separators=(",", ":")))
        print("activity of %d operations written to %s" % (len(work["layers"]), out))
    if args.report_json:
        Path = __import__("pathlib").Path
        Path(args.report_json).write_text(json.dumps(report, indent=1, default=float))
        print("report written to %s" % args.report_json)


def _models(argv):
    """Build or verify a PyTorch model as one linked ELF (models/, which needs
    torch). Runs models/build_model.py or verify_model.py with this Python."""
    import argparse
    import subprocess
    root = _repo_root()
    ap = argparse.ArgumentParser(
        prog="atalla-sim models",
        description="Trace a PyTorch model, schedule it onto the graph kernels in "
                    "models/kernels, compile and link it into one ELF with its DRAM image "
                    "(build), and run that ELF on atalla-sim checking every result against "
                    "PyTorch (verify). Needs torch. docs/model-elf.md has the details.")
    ap.add_argument("action", choices=("build", "verify"))
    ap.add_argument("model", choices=("basic", "alexnet_small"))
    ap.add_argument("--out-dir", metavar="DIR",
                    help="where the ELF, DRAM image and manifests go (default models/out/MODEL)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--scale", type=float, help="the model's width scale (default 0.01)")
    ap.add_argument("--limit", type=int, help="verify: cycles before giving up (default 10M)")
    ap.add_argument("--atol", type=float, help="verify: absolute tolerance (default 0.01)")
    ap.add_argument("--rtol", type=float, help="verify: relative tolerance (default 0.03)")
    args = ap.parse_args(argv)
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("atalla-sim models needs PyTorch: pip install torch")
    script = root / "models" / ("%s_model.py" % args.action)
    cmd = [sys.executable, str(script), "--model", args.model,
           "--out-dir", args.out_dir or str(root / "models" / "out" / args.model)]
    for opt in ("seed", "scale") + (("limit", "atol", "rtol") if args.action == "verify" else ()):
        if getattr(args, opt) is not None:
            cmd += ["--" + opt, str(getattr(args, opt))]
    return subprocess.call(cmd, cwd=str(root / "models"))


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
    "bench": (_bench, "cycles and simulator speed on the reference kernels, against the baseline"),
    "gallery": (_gallery, "a static site of traces for the reference kernels, ready to host"),
    "view": (_view, "the activity viewer: draw a trace, join a workload, or serve a folder"),
    "sweep": (_sweep, "run one program over a grid of machine parameters (CSV, plot)"),
    "new-component": (_new_component, "write a new clocked component and its test from the template"),
    "compare-rtl": (_compare_rtl, "compare two execution traces (RTL vs atalla-sim run --trace)"),
    "model": (_model, "run a model's operation plan (atalla-graph --plan-json) op by op"),
    "models": (_models, "build a PyTorch model into one ELF, or run it and check it against PyTorch"),
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
