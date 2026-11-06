#!/usr/bin/env python3
"""Run a kernel on the scheduler-driven platform.

The instruction stream goes to the VLIW scheduler model (fetch, decode,
issue, scalar execute), which drives the vector core's units, the systolic
array, the four scratchpad pads with their DMA backends, and DRAM
(src/atalla/atalla_platform.py).

    tools/run_kernel.py gemm                 # build kernels/build_gemm.py, run it
    tools/run_kernel.py path/to/prog.in      # an already assembled program
    tools/run_kernel.py path/to/prog.elf     # a linked program from the compiler
    tools/run_kernel.py softmax --golden     # also run the functional sim and compare
    tools/run_kernel.py gemm -- --rows 16    # arguments after -- go to the build script
    tools/run_kernel.py gemm --report        # where the cycles went, unit by unit
    tools/run_kernel.py gemm --perf-json gemm.json --timeline   # the monitor's report

A plain run prints the cycle count and nothing else: the performance monitor
and the other recorders are attached only for the options that need them
(--report, --perf-json, --timeline, --activity), so they cost nothing
otherwise.

A kernel name is built with the functional sim's own build script,
third_party/atalla-functional-sim/kernels/build_<name>.py.

A C program (a .c path, or a file name in the compiler's
test_validation_kernels/ or test_validation/) goes straight from the compiler
to atalla-sim: atalla_cc packetizes it, src/scheduler/assembler.py assembles
it, and its inputs and reference result come from the compiler's validation
harness (tools/c_kernels.py). It runs with the C loader's stack registers set.

    tools/run_kernel.py add.c                  # run, check against numpy
    tools/run_kernel.py relu.c --golden        # also compare with the functional sim
    tools/run_kernel.py add.c --save add.in    # keep the assembled image
    tools/run_kernel.py relu.c --elf           # compile and link to an ELF, load that

An ELF (src/scheduler/elf.py) runs with its code and data where its layout
put them, called through `jal x1, main; halt.s` at address 0, with the C
loader's stack registers. A .c program with --elf also gets its inputs and
reference result from the compiler's harness.
    tools/run_kernel.py gemm --activity gemm.html   # which blocks work when
    tools/run_kernel.py gemm --dram ramulator       # DRAM timed by Ramulator 2 (make ramulator)
    tools/run_kernel.py gemm --dram ramulator --dram-config ddr_cntrl --dram-param nREFI=6240
    tools/run_kernel.py gemm --dram ramulator --no-axi   # masters straight to the DRAM model
    tools/run_kernel.py gemm --axi --dram-latency 0     # the bus alone, an ideal DRAM behind it
"""

import sys
if sys.version_info < (3, 9):
    sys.exit("atalla-sim needs Python 3.9 or newer; %s is %s"
             % (sys.executable, sys.version.split()[0]))

import argparse
import os
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atalla import config, lockstep, tensors                      # noqa: E402
from atalla.hang import HANG_AFTER                                # noqa: E402
from atalla.kernel import entry_sregs                             # noqa: E402
from scheduler import golden                                     # noqa: E402
from scheduler.core import load_program_text                      # noqa: E402
from scheduler.elf import ElfError, load_elf                       # noqa: E402
from atalla.atalla_platform import build_atalla_platform, c_entry_sregs  # noqa: E402
from scheduler.assembler import to_in_text                        # noqa: E402
from memory.ramulator import DRAM_CONFIGS                                # noqa: E402
import c_kernels                                                  # noqa: E402

SIM = golden.SUBMODULE_PATH


def _sim_env():
    """The environment the functional sim's build scripts need."""
    paths = [str(SIM), str(SIM / "kernels"), str(SIM / "experiments" / "gemm")]
    return dict(os.environ, PYTHONPATH=os.pathsep.join(paths))


def find_c(name: str) -> Path:
    """A .c path as given, or by file name among the compiler's programs."""
    path = Path(name)
    if path.is_file():
        return path.resolve()
    for folder in (c_kernels.KERNELS, c_kernels.UNIT_TESTS):
        if (folder / path.name).is_file():
            return folder / path.name
    sys.exit("no C program %s (looked in %s and %s)"
             % (name, c_kernels.KERNELS, c_kernels.UNIT_TESTS))


def build_kernel(name: str, extra, out_dir: Path) -> Path:
    script = SIM / "kernels" / ("build_%s.py" % name)
    if not script.is_file():
        sys.exit("no build script %s" % script)
    out = out_dir / ("%s.in" % name)
    p = subprocess.run([sys.executable, str(script), "-o", str(out), *extra],
                       cwd=str(SIM), env=_sim_env(),
                       capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit("build failed:\n" + (p.stderr or p.stdout)[-3000:])
    return out


def compare(plat, instr, data, reductions: str = "lanes", init_sregs=None, writes=None) -> int:
    """Run the functional sim on the same program and print the first
    register write where the two part ways (atalla/lockstep.py, when `writes`
    recorded the platform's) and where the final state differs
    (golden.differences). Returns how many final-state differences.
    reductions="lanes": the functional sim reduces in the lane model's
    order; "functional": its own sequential fp32 sum."""
    trace = [] if writes is not None else None
    g = golden.run_golden_state(instr, data, reductions=reductions, init_sregs=init_sregs,
                                trace=trace)
    if writes is not None:
        div = lockstep.first_divergence(writes, trace)
        print("  " + lockstep.describe(div, instr, plat.describe_pc).replace("\n", "\n  "))
    diffs = golden.differences(plat, g)
    for line in diffs[:20]:
        print("  " + line)
    print("  %d differences" % len(diffs) if diffs else "  identical: registers and DRAM")
    return len(diffs)


class Loaded:
    """A program ready to run: what load_program returns."""

    def __init__(self, instr, data, init_sregs=None, describe_pc=None, c_path=None,
                 program=None):
        self.instr, self.data, self.init_sregs = instr, data, init_sregs
        self.describe_pc, self.c_path, self.program = describe_pc, c_path, program


def load_program(kernel: str, extra=(), elf: bool = False) -> Loaded:
    """A kernel name (built by the functional sim's build script, with
    `extra` arguments), a .c program (compiled, with the compiler harness's
    inputs), a .in image or a .elf. Exits with a message on errors."""
    extra = list(extra)
    program = None
    init_sregs = None
    c_path = None
    describe_pc = None
    if elf and not kernel.endswith(".c"):
        sys.exit("--elf is for a .c program; a .elf file is loaded as it is")
    if kernel.endswith(".c"):
        if extra:
            sys.exit("a C program takes no build arguments (its inputs come from the "
                     "compiler's harness)")
        c_path = find_c(kernel)
        try:
            program, data, init_sregs = c_kernels.build(c_path, elf=elf)
        except (c_kernels.CompileError, ValueError) as exc:
            sys.exit(str(exc))
        instr = program.instr
        describe_pc = getattr(program, "describe", None)
    elif kernel.endswith(".elf"):
        if extra:
            sys.exit("an ELF takes no build arguments")
        try:
            program = load_elf(kernel)
        except (OSError, ElfError) as exc:
            sys.exit("%s: %s" % (kernel, exc))
        instr, data = program.instr, program.data
        init_sregs = c_entry_sregs(data)
        describe_pc = program.describe
    else:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(kernel)
            if path.suffix != ".in":
                path = build_kernel(kernel, extra, Path(tmp))
            instr, data = load_program_text(path.read_text())
    return Loaded(instr, data, init_sregs, describe_pc, c_path, program)


def machine_config(args) -> dict:
    """The --config file with the command line's explicit flags applied."""
    cfg = config.load(args.config)
    core, mem = {}, {}
    if args.lanes is not None:
        core["lane_count"] = args.lanes
    if args.array is not None:
        core["systolic_array"] = args.array
    if args.warm_icache:
        core["warm_icache"] = True
    if args.dram_latency is not None:
        mem["dram_latency"] = args.dram_latency
    if args.burst_bytes is not None:
        mem["burst_bytes"] = args.burst_bytes
    if args.dram is not None:
        mem["dram"] = args.dram
    if args.axi is not None:
        mem["axi"] = args.axi
    if args.dram_config is not None:
        mem["ramulator_config"] = args.dram_config
    if args.dram_param:
        params = {}
        for item in args.dram_param:
            name, eq, value = item.partition("=")
            if not eq or not value:
                raise ValueError("--dram-param takes NAME=VALUE, not %r" % item)
            params[name] = int(value) if value.lstrip("-").isdigit() else value
        mem["ramulator"] = {**cfg["memory"]["ramulator"], **params}
    cfg = config.override(cfg, {"core": core, "memory": mem})
    if (args.dram_param or args.dram_config) and cfg["memory"]["dram"] != "ramulator":
        raise ValueError("--dram-config and --dram-param need --dram ramulator "
                         "(or a config with dram = \"ramulator\")")
    return cfg


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog="atalla-sim run", description=__doc__.split("\n\n")[0])
    ap.add_argument("kernel", help="a kernel name (kernels/build_<name>.py), a .c program, "
                                   "a .in file or a .elf file")
    ap.add_argument("--elf", action="store_true",
                    help="for a .c program: compile and link it to an ELF and run that, "
                         "instead of assembling the compiler's assembly")
    ap.add_argument("--golden", action="store_true",
                    help="also run the functional sim on the same image: report the first "
                         "register write where the two differ, and every difference in the "
                         "final registers and DRAM")
    ap.add_argument("--limit", type=int, default=5_000_000, help="cycle limit")
    ap.add_argument("--hang-after", type=int, default=HANG_AFTER, metavar="N",
                    help="stop, and report what every unit waits on, once nothing has moved "
                         "for N cycles (default %(default)d; 0 never)")
    ap.add_argument("--config", metavar="NAME|PATH",
                    help="the machine: a configs/NAME.toml or a .toml path (default "
                         "atalla-0x01); the flags below override it")
    ap.add_argument("--dram-latency", type=int, default=None,
                    help="cycles from a DRAM request launching to its data returning")
    ap.add_argument("--dram", choices=["fixed", "ramulator"], default=None,
                    help="DRAM timing: a fixed --dram-latency, or Ramulator 2's DDR4 model "
                         "(build it with `make ramulator`)")
    ap.add_argument("--axi", action=argparse.BooleanOptionalAction, default=None,
                    help="the memory_FA26 AXI bus between the DRAM masters and DRAM "
                         "(default: with --dram ramulator; --no-axi drives the DRAM "
                         "directly, --axi with --dram fixed puts a fixed-latency DRAM "
                         "behind the bus)")
    ap.add_argument("--dram-config", choices=list(DRAM_CONFIGS), default=None,
                    help="with --dram ramulator: rtl (memory_FA26's Ramulator wrapper, "
                         "DDR4-3200W), ddr4-1600 or ddr_cntrl (the DDR controller's "
                         "dram_pkg.vh timings, branch ddr_cntrl_final); src/memory/ramulator.py "
                         "DRAM_CONFIGS")
    ap.add_argument("--dram-param", metavar="NAME=VALUE", action="append", default=[],
                    help="with --dram ramulator: override one value of the config or of "
                         "Ramulator's DDR4 model, e.g. read_buffer=64, preset=DDR4_2400R, "
                         "nCL=12; repeatable")
    ap.add_argument("--lanes", type=int, default=None, help="vector lanes (must divide 32)")
    ap.add_argument("--array", choices=["meissa", "tpu"], default=None,
                    help="the systolic array: MEISSA (the RTL's) or the TPU-style array")
    ap.add_argument("--burst-bytes", type=int, default=None,
                    help="bytes per DRAM request, for every DRAM master "
                         "(default: the RTL's 64-bit interface)")
    ap.add_argument("--warm-icache", action="store_true",
                    help="start with every packet's line in the icache")
    ap.add_argument("--functional-reductions", action="store_true",
                    help="with --golden: compare against the functional sim's own "
                         "fp32 sums instead of the lane model's reduction order")
    ap.add_argument("--save", metavar="PATH",
                    help="write the program image (.in) that ran, e.g. for the functional sim")
    ap.add_argument("--activity", metavar="PATH",
                    help="record which blocks are busy each cycle: a .html path draws the "
                         "activity view (tools/activity_view.py), anything else saves the trace")
    ap.add_argument("--check", nargs="?", const="warn", choices=["warn", "error"],
                    help="report accesses nothing in the hardware orders (a vreg.ld before "
                         "its scpad.ld lands, a scpad.st before its vreg.st, misaligned "
                         "scalar accesses, packets the RTL can't run); =error also exits 1 "
                         "when there are any")
    ap.add_argument("--input", metavar="FILE@ADDR[:DTYPE]", action="append", default=[],
                    help="put an array into DRAM before the run: a .npy file (or numbers in a "
                         "text file) at byte address ADDR, as DTYPE (bf16, f32, i32, u32, i16, "
                         "u16, i8, u8; default bf16); repeatable, after the program's own data")
    ap.add_argument("--output", metavar="ADDR:SHAPE[:DTYPE][=FILE]", action="append", default=[],
                    help="read an array from DRAM after the run, e.g. 0x3000:32x32:bf16=y.npy; "
                         "prints a summary and saves FILE (.npy); repeatable. A C program's "
                         "stack goes above every --output")
    ap.add_argument("--expect", metavar="FILE@ADDR[:DTYPE]", action="append", default=[],
                    help="after the run, compare DRAM at ADDR with the array in FILE (its "
                         "shape); exits 1 if any element is more than --ulps BF16 steps off")
    ap.add_argument("--ulps", type=float, default=1,
                    help="with --expect: allowed error in BF16 steps (default %(default)g)")
    ap.add_argument("--profile", nargs="?", const=15, type=int, metavar="N",
                    help="charge every cycle to the packet responsible and print the N "
                         "packets that took the most (default 15), with what they waited on")
    ap.add_argument("--profile-json", metavar="PATH",
                    help="write the whole per-packet profile as JSON")
    ap.add_argument("--save-asm", metavar="PATH",
                    help="for a .c program: keep the compiler's assembly (the profile and "
                         "other reports refer to its line numbers)")
    ap.add_argument("--sdma-log", metavar="PATH",
                    help="log each SDMA's start and end (cycle, pc, addresses, shape, "
                         "duration; src/atalla/sdma_log.py) to PATH, or - for the terminal")
    ap.add_argument("--trace", metavar="PATH",
                    help="write the execution trace (packets issued, register writes; "
                         "src/atalla/trace.py) for atalla-sim compare-rtl")
    ap.add_argument("--report", action="store_true",
                    help="print the performance monitor's report: where the cycles went, "
                         "unit utilisation, and the AXI / DRAM summaries")
    ap.add_argument("--perf-json", metavar="PATH",
                    help="write the performance monitor's report as JSON")
    ap.add_argument("--timeline", action="store_true",
                    help="with --perf-json: include the per-cycle timeline")
    args = ap.parse_args(argv)

    loaded = load_program(args.kernel, extra, args.elf)
    instr, data, init_sregs = loaded.instr, loaded.data, loaded.init_sregs
    describe_pc, c_path, program = loaded.describe_pc, loaded.c_path, loaded.program
    try:
        inputs = [tensors.parse_input(spec) for spec in args.input]
        outputs = [tensors.parse_output(spec) for spec in args.output]
        expects = [tensors.parse_input(spec) for spec in args.expect]
    except ValueError as exc:
        sys.exit(str(exc))
    if inputs or outputs:
        data = dict(data)
        for path, addr, dtype in inputs:
            try:
                tensors.put(data, addr, tensors.load_array(path), dtype)
            except (OSError, ValueError) as exc:
                sys.exit("--input %s: %s" % (path, exc))
        if init_sregs is not None:
            init_sregs = entry_sregs(data, [(a, a + tensors.nbytes(shape, dt))
                                            for a, shape, dt, _ in outputs])
    if args.save_asm:
        source = getattr(program, "source", None) if c_path is not None else None
        if source is None:
            sys.exit("--save-asm is for a .c program assembled from the compiler's assembly")
        Path(args.save_asm).write_text(source)
        print("assembly written to %s" % args.save_asm)
    if args.save:
        Path(args.save).write_text(to_in_text(instr, data))
        print("program image written to %s" % args.save)

    try:
        cfg = machine_config(args)
        kw = config.platform_kwargs(cfg)
    except (config.ConfigError, RuntimeError, ValueError, TypeError) as exc:
        sys.exit(str(exc))
    dram_channel = kw.get("dram_channel")
    plat = build_atalla_platform(instr, data, **kw,
                                 perf=bool(args.report or args.perf_json),
                                 activity=bool(args.activity),
                                 check=bool(args.check),
                                 profile=bool(args.profile or args.profile_json),
                                 trace=bool(args.trace),
                                 sdma_log=bool(args.sdma_log),
                                 perf_timeline=args.timeline, init_sregs=init_sregs,
                                 describe_pc=describe_pc, strict=False)
    writes = lockstep.attach(plat) if args.golden else None
    t = time.time()
    try:
        cycles = plat.run_until_done(limit=args.limit, hang_after=args.hang_after or None)
    except RuntimeError as exc:
        sys.exit("%s: %s" % (args.kernel, exc))
    c = plat.core
    print("%s: %d cycles (%.1f s)" % (args.kernel, cycles, time.time() - t))
    if args.report:
        print("machine: %s" % config.summary(cfg))
        print()
        print(plat.perf.summary())
        print()
        if plat.axi is not None:
            print(plat.axi.summary(cycles))
            print()
        if dram_channel is not None:
            print(dram_channel.summary(through=plat.cycle - 1))
            print()
    if args.perf_json:
        plat.perf.to_json(args.perf_json, timeline=args.timeline, config=cfg)
        print("performance report written to %s" % args.perf_json)
    if args.activity:
        core, mem = cfg["core"], cfg["memory"]
        meta = dict(program=Path(args.kernel).name, array=core["systolic_array"],
                    lanes=core["lane_count"], burst_bytes=mem["burst_bytes"], dram=mem["dram"],
                    axi=plat.axi is not None, machine=config.summary(cfg), config=cfg)
        if mem["dram"] == "fixed":
            meta["dram_latency"] = mem["dram_latency"]
        if args.activity.endswith(".html"):
            import activity_view
            Path(args.activity).write_text(activity_view.render(plat.activity.trace(meta)))
            print("activity view written to %s" % args.activity)
        else:
            plat.activity.to_json(args.activity, meta)
            print("activity trace written to %s (tools/activity_view.py draws it)" % args.activity)
    if c.decode2.violations:
        print("  packets the RTL would mangle: %d (first at pc %#x: %s)"
              % (len(c.decode2.violations), c.decode2.violations[0][0],
                 "; ".join(c.decode2.violations[0][1])))
    if c_path is not None:
        ok, message = c_kernels.check(c_path, plat)
        print("reference result: %s -- %s"
              % ({True: "PASS", False: "FAIL", None: "not checked"}[ok], message))
        bug = c_kernels.known_bug(c_path)
        if bug and not ok:
            print("  a known compiler bug: %s" % bug)
    if args.golden:
        if not golden.HAVE_GOLDEN:
            sys.exit("functional sim unavailable: %s" % golden.load_error)
        red = "functional" if args.functional_reductions else "lanes"
        print("against the functional sim (%s reduction order):" % red)
        compare(plat, instr, data, red, init_sregs, writes)
    if args.check:
        print(plat.check.summary())
    if args.sdma_log == "-":
        print("SDMAs (%d):" % len(plat.sdma_log.records))
        sys.stdout.write(plat.sdma_log.text())
    elif args.sdma_log:
        plat.sdma_log.write(args.sdma_log)
        print("SDMA log written to %s (%d SDMAs)" % (args.sdma_log, len(plat.sdma_log.records)))
    if args.trace:
        plat.tracer.write(args.trace)
        print("trace written to %s" % args.trace)
    if args.profile:
        print(plat.profile.summary(top=args.profile))
    if args.profile_json:
        plat.profile.to_json(args.profile_json, config=cfg)
        print("profile written to %s" % args.profile_json)
    reader = tensors.plat_reader(plat)
    for addr, shape, dtype, path in outputs:
        y = tensors.read(reader, addr, shape, dtype)
        print("output %#x (%s): %s" % (addr, dtype, tensors.describe(y)))
        if path:
            import numpy as np
            np.save(path, y)
            print("  saved to %s" % path)
    failed = False
    for path, addr, dtype in expects:
        try:
            want = tensors.load_array(path)
        except (OSError, ValueError) as exc:
            sys.exit("--expect %s: %s" % (path, exc))
        got = tensors.read(reader, addr, want.shape, dtype)
        try:
            m = tensors.assert_close(got, want, ulps=args.ulps)
            print("expect %s at %#x: PASS (max abs %.3g, worst %.1f BF16 steps, cosine %.6f)"
                  % (path, addr, m["max_abs"], m["max_bf16_steps"], m["cosine"]))
        except AssertionError as exc:
            failed = True
            print("expect %s at %#x: FAIL -- %s" % (path, addr, exc))
    if failed or (args.check == "error" and plat.check.findings):
        sys.exit(1)


if __name__ == "__main__":
    main()
