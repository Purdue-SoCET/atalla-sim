"""C kernels: from the compiler straight to atalla-sim.

    C source --atalla_cc -p -S--> packetized assembly --assembler--> program image
                                  (third_party/aihw-ppci-compiler)  (src/scheduler/assembler.py)
    C source --atalla_cc -p ----> linked ELF ----------load_elf----> program image + data
                                                                    (src/scheduler/elf.py)

The compiler (branch atalla-arch) packetizes; atalla-sim only assembles and
runs what it was given. Two sets of C programs come with the compiler:

    test_validation_kernels/*.c   kernels (add, relu, softmax, conv, gemm, ...).
                                  Their inputs and numpy reference results come
                                  from the compiler's own validation harness,
                                  run_kernel_validation.py (seed_* / validate_*).
    test_validation/*.c           unit tests, one feature each; the expected
                                  result is in x10 (`EXPECT_X10:` in the header).

    from c_kernels import build, check
    program, data, regs = build(path_to_c)       # compile + assemble + inputs
    program, data, regs = build(path_to_c, elf=True)   # compile + link + load
    plat = build_atalla_platform(program.instr, data, init_sregs=regs)
    plat.run_until_done()
    ok, message = check(path_to_c, plat)         # the reference result

To add a kernel: put the .c file in test_validation_kernels/ and give it a
seed and validate function in the compiler's run_kernel_validation.py
(KERNEL_REGISTRY). To add a unit test: a .c file in test_validation/ with an
`EXPECT_X10:` line. Nothing here changes.

The ELF path places the code where the compiler's layout puts it (from
0x1000) and adds the program's own data (globals, from 0x80000). The kernels'
inputs are seeded at addresses they choose, and some fall inside that code:
relu's at 0x1000. atalla-sim keeps the program apart from DRAM, so the kernel
still runs, but on the RTL's single memory the inputs would overwrite the
code; build() warns when that happens (ElfLayoutWarning).
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atalla.atalla_platform import c_entry_sregs           # noqa: E402
from scheduler import golden                                # noqa: E402
from scheduler.assembler import Program, assemble           # noqa: E402
from scheduler.elf import ElfProgram, load_elf, overlaps    # noqa: E402

COMPILER = ROOT / "third_party" / "aihw-ppci-compiler"
KERNELS = COMPILER / "test_validation_kernels"
UNIT_TESTS = COMPILER / "test_validation"


class CompileError(RuntimeError):
    pass


class ElfLayoutWarning(UserWarning):
    """A kernel's inputs land on its own code or data (fine here, not on the RTL)."""


def compile_c(c_path: Path) -> str:
    """atalla_cc -p -S: the packetized assembly, as text."""
    if not (COMPILER / "ppci").is_dir():
        raise CompileError("the compiler submodule isn't checked out: %s" % COMPILER)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / (c_path.stem + ".s")
        # The compiler imports its packetizer and latency table from its root.
        env = dict(os.environ, PYTHONPATH=str(COMPILER))
        p = subprocess.run([sys.executable, "-m", "ppci.cli.atalla_cc", "--machine", "atalla",
                            "-O", "2", "-p", "-S", str(c_path), "-o", str(out)],
                           cwd=str(COMPILER), env=env, capture_output=True, text=True)
        if p.returncode != 0:
            raise CompileError("atalla_cc failed on %s:\n%s"
                               % (c_path.name, (p.stderr or p.stdout)[-3000:]))
        return out.read_text()


def compile_elf(c_path: Path) -> bytes:
    """atalla_cc -p -o kernel.elf: compiled and linked with the compiler's own
    driver and memory layout, as bytes."""
    if not (COMPILER / "ppci").is_dir():
        raise CompileError("the compiler submodule isn't checked out: %s" % COMPILER)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / (c_path.stem + ".elf")
        env = dict(os.environ, PYTHONPATH=str(COMPILER))
        p = subprocess.run([sys.executable, str(COMPILER / "atalla_cc"), "-O", "2", "-p",
                            str(c_path), "-o", str(out)],
                           cwd=str(COMPILER), env=env, capture_output=True, text=True)
        if p.returncode != 0 or not out.exists():
            raise CompileError("atalla_cc failed on %s:\n%s"
                               % (c_path.name, (p.stderr or p.stdout)[-3000:]))
        return out.read_bytes()


# -- the compiler's validation harness ----------------------------------------------------

_harness = None
harness_error: Optional[str] = None


def harness():
    """The compiler's run_kernel_validation module (seeds, references, known
    bugs), or None. Its references use the functional sim's GEMM model, so it
    needs that submodule too."""
    global _harness, harness_error
    if _harness is not None or harness_error is not None:
        return _harness
    if not golden.HAVE_GOLDEN:
        harness_error = "functional sim unavailable (%s)" % golden.load_error
        return None
    path = KERNELS / "run_kernel_validation.py"
    try:
        spec = importlib.util.spec_from_file_location("compiler_kernel_validation", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _harness = module
    except Exception as exc:                      # pragma: no cover - env dependent
        harness_error = "%s: %s" % (type(exc).__name__, exc)
    return _harness


def _unit_test_meta(c_path: Path):
    spec = importlib.util.spec_from_file_location("compiler_unit_validation",
                                                  UNIT_TESTS / "run_all_validation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_unit_test(c_path: Path) -> bool:
    return "EXPECT_X10:" in c_path.read_text()


def inputs(c_path: Path) -> Dict[int, int]:
    """The DRAM words the kernel starts with (empty if it needs none)."""
    words: Dict[int, int] = {}
    if is_unit_test(c_path):
        meta = _unit_test_meta(c_path)
        _, base, lane0, others = meta.parse_test_metadata(c_path)
        if base is not None:
            text = meta.render_bf16_vector_data(base, [lane0] + [others] * 31)
            for line in text.splitlines():
                a, w = line.split(":")
                words[int(a, 16)] = int(w, 16)
        return words
    h = harness()
    if h is not None and c_path.stem in h.KERNEL_REGISTRY:
        h.KERNEL_REGISTRY[c_path.stem][0](words)
    return words


def build(c_path: Path, elf: bool = False) -> Tuple[Program, Dict[int, int], Dict[int, int]]:
    """Compile and assemble (or, with `elf`, compile, link and load the ELF);
    returns (program, DRAM data, initial registers)."""
    if not elf:
        program = assemble(compile_c(c_path))
        data = inputs(c_path)
        return program, data, c_entry_sregs(data)
    program = load_elf(compile_elf(c_path))
    seeds = inputs(c_path)
    _warn_layout(c_path, program, seeds)
    data = {**program.data, **seeds}
    return program, data, c_entry_sregs(data)


def _warn_layout(c_path: Path, program: ElfProgram, seeds: Dict[int, int]) -> None:
    on_code = overlaps(program.code, seeds)
    if on_code:
        warnings.warn("%s: %d input words inside the code (%s), from %#x: they would "
                      "overwrite it in a single memory" % (
                          c_path.name, len(on_code),
                          ", ".join("%#x..%#x" % r for r in program.code), on_code[0]),
                      ElfLayoutWarning, stacklevel=3)
    on_data = sorted(set(seeds) & set(program.data))
    if on_data:
        warnings.warn("%s: %d input words replace the program's own data, from %#x"
                      % (c_path.name, len(on_data), on_data[0]), ElfLayoutWarning,
                      stacklevel=3)


# -- checking -----------------------------------------------------------------------------

def known_bug(c_path: Path) -> Optional[str]:
    """The compiler's note when a kernel fails because of a known compiler bug."""
    h = harness()
    return h.EXPECTED_FAILURES.get(c_path.stem) if h is not None else None


def _dram_dump(plat, path: Path) -> None:
    """The DRAM, in the functional sim's output format the references read."""
    dram = plat.dram
    lines = ["DATA MEM"]
    for block in sorted(dram._blocks):
        base = block * dram.block_bytes
        raw = bytes(dram._blocks[block])
        for off in range(0, len(raw), 4):
            lines.append("%08X: %08X" % (base + off, int.from_bytes(raw[off:off + 4], "little")))
    path.write_text("\n".join(lines) + "\n")


def check(c_path: Path, plat) -> Tuple[Optional[bool], str]:
    """(True, "...") on the reference result, (False, why) off it, (None, why)
    when there is no reference to compare with."""
    if is_unit_test(c_path):
        expected, *_ = _unit_test_meta(c_path).parse_test_metadata(c_path)
        got = plat.core.scalar_reg(10)
        if got == expected & 0xFFFFFFFF:
            return True, "x10 = %#x as expected" % got
        return False, "x10 = %#x, expected %#x" % (got, expected & 0xFFFFFFFF)
    h = harness()
    if h is None:
        return None, "no reference: %s" % harness_error
    if c_path.stem not in h.KERNEL_REGISTRY:
        return None, "no reference for %s in the compiler's harness" % c_path.stem
    with tempfile.TemporaryDirectory() as tmp:
        mem = Path(tmp) / "output_mem.out"
        _dram_dump(plat, mem)
        try:
            h.KERNEL_REGISTRY[c_path.stem][1](mem)
        except Exception as exc:
            return False, str(exc).splitlines()[0]
    return True, "matches the numpy reference"
