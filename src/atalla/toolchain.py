"""The C compiler (the third_party/aihw-ppci-compiler submodule), called from atalla-sim.

    compile_c(path)   atalla_cc -O 2 -p -S: the packetized assembly, as text
    compile_elf(path) atalla_cc -O 2 -p: a linked ELF, as bytes

The compiler packetizes; atalla-sim's assembler (scheduler/assembler.py)
only encodes. Both raise CompileError with the compiler's message.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Union

ROOT = Path(__file__).resolve().parents[2]
COMPILER = ROOT / "third_party" / "aihw-ppci-compiler"


class CompileError(RuntimeError):
    pass


def _need_compiler() -> None:
    if not (COMPILER / "ppci").is_dir():
        raise CompileError("the compiler submodule isn't checked out: %s "
                           "(git submodule update --init third_party/aihw-ppci-compiler)"
                           % COMPILER)


def _run(args, c_path: Path) -> None:
    # The compiler imports its packetizer and latency table from its root.
    env = dict(os.environ, PYTHONPATH=str(COMPILER))
    p = subprocess.run(args, cwd=str(COMPILER), env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, universal_newlines=True)
    if p.returncode != 0:
        raise CompileError("atalla_cc failed on %s:\n%s"
                           % (c_path.name, (p.stderr or p.stdout)[-3000:]))


def compile_c(c_path: Union[str, Path]) -> str:
    """atalla_cc -p -S: the packetized assembly, as text."""
    c_path = Path(c_path).resolve()
    _need_compiler()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / (c_path.stem + ".s")
        _run([sys.executable, "-m", "ppci.cli.atalla_cc", "--machine", "atalla",
              "-O", "2", "-p", "-S", str(c_path), "-o", str(out)], c_path)
        return out.read_text()


def compile_elf(c_path: Union[str, Path]) -> bytes:
    """atalla_cc -p -o kernel.elf: compiled and linked with the compiler's own
    driver and memory layout, as bytes."""
    c_path = Path(c_path).resolve()
    _need_compiler()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / (c_path.stem + ".elf")
        _run([sys.executable, str(COMPILER / "atalla_cc"), "-O", "2", "-p",
              str(c_path), "-o", str(out)], c_path)
        if not out.exists():
            raise CompileError("atalla_cc wrote no ELF for %s" % c_path.name)
        return out.read_bytes()
