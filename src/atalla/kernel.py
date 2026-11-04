"""Run one kernel with numpy inputs and read numpy outputs: the API for tests.

    from atalla.kernel import Kernel
    from atalla.tensors import assert_close

    k = Kernel(c=open("add.c").read())          # or asm=..., image=path.in, program=(instr, data)
    k.write(0x1000, a)                          # bf16 by default; dtype="f32", "i32", ...
    k.write(0x2000, b)
    k.output(0x3000, (32, 32))                  # reserve it: the stack goes above it
    r = k.run(golden=True)                      # atalla-sim, and the functional sim on the same image
    assert_close(r.read(0x3000, (32, 32)), a + b, ulps=1)
    assert_close(r.golden.read(0x3000, (32, 32)), a + b, ulps=1)
    r.cycles, r.plat                            # the platform, for its counters

C is compiled with the atalla-sim compiler (atalla/toolchain.py, `-O 2 -p`),
assembled by atalla-sim's assembler, and run with the C loader's stack
registers, placed above every input and every declared output (a stack that
lands on an output nobody initialised is a classic way to get garbage).
`run(**kw)` passes keywords to build_atalla_platform (check=True,
dram_channel=..., lane_count=..., perf=True ...).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from atalla import tensors
from atalla.atalla_platform import build_atalla_platform
from scheduler.assembler import assemble
from scheduler.core import load_program_text

PAGE = 0x1000


@dataclass
class Memory:
    """One simulator's DRAM after a run."""
    reader: tensors.Reader

    def read(self, addr: int, shape, dtype: str = "bf16"):
        return tensors.read(self.reader, addr, shape, dtype)

    def bytes(self, addr: int, n: int) -> bytes:
        return self.reader(int(addr), int(n))


@dataclass
class Result(Memory):
    cycles: int = 0
    plat: object = None
    #: The functional sim's DRAM, with run(golden=True).
    golden: Optional[Memory] = None


def entry_sregs(data: Dict[int, int], outputs=()) -> Dict[int, int]:
    """The C loader's registers: x2 (the stack) a page above the highest input
    word or declared output byte range [start, end), x33 a page above that."""
    top = max([a + 4 for a in data] + [end for _, end in outputs] + [0])
    stack = ((top + PAGE - 1) & ~(PAGE - 1)) + PAGE
    return {2: stack, 33: stack + PAGE}


def golden_reader(mem: Dict[int, int]) -> tensors.Reader:
    """Reads the functional sim's memory map through its own halfword reader
    (its map can hold entries at unaligned addresses too)."""
    from src.misc.memory import Memory as FsimMemory      # the functional sim's
    m = FsimMemory()
    m.data_mem = dict(mem)

    def rd(a: int, n: int) -> bytes:
        start, end = a & ~1, a + n
        raw = b"".join(m.read_bf16_le(h).to_bytes(2, "little") for h in range(start, end, 2))
        return raw[a - start:a - start + n]
    return rd


class Kernel:
    def __init__(self, c: Optional[str] = None, asm: Optional[str] = None,
                 image: Optional[str] = None,
                 program: Optional[Tuple[Dict[int, int], Dict[int, int]]] = None,
                 c_entry: Optional[bool] = None):
        given = [x is not None for x in (c, asm, image, program)]
        if sum(given) != 1:
            raise ValueError("give exactly one of c=, asm=, image=, program=")
        data: Dict[int, int] = {}
        if c is not None:
            from atalla.toolchain import compile_c
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "kernel.c"
                path.write_text(c)
                asm = compile_c(path)
        if asm is not None:
            self.instr = assemble(asm).instr
        elif image is not None:
            self.instr, data = load_program_text(Path(image).read_text())
        else:
            self.instr, data = dict(program[0]), dict(program[1] or {})
        self.asm = asm
        self.data: Dict[int, int] = dict(data)
        #: Byte ranges [start, end) the kernel writes but nobody initialises.
        self.outputs: List[Tuple[int, int]] = []
        # Compiled C expects the C loader's stack; hand-written asm by default doesn't.
        self.c_entry = (c is not None or asm is not None) if c_entry is None else c_entry

    def write(self, addr: int, x, dtype: str = "bf16") -> "Kernel":
        tensors.put(self.data, addr, x, dtype)
        return self

    def output(self, addr: int, shape, dtype: str = "bf16") -> "Kernel":
        self.outputs.append((int(addr), int(addr) + tensors.nbytes(shape, dtype)))
        return self

    def entry_sregs(self) -> Dict[int, int]:
        return entry_sregs(self.data, self.outputs)

    def run(self, golden: bool = False, limit: int = 5_000_000, **platform_kw) -> Result:
        regs = self.entry_sregs() if self.c_entry else None
        plat = build_atalla_platform(self.instr, self.data, init_sregs=regs,
                                     strict=platform_kw.pop("strict", False), **platform_kw)
        cycles = plat.run_until_done(limit=limit)
        res = Result(reader=tensors.plat_reader(plat), cycles=cycles, plat=plat)
        if golden:
            from scheduler import golden as g
            state = g.run_golden_state(self.instr, self.data, init_sregs=regs,
                                       reductions="lanes")
            res.golden = Memory(reader=golden_reader(state["mem"]))
        return res
