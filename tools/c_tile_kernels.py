"""C tile kernels as sweep baselines: the DRAM image their address table
expects, and the same tile the functional sim's tile kernels build.

tools/c_kernels.py runs the compiler's validation kernels, whose inputs come
from the compiler's harness. The kernels here (aihw-ppci-compiler/atalla_tests/
gelu.c, ...) have no harness entry, so this module writes their inputs:

    tile(rows, width, seed, scale)   the input, as kernels/tile_kernel.py
                                     builds it (standard normal x scale), so a
                                     C baseline and build_vloop.py get the
                                     same data for the same arguments
    C_KERNELS[name]                  how to build each kernel's DRAM image

gelu.c's address table, at 0x3C (60):
    +0 IN_GMEM   +4 OUT_GMEM   +8 tile_count   +12 rows   +16 SDMA config
    +20 tile bytes, then fp32 +24 c6  +28 c4  +32 c2  +36 c0  +40 0.5
    +44 3.0 (above: y = x)   +48 -3.0 (below: y = 0)
and it computes y = 0.5 x + P(x^2), P(u) = ((c6 u + c4) u + c2) u + c0,
clamped outside [-3, 3]. Its columns are WIDTH_M1 + 1 wide (read from the
.c), so the tile is too. The coefficients aren't given anywhere; GELU_POLY
fits them (least squares) to tanh GELU on [-3, 3] and rounds them to BF16,
since the vector unit rounds a scalar operand to BF16 anyway.
"""

import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
#: The compiler checkout the C baselines are compiled with: gelu.c needs a
#: newer atalla_cc than the submodule (5-argument vec_op_masked).
C_COMPILER = ROOT / "aihw-ppci-compiler"
C_TESTS = C_COMPILER / "atalla_tests"

CFG_BASE = 0x3C
INPUT_BASE = 0x00001000
OUTPUT_BASE = 0x00002000


def bf16_round(x: float) -> float:
    """Round to BF16, nearest-even, as the functional sim does."""
    u = struct.unpack("<I", struct.pack("<f", float(np.float32(x))))[0]
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", u))[0]


def bf16_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", bf16_round(x)))[0] >> 16


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def tile(rows: int, width: int, seed: int, scale: float) -> np.ndarray:
    """kernels/tile_kernel.tile(): the same draws, so the same data."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((rows, width)) * scale).astype(np.float32)


def sdma_config(sid: int, rows: int, cols: int) -> int:
    return (sid << 30) | ((rows - 1) << 25) | ((cols - 1) << 20)


def tile_words(base: int, values: np.ndarray) -> Dict[int, int]:
    """Row-major BF16, two to a little-endian word."""
    flat = [bf16_bits(v) for v in values.flatten(order="C")]
    if len(flat) % 2:
        flat.append(0)
    return {base + 2 * i: flat[i] | (flat[i + 1] << 16) for i in range(0, len(flat), 2)}


def read_tile(plat, base: int, rows: int, width: int) -> List[List[float]]:
    out = []
    for r in range(rows):
        row = []
        for c in range(width):
            addr = base + 2 * (r * width + c)
            w = int.from_bytes(plat.dram.read(addr & ~3, 4), "little")
            h = (w >> 16) if addr & 2 else (w & 0xFFFF)
            row.append(struct.unpack("<f", struct.pack("<I", h << 16))[0])
        out.append(row)
    return out


def c_define(c_path: Path, name: str) -> int:
    m = re.search(r"#define\s+%s\s+(\d+)" % name, Path(c_path).read_text())
    if not m:
        raise ValueError("%s has no #define %s" % (c_path, name))
    return int(m.group(1))


def tile_width(c_path: Path) -> int:
    """Columns the kernel moves: WIDTH_M1 + 1. WIDTH_M1 is the vreg/scpad
    num_cols field, 5 bits (0..31), so 32 columns is WIDTH_M1 31."""
    m1 = c_define(c_path, "WIDTH_M1")
    if not 0 <= m1 <= 31:
        raise ValueError("%s: WIDTH_M1 is %d, but it is columns - 1 in a 5-bit field "
                         "(0..31); for 32 columns use 31" % (Path(c_path).name, m1))
    return m1 + 1


# -- compiler workarounds -----------------------------------------------------------------
def split_double_mv_stm(asm: str, packet_size: int = 4) -> Tuple[str, int]:
    """Packetized assembly with each packet's second (third, ...) mv_stm
    moved to a packet of its own right after: (text, packets added).

    WORKAROUND for the newer atalla_cc's packetizer (aihw-ppci-compiler,
    aa7bdbf6), which puts two mv_stm in one packet. mv.stm runs on scalar
    unit EX2 and a packet may give each unit one instruction (decode2.py's
    contract); the RTL drops the extra one, and atalla-sim with strict=False
    then waits forever on its mask. Moving it a packet later is safe: two
    mv_stm read scalars and write masks, so neither reads what the other
    writes, and branches go by label, so the added packets move them. Report
    the packetizer bug and drop this once it's fixed."""
    out: List[str] = []
    cur: List[str] = []
    added = 0

    def mnemonic(line):
        code = re.split(r"[#;]", line, maxsplit=1)[0].strip()
        return code.split()[0].lower() if code else ""

    def flush():
        nonlocal added
        stm = [i for i, l in enumerate(cur) if mnemonic(l) == "mv_stm"]
        later = stm[1:]
        nop = "       nop"
        out.extend(nop if i in later else l for i, l in enumerate(cur))
        for i in later:
            out.extend([cur[i]] + [nop] * (packet_size - 1))
            added += 1
        cur.clear()

    for line in asm.splitlines():
        code = re.split(r"[#;]", line, maxsplit=1)[0].strip()
        if not code or code.startswith(".") or code.endswith(":") or \
                code.split()[0] in ("global", "type"):
            out.append(line)
            continue
        cur.append(line)
        if len(cur) == packet_size:
            flush()
    out.extend(cur)
    return "\n".join(out) + "\n", added


# -- gelu.c ------------------------------------------------------------------------------
def tanh_gelu(x: float) -> float:
    k = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + math.tanh(k * (x + 0.044715 * x ** 3)))


def fit_gelu_poly() -> Tuple[float, float, float, float]:
    """(c6, c4, c2, c0): P(x^2) ~ tanh_gelu(x) - x/2 on [-3, 3], least squares,
    rounded to BF16."""
    x = np.linspace(-3.0, 3.0, 4001)
    target = np.array([tanh_gelu(v) for v in x]) - 0.5 * x
    u = x * x
    basis = np.stack([u ** 3, u ** 2, u, np.ones_like(u)], axis=1)
    coeffs = np.linalg.lstsq(basis, target, rcond=None)[0]
    return tuple(bf16_round(c) for c in coeffs)


GELU_POLY = fit_gelu_poly()


def gelu_c_image(c_path: Path, rows: int, seed: int, scale: float) -> Tuple[Dict[int, int], int]:
    """gelu.c's DRAM: (data words, tile width)."""
    width = tile_width(c_path)
    data = tile_words(INPUT_BASE, tile(rows, width, seed, scale))
    c6, c4, c2, c0 = GELU_POLY
    words = [INPUT_BASE, OUTPUT_BASE, 1, rows, sdma_config(0, rows, width), rows * width * 2]
    floats = [c6, c4, c2, c0, 0.5, 3.0, -3.0]
    for i, w in enumerate(words):
        data[CFG_BASE + 4 * i] = w
    for i, f in enumerate(floats):
        data[CFG_BASE + 24 + 4 * i] = f32_bits(f)
    return data, width


@dataclass(frozen=True)
class CKernel:
    """A C baseline: its source, its DRAM image builder (c_path, rows, seed,
    scale) -> (data, width), and the function its output should be."""
    path: Path
    image: Callable
    true: Callable[[float], float]


C_KERNELS = {
    "gelu": CKernel(C_TESTS / "gelu.c", gelu_c_image, tanh_gelu),
}


#: Default input scale per function; must match the functional sim's
#: kernels/tile_kernel.DATA_SCALE (the sweep passes it to both explicitly).
DEFAULT_SCALE = {"relu": 1.0, "gelu": 2.0}

_compiled: Dict[Path, object] = {}


def program(name: str):
    """A C baseline compiled with C_COMPILER, the mv_stm workaround applied,
    and assembled (cached): a scheduler.assembler.Program."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "tools"))
    import c_kernels
    from scheduler.assembler import assemble
    path = C_KERNELS[name].path
    if path not in _compiled:
        text, _ = split_double_mv_stm(c_kernels.compile_c(path, C_COMPILER))
        _compiled[path] = assemble(text)
    return _compiled[path]
