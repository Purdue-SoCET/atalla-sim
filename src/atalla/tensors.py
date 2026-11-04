"""Tensors in Atalla DRAM: numpy arrays in, numpy arrays out.

DRAM is byte-addressed and little-endian; programs see 32-bit words. These
helpers put an array into a program's initial data and read one back, in the
element types the hardware uses:

    bf16   the vector datapath's type (round to nearest even from float32)
    f32, i32, u32, i16, u16, i8, u8

    data = {}
    put(data, 0x1000, x, "bf16")              # merges into {word address: word}
    y = read(plat_reader(plat), 0x3000, (32, 32), "bf16")
    assert_close(y, ref, ulps=2)              # within 2 BF16 steps of ref

`parse_input` / `parse_output` read the command line's `FILE@ADDR[:DTYPE]`
and `ADDR:SHAPE[:DTYPE][=FILE]`.
"""

from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

DTYPES = {"bf16": np.uint16, "f32": np.float32, "i32": np.int32, "u32": np.uint32,
          "i16": np.int16, "u16": np.uint16, "i8": np.int8, "u8": np.uint8}

Reader = Callable[[int, int], bytes]           # (address, length) -> bytes


def _check_dtype(dtype: str) -> None:
    if dtype not in DTYPES:
        raise ValueError("unknown dtype %r; one of %s" % (dtype, ", ".join(DTYPES)))


def to_bf16_bits(x) -> np.ndarray:
    """float -> BF16 bit patterns, rounded to nearest even (NaN stays NaN)."""
    f = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    b = f.view(np.uint32).astype(np.uint64)
    rounded = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(f)
    if nan.any():
        rounded[nan] = 0x7FC0
    return rounded


def from_bf16_bits(bits) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint32) << 16).astype(np.uint32).view(np.float32)


def bf16_round(x) -> np.ndarray:
    """x rounded to BF16, as float32."""
    return from_bf16_bits(to_bf16_bits(x))


def to_bytes(x, dtype: str = "bf16") -> bytes:
    _check_dtype(dtype)
    if dtype == "bf16":
        return to_bf16_bits(x).astype("<u2").tobytes()
    return np.ascontiguousarray(np.asarray(x).astype(DTYPES[dtype])).astype(
        np.dtype(DTYPES[dtype]).newbyteorder("<")).tobytes()


def from_bytes(raw: bytes, shape, dtype: str = "bf16") -> np.ndarray:
    _check_dtype(dtype)
    if dtype == "bf16":
        return from_bf16_bits(np.frombuffer(raw, dtype="<u2")).reshape(shape)
    return np.frombuffer(raw, dtype=np.dtype(DTYPES[dtype]).newbyteorder("<")).astype(
        DTYPES[dtype]).reshape(shape)


def nbytes(shape, dtype: str) -> int:
    _check_dtype(dtype)
    return int(np.prod(shape, dtype=np.int64)) * np.dtype(DTYPES[dtype]).itemsize


def put(data: Dict[int, int], addr: int, x, dtype: str = "bf16") -> Dict[int, int]:
    """Merge array `x` at byte address `addr` into `data` ({word address: 32-bit
    word}, as load_program_text returns and build_atalla_platform takes)."""
    raw = to_bytes(x, dtype)
    addr = int(addr)
    for k, byte in enumerate(raw):
        a = addr + k
        w = a & ~3
        shift = (a & 3) * 8
        word = data.get(w, 0) & ~(0xFF << shift)
        data[w] = (word | (byte << shift)) & 0xFFFFFFFF
    return data


def read(reader: Reader, addr: int, shape, dtype: str = "bf16") -> np.ndarray:
    shape = tuple(shape) if not isinstance(shape, int) else (shape,)
    return from_bytes(reader(int(addr), nbytes(shape, dtype)), shape, dtype)


def plat_reader(plat) -> Reader:
    """Reads an AtallaPlatform's DRAM."""
    return lambda a, n: plat.dram.read(int(a), int(n))


def words_reader(words: Dict[int, int]) -> Reader:
    """Reads a {word address: word} map (the functional sim's memory)."""
    def rd(a: int, n: int) -> bytes:
        out = bytearray(n)
        for k in range(n):
            w = words.get((a + k) & ~3, 0)
            out[k] = (w >> (((a + k) & 3) * 8)) & 0xFF
        return bytes(out)
    return rd


# -- comparing ----------------------------------------------------------------------------

def bf16_steps(got, want) -> np.ndarray:
    """|got - want| in BF16 steps at want's magnitude (one step: 2^(e-7) for
    want in [2^e, 2^(e+1)); want = 0 uses the smallest normal's step)."""
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    mag = np.maximum(np.abs(want), np.finfo(np.float32).tiny)
    step = np.exp2(np.floor(np.log2(mag)) - 7)
    return np.abs(got - want) / step


def compare(got, want) -> dict:
    """Error measures of got against want: max abs, rel_l2, cosine, worst BF16 steps."""
    g = np.asarray(got, dtype=np.float64).ravel()
    w = np.asarray(want, dtype=np.float64).ravel()
    if g.shape != w.shape:
        raise ValueError("shapes differ: %s vs %s" % (np.shape(got), np.shape(want)))
    diff = np.abs(g - w)
    steps = bf16_steps(g, w)
    nw, ng = np.linalg.norm(w), np.linalg.norm(g)
    worst = int(np.argmax(steps)) if steps.size else 0
    return dict(max_abs=float(diff.max()) if diff.size else 0.0,
                rel_l2=float(np.linalg.norm(g - w) / nw) if nw else float(np.linalg.norm(g)),
                cosine=float(g @ w / (ng * nw)) if ng and nw else (1.0 if ng == nw else 0.0),
                max_bf16_steps=float(steps.max()) if steps.size else 0.0,
                worst_index=np.unravel_index(worst, np.shape(want)) if steps.size else (),
                mismatches=int((steps > 0).sum()))


def assert_close(got, want, ulps: float = 1, atol: float = 0.0) -> dict:
    """Raise AssertionError unless every element of got is within `ulps` BF16
    steps of want (or within `atol`). Returns compare()'s measures."""
    m = compare(got, want)
    g = np.asarray(got, dtype=np.float64)
    w = np.asarray(want, dtype=np.float64)
    bad = (bf16_steps(g, w) > ulps) & (np.abs(g - w) > atol)
    if bad.any():
        idx = tuple(int(i) for i in np.argwhere(bad)[0])
        raise AssertionError(
            "%d of %d elements more than %g BF16 steps off (atol %g); first at %s: got %r, "
            "want %r; max abs %.3g, rel_l2 %.3g, worst %.1f steps"
            % (int(bad.sum()), bad.size, ulps, atol, idx, float(g[idx]), float(w[idx]),
               m["max_abs"], m["rel_l2"], m["max_bf16_steps"]))
    return m


# -- the command line ---------------------------------------------------------------------

def _int(text: str) -> int:
    return int(text, 0)


def parse_input(spec: str) -> Tuple[str, int, str]:
    """`FILE@ADDR[:DTYPE]` -> (file, address, dtype)."""
    path, at, rest = spec.rpartition("@")
    if not at or not path:
        raise ValueError("--input takes FILE@ADDR[:DTYPE], e.g. x.npy@0x1000:bf16; not %r"
                         % spec)
    addr, _, dtype = rest.partition(":")
    dtype = dtype or "bf16"
    _check_dtype(dtype)
    return path, _int(addr), dtype


def parse_output(spec: str) -> Tuple[int, Tuple[int, ...], str, Optional[str]]:
    """`ADDR:SHAPE[:DTYPE][=FILE]`, SHAPE like 32x32 -> (address, shape, dtype, file)."""
    body, eq, path = spec.partition("=")
    parts = body.split(":")
    if len(parts) not in (2, 3):
        raise ValueError("--output takes ADDR:SHAPE[:DTYPE][=FILE], e.g. 0x3000:32x32:bf16="
                         "y.npy; not %r" % spec)
    shape = tuple(int(n) for n in parts[1].lower().split("x"))
    dtype = parts[2] if len(parts) == 3 else "bf16"
    _check_dtype(dtype)
    return _int(parts[0]), shape, dtype, (path or None) if eq else None


def load_array(path: str) -> np.ndarray:
    """An .npy file, or text (whitespace or comma separated) for anything else."""
    if path.endswith(".npy"):
        return np.load(path)
    text = open(path).read().replace(",", " ")
    return np.array([float(t) for t in text.split()], dtype=np.float64)


def describe(x: np.ndarray) -> str:
    flat = np.asarray(x, dtype=np.float64).ravel()
    head = ", ".join("%.4g" % v for v in flat[:6])
    return "shape %s, min %.4g, max %.4g, mean %.4g: [%s%s]" % (
        "x".join(map(str, np.shape(x))), flat.min(), flat.max(), flat.mean(), head,
        ", ..." if flat.size > 6 else "")
