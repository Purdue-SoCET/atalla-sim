"""ELF loader: a linked program from the C compiler to a program image.

`atalla_cc kernel.c -o kernel.elf` (third_party/aihw-ppci-compiler) compiles
each source to an object and links them with its memory layout
(atalla_layout.mmap: code from 0x1000, data from 0x80000). This module reads
the result into what the scheduler runs -- the same Program the assembler
produces, plus the program's data and symbols:

    program = load_elf("kernel.elf")
    program.instr        {packet address: 160-bit packet}, for SchedulerCore
    program.data         {word address: 32-bit word}, for DRAM
    program.labels       {symbol: address}, every function and label in code
    program.symbols      Symbol(name, addr, kind, section, size), all of them
    program.code         [(start, end)] address ranges of the code sections
    program.describe(pc) "main_block3+20", for error messages and traces

What the compiler writes, and what this does with it:

- **Header.** ELF32, little-endian, machine 0x270F (Atalla's own ID).
  The linker marks its output REL (relocatable), with no program headers
  and entry 0, although every address is final; REL and EXEC are accepted.
  The layout is read from the section headers.
- **Sections.** Every allocated section of program content (PROGBITS or
  NOBITS; the compiler also marks its symbol tables allocated) is loaded at
  its address:
  executable ones (`code`) into the program image, the others (`data`)
  into DRAM, zero-filled ones as zeros. Sections are told apart by their
  flags, not their names, which come from the layout file. `.relacode`
  is skipped: the linker has already applied it.
- **Code.** Instructions are 40 bits, 5 bytes each, little-endian, packed
  back to back; every four starting at the section's address are one
  packet, slot 0 at the lowest address. The image keys packets by address
  with slot 0 in the top 40 bits, as the assembler does. Every function and
  label must start a packet (a whole number of packets into its section),
  or the program can't be fetched as the compiler packed it.
- **Data.** 32-bit little-endian words at their addresses. A section that
  doesn't end on a word boundary has its last word zero-filled.
- **Startup.** As the assembler does for C, two packets go at address 0:
  `jal x1, main` and `halt.s`, so the program calls main and halts when main
  returns. The code must therefore start at or after address 40. The
  initial registers (stack pointer x2, and x33) aren't in the file; the
  caller sets them (atalla.atalla_platform.c_entry_sregs).

Branch and jal offsets in the code are used as encoded: instruction counts
from the packet (scheduler/semantics.py).

The parser is written with `struct`: the subset needed is small, and a
general ELF library would add a dependency for a machine it doesn't know.
"""

import struct
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from scheduler.assembler import BOOTSTRAP_PACKETS, Instr, Program, encode
from scheduler.isa import INST_BYTES, INST_W, PACKET_BYTE_W, PACKET_SIZE

#: e_machine of the compiler's ELFs (ppci/format/elf/headers.py).
EM_ATALLA = 0x270F
ET_REL, ET_EXEC = 1, 2
SHT_PROGBITS, SHT_SYMTAB, SHT_NOBITS = 1, 2, 8
SHF_ALLOC, SHF_EXECINSTR = 0x2, 0x4
STT_OBJECT, STT_FUNC = 1, 2
SHN_UNDEF, SHN_LORESERVE = 0, 0xFF00

_EHDR = struct.Struct("<HHIIIIIHHHHHH")     # after the 16-byte e_ident
_SHDR = struct.Struct("<10I")
_SYM = struct.Struct("<IIIBBH")

ENTRY = "main"


class ElfError(ValueError):
    """A file this loader can't run, and why."""


@dataclass(frozen=True)
class Symbol:
    name: str
    addr: int
    #: "function", "label" (a block label in code) or "data" (a global).
    kind: str
    section: str
    size: int = 0


@dataclass
class _Section:
    name: str
    type: int
    flags: int
    addr: int
    offset: int
    size: int
    link: int


@dataclass
class ElfProgram(Program):
    data: Dict[int, int] = field(default_factory=dict)
    symbols: List[Symbol] = field(default_factory=list)
    code: List[Tuple[int, int]] = field(default_factory=list)
    entry: int = 0

    def describe(self, pc: int) -> str:
        """`pc` as the nearest code symbol at or below it, e.g. "main_block3+20"."""
        code = sorted((s.addr, s.name) for s in self.symbols if s.kind != "data")
        i = bisect_right([a for a, _ in code], int(pc)) - 1
        if i < 0 or not any(lo <= int(pc) < hi for lo, hi in self.code):
            return "%#x" % int(pc)
        addr, name = code[i]
        return name if addr == int(pc) else "%s+%d" % (name, int(pc) - addr)


def load_elf(source: Union[str, Path, bytes], entry: str = ENTRY) -> ElfProgram:
    """Read a linked Atalla ELF (a path or its bytes) into a program image."""
    raw = source if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
    raw = bytes(raw)
    sections = _sections(raw)
    prog = ElfProgram()
    nop = encode(Instr("nop.s", [], 0))

    for sec in sections:
        if not _loaded(sec):
            continue
        body = bytes(sec.size) if sec.type == SHT_NOBITS else _slice(raw, sec)
        if sec.flags & SHF_EXECINSTR:
            _load_code(prog, sec, body, nop)
        else:
            _load_data(prog, sec, body)
    if not prog.code:
        raise ElfError("no code section")
    _check_disjoint(prog, sections)

    prog.symbols = _symbols(raw, sections)
    for sym in prog.symbols:
        if sym.kind == "data":
            continue
        start = next((lo for lo, hi in prog.code if lo <= sym.addr <= hi), None)
        if start is None:
            raise ElfError("%s at %#x is outside the code it belongs to" % (sym.name, sym.addr))
        if (sym.addr - start) % PACKET_BYTE_W:
            raise ElfError("%s at %#x is not at a packet boundary: it is %d bytes into its "
                           "section's packets of %d" % (sym.name, sym.addr,
                                                        (sym.addr - start) % PACKET_BYTE_W,
                                                        PACKET_BYTE_W))
        prog.labels.setdefault(sym.name, sym.addr)

    if entry not in prog.labels:
        raise ElfError("no %r symbol in the code: nothing to call" % entry)
    prog.entry = prog.labels[entry]
    _bootstrap(prog, entry, nop)
    return prog


# -- parsing ------------------------------------------------------------------------------

def _sections(raw: bytes) -> List[_Section]:
    if len(raw) < 52 or raw[:4] != b"\x7fELF":
        raise ElfError("not an ELF file")
    if raw[4] != 1:
        raise ElfError("not a 32-bit ELF (EI_CLASS %d)" % raw[4])
    if raw[5] != 1:
        raise ElfError("not little-endian (EI_DATA %d)" % raw[5])
    (e_type, machine, version, _entry, _phoff, shoff, _flags, _ehsize, _phentsize, _phnum,
     shentsize, shnum, shstrndx) = _EHDR.unpack_from(raw, 16)
    if machine != EM_ATALLA:
        raise ElfError("machine %#x, not Atalla (%#x)" % (machine, EM_ATALLA))
    if e_type not in (ET_REL, ET_EXEC):
        raise ElfError("ELF type %d: expected a linked program (REL or EXEC)" % e_type)
    if version != 1:
        raise ElfError("ELF version %d" % version)
    if shentsize != _SHDR.size or not shnum or shoff + shnum * shentsize > len(raw):
        raise ElfError("bad section header table (%d entries of %d bytes at %#x)"
                       % (shnum, shentsize, shoff))
    if shstrndx >= shnum:
        raise ElfError("bad section name table index %d" % shstrndx)
    hdrs = [_SHDR.unpack_from(raw, shoff + i * shentsize) for i in range(shnum)]
    names = hdrs[shstrndx]
    out = []
    for name, typ, flags, addr, offset, size, link, _info, _align, _entsize in hdrs:
        out.append(_Section(_cstr(raw, names[4], name), typ, flags, addr, offset, size, link))
    return out


def _loaded(sec: _Section) -> bool:
    """Code or data that goes into memory. (The compiler also marks its symbol
    and string tables allocated, at address 0; they aren't program content.)"""
    return bool(sec.flags & SHF_ALLOC) and sec.type in (SHT_PROGBITS, SHT_NOBITS)


def _slice(raw: bytes, sec: _Section) -> bytes:
    if sec.offset + sec.size > len(raw):
        raise ElfError("section %s runs past the end of the file" % sec.name)
    return raw[sec.offset: sec.offset + sec.size]


def _cstr(raw: bytes, table: int, offset: int) -> str:
    start = table + offset
    end = raw.find(b"\0", start)
    if start >= len(raw) or end < 0:
        raise ElfError("bad string table offset %d" % offset)
    return raw[start:end].decode("ascii", "replace")


def _symbols(raw: bytes, sections: List[_Section]) -> List[Symbol]:
    out = []
    for tab in (s for s in sections if s.type == SHT_SYMTAB):
        if tab.link >= len(sections):
            raise ElfError("symbol table %s names a bad string table" % tab.name)
        strtab = sections[tab.link]
        body = _slice(raw, tab)
        for i in range(len(body) // _SYM.size):
            name_off, value, size, info, _other, shndx = _SYM.unpack_from(body, i * _SYM.size)
            if not name_off:
                continue
            name = _cstr(raw, strtab.offset, name_off)
            if shndx == SHN_UNDEF:
                raise ElfError("symbol %s is undefined: the program isn't fully linked" % name)
            if shndx >= SHN_LORESERVE or shndx >= len(sections):
                continue                                  # absolute, common: not an address
            sec = sections[shndx]
            if not _loaded(sec) or info & 0xF not in (0, STT_OBJECT, STT_FUNC):
                continue
            if sec.flags & SHF_EXECINSTR:
                kind = "function" if info & 0xF == STT_FUNC else "label"
            else:
                kind = "data"
            out.append(Symbol(name, value, kind, sec.name, size))
    return sorted(out, key=lambda s: (s.addr, s.kind != "function", s.name))


# -- loading ------------------------------------------------------------------------------

def _load_code(prog: ElfProgram, sec: _Section, body: bytes, nop: int) -> None:
    if sec.addr < BOOTSTRAP_PACKETS * PACKET_BYTE_W:
        raise ElfError("code section %s at %#x overlaps the startup packets at 0..%d"
                       % (sec.name, sec.addr, BOOTSTRAP_PACKETS * PACKET_BYTE_W - 1))
    if len(body) % INST_BYTES:
        raise ElfError("code section %s is %d bytes, not a whole number of %d-byte "
                       "instructions" % (sec.name, len(body), INST_BYTES))
    words = [int.from_bytes(body[i: i + INST_BYTES], "little")
             for i in range(0, len(body), INST_BYTES)]
    words += [nop] * (-len(words) % PACKET_SIZE)          # a partial last packet: pad
    for k in range(0, len(words), PACKET_SIZE):
        value = 0
        for w in words[k: k + PACKET_SIZE]:
            value = (value << INST_W) | w
        prog.instr[sec.addr + (k // PACKET_SIZE) * PACKET_BYTE_W] = value
    prog.packets += len(words) // PACKET_SIZE
    prog.code.append((sec.addr, sec.addr + len(body)))


def _load_data(prog: ElfProgram, sec: _Section, body: bytes) -> None:
    for i, byte in enumerate(body):
        a = sec.addr + i
        word = a & ~3
        prog.data[word] = prog.data.get(word, 0) | (byte << (8 * (a & 3)))


def _check_disjoint(prog: ElfProgram, sections: List[_Section]) -> None:
    spans = sorted((s.addr, s.addr + s.size, s.name) for s in sections if _loaded(s) and s.size)
    for (lo0, hi0, n0), (lo1, hi1, n1) in zip(spans, spans[1:]):
        if lo1 < hi0:
            raise ElfError("sections %s and %s overlap (%#x..%#x and %#x..%#x)"
                           % (n0, n1, lo0, hi0, lo1, hi1))


def _bootstrap(prog: ElfProgram, entry: str, nop: int) -> None:
    """jal x1, main at 0 and halt.s at 20, as the assembler puts them."""
    target = prog.labels[entry]
    for i, ins in enumerate([Instr("jal", ["x1", str(target)], 0), Instr("halt.s", [], 0)]):
        value = 0
        for w in [encode(ins, i * PACKET_BYTE_W)] + [nop] * (PACKET_SIZE - 1):
            value = (value << INST_W) | w
        prog.instr[i * PACKET_BYTE_W] = value
    prog.added_packets = BOOTSTRAP_PACKETS


def overlaps(ranges: List[Tuple[int, int]], addrs) -> List[int]:
    """The addresses (e.g. a kernel's input words) that fall inside `ranges`."""
    spans = sorted(ranges)
    return sorted(a for a in addrs if any(lo <= a < hi for lo, hi in spans))
