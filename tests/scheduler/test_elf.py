"""The ELF loader (src/scheduler/elf.py).

Most tests build small ELFs here, laid out as the compiler's linker writes
them (data, code, .symtab, .strtab as the section-name table too), so they
don't depend on the compiler. The last ones link real programs with the
compiler and run them, when it is checked out.
"""
import shutil
import struct
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from atalla.atalla_platform import build_atalla_platform, c_entry_sregs
from scheduler.assembler import Instr, encode
from scheduler.elf import (
    EM_ATALLA, ElfError, ElfProgram, SHF_ALLOC, SHF_EXECINSTR, STT_FUNC, STT_OBJECT, load_elf,
    overlaps)
from scheduler.isa import INST_W, decode_instruction, decode_packet

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import c_kernels  # noqa: E402

NOP = encode(Instr("nop.s", [], 0))
HALT = encode(Instr("halt.s", [], 0))
SHF_WRITE = 0x1


def ins(text, pc=0, labels=None):
    head, _, rest = text.partition(" ")
    return encode(Instr(head, [o.strip() for o in rest.split(",") if o.strip()], 0), pc,
                  labels)


def code_bytes(*words):
    return b"".join(w.to_bytes(5, "little") for w in words)


def make_elf(code=(0x1004, b""), data=(0x80000, b""), symbols=(), *, e_type=1,
             machine=EM_ATALLA, ei_class=1, ei_data=1, extra_sections=()):
    """An ELF as ppci's linker writes it. `symbols`: (name, value, type, section)
    with section "code", "data" or None (undefined)."""
    strtab = bytearray(b"\0")

    def name(s):
        off = len(strtab)
        strtab.extend(s.encode() + b"\0")
        return off

    sections = [("", 0, 0, 0, b"", 0)]                        # name, type, flags, addr, body, link
    sections.append(("data", 1, SHF_ALLOC | SHF_WRITE, data[0], data[1], 0))
    sections.append(("code", 1, SHF_ALLOC | SHF_EXECINSTR, code[0], code[1], 0))
    sections.extend(extra_sections)
    index = {s[0]: i for i, s in enumerate(sections)}
    sym = bytearray(16)
    for sname, value, typ, sec in symbols:
        shndx = index[sec] if sec else 0
        sym += struct.pack("<IIIBBH", name(sname), value, 0, (1 << 4) | typ, 0, shndx)
    # ppci marks its symbol and string tables allocated (flag A), at address 0.
    sections.append((".symtab", 2, SHF_ALLOC, 0, bytes(sym), len(sections) + 1))
    sections.append((".strtab", 3, SHF_ALLOC, 0, None, 0))
    names = [name(s[0]) if s[0] else 0 for s in sections]
    sections[-1] = sections[-1][:4] + (bytes(strtab),) + (0,)

    out = bytearray(52)
    offsets = []
    for s in sections:
        offsets.append(len(out))
        out += s[4]
    shoff = len(out)
    for (sname, typ, flags, addr, body, link), off, n in zip(sections, offsets, names):
        out += struct.pack("<10I", n, typ, flags, addr, off if body else 0, len(body), link,
                           0, 4, 16 if typ == 2 else 0)
    ident = b"\x7fELF" + bytes([ei_class, ei_data, 1]) + bytes(9)
    out[:52] = ident + struct.pack("<HHIIIIIHHHHHH", e_type, machine, 1, 0, 0, shoff, 0, 52,
                                   0, 0, 40, len(sections), len(sections) - 1)
    return bytes(out)


def slots(packet):
    return [(packet >> (INST_W * (3 - k))) & ((1 << INST_W) - 1) for k in range(4)]


# -- reading the file ---------------------------------------------------------------------

def simple(**kw):
    words = [ins("addi.s x5, x0, 7"), NOP, NOP, NOP, ins("jalr x0, x1, 0"), NOP, NOP, NOP]
    return make_elf(code=(0x1004, code_bytes(*words)),
                    symbols=[("main", 0x1004, STT_FUNC, "code"),
                             ("main_epilog", 0x1018, STT_OBJECT, "code")], **kw)


def test_packets_put_slot_0_at_the_lowest_address_and_the_top_bits():
    a, b, c, d = (ins("addi.s x%d, x0, %d" % (r, r)) for r in (5, 6, 7, 8))
    prog = load_elf(make_elf(code=(0x1004, code_bytes(a, b, c, d)),
                             symbols=[("main", 0x1004, STT_FUNC, "code")]))
    assert slots(prog.instr[0x1004]) == [a, b, c, d]
    assert [x["rd"] for x in decode_packet(prog.instr[0x1004])] == [5, 6, 7, 8]


@pytest.mark.parametrize("bad,why", [
    (dict(machine=0xF3), "machine 0xf3"),
    (dict(ei_class=2), "32-bit"),
    (dict(ei_data=2), "little-endian"),
    (dict(e_type=3), "ELF type 3"),
])
def test_the_header_is_checked(bad, why):
    with pytest.raises(ElfError, match=why):
        load_elf(simple(**bad))


def test_an_executable_is_accepted_too():
    assert load_elf(simple(e_type=2)).entry == 0x1004


def test_not_an_elf():
    with pytest.raises(ElfError, match="not an ELF"):
        load_elf(b"{\n  \"arch\": \"atalla\"\n}")      # the compiler's .o is JSON


def test_main_is_called_at_0_and_halts_on_return():
    prog = load_elf(simple())
    jal, halt = decode_instruction(slots(prog.instr[0])[0]), slots(prog.instr[20])
    # jal counts instructions: 0x1004 is 820 of them from address 0.
    assert (jal["mnemonic"], jal["rd"], jal["imm"]) == ("jal", 1, 0x1004 // 5)
    assert halt == [HALT, NOP, NOP, NOP]
    assert sorted(prog.instr) == [0, 20, 0x1004, 0x1018]
    assert (prog.packets, prog.added_packets) == (2, 2)


def test_it_runs():
    prog = load_elf(simple())
    plat = build_atalla_platform(prog.instr, prog.data, init_sregs=c_entry_sregs(prog.data))
    plat.run_until_done(limit=2000)
    assert plat.core.scalar_reg(5) == 7


def test_data_becomes_little_endian_words_and_a_partial_word_is_zero_filled():
    prog = load_elf(make_elf(code=(0x1004, code_bytes(HALT, NOP, NOP, NOP)),
                             data=(0x80000, bytes([1, 2, 3, 4, 5, 6])),
                             symbols=[("main", 0x1004, STT_FUNC, "code"),
                                      ("table", 0x80000, STT_OBJECT, "data")]))
    assert prog.data == {0x80000: 0x04030201, 0x80004: 0x0605}
    assert [(s.name, s.kind) for s in prog.symbols] == [("main", "function"),
                                                       ("table", "data")]
    assert "table" not in prog.labels


def test_a_zero_filled_section_is_loaded_as_zeros():
    bss = ("bss", 8, SHF_ALLOC | SHF_WRITE, 0x90000, b"", 0)
    raw = bytearray(make_elf(code=(0x1004, code_bytes(HALT, NOP, NOP, NOP)),
                             symbols=[("main", 0x1004, STT_FUNC, "code")],
                             extra_sections=[bss]))
    # NOBITS has no body in the file; give it a size of 8 in its header.
    shoff, = struct.unpack_from("<I", raw, 0x20)
    struct.pack_into("<I", raw, shoff + 3 * 40 + 20, 8)
    assert load_elf(bytes(raw)).data == {0x90000: 0, 0x90004: 0}


def test_a_partial_last_packet_is_padded_with_nops():
    prog = load_elf(make_elf(code=(0x1004, code_bytes(HALT)),
                             symbols=[("main", 0x1004, STT_FUNC, "code")]))
    assert slots(prog.instr[0x1004]) == [HALT, NOP, NOP, NOP]


def test_symbols_name_pcs():
    prog = load_elf(simple())
    assert prog.labels == {"main": 0x1004, "main_epilog": 0x1018}
    assert [(s.name, s.kind) for s in prog.symbols] == [("main", "function"),
                                                       ("main_epilog", "label")]
    assert prog.describe(0x1004) == "main"
    assert prog.describe(0x1018 + 10) == "main_epilog+10"
    assert prog.describe(0x102c) == "0x102c"                 # past the end of the code
    assert prog.describe(20) == "0x14"                       # the startup packets


@pytest.mark.parametrize("elf,why", [
    (make_elf(code=(0x1004, code_bytes(*[NOP] * 8)), symbols=[("main", 0x1009, STT_FUNC, "code")]),
     "not at a packet boundary"),
    (make_elf(code=(0x1004, code_bytes(NOP) + b"\0")), "whole number of 5-byte"),
    (make_elf(code=(0, code_bytes(*[NOP] * 4)), symbols=[("main", 0, STT_FUNC, "code")]),
     "overlaps the startup packets"),
    (make_elf(code=(0x1004, code_bytes(*[NOP] * 4)), symbols=[("other", 0x1004, STT_FUNC, "code")]),
     "no 'main'"),
    (make_elf(code=(0x1004, code_bytes(*[NOP] * 4)),
              symbols=[("main", 0x1004, STT_FUNC, "code"), ("printf", 0, STT_FUNC, None)]),
     "undefined"),
    (make_elf(code=(0x1004, code_bytes(*[NOP] * 4)), data=(0x1008, b"\1\2\3\4"),
              symbols=[("main", 0x1004, STT_FUNC, "code")]),
     "overlap"),
])
def test_programs_it_cant_run_are_refused(elf, why):
    with pytest.raises(ElfError, match=why):
        load_elf(elf)


def test_overlaps_finds_inputs_inside_the_code():
    assert overlaps([(0x1004, 0x15b8)], [0x3c, 0x1000, 0x1004, 0x1400, 0x15b8]) == [0x1004,
                                                                                   0x1400]


# -- with the compiler --------------------------------------------------------------------

needs_compiler = pytest.mark.skipif(not (c_kernels.COMPILER / "ppci").is_dir(),
                                    reason="compiler submodule not checked out")


@pytest.fixture(scope="module")
def sum_loop_elf():
    try:
        return c_kernels.compile_elf(c_kernels.UNIT_TESTS / "scalar_sum_loop.c")
    except c_kernels.CompileError as exc:               # pragma: no cover - env dependent
        pytest.skip(str(exc))


@needs_compiler
def test_a_linked_program_runs_and_gives_its_result(sum_loop_elf):
    path = c_kernels.UNIT_TESTS / "scalar_sum_loop.c"
    prog = load_elf(sum_loop_elf)
    assert isinstance(prog, ElfProgram) and prog.entry == prog.labels["main"]
    plat = build_atalla_platform(prog.instr, prog.data, init_sregs=c_entry_sregs(prog.data))
    plat.run_until_done(limit=100_000)
    ok, message = c_kernels.check(path, plat)
    assert ok, message


@needs_compiler
@pytest.mark.skipif(shutil.which("readelf") is None, reason="no readelf")
def test_sections_and_symbols_agree_with_readelf(sum_loop_elf, tmp_path):
    path = tmp_path / "k.elf"
    path.write_bytes(sum_loop_elf)
    out = subprocess.run(["readelf", "-W", "-S", "-s", str(path)], capture_output=True,
                         text=True).stdout
    prog = load_elf(sum_loop_elf)
    for lo, hi in prog.code:
        assert ("%08x" % lo) in out and ("%06x" % (hi - lo)) in out
    for s in prog.symbols:
        assert ("%08x" % s.addr) in out and s.name in out


@needs_compiler
def test_a_kernels_inputs_on_its_code_are_reported():
    if c_kernels.harness() is None:
        pytest.skip("no harness: %s" % c_kernels.harness_error)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        program, data, regs = c_kernels.build(c_kernels.KERNELS / "relu.c", elf=True)
    assert any(issubclass(w.category, c_kernels.ElfLayoutWarning) and "inside the code"
               in str(w.message) for w in caught)
    plat = build_atalla_platform(program.instr, data, init_sregs=regs,
                                 describe_pc=program.describe)
    plat.run_until_done(limit=200_000)
    ok, message = c_kernels.check(c_kernels.KERNELS / "relu.c", plat)
    assert ok, message
