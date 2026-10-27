"""Assembler: the compiler's packetized assembly to a program image.

The C compiler (third_party/aihw-ppci-compiler, `atalla_cc -p -S`) already
groups instructions into VLIW packets: every four instruction lines are one
packet, slot 0 first, padded with `nop`, and every label starts a new packet.
This module turns that text into the image the scheduler fetches -- packets
of four 40-bit words at 20-byte addresses -- without moving anything: the
compiler's packets are the packets that run.

    program = assemble(open("kernel.s").read())
    program.instr        {packet address: 160-bit packet}, for SchedulerCore
    program.labels       {label: packet address}
    to_in_text(program.instr, data)   the functional sim's .in format, so it
                                      can run the identical image as a check

What it does besides encoding, and why:

- **Bootstrap.** Two packets go first: `jal x1, main` and `halt.s`, so the
  program calls main and halts when main returns (as the functional sim's
  assembler does). main starts at address 40.
- **li.s is expanded**: it is a pseudo-instruction the hardware never sees.
  An immediate that fits addi.s's 12 bits becomes `addi.s rd, x0, imm` in the
  same slot. A larger one becomes `lui.s rd, imm >> 7` in the same slot and
  `addi.s rd, rd, imm & 0x7F` in a packet inserted right after. That is safe
  because the compiler never puts an instruction in the same packet as one
  whose result it reads.
- **Padding is dropped.** The compiler's printer puts a few `nop`s before the
  first label and after the last packet for alignment; a partial packet of
  nops is discarded. A partial packet with anything else is an error.

Encoding is decode's inverse: each instruction's operands, in order, fill its
type's fields in scheduler/isa.py's FIELDS. Memory operands are `imm(xN)`;
branch and jal targets are labels (or byte offsets), encoded as instruction
counts from the packet: the byte offset divided by 5.
To support a new instruction: add it to isa.OPCODES (and FIELDS, for a new
type); nothing here needs to change unless its assembly syntax is unusual.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from scheduler.isa import FIELDS, INST_BYTES, INST_W, OPCODES, PACKET_BYTE_W, PACKET_SIZE

#: mnemonic -> (opcode, type)
_BY_NAME = {name: (op, kind) for op, (name, kind) in OPCODES.items()}
#: Lines that carry no instruction: section, alignment and symbol directives.
_DIRECTIVES = ("global", "type")
_REG = re.compile(r"^\$?[xvm]?(\d+)$")
_MEM = re.compile(r"^(-?(?:0x[0-9a-fA-F]+|\d+))\((.+)\)$")
_ENTRY = "main"
#: Packets placed before the program: call main, then halt when it returns.
BOOTSTRAP_PACKETS = 2


class AssemblyError(ValueError):
    """An instruction the assembler can't encode, with its source line."""


@dataclass
class Instr:
    mnemonic: str              # normalised: "addi.s", "vreg.ld", ...
    ops: List[str]
    line: int                  # source line, for error messages


@dataclass
class Program:
    instr: Dict[int, int] = field(default_factory=dict)
    labels: Dict[str, int] = field(default_factory=dict)
    #: Packets the compiler emitted, and packets added (bootstrap, li.s).
    packets: int = 0
    added_packets: int = 0


# -- parsing ------------------------------------------------------------------------------

def _mnemonic(raw: str) -> str:
    """`addi_s` -> `addi.s`, `vreg_ld` -> `vreg.ld`, `halt` -> `halt.s`."""
    m = raw.strip().lower().replace("_", ".")
    if m not in _BY_NAME and m + ".s" in _BY_NAME:
        m += ".s"
    return m


def parse(text: str) -> List[object]:
    """The source as a list of label names (str) and Instr, in order."""
    items: List[object] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        code = re.split(r"[#;]", line, maxsplit=1)[0].strip()
        if not code or code.startswith("."):
            continue
        if code.endswith(":"):
            items.append(code[:-1].strip())
            continue
        head, _, rest = code.partition(" ")
        if head in _DIRECTIVES:
            continue
        ops = [o.strip() for o in rest.split(",") if o.strip()]
        items.append(Instr(_mnemonic(head), ops, lineno))
    return items


def _packetize(items: List[object]) -> Tuple[List[List[Instr]], Dict[str, int]]:
    """Group into the compiler's packets; labels map to packet indices."""
    packets: List[List[Instr]] = []
    labels: Dict[str, int] = {}
    cur: List[Instr] = []

    def drop_padding():
        if any(i.mnemonic != "nop.s" for i in cur):
            raise AssemblyError("line %d: a packet of %d instructions (packets are %d): "
                                "was this compiled with -p?"
                                % (cur[0].line, len(cur), PACKET_SIZE))
        cur.clear()

    for item in items:
        if isinstance(item, str):
            drop_padding()
            labels[item] = len(packets)
            continue
        cur.append(item)
        if len(cur) == PACKET_SIZE:
            packets.append(cur)
            cur = []
    drop_padding()
    return packets, labels


# -- li.s ---------------------------------------------------------------------------------

def _fits(value: int, bits: int) -> bool:
    return -(1 << (bits - 1)) <= value < (1 << (bits - 1))


def _expand_li(packets: List[List[Instr]], labels: Dict[str, int]):
    """Replace li.s as the module docstring describes. Returns the new packet
    list, labels moved past inserted packets, and how many were inserted."""
    out: List[List[Instr]] = []
    old_to_new: Dict[int, int] = {}
    for idx, pkt in enumerate(packets):
        old_to_new[idx] = len(out)
        follow: List[Instr] = []
        new_pkt = []
        for ins in pkt:
            if ins.mnemonic != "li.s":
                new_pkt.append(ins)
                continue
            if len(ins.ops) != 2:
                raise AssemblyError("line %d: li.s takes rd, imm" % ins.line)
            rd, imm = ins.ops[0], _int(ins.ops[1], ins)
            if _fits(imm, 12):
                new_pkt.append(Instr("addi.s", [rd, "x0", str(imm)], ins.line))
            else:
                imm &= 0xFFFFFFFF
                new_pkt.append(Instr("lui.s", [rd, str(imm >> 7)], ins.line))
                follow.append(Instr("addi.s", [rd, rd, str(imm & 0x7F)], ins.line))
        out.append(new_pkt)
        # One addi.s per inserted packet, so two expansions never share one.
        for ins in follow:
            out.append([ins] + [Instr("nop.s", [], ins.line)] * (PACKET_SIZE - 1))
    old_to_new[len(packets)] = len(out)
    moved = {name: old_to_new[i] for name, i in labels.items()}
    return out, moved, len(out) - len(packets)


# -- encoding -----------------------------------------------------------------------------

def _int(text: str, ins: Instr) -> int:
    try:
        return int(text, 0)
    except ValueError:
        raise AssemblyError("line %d: %s: not a number: %r" % (ins.line, ins.mnemonic, text))


def _reg(text: str, ins: Instr) -> int:
    m = _REG.match(text.strip())
    if not m:
        raise AssemblyError("line %d: %s: not a register: %r" % (ins.line, ins.mnemonic, text))
    return int(m.group(1))


def _place(word: int, value: int, high: int, low: int, signed: Optional[int], ins: Instr,
           name: str) -> int:
    width = high - low + 1
    ok = _fits(value, width) if signed else 0 <= value < (1 << width)
    if not ok:
        raise AssemblyError("line %d: %s: %s=%d does not fit %d bits"
                            % (ins.line, ins.mnemonic, name, value, width))
    return word | ((value & ((1 << width) - 1)) << low)


def _target(text: str, pc: int, labels: Dict[str, int], ins: Instr) -> int:
    """A label (or a number) as a byte offset from this packet."""
    if text in labels:
        return labels[text] - pc
    try:
        return int(text, 0)
    except ValueError:
        raise AssemblyError("line %d: %s: unknown label %r" % (ins.line, ins.mnemonic, text))


def _instructions(off: int, bits: int, ins: Instr) -> int:
    """A byte offset as the signed instruction count a branch or jal holds."""
    count, rest = divmod(off, INST_BYTES)
    if rest or not _fits(count, bits):
        raise AssemblyError("line %d: %s: offset %d bytes out of range or not a whole "
                            "number of instructions (+-%d bytes, in steps of %d)"
                            % (ins.line, ins.mnemonic, off, (1 << (bits - 1)) * INST_BYTES,
                               INST_BYTES))
    return count


def encode(ins: Instr, pc: int = 0, labels: Optional[Dict[str, int]] = None) -> int:
    """One instruction to its 40-bit word. `pc` is its packet's address and
    `labels` label addresses, for branch and jal targets."""
    labels = labels or {}
    if ins.mnemonic not in _BY_NAME:
        raise AssemblyError("line %d: unknown instruction %r" % (ins.line, ins.mnemonic))
    opcode, kind = _BY_NAME[ins.mnemonic]
    word = opcode
    ops = list(ins.ops)

    if kind == "BR":
        # rs1, rs2, target[, incr]: the offset counts instructions, {[14], [39:31]}.
        if len(ops) not in (3, 4):
            raise AssemblyError("line %d: %s takes rs1, rs2, label" % (ins.line, ins.mnemonic))
        v = _instructions(_target(ops[2], pc, labels, ins), 10, ins) & 0x3FF
        word |= (v >> 9) << 14 | (v & 0x1FF) << 31
        word |= _reg(ops[0], ins) << 15 | _reg(ops[1], ins) << 23
        if len(ops) == 4:
            word = _place(word, _int(ops[3], ins), 13, 7, None, ins, "incr")
        return word

    layout = FIELDS.get(kind, ())
    if kind == "M":
        # lw.s rd, imm(rs1) / sw.s rs2, imm(rs1): the first register goes in rd's field.
        m = _MEM.match(ops[1].replace(" ", "")) if len(ops) == 2 else None
        if not m:
            raise AssemblyError("line %d: %s takes reg, imm(base)" % (ins.line, ins.mnemonic))
        ops = [ops[0], m.group(2), m.group(1)]
    if ins.mnemonic in ("halt.s", "nop.s") and not ops:
        return word
    if ins.mnemonic == "jal":
        ops = ([ops[0], str(_instructions(_target(ops[1], pc, labels, ins), 25, ins))]
               if len(ops) == 2 else ops)
    if len(ops) != len(layout):
        raise AssemblyError("line %d: %s takes %d operands (%s), got %d"
                            % (ins.line, ins.mnemonic, len(layout),
                               ", ".join(f[0] for f in layout), len(ops)))
    for (name, high, low, signed), text in zip(layout, ops):
        is_value = name.startswith("imm") or name in ("num_cols", "sid")
        value = _int(text, ins) if is_value else _reg(text, ins)
        if ins.mnemonic == "lui.s":
            signed = None                    # rd = imm << 7: all 25 bits unsigned
        word = _place(word, value, high, low, signed, ins, name)
    return word


def assemble(text: str) -> Program:
    """Packetized assembly (atalla_cc -p -S) to a program image."""
    packets, labels = _packetize(parse(text))
    if _ENTRY not in labels:
        raise AssemblyError("no %r label: nothing to call" % _ENTRY)
    packets, labels, inserted = _expand_li(packets, labels)
    boot = [[Instr("jal", ["x1", _ENTRY], 0)], [Instr("halt.s", [], 0)]]
    packets = boot + packets
    addr = {name: (i + BOOTSTRAP_PACKETS) * PACKET_BYTE_W for name, i in labels.items()}

    prog = Program(labels=addr, packets=len(packets) - BOOTSTRAP_PACKETS - inserted,
                   added_packets=BOOTSTRAP_PACKETS + inserted)
    nop = encode(Instr("nop.s", [], 0))
    for i, pkt in enumerate(packets):
        pc = i * PACKET_BYTE_W
        words = [encode(ins, pc, addr) for ins in pkt]
        words += [nop] * (PACKET_SIZE - len(words))
        value = 0
        for w in words:
            value = (value << INST_W) | w
        prog.instr[pc] = value
    return prog


def to_in_text(instr: Dict[int, int], data: Optional[Dict[int, int]] = None) -> str:
    """The functional sim's .in format (scheduler.core.load_program_text reads it)."""
    hex_w = INST_W // 4
    lines = []
    for addr in sorted(instr):
        words = [(instr[addr] >> (INST_W * (PACKET_SIZE - 1 - s))) & ((1 << INST_W) - 1)
                 for s in range(PACKET_SIZE)]
        lines.append("%08X: %s" % (addr, " ".join("%0*X" % (hex_w, w) for w in words)))
    lines.append(".data")
    for addr in sorted(data or {}):
        lines.append("%08X: %08X" % (addr, data[addr] & 0xFFFFFFFF))
    return "\n".join(lines) + "\n"
