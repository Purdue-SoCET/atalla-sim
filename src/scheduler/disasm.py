"""Packets as text, for diagnostics: `addi.s x5, x5, 16 | vreg.ld v3, x9 [pad 1]`.

Only for people reading reports (hang diagnosis, the dependency checker, the
per-PC profile); the format is not meant to be parsed or reassembled.
"""

from typing import List

from scheduler.control import ScalarOp, SdmaOp, VectorOp


def format_op(o) -> str:
    regs: List[str] = []
    if isinstance(o, ScalarOp):
        if o.reg_write or o.mask_reg_write:
            regs.append(("m%d" if o.mask_reg_write else "x%d") % o.rd)
        regs += ["x%d" % r for r in o.reads()]
        if o.imm_src:
            regs.append(str(o.imm))
    elif isinstance(o, VectorOp):
        for r, use in ((o.vd, o.vector_reg_write),):
            if use:
                regs.append("v%d" % r)
        if o.scalar_reg_write:
            regs.append("x%d" % o.rd)
        if o.mask_reg_write:
            regs.append("m%d" % o.vmd)
        regs += ["v%d" % r for r in o.vector_reads()]
        regs += ["x%d" % r for r in o.scalar_reads()]
        if o.use_vms:
            regs.append("m%d" % o.vms)
    elif isinstance(o, SdmaOp):
        regs += ["x%d" % r for r in o.reads()]
    return ("%s %s" % (o.mnemonic, ", ".join(regs))).strip()


def format_packet(pkt) -> str:
    """Every valid op of a DecodedPacket, scalar then vector then SDMA."""
    s, v, d = pkt.valid_ops()
    ops = [format_op(o) for o in (*s, *v, *d)]
    return " | ".join(ops) if ops else "(empty)"
