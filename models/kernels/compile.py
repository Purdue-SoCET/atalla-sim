#!/usr/bin/env python3
"""Runner compiler policy for the pinned Atalla backend.

Uses PPCI's frontend, optimizers, packetizer and ELF object format unchanged.
Configure the callee-save set to match the registers its callers assume survive.
The spatial orchestration avoids CleanPass's broken phi-edge rewrite.
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'aihw-ppci-compiler'))
from ppci import api
from ppci.arch.atalla.registers import get_register
from ppci.irutils import verify_module
from ppci.lang.atalla_c.options import COptions


def compile_source(source: Path, output: Path, level: str = '2'):
    arch = api.get_arch('atalla')
    arch.callee_save = tuple(get_register(i) for i in (9, 11, *range(18, 28)))
    options = COptions()
    options.add_include_path(ROOT)
    with source.open() as stream:
        module = api.atalla_c_to_ir(stream, arch, coptions=options)
    if source.name in ('spatial.c', 'spatial_tiles.c') and level != '0':
        # This is the normal O2 sequence with only the failing CFG cleanup
        # omitted. Preserve and verify the original phi edges instead.
        passes = [api.Mem2RegPromotor(), api.RemoveAddZeroPass(),
            api.ConstantFolder(), api.CommonSubexpressionEliminationPass(),
            api.TailCallOptimization(), api.LoadAfterStorePass(),
            api.DeleteUnusedInstructionsPass()]
        verify_module(module)
        for _ in range(3):
            for optimization in passes:
                optimization.run(module)
        verify_module(module)
    else:
        api.optimize(module, level=level)
    obj = api.ir_to_object([module], arch, packetize=True)
    with output.open('w') as stream:
        obj.save(stream)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('-o', required=True, type=Path)
    parser.add_argument('-O', default='2', choices=('0','1','2','3'))
    args = parser.parse_args()
    compile_source(args.source, args.o, args.O)
