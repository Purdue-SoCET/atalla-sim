#!/usr/bin/env python3
"""Freeze a built model ELF as a reference program for atalla-sim's tests.

    python freeze.py basic            # after build_model.py / atalla-sim models build basic

writes, from models/out/<model>:

    tests/equivalence/programs/model_<model>.c.in.gz   the ELF as a program image, which
                                                        atalla-sim bench and gallery run
    tests/models/model_<model>.expected.json           every checked result's DRAM address,
                                                        shape and BF16 PyTorch value

so the model can be run and checked without torch or the compiler
(tests/models/test_frozen_models.py). The compiler doesn't emit the same
code on every run, so a frozen image also keeps the cycle baseline steady.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT / "atalla-graph"), str(REPO / "src")]


def bf16_bits(a):
    """Round to nearest even BF16 and return the 16-bit patterns."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", choices=("basic", "alexnet_small"))
    ap.add_argument("--out-dir", type=Path, help="the build (default models/out/MODEL)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=.01)
    args = ap.parse_args(argv)
    out_dir = (args.out_dir or ROOT / "out" / args.model).resolve()
    import verify_model
    from scheduler.assembler import to_in_text
    from scheduler.elf import load_elf
    elf = out_dir / "output.elf"
    program = load_elf(elf)
    name = "model_%s.c" % args.model
    image = REPO / "tests" / "equivalence" / "programs" / (name + ".in.gz")
    image.write_bytes(gzip.compress(to_in_text(program.instr, program.data).encode(), mtime=0))
    results = []
    for node, operation, info, reference in verify_model.expected_results(
            args.model, out_dir, args.seed, args.scale):
        results.append(dict(node=node, operation=operation, address=info["address"],
                            shape=list(info["shape"]),
                            bf16=bf16_bits(reference).ravel().tolist()))
    import hashlib
    doc = dict(model=args.model, seed=args.seed, scale=args.scale, program=image.name,
               elf_sha256=hashlib.sha256(elf.read_bytes()).hexdigest(),
               atol=.01, rtol=.03, results=results)
    expected = REPO / "tests" / "models" / ("model_%s.expected.json" % args.model)
    expected.parent.mkdir(exist_ok=True)
    expected.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print("%s and %s (%d results)" % (image.relative_to(REPO), expected.relative_to(REPO),
                                      len(results)))


if __name__ == "__main__":
    main()
