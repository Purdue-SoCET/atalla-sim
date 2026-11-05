# Atalla graph kernels

This directory is the shared, callable kernel package for `atalla-model-runner`.
Kernel code here is independent of FX/PyTorch. The runner owns graph lowering,
node ordering, addresses, and per-node descriptors; this package owns the graph
kernel ABI and Atalla-specific tile transfers and compute.

## Layout

- `kernels.h`: public descriptor and function ABI used by generated schedules.
- `include/sdma.h`: shared SDMA control-word field encoding and tile presets.
- `src/elementwise.c`: shared add, multiply, scalar multiply, and ReLU tile kernels.
- `prompt.md`: compiler-team workflow for analyzing compiler and validation-kernel changes.

The elementwise routines consume padded row-major 32x32 BF16 tiles, matching the
current runner allocator. Add and multiply require matching shapes/tile grids; add also supports a rank-1
right-hand bias matching the output’s trailing dimension. Scalar multiply and ReLU
require matching input/output shapes. ReLU ignores its optional bias argument. Each function returns
zero on success and a negative value when the descriptor tile counts disagree.

## Build

From the repository root, compile the current library object with:

```bash
make -C kernels
```

The object is written under the ignored `kernels/build/` directory. This is a
compile-only step; graph-level golden validation is not wired to the package yet. Override
`PYTHON`, `COMPILER`, or `BUILD_DIR` for a local compiler checkout or build path.

## Compiler-team workflow

Kernel sources are ordinary Atalla C and can be optimized independently from the
FX graph pipeline. Keep public signatures in `kernels.h` stable; record a deliberate
ABI change there and update the generated schedule and manifest together. Preserve
a clear baseline implementation when adding a specialized variant, and document the
variant's required tile layout and geometry beside its dispatch entry.

The current compiler validation programs remain standalone goldens and optimization
references. They are not included into this library and are not copied per graph
node. The package currently implements elementwise kernels; matrix, convolution,
and pooling declarations remain unavailable until their graph tile/data-layout
contracts are implemented. The reviewed compiler revision is tracked in `COMPILER_BASELINE` and
`capabilities.json`; the compiler submodule remains unchanged. The runner manifest must keep those operations marked
unsupported so an ELF build cannot silently resolve them to placeholder behavior.
