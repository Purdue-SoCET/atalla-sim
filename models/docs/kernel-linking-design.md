# Graph Kernel Linking Design

**Intent:** Compile one statically scheduled model graph into one Atalla ELF, using the optimized compiler validation kernels and carrying the graph's DRAM image with the build output.

**Goals:** Keep one shared implementation per operation type; bind graph-specific addresses, tensor shapes, tile counts, and operation parameters at graph build time; resolve function and global relocations in one link; make the DRAM and config address map explicit; do all implementation work in `atalla-model-runner`.

**Non-goals:** Modify or commit the compiler submodule; emit a separate C kernel implementation for every graph node; imply that arbitrary graph shapes are supported when a selected kernel template cannot represent them.

## Approaches considered

1. **Compile existing validation sources unchanged.** Rejected: each source is an executable with `main()` and `atalla_halt()`, hard-coded geometry, and a config table at `0x3c`; it cannot serve the `GlobalTile*` graph-call ABI.
2. **Generate one modified kernel file per graph node.** Rejected: repeats kernel bodies and creates the same maintenance problem as `add_1.c`, `add_2.c`, etc.
3. **Shared graph kernel package in the runner repository.** Selected: extract and generalize validated Atalla kernel logic into `kernels/`, define a callable descriptor ABI, and keep per-node addresses/shapes in graph descriptors. Compiler validation programs remain optimization and correctness references. The package is isolated so compiler contributors can optimize it without working in FX or allocator code.

## Design

The graph pass retains `GlobalTile` metadata and emits calls through `kernels/kernels.h`. The library owns tile traversal, SDMA controls, and kernel computation; the runner owns graph ordering, address allocation, and data packaging. Compiler validation programs stay standalone and unchanged. The first library implementations cover elementwise add, multiply, scalar multiply, and ReLU over padded 32x32 BF16 tiles. Matrix, convolution, and pooling calls remain unavailable until their algorithms are adapted to this ABI and their data layouts are supported. `kernel_backend.py` records capabilities and prevents an incomplete operation set from being reported linkable.

A future build tool must emit graph code, the selected shared kernel translation units, and the model DRAM payload, then compile/link them with the selected `aihw-ppci-compiler` revision. Before linking, the graph allocator and linker layout must agree on a non-overlapping config and tensor address map. The ELF and standalone `dram.bin` must represent the same payload and address map until the ELF loader path is confirmed end to end.

## Constraints and risks

- The latest `atalla-arch` checkout reports all 12 validation kernels passing numerical validation, but the test files remain validation-sized rather than a generic kernel library. Their fixed SDMA controls and tile geometry cannot be made graph-callable by renaming `main` or changing config words. A generic implementation and explicit layout contract are required; unsupported operation geometry/layout must be rejected.
- GEMM expects weight/layout and SDMA strides that may differ from the graph's tiled representation; convolution tests may expect im2col-like inputs. These must be explicitly reconciled.
- The graph allocator currently assigns monotonically increasing 2 KiB tiles from address zero, while the report describes liveness reuse. The config table at `0x3c`, graph tensors, and compiler `.data` at `0x80000` must not overlap.
- The current schedule generator calls generic `*_kernel` symbols, while the old `xmas` per-node files export `*_main`; linking objects alone does not connect them.
