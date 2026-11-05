# Graph Kernel Linking Design

**Intent:** Compile one statically scheduled model graph into one Atalla ELF, using the optimized compiler validation kernels and carrying the graph's DRAM image with the build output.

**Goals:** Keep one shared implementation per operation type; bind graph-specific addresses, tensor shapes, tile counts, and operation parameters at graph build time; resolve function and global relocations in one link; make the DRAM and config address map explicit; do all implementation work in `atalla-model-runner`.

**Non-goals:** Modify or commit the compiler submodule; emit a separate C kernel implementation for every graph node; imply that arbitrary graph shapes are supported when a selected kernel template cannot represent them.

## Approaches considered

1. **Compile existing validation sources unchanged.** Rejected: each source is an executable with `main()` and `atalla_halt()`, hard-coded geometry, and a config table at `0x3c`; it cannot serve the `GlobalTile*` graph-call ABI.
2. **Generate one modified kernel file per graph node.** Rejected: repeats kernel bodies and creates the same maintenance problem as `add_1.c`, `add_2.c`, etc.
3. **Runner-owned shared adapter ABI.** Selected: use the compiler validation files as the canonical kernel sources, adapt each source once per operation into a callable entry point, and pass per-node addresses and shape/control data through graph-owned config records. The graph schedule contains calls/configuration, not per-node kernel bodies. Add compatibility checks for kernel geometry and layout.

## Design

The graph pass retains `GlobalTile` metadata (logical shape, tile grid, tile addresses) and lowers each supported node to a typed invocation record. A shared entry point for each operation consumes records and routes to the selected optimized implementation. The adapter isolates the test-kernel ABI (`main`, fixed config base, shape constants) from the graph ABI; the compiler repository remains unchanged.

The build tool emits graph code, adapter translation units, and the model DRAM payload, compiles them with the pinned `aihw-ppci-compiler`, and links one ELF. The linker/data layout reserves the fixed config area and places graph payload at a documented base; generated references must be relocatable or derived from linker symbols. The ELF and standalone `dram.bin` must represent the same payload and address map until the ELF loader path is confirmed end to end.

## Constraints and risks

- Current test files are validation-sized, not a complete generic kernel library. The adapter must reject unsupported operation geometry/layout instead of silently generating incorrect code.
- GEMM expects weight/layout and SDMA strides that may differ from the graph's tiled representation; convolution tests may expect im2col-like inputs. These must be explicitly reconciled.
- The graph allocator currently assigns monotonically increasing 2 KiB tiles from address zero, while the report describes liveness reuse. The config table at `0x3c`, graph tensors, and compiler `.data` at `0x80000` must not overlap.
- The current schedule generator calls generic `*_kernel` symbols, while the old `xmas` per-node files export `*_main`; linking objects alone does not connect them.
