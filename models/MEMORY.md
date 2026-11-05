# Atalla Model Runner Memory

Living codebase notes for the graph-to-ELF work. Update this file as new behavior is confirmed.

## Repository boundaries

- `atalla-model-runner` is the active implementation repository. Keep commits here only.
- `aihw-ppci-compiler` is a submodule pinned at `fda4960406fe17ca2a2379be20ea875462b321d2` (`gemm 512, 1024 variants`). Do not commit changes in that repository.
- Current runner support is schedule emission only: `atalla-graph/run_schedule.py` writes `graph_schedule.c` and `dram.bin`.
- The runner has no `codegen/c_emitter.py` or per-op kernel builders. In `atalla-models`, the per-node `.c` generation used by validation is in `atalla-graph/codegen/c_emitter.py`; its C builders are loaded from `functional_sim/kernels`.

## Current graph schedule and memory

- `run_schedule.py`: converts model to BF16/eval, FX traces, lowers Linear to matmul/add, performs shape propagation, normalizes ops, plans tiles, allocates DRAM, and calls `scripts/generate_schedule.emit`.
- `graph/memoryallocator.py`: `TILE_HEIGHT = TILE_WIDTH = 32`; one BF16 tile is 2,048 bytes. Tensor bytes are laid out as row-major 32x32 tiles, with edge padding. Current runner allocator assigns monotonically increasing addresses from zero; it does not implement the liveness reuse described in the AtallaGraph report.
- `generate_schedule.py`: reads each node's `dram_addr`, `bytes`, BF16 shape and tile counts. It emits C `GlobalTile` descriptors whose `tiles[]` contain the DRAM base address of each tile. The descriptor also records logical shape and tiles-per-dimension. A view/transpose may alias its source address.
- Schedule calls are graph operations such as `matmul_kernel(&A, &B, &C)` and `add_kernel(&lhs, &rhs, &dst, alpha)`. These are generic graph-kernel APIs, not entry points in `test_validation_kernels`.
- The per-node C files in `../xmas` have config loads from address `0x3c` (decimal 60). Their config words are per-kernel runtime GMEM addresses and dimensions; constants such as SDMA control values, row/column bounds and loop counts encode the selected memory geometry.

## Existing standalone compiler kernels

`aihw-ppci-compiler/test_validation_kernels/*.c` are validation programs, not a linkable graph-kernel library:

- Each file defines `main()` and calls `atalla_halt()`; they are standalone programs.
- They read a fixed config-table base `0x3c` and several use compile-time shape/layout constants.
- Examples at the pinned compiler revision: `add.c` uses 4 rows x 32 columns; `relu.c` uses 4 x 32; `softmax.c` uses one 32-column row; `maxpool.c` uses an 8x8 input and 4x8 output; `conv_baseline.c` fixes M=4, K_FLAT=27, K_OUT=4; `gemm_tiled_*` fixes TILE=4 while reading matrix dimensions/tile counts from the config table. `maxpool_2x2.c` is another specialized 8x8 test.
- `gemm_tiled_pipelined.c` and `conv_pipelined.c` are selected optimized variants; the test files still have validation-sized geometry and some tests are known to have limitations in later compiler history.
- They cannot be called from the emitted graph as-is: their `main`/halt behavior, fixed geometries, and config ABI do not match `GlobalTile*` arguments. Reuse needs a callable adapter/config ABI, plus shape/layout compatibility checks.

## Previous whole-graph attempt (`../xmas`)

- `graph_schedule.c` defines graph `GlobalTile` descriptors and calls generic `*_kernel` names.
- `kernels.h` declares these generic names. `kernels.c` in the attempt contains no-op stubs, so graph calls do not dispatch to the generated `add_1_main`, `relu_main`, etc.
- Generated per-node functions have unique symbols (`add_1_main`, `add_2_main`) and hard-coded operation geometry plus config-table reads. This explains why compiling all files can link while still not providing a coherent reusable kernel ABI.
- `dram.bin` is a separate raw image in that experiment; it is not automatically merged into `output.elf` by `atalla_cc`.
- Compiler default `atalla_layout.mmap` places `code` at `0x1000` and `data` at `0x80000`. This is distinct from the graph's raw DRAM offsets and from the standalone kernels' config base `0x3c`; the build must explicitly define how graph tensor bytes and runtime config are loaded/addressed.

## Architecture report findings

Read `past-years-reports/AtallaGraph_SystemsSoftware_Report_Final.pdf` (and compiler/scheduler reports):

- Atalla uses 32x32 BF16 tiles, explicit DRAM-to-scratchpad movement, and statically planned addresses; the report describes tensor inputs/parameters as preinitialized DRAM data and intermediates as graph-allocated addresses.
- Tile layout is row-major over tiles and row-major within each tile, with padded edge tiles occupying a full 2 KiB.
- Kernel work and transfer metadata must agree on tile shape, valid rows/columns, and DRAM stride. Shape-specific SDMA control words in generated sources are meaningful transfer descriptors, not arbitrary constants.
- The report describes liveness-based DRAM reuse, but the runner's current allocator is monotonic. Keep this difference visible in the design until the allocator is updated or explicitly accepted.

## Open integration questions to resolve in code

- Define one reusable callable ABI per operation (or another explicit graph dispatch ABI) that maps `GlobalTile` shape/tile/address data into the validation kernel's expected config and SDMA geometry.
- Ensure each required graph op/model shape is compatible with the selected pre-made kernel algorithm; fail early for unsupported layouts instead of emitting a misleading ELF.
- Produce a single ELF with resolved function/global relocations and a documented mapping for the DRAM image and config table. Do not assume linking `dram.bin` happens automatically.
- Keep the compiler submodule unchanged; runner-owned adapters/build scripts should contain this integration work.
