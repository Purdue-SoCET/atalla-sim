# Atalla Model Runner Memory

Living codebase notes for the graph-to-ELF work. Update this file as new behavior is confirmed.

## Repository boundaries

- `atalla-model-runner` is the active implementation repository. Keep commits here only.
- `aihw-ppci-compiler` is checked out on `atalla-arch` at `351b5fabf76fa3352ebc2e9571e9793d53c7cbe6` (latest kernel validation status is documented in `test_validation_kernels/KERNEL_STATUS.md`). Keep compiler source changes out of this task; only runner commits are allowed.
- Current runner support is schedule emission plus source-binding metadata: `atalla-graph/run_schedule.py` writes `graph_schedule.c`, `dram.bin`, and `kernel_manifest.json`. The manifest references one compiler source per operation type and records that adapters are still required.
- The runner has no `codegen/c_emitter.py` or per-op kernel builders. `atalla-graph/kernel_backend.py` now maps graph ops to shared compiler validation source paths; it does not copy or adapt the source bodies. In `atalla-models`, the per-node `.c` generation used by validation is in `atalla-graph/codegen/c_emitter.py`; its C builders are loaded from `functional_sim/kernels`.

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
- Examples at the runner’s recorded compiler revision: `add.c` uses 4 rows x 32 columns; `relu.c` uses 4 x 32; `softmax.c` uses one 32-column row; `maxpool.c` uses an 8x8 input and 4x8 output; `conv_baseline.c` fixes M=4, K_FLAT=27, K_OUT=4; `gemm_tiled_*` fixes TILE=4 while reading matrix dimensions/tile counts from the config table. `maxpool_2x2.c` is another specialized 8x8 test.
- `gemm_tiled_pipelined.c` and `conv_pipelined.c` are optimized validation variants. The current latest `atalla-arch` checkout reports all 12 validation kernels passing against numerical goldens, but they remain fixed-geometry standalone programs rather than graph-callable kernels.
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


## Adapter feasibility update (2026-11-05, latest atalla-arch checkout)

- The validation C files are not parameterized kernel libraries. Their SDMA control words encode row count, vector width, and DRAM row stride in immediates; changing only `ROWS` or the config table does not retarget the transfer geometry.
- The runner allocator stores every rank-2 tensor as 32x32 tiles with a 32-element row stride, while validation `add.c`/`relu.c` process 4x32 contiguous rows (the encoded control words are fixed). Even when a graph tensor has a 4x32 logical shape, its storage is still a padded 32x32 tile, so its row stride differs.
- `gemm_tiled_pipelined.c` fixes the compute tile to 4x4 and its SDMA controls assume 8-wide matrices. Runner descriptors use 32x32 tiles and contain only tile base addresses, not the packed matrix buffers expected by that test.
- `conv_pipelined.c` is a GEMM over an im2col matrix, not a direct NCHW convolution kernel. The old emitter explicitly materializes im2col and transposed/flattened weights; current runner allocation does neither.
- Therefore a callable wrapper alone cannot safely turn these files into graph kernels. A valid adapter needs either a compiler-side generic kernel ABI, or runner data-layout/packing and schedule changes that reproduce each template's expected SDMA layout. It must reject incompatible graphs. No compiler files were changed.
- The PPCI linker layout puts `.data` at `0x80000`, while graph addresses begin at zero; including `dram.bin` as an ELF C object is not sufficient without moving graph allocations/config to a non-overlapping linker region and proving emitted section addresses.
