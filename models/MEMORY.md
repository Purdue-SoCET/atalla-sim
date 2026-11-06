# Atalla Model Runner Memory

Living codebase notes for the graph-to-ELF work. Update this file as new behavior is confirmed.

## Repository boundaries

- `atalla-model-runner` is the active implementation repository. Keep commits here only.
- `aihw-ppci-compiler` is pinned at `351b5fabf76fa3352ebc2e9571e9793d53c7cbe6` (latest kernel validation status is documented in `test_validation_kernels/KERNEL_STATUS.md`). Keep compiler source changes out of this task; only runner commits are allowed.
- Current runner support writes `graph_schedule.c`, `dram.bin`, and `kernel_manifest.json`. Shared callable add/multiply/scalar-multiply/ReLU implementations now live in `kernels/src/elementwise.c`; the manifest gates linkability by operation and shape support in `kernels/capabilities.json`.
- The runner has no legacy `codegen/c_emitter.py` or per-node kernel builders. `atalla-graph/kernel_backend.py` binds graph ops to shared library sources and records compiler validation sources as references only. `kernels/` defines the schedule ABI, capability table, and first shared elementwise implementations. In `atalla-models`, legacy per-node `.c` generation was in `atalla-graph/codegen/c_emitter.py`; its C builders were loaded from `functional_sim/kernels`.

## Current graph schedule and memory

- `run_schedule.py`: converts model to BF16/eval, FX traces, lowers Linear to matmul/add, performs shape propagation, normalizes ops, plans tiles, allocates DRAM, and calls `scripts/generate_schedule.emit`.
- `graph/memoryallocator.py`: `TILE_HEIGHT = TILE_WIDTH = 32`; one BF16 tile is 2,048 bytes. Tensor bytes are laid out as row-major 32x32 tiles, with edge padding. Current runner allocator assigns monotonically increasing addresses from zero; it does not implement the liveness reuse described in the AtallaGraph report.
- `generate_schedule.py`: reads each node's `dram_addr`, `bytes`, BF16 shape and tile counts. It emits C `GlobalTile` descriptors whose `tiles[]` contain the DRAM base address of each tile. The descriptor also records logical shape and tiles-per-dimension. A view/transpose may alias its source address.
- Schedule calls are graph operations such as `matmul_kernel(&A, &B, &C)` and `add_kernel(&lhs, &rhs, &dst, alpha)`. These are generic graph-kernel APIs, not entry points in `test_validation_kernels`.
- The per-node C files in the former external `xmas` experiment have config loads from address `0x3c` (decimal 60). Their config words are per-kernel runtime GMEM addresses and dimensions; constants such as SDMA control values, row/column bounds and loop counts encode the selected memory geometry.

## Existing standalone compiler kernels

`aihw-ppci-compiler/test_validation_kernels/*.c` are validation programs, not a linkable graph-kernel library:

- Each file defines `main()` and calls `atalla_halt()`; they are standalone programs.
- They read a fixed config-table base `0x3c` and several use compile-time shape/layout constants.
- Examples at the runner’s recorded compiler revision: `add.c` uses 4 rows x 32 columns; `relu.c` uses 4 x 32; `softmax.c` uses one 32-column row; `maxpool.c` uses an 8x8 input and 4x8 output; `conv_baseline.c` fixes M=4, K_FLAT=27, K_OUT=4; `gemm_tiled_*` fixes TILE=4 while reading matrix dimensions/tile counts from the config table. `maxpool_2x2.c` is another specialized 8x8 test.
- `gemm_tiled_pipelined.c` and `conv_pipelined.c` are optimized validation variants. The current latest `atalla-arch` checkout reports all 12 validation kernels passing against numerical goldens, but they remain fixed-geometry standalone programs rather than graph-callable kernels.
- They cannot be called from the emitted graph as-is: their `main`/halt behavior, fixed geometries, and config ABI do not match `GlobalTile*` arguments. Reuse needs a callable adapter/config ABI, plus shape/layout compatibility checks.

## Previous whole-graph attempt (the former external `xmas` experiment)

- `graph_schedule.c` defines graph `GlobalTile` descriptors and calls generic `*_kernel` names.
- `kernels.h` declares these generic names. `kernels.c` in the attempt contains no-op stubs, so graph calls do not dispatch to the generated `add_1_main`, `relu_main`, etc.
- Generated per-node functions have unique symbols (`add_1_main`, `add_2_main`) and hard-coded operation geometry plus config-table reads. This explains why compiling all files can link while still not providing a coherent reusable kernel ABI.
- `dram.bin` is a separate raw image in that experiment; it is not automatically merged into `output.elf` by `atalla_cc`.
- Compiler default `atalla_layout.mmap` places `code` at `0x1000` and `data` at `0x80000`. This is distinct from the graph's raw DRAM offsets and from the standalone kernels' config base `0x3c`; the build must explicitly define how graph tensor bytes and runtime config are loaded/addressed.

## Architecture report findings

The earlier AtallaGraph systems software report and companion compiler/scheduler reports describe:

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
- Therefore a callable wrapper alone cannot safely turn these files into graph kernels. The runner now has its own callable ABI and elementwise implementation derived from the validated vector patterns. Matrix, convolution, and pooling remain unsupported pending library implementations and explicit graph-layout handling. No compiler files were changed.
- The PPCI linker layout puts `.data` at `0x80000`, while graph addresses begin at zero; including `dram.bin` as an ELF C object is not sufficient without moving graph allocations/config to a non-overlapping linker region and proving emitted section addresses.


## Kernel package and compiler-team workflow

- `kernels/kernels.h` is the C descriptor/function ABI included by generated schedules. Descriptors carry shape, tile grid, and per-tile DRAM byte addresses. `kernels/include/sdma.h` centralizes the SDMA control-word fields and 32x32 presets.
- `kernels/src/elementwise.c` processes padded 32x32 BF16 tiles; supported broadcasts and shape preconditions are declared in `kernels/capabilities.json`. `kernel_backend.py` rejects view inputs because the current `GlobalTile` ABI has no stride/offset fields and the allocator aliases views.
- `kernels/Makefile` compiles the shared object through the selected compiler submodule; generated objects stay under ignored `kernels/build/`.
- `kernels/prompt.md` guides compiler optimization review and propagation into the library. `kernels/COMPILER_BASELINE` records the last reviewed compiler source revision; advance it only after review and validation.
- GEMM, convolution, and pooling remain unavailable; their declarations reserve the schedule ABI, not implementations. `kernel_manifest.json` must continue to reject graphs requiring them.

## End-to-end linking update (2026-11-05)

- `build_model.py` now runs FX scheduling, compiles schedule/entry/shared kernels, wraps `dram.bin` as a PPCI data object, links with a generated layout, and writes `output.elf`. The data object is first in the `.data` section at `0x80000`, followed by graph descriptors and kernel scratch buffers. The build reopens the ELF and checks that its data begins with the exact DRAM image. `readelf -r` on both built-in model ELFs reported no relocation sections.
- The graph allocator now starts addresses at `0x80000` while keeping `dram.bin` relative to that base. Constant transpose views are materialized as padded tiles so rank-2 GEMM sees KxN weights. Other aliased views are still rejected when used as kernel inputs. `torch.flatten` gets a separate runtime materialization kernel.
- `kernels/src/matmul.c` implements callable rank-2 32x32 tiled GEMM using the validation kernel's `load_weights`/`gemm` sequence. `kernels/src/spatial.c` adds runtime im2col packing for NCHW convolution, max pooling, and flattening. Spatial BF16 tensor access uses packed 32-bit words because Atalla C does not parse `short`; `spatial.c` compiles at `-O0` because the compiler optimizer's clean pass crashes on its nested loops at `-O2`. Compiler source was not changed.
- `basic` and `alexnet_small` both compile and link into an executable ELF with code and writable data load segments. This verifies parsing, code generation, section placement, and symbol resolution. Their graph kernels have not yet been run against PyTorch numerical goldens on simulator or hardware; standalone compiler validation does not prove graph numerical correctness.
- Current unsupported operations include average pooling, layer normalization, GELU, and attention. Capability gating is required for graphs beyond the two built-in models.

## Simulator packet-boundary finding (2026-11-05)

- `atalla-sim` loads four 5-byte instructions per 20-byte packet and requires every code symbol to start at a packet boundary relative to the code section. The initial runner ELF failed at `run_graph_block0` because `build_model.py` compiled objects without `-p`; linking and `readelf -r` alone did not reveal this.
- The runner now passes `-p` to PPCI for every C object and checks the linked code size and all code-symbol addresses before writing the ELF. A comparison ELF from the compiler's `atalla_cc -p` passed the same simulator loader. No compiler or simulator source was changed.

- Simulator execution exposed a separate compiler LABEL-lowering issue: passing `&tensor_gt` loaded its first word (`rank`) as the argument. Generated schedules now declare a global pointer holding each descriptor address and pass the address of that pointer; the compiler's one implicit LABEL dereference then supplies the descriptor pointer.
- The optimized elementwise library lost a descriptor pointer across nested `same_shape` calls in simulator execution and returned `-1`. Building `elementwise.c` at `-O0` allows `basic` to run all three GEMMs and return status 0. This is a runner build workaround; compiler source remains unchanged.
- A numerical comparison of the executed `basic` ELF against PyTorch still fails. In particular, the first matrix result differs, and the first bias add leaves the first output unchanged. The simulator documentation identifies standalone compiler GEMM weight orientation and SDMA dependency issues. Do not claim graph numerical correctness from an ELF load or clean halt; further kernel/compiler work is required.

- A temporary simulator probe repacked each constant matrix tile as reversed rows of its transpose (matching the documented `lw.vi` shift order) while keeping the original GEMM code. The first `basic` GEMM still differed from PyTorch (`max abs` about 1.88), so weight order alone is not the remaining numerical fix. No probe changes were committed to kernel code or the compiler.

## Numerical correctness work (2026-11-05)

The following findings supersede the earlier numerical limitations:

- `basic` now runs in `atalla-sim` with strict packet checks, returns zero,
  and every one of its nine scheduled results matches BF16 PyTorch exactly.
  `verify_model.py` reproduces this and writes `numerical_report.json`.
- Scalar vector operands must carry FP32 register bits. The selected frontend
  lowers `float` to BF16 bits, which made the add kernel's scalar multiplier
  effectively zero. The ABI now passes FP32 bits as int and uses explicit
  register bit preservation in the leaf kernel.
- Hardware `lw.vi` shifts each vector into weight column zero. GEMM must load
  logical weight columns in reverse order. The allocator prepackages constant
  rank-2 weights as layout 1 and grouped convolution weights as layout 2. The
  original OIHW logical shape and packed physical tile count are both explicit.
- The GEMM test proved activation DMA read stale weight data when scratch base
  zero was put in a different register. Inline assembly now returns a token
  after reading the held DMA register; consumers cannot run before completion.
- Scalar staging writes remain dirty in the write-back dcache. They cannot
  feed DMA, and scalar result stores are invisible to later graph DMA until
  eviction. Spatial staging and result writes now use vector scratchpad paths.
- `kernels/compile.py` configures the target's existing callee-save mechanism
  (x9, x11, x18..x27). The backend's empty callee-save set corrupted live
  variables across calls. No compiler files are modified.
- CleanPass also discarded the non-first-K accumulator phi edge in the spatial
  leaf function. The multi-K convolution test observed exactly the final K
  contribution plus bias. Both spatial sources now use the normal O2 passes
  with only this cleanup omitted, preserving and verifying the phi edges.
- Convolution, max pooling, flattening, a spatial-to-linear graph, grouped
  convolution without bias, tensor multiply tails, scalar scaling, and matrix
  tails have focused simulator/PyTorch tests in the runner. Full spatial-model
  verification is performed with the same verifier, not the simulator's broad
  pytest suite.
- Spatial staging is unrolled four lanes at a time; 32-lane unrolling hit
  unsupported mask-register spilling. These are shared source routines,
  not per-node generated kernel copies.

- Convolution now stores an address-only im2col plan after the tensor payloads
  in `dram.bin`. `GlobalTile.gather_addr` points to this plan. Each entry is an
  absolute BF16 element address or `0xffffffff` for padding. Plans order batch,
  group, spatial blocks, K blocks, active rows, and 32 lanes; output-channel
  tiles reuse them. The runtime still reads current activations and computes
  every result. This moves repeated coordinate arithmetic out of the scalar
  simulator without generating per-model kernel code or caching results.

## Final simulator numerical verification (2026-11-05)

- `python verify_model.py --model basic --out-dir out/basic`: status 0,
  41,220 cycles, all nine scheduled results exactly equal BF16 PyTorch.
- `python verify_model.py --model alexnet_small --out-dir out/graph --limit 20000000`
  (default seed 42, scale 0.01): status 0, 5,222,676 cycles, all 22 scheduled
  results passed at atol 0.01 / rtol 0.03. Largest intermediate absolute error
  was 0.0078125; final output maximum absolute error was 0.00048828125. This is
  numerically checked BF16 execution, not a claim of bit-exact AlexNet output.
  The full simulator run took about 21 minutes; avoid repeating it for changes
  covered by focused cases. Reports and ELF hashes are in the output directories.
- Focused runner pytest: 15 tests passed together, then the added batched /
  channel-tiled / strided / dilated convolution and multiple-spatial-block
  convolution tests each passed (17 total). `make -B -C kernels`, Python syntax
  checks, and `git diff --check` passed. No broad atalla-sim pytest suite was run.
- Compiler remains at 351b5fabf76fa3352ebc2e9571e9793d53c7cbe6 with no source
  changes. Simulator source is unchanged; its documented native accelerator
  extension was built locally for execution.

## Cosine metrics in runner validation (2026-11-05)

- The functional simulator's `core/kernel_ops.py::compare_cosine_max_mean`
  returns cosine similarity, max absolute error, and mean absolute error in
  float64; cosine is zero when either operand has zero norm. Runner-owned
  `validation_metrics.py` follows this convention and rejects shape mismatches.
- `verify_model.py` now writes/prints those three metrics per scheduled output
  while retaining its PyTorch `allclose` correctness test. The focused pytest
  suite checks cosine >0.99 for nonzero outputs plus absolute error, and tests
  helper behavior on perturbed vectors, zero norms, and shape mismatches.
- Verification: combined metric and kernel suite reported 20 passed in 177.24s;
  rebuilt `basic` reported cosine 1.0 and zero max/mean error for all nine
  scheduled results. The shell wrapper used a reserved zsh variable after
  pytest had completed, so its shell exit was nonzero despite pytest's explicit
  `20 passed` result; use no `status` variable in zsh command wrappers.

## Self-contained checkout (2026-11-05)

- The runner now mounts `atalla-sim` beside `aihw-ppci-compiler` as a pinned submodule. `verify_model.py` and focused numerical tests default to the runner-local simulator.
- Build and numerical verification require only the two direct submodules. The simulator's nested compiler, functional simulator, and Ramulator submodules serve its own optional tooling; the runner uses its direct compiler submodule.

## Compiler baseline review (2026-11-05)

- The pinned `aihw-ppci-compiler` `atalla-arch` revision is `351b5fabf76fa3352ebc2e9571e9793d53c7cbe6`, exactly the revision in `kernels/COMPILER_BASELINE`; `git diff <baseline>..HEAD` is empty. The submodule URL changed, but the pinned source revision did not.
- Reviewed `test_validation_kernels/KERNEL_STATUS.md` and the current `add.c`, `relu.c`, `gemm_tiled_pipelined.c`, `conv_pipelined.c`, and `maxpool_2x2.c` against runner elementwise, matmul, and spatial code. Their standalone fixed-geometry setup is not a graph ABI change. No new compiler optimization, ISA, syntax, or packetization delta is available to propagate. The runner retains explicit masked merges, padded 32x32 tile movement, packed weight order, and DMA dependencies.
- Focused `test_matmul`, `test_convolution`, `test_maxpool`, and `test_relu`: 4 passed in 23.09s. `verify_model.py --model basic` linked and ran the local simulator: status 0, 41,063 cycles, all nine scheduled results passed with cosine 1.0 and zero absolute error. No full spatial-model run was needed because no library or compiler source changed.
- `kernels/COMPILER_BASELINE`, capabilities, and ABI remain unchanged. The existing unsupported operations and layout limits still apply.
