# Prompt: propagate compiler optimizations into the graph kernel library

You are reviewing compiler work for `atalla-model-runner`. Your task is to analyze
new compiler behavior and decide whether a validated optimization or architectural,
syntactic, instruction-selection, packetization, or ISA change should be reflected
in the shared kernels under `kernels/`.

## Scope and boundaries

- Treat `kernels/` as the callable graph-kernel library and `aihw-ppci-compiler/`
  as the compiler and validation reference.
- Do not edit or commit the compiler submodule while doing this propagation task.
  Make implementation changes in the `atalla-model-runner` repository only.
- Do not copy standalone validation programs wholesale. They contain test setup,
  fixed addresses/config tables, `main()`, and `atalla_halt()` behavior. Extract
  the relevant algorithm or optimized instruction pattern into the library ABI.
- Keep graph planning, FX/PyTorch imports, tensor allocation, node ordering, and ELF
  payload packaging in the runner. Keep Atalla tile movement and kernel computation
  in `kernels/`.
- Never mark an operation linkable until its callable implementation and data layout
  are supported. Missing operations must remain explicit in the capability manifest.

## Review procedure

1. Read the previous reviewed revision from `kernels/COMPILER_BASELINE` and the
   current submodule revision with `git -C aihw-ppci-compiler rev-parse HEAD`.
   Read the current `test_validation_kernels/KERNEL_STATUS.md`, relevant validation
   C files, and `git -C aihw-ppci-compiler diff <baseline>..HEAD`. Inspect
   associated compiler changes when the kernel
   diff depends on frontend syntax, instruction selection, lowering, packetization,
   latency/resource constraints, or ISA semantics.
2. For each change, explain the semantic or architectural effect in plain terms.
   Separate algorithm changes from test-only setup, geometry constants, and harness
   changes. Cite file names and the specific expressions or instructions involved.
3. Check whether `kernels/kernels.h` can express the required addresses, shapes,
   strides, tile layout, and edge behavior. Check the current runner allocation
   contract: padded row-major 32x32 BF16 tiles, 2,048 bytes per tile, with tile base
   addresses in each `GlobalTile` descriptor.
4. Decide whether to propagate the change. Preserve the existing callable ABI when
   possible. If an ABI change is necessary, enumerate every caller and generated
   artifact that must change before editing.
5. Implement the minimal library change. Prefer shared helpers and one implementation
   per operation/signature. Add a specialized variant only when its preconditions
   can be selected and checked from graph metadata. Do not create per-node kernel
   source copies or textual placeholder substitution for runtime addresses.
6. Update `kernels/README.md`, `kernels/capabilities.json`, `MEMORY.md`, and
   `kernels/COMPILER_BASELINE` with the supported geometry, data layout, selection
   rule, and reviewed compiler revision. Advance the baseline only after the review
   and required validation are complete. Keep unsupported cases explicit.
7. Describe correctness and performance validation needed. Use the compiler's
   numerical golden and packetization/performance data as evidence for the source
   change, then validate the library entry point with equivalent addresses, tile
   strides, tails, and graph descriptor shapes. Do not claim the library inherited
   an optimization merely because the standalone validation file passes.

## Required response

Report:

- compiler revision and files reviewed;
- each relevant optimization/architecture change and its expected effect;
- library changes made, or why no propagation is safe/useful;
- ABI and supported-layout impact;
- validation commands/results, including any unrun checks;
- remaining unsupported shapes or operations.

If the compiler change is ambiguous, state the uncertainty and the evidence needed.
Do not invent ISA semantics from instruction names; trace the compiler lowering or
validation behavior first.
