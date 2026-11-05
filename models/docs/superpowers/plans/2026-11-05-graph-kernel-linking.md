# Graph Kernel Linking Implementation Plan

> **For agentic workers:** Use the plan task-by-task. The user's instruction is to keep commits in `atalla-model-runner` only.

**Goal:** Emit a single Atalla ELF for a model graph using shared optimized compiler kernels and an integrated graph DRAM image.

**Architecture:** The FX graph emits shared-kernel calls through a stable descriptor ABI. `kernels/` contains callable Atalla C implementations and capability metadata; compiler validation sources remain read-only optimization references. The runner owns graph metadata and data packaging, while compiler contributors can change kernel implementations in the isolated package.

**Tech Stack:** Python, PyTorch FX, PPCI `atalla_cc`/`ld`, C, Atalla ISA.

**Spec:** `docs/kernel-linking-design.md`

## Global constraints

- Make commits only in `atalla-model-runner`.
- Keep the compiler submodule contents unchanged at the selected `atalla-arch` revision.
- Do not generate a distinct kernel body for every FX node.
- Surface unsupported op geometry/layout at build time.
- Preserve the existing schedule-only CLI while adding an explicit link/build CLI.

## Review focus

- Config-table address `0x3c` must not overlap model tensors.
- 32x32 BF16 graph tile layout and edge padding must match each selected kernel's SDMA control/stride.
- Transposed GEMM weights and convolution lowering must preserve logical indexing.
- Every graph call must resolve to a real kernel symbol; no no-op stubs or dead per-node entry points.
- ELF data layout must include all graph bytes at the same addresses referenced by the schedule.

---

### Task 1: Record architecture memory and kernel compatibility

**Files:** `MEMORY.md`, `docs/kernel-linking-design.md`

- [x] Record current runner flow, graph allocator layout, old `xmas` wiring, validation source ABI, and report findings.
- [x] Keep verified mismatches and unsupported geometry visible in the memory file.

### Task 2: Build the shared kernel package and invocation manifest

**Files:** `kernels/kernels.h`, `kernels/include/sdma.h`, `kernels/capabilities.json`, `kernels/src/elementwise.c`, `kernels/README.md`, `kernels/prompt.md`, `atalla-graph/kernel_backend.py`, `atalla-graph/run_schedule.py`

- [x] Define a public `GlobalTile` descriptor ABI and schedule-compatible kernel declarations.
- [x] Implement shared add, multiply, scalar multiply, and ReLU tile kernels; publish their geometry in `capabilities.json`.
- [x] Emit per-node bindings and gate kernel coverage based on implemented operation and shape compatibility; report ELF linkability separately.
- [x] Add a standalone object build target, compiler contributor docs, and optimization propagation prompt.
- [ ] Implement GEMM, convolution, pooling, and other mapped graph operations against the shared tile ABI.
- [x] Make schedule calls propagate kernel error status.
- [ ] Reject every unsupported broadcast/layout before C emission.

### Task 3: Package graph data and link the ELF

**Files:** `atalla-graph/compile_graph.py`, `atalla-graph/linker/atalla_graph.mmap`, graph emitter

- [ ] Reserve a non-overlapping runtime config range and assign graph payload a stable base.
- [ ] Emit or import the DRAM payload into the linkable data image and relocate descriptor addresses.
- [ ] Compile shared kernel units plus graph schedule and link them into one ELF with the compiler submodule.

### Task 4: Document build artifacts and remaining limits

**Files:** `README.md`, `MEMORY.md`

- [ ] Document the one-command model-to-ELF workflow, output map, and supported operator geometries.
- [ ] Verify runner-only git history and leave other repositories unchanged.
