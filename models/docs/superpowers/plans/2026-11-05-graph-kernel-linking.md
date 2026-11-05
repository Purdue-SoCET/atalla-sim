# Graph Kernel Linking Implementation Plan

> **For agentic workers:** Use the plan task-by-task. The user's instruction is to keep commits in `atalla-model-runner` only.

**Goal:** Emit a single Atalla ELF for a model graph using shared optimized compiler kernels and an integrated graph DRAM image.

**Architecture:** The FX graph becomes a graph manifest/config table and one schedule entry point. Runner-owned adapters reconcile graph tensor metadata with the standalone validation-kernel ABI; the compiler submodule is read-only. A single build command compiles the graph and shared kernel entry points, then links them with a known data layout.

**Tech Stack:** Python, PyTorch FX, PPCI `atalla_cc`/`ld`, C, Atalla ISA.

**Spec:** `docs/kernel-linking-design.md`

## Global constraints

- Make commits only in `atalla-model-runner`.
- Keep the compiler submodule unchanged and pinned.
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
- [ ] Keep verified mismatches and unsupported geometry visible in the memory file.

### Task 2: Add a shared kernel adapter and invocation manifest

**Files:** `atalla-graph/kernel_backend.py`, `atalla-graph/scripts/generate_schedule.py`, runner-owned adapter source/templates

- [ ] Map each supported FX op to a compiler validation-kernel source and optimized variant.
- [ ] Represent node-specific tensor addresses, shapes, strides, and op parameters as invocation records.
- [ ] Adapt each source once into a callable symbol and reject geometry the source cannot cover.

### Task 3: Package graph data and link the ELF

**Files:** `atalla-graph/compile_graph.py`, `atalla-graph/linker/atalla_graph.mmap`, graph emitter

- [ ] Reserve a non-overlapping runtime config range and assign graph payload a stable base.
- [ ] Emit or import the DRAM payload into the linkable data image and relocate descriptor addresses.
- [ ] Compile shared kernel units plus graph schedule and link them into one ELF with the compiler submodule.

### Task 4: Document build artifacts and remaining limits

**Files:** `README.md`, `MEMORY.md`

- [ ] Document the one-command model-to-ELF workflow, output map, and supported operator geometries.
- [ ] Verify runner-only git history and leave other repositories unchanged.
