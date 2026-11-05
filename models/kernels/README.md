# Atalla graph kernels

The library is independent of FX and PyTorch. `kernels.h` defines the callable
ABI: logical shape, tile grid, physical tile addresses, and a storage-layout
tag and an optional address-only gather plan. The runner supplies descriptors and initialized data; shared C sources
perform memory transfers and computation.

- `src/elementwise.c`: add, multiply, scalar multiply, and ReLU.
- `src/matmul.c`: rank-2 tiled GEMM.
- `src/spatial.c`: NCHW convolution orchestration, max pooling, and flattening.
- `src/spatial_tiles.c`: shared vector staging, GEMM, result extraction, and
  coherent result writes for spatial operations.
- `compile.py`: explicit compatibility policy for the pinned compiler.
- `capabilities.json`: implemented operations, source dependencies, and limits.

## Numerical contracts

`GlobalTile.storage_layout` is part of the public ABI:

| Value | Physical storage |
|---|---|
| 0 | Canonical padded row-major 32x32 BF16 tiles. |
| 1 | Constant rank-2 GEMM weights: each logical KxN tile stores its columns as rows, last column first. Tile order remains K-major, N-minor. |
| 2 | Constant OIHW convolution weights: flatten K within each group, form KxO matrices, then pack as layout 1; group-major, K-major, O-minor tiles. |

Matrix and convolution outputs use layout 0. Convolution descriptors retain
the logical OIHW shape but carry the actual packed tile count. Dynamic RHS
matrices and weights shared with incompatible consumers are rejected until a
runtime weight-transform path is provided.

Vector-scalar operands use an integer containing the IEEE FP32 bit pattern.
The selected C frontend's `float` is BF16, but the vector scalar port reads
FP32 register bits. Inline assembly in the leaf kernels preserves those bits;
passing a BF16 immediate directly would turn the multiplier into near-zero.

A vector load must wait for the actual DMA completion. The inline assembly
at each load returns an opaque scratch-base token after reading the DMA-held
register; vector loads use that token rather than rematerializing zero.
Keep these dependencies when changing scheduling or register allocation.

Scalar stores are write-back cached and cannot safely feed DMA staging.
Spatial input rows are constructed through vector operations into scratchpad
0; accumulators remain in scratchpad 1. Weights use pad 2 and bias/result
writes use pad 3. Convolution address plans contain input element addresses or padding markers,
ordered by batch, group, spatial block, reduction block, row, and lane. The
shared staging kernel reads current tensor values through those addresses;
no convolution values are precomputed. Output-channel tiles reuse the plan.
Scalar arrays passed between CPU functions are read through the same CPU cache. Final spatial results are written through scratchpad DMA,
so later graph kernels see completed data. The graph uses distinct tensor
allocations; no in-place scalar-cache/DMA aliases are assumed.

Masked operations pass their merge vector explicitly. ReLU therefore does
not depend on whichever vector register the allocator happens to choose.

## Compiler policy and optimization

```bash
make -C kernels
```

`compile.py` uses the selected PPCI frontend, code generator, packetizer,
relocations, and object format. It sets callee-save registers x9, x11, x18..x27
because callers already assume those survive calls. The normal backend leaves
that set empty. For spatial sources it runs the normal O2 sequence without
CleanPass: that pass can crash or discard the accumulator's phi edge. These
workarounds are explicit and should be reviewed against future compiler
updates; do not silently patch the compiler submodule.

The fixed staging loop handles four lanes per iteration. Fully unrolling all
32 lanes causes this backend to spill masks, which it cannot encode. The
result extraction uses explicit lane indices because the ISA requires an
immediate index.

Use focused cases in `tests/test_kernel_numerics.py`, then `verify_model.py`
for the affected complete model. The reports record the exact ELF hash,
tolerances, per-operation error, and cycle count. `prompt.md` describes how to
propagate optimizations while preserving these contracts.

Current limits: rank-2 matrix operands; rank-4 NCHW spatial operands; constant
prepacked weights; max pooling with unit dilation and no ceil mode. Grouped
convolution with bias requires channels per group divisible by 32. Average
pooling, normalization, GELU, and attention remain unavailable.
