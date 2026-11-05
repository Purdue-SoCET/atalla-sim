# Atalla graph kernels

`kernels/` is the callable library linked into graph ELFs. `kernels.h` is the
public `GlobalTile` ABI; each descriptor has logical shape, tile grid, and
absolute DRAM addresses for padded 32x32 BF16 tiles. The runner creates the
schedule and data image; kernel code contains Atalla memory and compute logic.

- `src/elementwise.c`: add, multiply, scalar multiply, and ReLU.
- `src/matmul.c`: rank-2 tiled GEMM, using `load_weights` and `gemm` from the
  compiler's tiled validation program.
- `src/spatial.c`: rank-4 NCHW convolution through runtime im2col packing and
  GEMM, rank-4 max pooling, and rank-4 to rank-2 flattening.
- `include/sdma.h`: SDMA control-word encoding and 32x32 presets.
- `capabilities.json`: operation/source map used to reject unsupported graphs.
- `prompt.md`: compiler-team workflow for propagating new optimizations.

The spatial adapter uses packed 32-bit scalar loads/stores for BF16 values
because the selected Atalla C frontend does not parse `short`. Its 32x32 GEMM
scratch buffers are shared across sequential graph calls. `spatial.c` builds at
`-O0` because the current compiler's `-O2` clean pass crashes on its nested
loops; other sources build at `-O2`. The graph schedule remains the only per-model
C source, and no per-node kernel copies are created.

Build the library with `make -C kernels`, or build a whole model with
`python build_model.py --model basic`. The compiler submodule stays unchanged.
Standalone validation files under `aihw-ppci-compiler/test_validation_kernels`
are optimization references, not callable graph kernels. Compiler contributors
can edit these library sources directly and use `prompt.md` to review and
propagate new compiler behavior.

Current boundaries: matrix kernels support rank-2 BF16 tensors; spatial kernels
support rank-4 NCHW inputs. Max pooling accepts unit dilation and no ceil mode.
Average pooling, layer normalization, GELU, attention, and other operators are
not implemented. The build manifest must reject such graphs rather than link a
placeholder function. A linked ELF proves layout and symbol resolution; graph
numerical equivalence still needs simulator or hardware execution.
