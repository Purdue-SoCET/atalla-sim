# MEISSA Systolic Array

`src/systolic_array/systolic_array_meissa.py` — a multiplier grid with one
pipelined adder tree per column.

A standard systolic array interleaves multiply and accumulate in one MAC cell.
MEISSA splits them apart: plain multipliers in the grid, and column-wise adder
trees that reduce each column's N products to a single value.

## Shape

```
   a[0] ->[x]->[x]->[x]->[x]      one column per cycle, left to right
   a[1] ->[x]->[x]->[x]->[x]
   a[2] ->[x]->[x]->[x]->[x]
   a[3] ->[x]->[x]->[x]->[x]
          |    |    |    |
         tree tree tree tree      N products -> 1 sum, pipelined
          |    |    |    |
         red  red  red  red       FP32 -> BF16, before the buffer
          |    |    |    |
         ----- output buffer ---- one bank per column, de-skews
                  |
                 GSAU
```

## No input skew, output skew instead

Because product generation and reduction are both pipelined, a whole activation
vector is injected on one cycle — no temporal staggering on the way in.

The skew moves to the output. An activation reaches column `j` on cycle `j`, so
column `j` finishes `j` cycles after column 0. The output buffer removes it by
giving each column **its own bank with its own write pointer**; a read takes one
address across all banks, which is one complete result vector.

Once the pipeline is full the array retires **one output vector per cycle**.

## Cycles

```
pipeline_depth = mul_latency + (adder levels x their latency)
first output   = N + pipeline_depth
thereafter     = 1 per cycle
```

| | levels | `pipeline_depth` (mul 1, add2 1, add4 3) |
|---|---|---|
| N=32, pure 2-input | `log2(32)` = 5 x add2 | 6 |
| N=32, mixed | 2 x add4 + 1 x add2 | 8 |
| N=16, pure 2-input | `log2(16)` = 4 x add2 | 5 |
| N=16, mixed | 2 x add4, no add2 | 7 |

## The two adder trees

A pure binary tree is `log2(N)` deep; four-input adders reduce that to
`log4(N)`. Four-input adders are the cheaper way to reduce — roughly **2.4x the
area efficiency** of three cascaded two-input adders — so the mixed tree uses
them wherever it can.

Whether a pure four-input tree is possible depends on the array dimension:

- **16** is a power of four, so `log4(16) = 2` levels reduce it completely.
- **32** is not, so two four-input levels leave two terms and a final
  **two-input** level is needed.

`is_power_of_four(N)` decides, matching the RTL's `LOG4_IS_WHOLE`.

The two shapes are not interchangeable numerically: a four-input adder rounds
once where three cascaded two-input adders round three times. On values that are
exact in FP32 they agree; otherwise they can differ in the last bits.

## Precision

Inputs BF16, products and every tree sum FP32, then a reducer converts to BF16
**before** the output buffer — so the banks store 16-bit values rather than
32-bit ones, halving the buffer.

## Weights load through a shift register

The first weight vector pushed ends up in the **last** column:

```python
# to land column j of W in array column j
stream = [w[:, size - 1 - k] for k in range(size)]
```

Get this wrong and the array computes `A @ W.T` with reversed columns, quietly.
Nothing in the interface can catch it — the shapes are identical.

## No partial-sum input

`pipelined_adder_tree.sv` drives `sum_out` from the tree alone; the `psum_in`
port is disconnected. `enqueue_psums` therefore accepts zeros and raises on
anything else, rather than dropping a partial sum on the floor.

## Using it

```python
platform = build_tpu_platform(size=32, systolic_array="meissa")
```

The interface matches `SystolicArrayTPU`, so the same `GSAUTPUBridge` drives
either. One difference the bridge cares about: `flush_cycles()` is **0**. A
skewed array needs zero rows pushed in behind the last real one to shift its
results out; MEISSA does not, and pushing them would only burn its input
credits. `drain_cycles()` reports the time the last vector still needs.

Input backpressure is credit-based, as in the RTL: `pipeline_depth + N - 1`
credits, one spent per vector injected and returned when one retires.

## Speed

The model's arithmetic is trivial — 1024 multiplies and ~1024 adds per cycle at
N=32. Everything else was overhead, and it came off in four layers:

| | µs/cycle | vs start |
|---|---|---|
| original, scalar numpy per column | 942 | 1.0x |
| vectorised tree reduction | 171 | 5.5x |
| lazy activity stats | 128 | 7.4x |
| AVX kernel, per-cycle `tick()` | 65 | 14.6x |
| AVX kernel, `run_cycles(128)` | 20 | 47.5x |

**Vectorising the reduction** was the single biggest step and needed no C++: the
tree is identical for every column, so one array operation per level replaces N
scalar ones while leaving the pairing and order untouched.

**Batching is what breaks the per-cycle floor.** A cycle is ~25 small numpy or
ctypes operations, each costing more in dispatch than the arithmetic it
performs. `tick()` still works and is bit-identical, but `run_cycles(k)` hands
k cycles to the kernel in one call and amortises all of it — which is why it is
3x faster again than the per-cycle kernel path.

```python
sa.run_cycles(n, activations=rows, weights=weight_stream)
```

Use it wherever the array runs unattended: a tile streamed in one burst, or the
drain after the last vector. `activations[c]` is what to inject on cycle c, or
`None` for a bubble.

### The kernel

`atalla_meissa_run` in `src/native/atalla_kernels.cpp`, built by `make native`.
AVX-512F, AVX2+F16C and scalar paths, chosen by CPUID at load; the Python
struct points at the same buffers the numpy path uses, so there is one state,
not two.

The vectorisation axis is the **column**. The tree reduces across the *term*
index, whose order must be preserved, so that axis is never vectorised —
columns are mutually independent, and laying the grid out with the column index
contiguous turns each tree level into a plain elementwise vector add.

Bit-exactness is a test, not a hope: the kernel and the numpy reference are
compared bit for bit across both tree shapes, with and without the reducer, at
three array sizes. `ATALLA_NO_NATIVE=1` forces the reference path.

Two details that matter for that:

- a 2-input level is a float32 add; a 4-input level accumulates in double and
  rounds **once**, matching the reference's evaluation order exactly;
- no FMA anywhere — products are rounded to float32 before they are summed,
  which is why the build sets `-ffp-contract=off`.

Transcribed from `sysarr_MEISSA_top.sv`, `mul_grid.sv`,
`pipelined_adder_tree.sv`, `mixed_pipelined_adder_tree.sv` and
`output_buffer.sv` on the atalla repo's `systolic_array_arch` branch.
