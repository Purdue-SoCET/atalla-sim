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

Transcribed from `sysarr_MEISSA_top.sv`, `mul_grid.sv`,
`pipelined_adder_tree.sv`, `mixed_pipelined_adder_tree.sv` and
`output_buffer.sv` on the atalla repo's `systolic_array_arch` branch.
