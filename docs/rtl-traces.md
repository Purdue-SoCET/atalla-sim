# Comparing atalla-sim with an RTL simulation

atalla-sim writes a per-cycle execution trace. An RTL testbench can write
the same format, and `atalla-sim compare-rtl` finds where the two part ways:

```bash
atalla-sim run prog.in --trace sim.trace          # atalla-sim's
# ... the RTL testbench writes rtl.trace for the same program image ...
atalla-sim compare-rtl rtl.trace sim.trace
```

```
comparing rtl.trace (486 events) with sim.trace (486 events); compared: halt, issue, wb
sim.trace's cycles shifted by +5 so the first packets issue together (--no-align keeps them)
1 difference; earliest first:
  cycle 380: write #3 to x27: value 00000010 in rtl.trace (cycle 380), 00000013 in sim.trace (cycle 380)
around cycle 380 in rtl.trace:
     377 issue  17c
     378 issue  190
     380 wb    x27 00000010
     382 issue  154
around cycle 380 in sim.trace:
     377 issue  17c
     378 issue  190
     380 wb    x27 00000013
     382 issue  154
```

(Here `rtl.trace` is gemm's `sim.trace` with five more reset cycles and one
value changed, to show the output; no RTL tracer exists yet.)

It reports, earliest first:

- the first packet issued at a **different pc** (control flow went another
  way), and the first issued at a **different cycle** (timing), with the
  offset;
- for each register, the first write with a different **value** or a
  different **cycle**;
- the halt cycle;

and then the events around the first difference in both traces. The two
traces are lined up on their first issued packet, because testbenches count
reset cycles differently; `--no-align` compares cycles as written. Exit
status 1 when they differ.

Before reading a difference as a bug, check
[known-differences.md](known-differences.md): some are intended (the model
doesn't reproduce RTL bugs, and keeps scalar BF16 in the upper half of a
register where the RTL uses the lower half).

## The format

Text, one event per line; `#` starts a comment.

```
# atalla-trace v1
<cycle> issue <pc>              decode 2 issued the packet at <pc> (hex)
<cycle> wb x<r> <value>         scalar register r written: 8 hex digits
<cycle> wb m<r> <value>         mask register r written: 8 hex digits
<cycle> wb v<r> <h0>,...,<h31>  vector register r written: 32 BF16 bit patterns,
                                4 hex digits each, element 0 first
<cycle> halt                    the program finished
```

`<cycle>` is decimal. A write's cycle is the clock edge it reaches the
register file. Writes to `x0` and `v0` are left out. A trace may contain
only some kinds of event: only kinds both traces contain are compared, so an
RTL tracer can start with `issue` and add writes later.

## Writing it from the RTL

On the scheduler branch (`scheduler_integration_SP26_joshklug`), sample at
each rising clock edge, in a separate tracer module so the testbench itself
stays unchanged:

| event | signals (`decode_2_if` in decode_2.sv) |
|---|---|
| `issue` | `ready` while the D1/D2 latch holds a valid packet (scheduler_core.sv); the pc is `pc_out` |
| `wb x` | per write port: `scalar_WB_WEN[i]`, `scalar_WB_wsel[i]`, `scalar_WB_wdata[i]` |
| `wb v` | per write port: `vector_WB_WEN[i]`, `vector_WB_wsel[i]`, `vector_WB_wdata[i]` (32 x 16 bits) |
| `wb m` | per write port: `mask_WB_WEN[i]`, `mask_WB_wsel[i]`, `mask_WB_wdata[i]` |
| `halt` | the core's halt output, once the work in flight has drained |

`$fdisplay(fd, "%0d issue %0h", cycle, pc_out)` and the like, one line per
event, with `cycle` counted from reset.
