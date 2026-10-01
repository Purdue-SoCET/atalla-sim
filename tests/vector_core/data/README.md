# Transpose unit RTL trace

`transpose_unit_rtl_trace.txt.xz` is a Questa run of the atalla repo's unit
testbench `tb/unit/vector/transpose_unit_tb.sv` on `rtl/modules/vector/transpose_unit.sv`,
branch `transpose_integration` at `b1ba35ff`. `tracer.sv` logged it: one line
per cycle after reset, sampled on the falling edge so every signal is settled,
plus `vec_in` on cycles that push and `vec_out` on cycles in `DONE`.
`test_transpose_rtl_trace.py` replays it against `TransposeUnit`.

To regenerate it, from a checkout of that commit (the license server is the
one `module load siemens/questa/2021.4` sets):

```bash
export PATH=/package/eda/mg/questa2021.4/questasim/bin:$PATH
export MGLS_LICENSE_FILE=28000@marina.ecn.purdue.edu
vlib work
vlog -sv +incdir+rtl/include/common/xbar +incdir+rtl/include/vector +incdir+rtl/include \
    rtl/include/common/xbar/xbar_pkg.sv rtl/modules/common/xbar/param_switch.sv \
    rtl/modules/common/xbar/clos.sv rtl/modules/common/memory/sram_bank.sv \
    rtl/modules/vector/transpose_unit.sv tb/unit/vector/transpose_unit_tb.sv \
    <atalla-sim>/tests/vector_core/data/tracer.sv
vsim -c transpose_unit_tb tracer -do "run -all; quit -f"
xz -9e -c trace.txt > transpose_unit_rtl_trace.txt.xz
```

The testbench should end with `RESULT: ALL TESTS PASSED`. If the RTL changes,
regenerate the trace; the replay test then shows the first cycle where the
model and the new RTL part ways.
