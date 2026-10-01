# Scratchpad RTL trace

`scratchpad_rtl_trace.txt.xz` is a Questa run of `scratchpad_meas_tb.sv` on
the atalla scratchpad, `rtl/modules/memory/scratchpad/` at branch
`transpose_integration`, `b1ba35ff`. The testbench drives pad 0:
1. one write, then one read;
2. 16 back-to-back writes, then 16 back-to-back reads;
3. alternating writes and reads;
4. a read the cycle after a write to the same row;
5. 4-row backend DMA loads and stores, against an immediate DRAM responder.

It logs one line per cycle, sampled on the falling edge. `fe_acc` marks a
frontend request the next rising edge accepts; `rd_en`/`wr_en` are the
controller's bank enables; `res`/`be_res` are data back at the frontend and
backend. `test_scratchpad_rtl_timing.py` replays it against `Scratchpad`.

To regenerate it, from a checkout of that commit:

```bash
export PATH=/package/eda/mg/questa2021.4/questasim/bin:$PATH
export MGLS_LICENSE_FILE=28000@marina.ecn.purdue.edu
vlib work
vlog -sv +incdir+rtl/include/memory/scratchpad +incdir+rtl/include/common/xbar +incdir+rtl/include \
    rtl/include/common/xbar/xbar_pkg.sv rtl/include/memory/scratchpad/scpad_pkg.sv \
    rtl/include/memory/scratchpad/scpad_if.sv rtl/modules/common/general/fifo.sv \
    rtl/modules/common/memory/sram_bank.sv rtl/modules/memory/scratchpad/*.sv \
    <atalla-sim>/tests/memory/data/scratchpad_meas_tb.sv
vsim -c spad_meas_tb -do "run -all; quit -f"
xz -9e -c spad_trace.txt > scratchpad_rtl_trace.txt.xz
```

The RTL's own `tb/unit/memory/scratchpad/scratchpad_tb.sv` passes all 60 of its
checks on the same build.
