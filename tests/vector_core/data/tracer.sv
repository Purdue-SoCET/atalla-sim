// Logs the transpose unit every cycle, sampled at the falling edge so every
// signal is settled. One line per cycle after reset.
module tracer;
  int fd, cyc;
  initial begin fd = $fopen("trace.txt", "w"); cyc = 0; end
  always @(negedge transpose_unit_tb.CLK) begin
    if (transpose_unit_tb.nRST) begin
      $fdisplay(fd, "%0d %s cnt=%0d lat=%0d rdy_in=%0d vld_out=%0d push=%0d pop=%0d rdy_out=%0d ren=%0d wen=%0d rdone=%0d wdone=%0d",
        cyc, transpose_unit_tb.DUT.state.name(), transpose_unit_tb.DUT.count,
        transpose_unit_tb.DUT.lat_count,
        transpose_unit_tb.tif.out.ready_in, transpose_unit_tb.tif.out.valid_out,
        transpose_unit_tb.tif.in.push_req && transpose_unit_tb.tif.in.valid_in,
        transpose_unit_tb.tif.in.pop_req, transpose_unit_tb.tif.in.ready_out,
        transpose_unit_tb.DUT.ren, transpose_unit_tb.DUT.wen,
        transpose_unit_tb.DUT.sram_rdone[0], transpose_unit_tb.DUT.sram_wdone[0]);
      if (transpose_unit_tb.tif.in.push_req && transpose_unit_tb.tif.in.valid_in && transpose_unit_tb.tif.out.ready_in)
        $fdisplay(fd, "  vec_in=%h", transpose_unit_tb.tif.in.vec_in);
      if (transpose_unit_tb.tif.out.valid_out)
        $fdisplay(fd, "  vec_out=%h", transpose_unit_tb.tif.out.vec_out);
      cyc++;
    end
  end
endmodule
