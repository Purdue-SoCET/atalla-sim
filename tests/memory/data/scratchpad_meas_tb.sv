`timescale 1ps/1ps
`include "scpad_if.sv"

// Measures scratchpad pad-0 timing: front-end (VLSU) reads and writes, alone
// and back to back, and backend DMA. Logs one line per cycle, sampled on the
// falling edge. "fe_acc" is a request that the next rising edge accepts.
module spad_meas_tb;
    import scpad_pkg::*;
    logic clk = 0;
    logic n_rst;
    always #5 clk = ~clk;

    scpad_if sif(clk, n_rst);
    scratchpad DUT (.sif(sif));

    int fd, cyc = 0;
    string phase = "reset";
    int salt = 0;

    always @(negedge clk) if (n_rst) begin
        $fdisplay(fd, "%0d %s fe_acc=%0d fe_stall=%0d fe_w=%0d fe_row=%0d res=%0d res_d0=%0d rd_en=%0d wr_en=%0d bank_done=%0d be_req=%0d be_w=%0d be_res=%0d dreq=%0d dreq_w=%0d dres=%0d sstall=%0d sdone=%0d",
            cyc, phase,
            sif.vec_req[0].valid && !sif.fe_vec_stall[0], sif.fe_vec_stall[0],
            sif.vec_req[0].write, sif.vec_req[0].row_id,
            sif.vec_res[0].valid, sif.vec_res[0].rdata[0],
            sif.cntrl_spad_req[0].valid, sif.cntrl_spad_wr_req[0].valid,
            |sif.spad_cntrl_res[0],
            sif.be_req[0].valid, sif.be_req[0].write, sif.be_res[0].valid,
            sif.be_dram_req[0].valid, sif.be_dram_req[0].write,
            sif.dram_be_res[0].valid, sif.sched_stall[0], sif.sdma_done[0]);
        cyc++;
    end

    task automatic set_req(input bit write, input int row);
        sif.vec_req[0].valid     = 1'b1;
        sif.vec_req[0].write     = write;
        sif.vec_req[0].spad_addr = '0;
        sif.vec_req[0].num_rows  = 5'(0);
        sif.vec_req[0].num_cols  = 5'(31);
        sif.vec_req[0].row_id    = 5'(row);
        for (int c = 0; c < NUM_COLS; c++)
            sif.vec_req[0].wdata[c] = 16'(salt + row * 32 + c + 1);
    endtask

    // Present requests back to back: the next one goes up the cycle after the
    // previous one is accepted. writes[i] / rows[i] describe request i.
    task automatic fe_stream(input bit writes[], input int rows[]);
        int i = 0;
        bit accepted;
        set_req(writes[0], rows[0]);
        while (i < writes.size()) begin
            @(negedge clk);
            accepted = !sif.fe_vec_stall[0];
            @(posedge clk); #1;
            if (accepted) begin
                i++;
                if (i < writes.size()) set_req(writes[i], rows[i]);
                else sif.vec_req[0].valid = 1'b0;
            end
        end
    endtask

    task automatic idle(input int n);
        repeat (n) @(posedge clk);
        #1;
    endtask

    task automatic dma(input bit store, input int rows);
        int t = 0;
        logic [7:0] id;
        sif.sched_req[0].valid = 1'b1;
        sif.sched_req[0].write = store;
        sif.sched_req[0].spad_addr = 20'd0;
        sif.sched_req[0].dram_addr = 32'd0;
        sif.sched_req[0].num_rows = 5'(rows - 1);
        sif.sched_req[0].num_cols = 5'(31);
        sif.sched_req[0].full_num_cols = 20'(31);
        sif.sched_req[0].scpad_id = '0;
        sif.dram_be_stall[0] = 1'b0;
        do begin
            @(posedge clk); #1;
            sif.sched_req[0].valid = 1'b0;
            if (sif.be_dram_req[0].valid && !sif.be_dram_req[0].write) begin
                id = sif.be_dram_req[0].id;
                sif.dram_be_res[0].valid = 1'b1;
                sif.dram_be_res[0].id = id;
                sif.dram_be_res[0].dram_vector_mask = sif.be_dram_req[0].dram_vector_mask;
                sif.dram_be_res[0].rdata = {16'(id), 16'(id), 16'(id), 16'(id)};
            end else begin
                sif.dram_be_res[0].valid = 1'b0;
            end
            t++;
        end while ((sif.sched_stall[0] || t < 3) && t < 2000);
        sif.dram_be_res[0] = '0;
    endtask

    bit w[];
    int r[];
    initial begin
        fd = $fopen("spad_trace.txt", "w");
        n_rst = 0;
        sif.vec_req[0] = '0; sif.vec_req[1] = '0; sif.vec_req[2] = '0; sif.vec_req[3] = '0;
        sif.sched_req[0] = '0; sif.sched_req[1] = '0; sif.sched_req[2] = '0; sif.sched_req[3] = '0;
        for (int p = 0; p < 4; p++) begin
            sif.dram_be_stall[p] = 1'b0; sif.dram_be_res[p] = '0; sif.fe_vec_res_stall[p] = 1'b0;
        end
        repeat (5) @(posedge clk);
        n_rst = 1;
        idle(5);

        phase = "wr1";     w = '{1}; r = '{0}; fe_stream(w, r); idle(30);
        phase = "rd1";     w = '{0}; r = '{0}; fe_stream(w, r); idle(30);
        phase = "wr_stream";
        w = new[16]; r = new[16];
        foreach (w[i]) begin w[i] = 1; r[i] = i; end
        fe_stream(w, r); idle(40);
        phase = "rd_stream";
        foreach (w[i]) begin w[i] = 0; r[i] = i; end
        fe_stream(w, r); idle(40);
        phase = "mix";
        foreach (w[i]) begin w[i] = i % 2 == 0; r[i] = (i % 2 == 0) ? 20 + i : i; end
        fe_stream(w, r); idle(40);
        phase = "raw";     w = new[2]; r = new[2]; w = '{1, 0}; r = '{9, 9};
        salt = 5000;
        fe_stream(w, r); idle(40);
        phase = "dma_load";  dma(0, 4); idle(40);
        phase = "dma_store"; dma(1, 4); idle(40);
        phase = "end";
        idle(2);
        $fclose(fd);
        $finish;
    end
endmodule
