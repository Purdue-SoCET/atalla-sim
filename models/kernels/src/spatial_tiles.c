#include "../include/spatial.h"
#include "../include/sdma.h"

/* Construct an im2col row directly in scratchpad 0. No CPU-to-DMA cache
 * handoff: the CPU reads an address-only plan and gathers current input values. */
void atalla_stage_row(int *addresses, int row) {
    int sp = 0;
    int all = -1;
    vec zero = vector_load(sp, 0, 31, 0);
    zero = vec_op_masked("*", zero, 0.0, all);
    vec result = zero;
    int lane = 0;
    while (lane < 32) {
        int bits;
        float value;
        int selected;
        bits = 0;
        int address_0 = addresses[lane + 0];
        if (address_0 != -1) {
            int *source_0 = (int *)(address_0 & -4);
            bits = (source_0[0] >> ((address_0 & 2) << 3)) << 16;
        }
        asm("addi_s %0, %1, 0" : "=r"(value) : "r"(bits));
        selected = 1 << (lane + 0);
        result = vec_op_masked("+", zero, value, selected, result);
        bits = 0;
        int address_1 = addresses[lane + 1];
        if (address_1 != -1) {
            int *source_1 = (int *)(address_1 & -4);
            bits = (source_1[0] >> ((address_1 & 2) << 3)) << 16;
        }
        asm("addi_s %0, %1, 0" : "=r"(value) : "r"(bits));
        selected = 1 << (lane + 1);
        result = vec_op_masked("+", zero, value, selected, result);
        bits = 0;
        int address_2 = addresses[lane + 2];
        if (address_2 != -1) {
            int *source_2 = (int *)(address_2 & -4);
            bits = (source_2[0] >> ((address_2 & 2) << 3)) << 16;
        }
        asm("addi_s %0, %1, 0" : "=r"(value) : "r"(bits));
        selected = 1 << (lane + 2);
        result = vec_op_masked("+", zero, value, selected, result);
        bits = 0;
        int address_3 = addresses[lane + 3];
        if (address_3 != -1) {
            int *source_3 = (int *)(address_3 & -4);
            bits = (source_3[0] >> ((address_3 & 2) << 3)) << 16;
        }
        asm("addi_s %0, %1, 0" : "=r"(value) : "r"(bits));
        selected = 1 << (lane + 3);
        result = vec_op_masked("+", zero, value, selected, result);
        lane = lane + 4;
    }
    vector_store(result, sp, row, 31, 0);
}

/* A is already in scratchpad 0; C remains in scratchpad 1 across K tiles.
 * Constant W streams are packed by the runner into the ELF data image. */
void atalla_compute_tile(int weight_addr, int bias_addr, int rows,
                         int first_k, int last_k) {
    int sp = 0;
    int all = -1;
    asm("scpad_ld %3, %1, %2\naddi_s %0, %3, 0" : "=r"(sp)
        : "r"(weight_addr), "r"(ATALLA_SDMA_CONTROL(2u,32,32,32)), "r"(sp));
    int wi = 0;
    while (wi < 32) {
        vec w = vector_load(sp, wi, 31, 2);
        load_weights(w);
        wi = wi + 1;
    }
    vec zero = vector_load(sp, 0, 31, 2);
    zero = vec_op_masked("*", zero, 0.0, all);
    vec bias = zero;
    if (last_k && bias_addr >= 0) {
        asm("scpad_ld %3, %1, %2\naddi_s %0, %3, 0" : "=r"(sp)
            : "r"(bias_addr), "r"(ATALLA_SDMA_CONTROL(3u,1,32,32)), "r"(sp));
        bias = vector_load(sp, 0, 31, 3);
    }
    int row = 0;
    while (row < rows) {
        vec a = vector_load(sp, row, 31, 0);
        vec c = vector_load(sp, row, 31, 1);
        if (first_k) c = zero;
        vec result = gemm(a, c, all);
        if (last_k) result = vec_op_masked("+", result, bias, all);
        vector_store(result, sp, row, 31, 1);
        row = row + 1;
    }
}

/* Write a scalar result through the accelerator path. A CPU store here would
 * remain dirty in the data cache and be invisible to the next graph DMA. */
void atalla_write_value(int address, int bits) {
    int sp = 0;
    int all = -1;
    float value;
    asm("addi_s %0, %1, 0" : "=r"(value) : "r"(bits));
    vec zero = vector_load(sp, 0, 31, 3);
    zero = vec_op_masked("*", zero, 0.0, all);
    vec result = vec_op_masked("+", zero, value, all);
    vector_store(result, sp, 0, 31, 3);
    asm("scpad_st %3, %1, %2\naddi_s %0, %3, 0" : "=r"(sp)
        : "r"(address), "r"(ATALLA_SDMA_CONTROL(3u,1,1,1)), "r"(sp));
}

void atalla_extract_row(int row, int *bits, int columns) {
    int sp = 0;
    vec result = vector_load(sp, row, 31, 1);
    int value;
    if (columns > 0) {
        asm("vmov_vts %0, %1, 0" : "=r"(value) : "v"(result));
        bits[0] = value;
    }
    if (columns > 1) {
        asm("vmov_vts %0, %1, 1" : "=r"(value) : "v"(result));
        bits[1] = value;
    }
    if (columns > 2) {
        asm("vmov_vts %0, %1, 2" : "=r"(value) : "v"(result));
        bits[2] = value;
    }
    if (columns > 3) {
        asm("vmov_vts %0, %1, 3" : "=r"(value) : "v"(result));
        bits[3] = value;
    }
    if (columns > 4) {
        asm("vmov_vts %0, %1, 4" : "=r"(value) : "v"(result));
        bits[4] = value;
    }
    if (columns > 5) {
        asm("vmov_vts %0, %1, 5" : "=r"(value) : "v"(result));
        bits[5] = value;
    }
    if (columns > 6) {
        asm("vmov_vts %0, %1, 6" : "=r"(value) : "v"(result));
        bits[6] = value;
    }
    if (columns > 7) {
        asm("vmov_vts %0, %1, 7" : "=r"(value) : "v"(result));
        bits[7] = value;
    }
    if (columns > 8) {
        asm("vmov_vts %0, %1, 8" : "=r"(value) : "v"(result));
        bits[8] = value;
    }
    if (columns > 9) {
        asm("vmov_vts %0, %1, 9" : "=r"(value) : "v"(result));
        bits[9] = value;
    }
    if (columns > 10) {
        asm("vmov_vts %0, %1, 10" : "=r"(value) : "v"(result));
        bits[10] = value;
    }
    if (columns > 11) {
        asm("vmov_vts %0, %1, 11" : "=r"(value) : "v"(result));
        bits[11] = value;
    }
    if (columns > 12) {
        asm("vmov_vts %0, %1, 12" : "=r"(value) : "v"(result));
        bits[12] = value;
    }
    if (columns > 13) {
        asm("vmov_vts %0, %1, 13" : "=r"(value) : "v"(result));
        bits[13] = value;
    }
    if (columns > 14) {
        asm("vmov_vts %0, %1, 14" : "=r"(value) : "v"(result));
        bits[14] = value;
    }
    if (columns > 15) {
        asm("vmov_vts %0, %1, 15" : "=r"(value) : "v"(result));
        bits[15] = value;
    }
    if (columns > 16) {
        asm("vmov_vts %0, %1, 16" : "=r"(value) : "v"(result));
        bits[16] = value;
    }
    if (columns > 17) {
        asm("vmov_vts %0, %1, 17" : "=r"(value) : "v"(result));
        bits[17] = value;
    }
    if (columns > 18) {
        asm("vmov_vts %0, %1, 18" : "=r"(value) : "v"(result));
        bits[18] = value;
    }
    if (columns > 19) {
        asm("vmov_vts %0, %1, 19" : "=r"(value) : "v"(result));
        bits[19] = value;
    }
    if (columns > 20) {
        asm("vmov_vts %0, %1, 20" : "=r"(value) : "v"(result));
        bits[20] = value;
    }
    if (columns > 21) {
        asm("vmov_vts %0, %1, 21" : "=r"(value) : "v"(result));
        bits[21] = value;
    }
    if (columns > 22) {
        asm("vmov_vts %0, %1, 22" : "=r"(value) : "v"(result));
        bits[22] = value;
    }
    if (columns > 23) {
        asm("vmov_vts %0, %1, 23" : "=r"(value) : "v"(result));
        bits[23] = value;
    }
    if (columns > 24) {
        asm("vmov_vts %0, %1, 24" : "=r"(value) : "v"(result));
        bits[24] = value;
    }
    if (columns > 25) {
        asm("vmov_vts %0, %1, 25" : "=r"(value) : "v"(result));
        bits[25] = value;
    }
    if (columns > 26) {
        asm("vmov_vts %0, %1, 26" : "=r"(value) : "v"(result));
        bits[26] = value;
    }
    if (columns > 27) {
        asm("vmov_vts %0, %1, 27" : "=r"(value) : "v"(result));
        bits[27] = value;
    }
    if (columns > 28) {
        asm("vmov_vts %0, %1, 28" : "=r"(value) : "v"(result));
        bits[28] = value;
    }
    if (columns > 29) {
        asm("vmov_vts %0, %1, 29" : "=r"(value) : "v"(result));
        bits[29] = value;
    }
    if (columns > 30) {
        asm("vmov_vts %0, %1, 30" : "=r"(value) : "v"(result));
        bits[30] = value;
    }
    if (columns > 31) {
        asm("vmov_vts %0, %1, 31" : "=r"(value) : "v"(result));
        bits[31] = value;
    }
}
