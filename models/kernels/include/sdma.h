#ifndef ATALLA_KERNELS_SDMA_H
#define ATALLA_KERNELS_SDMA_H

/*
 * Atalla SDMA control word fields used by the validation kernels:
 * [31:30] scratchpad SID, [29:25] rows-1, [24:20] columns-1,
 * [19:0] DRAM row stride-1 in elements.
 */
#define ATALLA_SDMA_CONTROL(sid, rows, columns, row_stride) \
    (((sid) << 30) | (((rows) - 1) << 25) | (((columns) - 1) << 20) | \
     ((row_stride) - 1))

#define ATALLA_SDMA_TILE32_SID0 ATALLA_SDMA_CONTROL(0, 32, 32, 32)
#define ATALLA_SDMA_TILE32_SID1 ATALLA_SDMA_CONTROL(1, 32, 32, 32)

/* DMA call sites return an opaque scratch-base token through inline assembly.
 * Vector loads consume that token to wait for the DMA's held rs1 register.
 * Keep asm strings at the call sites: this compiler loses string constraints
 * when they are expanded inside a preprocessor macro. */
#endif
