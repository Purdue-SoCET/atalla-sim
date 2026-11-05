#ifndef ATALLA_SPATIAL_H
#define ATALLA_SPATIAL_H
/* Scalar orchestration passes packed BF16 bits through the CPU cache; these
 * leaf routines keep all accelerator staging in scratchpads, never dirty DRAM. */
void atalla_stage_row(int *addresses, int row);
void atalla_compute_tile(int weight_addr, int bias_addr, int rows, int first_k, int last_k);
void atalla_extract_row(int row, int *fp32_bits, int columns);
void atalla_write_value(int address, int fp32_bits);
#endif
