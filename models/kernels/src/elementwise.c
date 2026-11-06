#include "../kernels.h"
#include "../include/sdma.h"

/*
 * Shared 32x32 BF16 tile operations derived from the validated Atalla vector
 * kernels in aihw-ppci-compiler/test_validation_kernels. The validation sources
 * remain standalone; this file owns the callable graph ABI.
 *
 * The current schedule allocator pads every edge tile to 32x32. These routines
 * operate on the full padded tile, so callers must provide matching tile grids.
 */
#define ALL_MASK         -1
#define TILE_LAST_COL    31

/* Expand descriptor validation inside each leaf kernel. The selected compiler
 * does not preserve all live scalar registers across nested helper calls. */
#define CHECK_SAME_SHAPE(a, b) do { \
    int dim = 0; \
    if ((a)->rank != (b)->rank || (a)->count != (b)->count) return -1; \
    while (dim < (a)->rank) { \
        if ((a)->shape[dim] != (b)->shape[dim] || \
            (a)->tiles_per_dim[dim] != (b)->tiles_per_dim[dim]) return -1; \
        dim = dim + 1; \
    } \
} while (0)

int add_kernel(const GlobalTile *lhs, const GlobalTile *rhs,
               GlobalTile *output, int alpha_bits) {
    float alpha;
    asm("addi_s %0, %1, 0" : "=r"(alpha) : "r"(alpha_bits));
    int tile = 0;
    int mask = ALL_MASK;
    int broadcast_rhs = 1;
    CHECK_SAME_SHAPE(lhs, output);
    if (rhs->rank == 1 && output->rank > 1 && rhs->shape[0] == output->shape[output->rank - 1]) {
        broadcast_rhs = 2;
    } else {
        CHECK_SAME_SHAPE(rhs, output);
    }

    while (tile < output->count) {
        int sp = 0;
        int right_tile = broadcast_rhs == 2 ? tile % rhs->count : tile;
        int left_addr = lhs->tiles[tile].base_addr;
        int right_addr = rhs->tiles[right_tile].base_addr;
        int output_addr = output->tiles[tile].base_addr;
        asm("scpad_ld %3, %1, %2\naddi_s %0, %3, 0" : "=r"(sp)
            : "r"(left_addr), "r"(ATALLA_SDMA_TILE32_SID0), "r"(sp));
        asm("scpad_ld %3, %1, %2\naddi_s %0, %3, 0" : "=r"(sp)
            : "r"(right_addr), "r"(ATALLA_SDMA_TILE32_SID1), "r"(sp));

        int row = 0;
        while (row < ATALLA_TILE_HEIGHT) {
            int rhs_row = broadcast_rhs == 2 ? 0 : row;
            vec left = vector_load(sp, row, TILE_LAST_COL, 0);
            vec right = vector_load(sp, rhs_row, TILE_LAST_COL, 1);
            vec scaled = vec_op_masked("*", right, alpha, mask);
            vec result = vec_op_masked("+", left, scaled, mask);
            vector_store(result, sp, row, TILE_LAST_COL, 0);
            row = row + 1;
        }
        scpad_store(sp, output_addr, ATALLA_SDMA_TILE32_SID0);
        tile = tile + 1;
    }
    return 0;
}

int mul_kernel(const GlobalTile *lhs, const GlobalTile *rhs,
               GlobalTile *output) {
    int tile = 0;
    int mask = ALL_MASK;
    CHECK_SAME_SHAPE(lhs, output);
    CHECK_SAME_SHAPE(rhs, output);

    while (tile < output->count) {
        int sp = 0;
        int left_addr = lhs->tiles[tile].base_addr;
        int right_addr = rhs->tiles[tile].base_addr;
        int output_addr = output->tiles[tile].base_addr;
        asm("scpad_ld %3, %1, %2\naddi_s %0, %3, 0" : "=r"(sp)
            : "r"(left_addr), "r"(ATALLA_SDMA_TILE32_SID0), "r"(sp));
        asm("scpad_ld %3, %1, %2\naddi_s %0, %3, 0" : "=r"(sp)
            : "r"(right_addr), "r"(ATALLA_SDMA_TILE32_SID1), "r"(sp));

        int row = 0;
        while (row < ATALLA_TILE_HEIGHT) {
            vec left = vector_load(sp, row, TILE_LAST_COL, 0);
            vec right = vector_load(sp, row, TILE_LAST_COL, 1);
            vec result = vec_op_masked("*", left, right, mask);
            vector_store(result, sp, row, TILE_LAST_COL, 0);
            row = row + 1;
        }
        scpad_store(sp, output_addr, ATALLA_SDMA_TILE32_SID0);
        tile = tile + 1;
    }
    return 0;
}

int mul_scalar_kernel(const GlobalTile *input, GlobalTile *output,
                      int scalar_bits) {
    float scalar;
    asm("addi_s %0, %1, 0" : "=r"(scalar) : "r"(scalar_bits));
    int tile = 0;
    int mask = ALL_MASK;
    CHECK_SAME_SHAPE(input, output);

    while (tile < output->count) {
        int sp = 0;
        int input_addr = input->tiles[tile].base_addr;
        int output_addr = output->tiles[tile].base_addr;
        asm("scpad_ld %3, %1, %2\naddi_s %0, %3, 0" : "=r"(sp)
            : "r"(input_addr), "r"(ATALLA_SDMA_TILE32_SID0), "r"(sp));

        int row = 0;
        while (row < ATALLA_TILE_HEIGHT) {
            vec value = vector_load(sp, row, TILE_LAST_COL, 0);
            vec result = vec_op_masked("*", value, scalar, mask);
            vector_store(result, sp, row, TILE_LAST_COL, 0);
            row = row + 1;
        }
        scpad_store(sp, output_addr, ATALLA_SDMA_TILE32_SID0);
        tile = tile + 1;
    }
    return 0;
}

int relu_kernel(const GlobalTile *input, GlobalTile *output,
                const GlobalTile *bias) {
    int tile = 0;
    int mask = ALL_MASK;
    (void)bias;
    CHECK_SAME_SHAPE(input, output);

    while (tile < output->count) {
        int sp = 0;
        int input_addr = input->tiles[tile].base_addr;
        int output_addr = output->tiles[tile].base_addr;
        asm("scpad_ld %3, %1, %2\naddi_s %0, %3, 0" : "=r"(sp)
            : "r"(input_addr), "r"(ATALLA_SDMA_TILE32_SID0), "r"(sp));

        vec zero_vec = vector_load(sp, 0, TILE_LAST_COL, 0);
        zero_vec = vec_op_masked("*", zero_vec, 0.0, mask);
        int row = 0;
        while (row < ATALLA_TILE_HEIGHT) {
            vec value = vector_load(sp, row, TILE_LAST_COL, 0);
            int negative = make_mask("<", value, zero_vec, mask);
            vec result = vec_op_masked("*", value, 0.0, negative, value);
            vector_store(result, sp, row, TILE_LAST_COL, 0);
            row = row + 1;
        }
        scpad_store(sp, output_addr, ATALLA_SDMA_TILE32_SID0);
        tile = tile + 1;
    }
    return 0;
}
