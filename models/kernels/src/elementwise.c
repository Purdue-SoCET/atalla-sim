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

static int same_shape(const GlobalTile *a, const GlobalTile *b) {
    int dim = 0;
    if (a->rank != b->rank) return 0;
    while (dim < a->rank) {
        if (a->shape[dim] != b->shape[dim]) return 0;
        if (a->tiles_per_dim[dim] != b->tiles_per_dim[dim]) return 0;
        dim = dim + 1;
    }
    return a->count == b->count;
}

static int same_tile_grid(const GlobalTile *a, const GlobalTile *b,
                          const GlobalTile *out) {
    return same_shape(a, out) && same_shape(b, out);
}

static int add_shapes_supported(const GlobalTile *lhs, const GlobalTile *rhs,
                                const GlobalTile *out) {
    if (!same_shape(lhs, out)) return 0;
    if (same_shape(rhs, out)) return 1;
    if (rhs->rank == 1 && out->rank > 1 &&
        rhs->shape[0] == out->shape[out->rank - 1] &&
        rhs->tiles_per_dim[0] == out->tiles_per_dim[out->rank - 1]) return 2;
    return 0;
}

int add_kernel(const GlobalTile *lhs, const GlobalTile *rhs,
               GlobalTile *output, float alpha) {
    int tile = 0;
    int mask = ALL_MASK;
    int broadcast_rhs = add_shapes_supported(lhs, rhs, output);
    if (!broadcast_rhs) return -1;

    while (tile < output->count) {
        int sp = 0;
        int right_tile = broadcast_rhs == 2 ? tile % rhs->count : tile;
        int left_addr = lhs->tiles[tile].base_addr;
        int right_addr = rhs->tiles[right_tile].base_addr;
        int output_addr = output->tiles[tile].base_addr;
        scpad_load(sp, left_addr, ATALLA_SDMA_TILE32_SID0);
        scpad_load(sp, right_addr, ATALLA_SDMA_TILE32_SID1);

        int row = 0;
        while (row < ATALLA_TILE_HEIGHT) {
            int rhs_row = broadcast_rhs == 2 ? 0 : row;
            vec left = vector_load(0, row, TILE_LAST_COL, 0);
            vec right = vector_load(0, rhs_row, TILE_LAST_COL, 1);
            vec scaled = vec_op_masked("*", right, alpha, mask);
            vec result = vec_op_masked("+", left, scaled, mask);
            vector_store(result, 0, row, TILE_LAST_COL, 0);
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
    if (!same_tile_grid(lhs, rhs, output)) return -1;

    while (tile < output->count) {
        int sp = 0;
        int left_addr = lhs->tiles[tile].base_addr;
        int right_addr = rhs->tiles[tile].base_addr;
        int output_addr = output->tiles[tile].base_addr;
        scpad_load(sp, left_addr, ATALLA_SDMA_TILE32_SID0);
        scpad_load(sp, right_addr, ATALLA_SDMA_TILE32_SID1);

        int row = 0;
        while (row < ATALLA_TILE_HEIGHT) {
            vec left = vector_load(0, row, TILE_LAST_COL, 0);
            vec right = vector_load(0, row, TILE_LAST_COL, 1);
            vec result = vec_op_masked("*", left, right, mask);
            vector_store(result, 0, row, TILE_LAST_COL, 0);
            row = row + 1;
        }
        scpad_store(sp, output_addr, ATALLA_SDMA_TILE32_SID0);
        tile = tile + 1;
    }
    return 0;
}

int mul_scalar_kernel(const GlobalTile *input, GlobalTile *output,
                      float scalar) {
    int tile = 0;
    int mask = ALL_MASK;
    if (!same_shape(input, output)) return -1;

    while (tile < output->count) {
        int sp = 0;
        int input_addr = input->tiles[tile].base_addr;
        int output_addr = output->tiles[tile].base_addr;
        scpad_load(sp, input_addr, ATALLA_SDMA_TILE32_SID0);

        int row = 0;
        while (row < ATALLA_TILE_HEIGHT) {
            vec value = vector_load(0, row, TILE_LAST_COL, 0);
            vec result = vec_op_masked("*", value, scalar, mask);
            vector_store(result, 0, row, TILE_LAST_COL, 0);
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
    if (!same_shape(input, output)) return -1;

    while (tile < output->count) {
        int sp = 0;
        int input_addr = input->tiles[tile].base_addr;
        int output_addr = output->tiles[tile].base_addr;
        scpad_load(sp, input_addr, ATALLA_SDMA_TILE32_SID0);

        vec zero_vec = vector_load(0, 0, TILE_LAST_COL, 0);
        zero_vec = vec_op_masked("*", zero_vec, 0.0, mask);
        int row = 0;
        while (row < ATALLA_TILE_HEIGHT) {
            vec value = vector_load(0, row, TILE_LAST_COL, 0);
            int negative = make_mask("<", value, zero_vec, mask);
            vec result = vec_op_masked("*", value, 0.0, negative);
            vector_store(result, 0, row, TILE_LAST_COL, 0);
            row = row + 1;
        }
        scpad_store(sp, output_addr, ATALLA_SDMA_TILE32_SID0);
        tile = tile + 1;
    }
    return 0;
}
