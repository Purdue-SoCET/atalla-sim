#include "../kernels.h"
#include "../include/sdma.h"

/* One callable GEMM for every graph tile. Weights are K x N after lowering.
 * Each 32 x 32 tile is physically padded in DRAM by the graph allocator.
 * The K loop follows gemm_tiled_pipelined.c's load_weights/gemm sequence. */
int matmul_kernel(const GlobalTile *lhs, const GlobalTile *rhs,
                  GlobalTile *output) {
    int mi = 0;
    if (rhs->storage_layout != 1) return -1;
    int m_tiles;
    int n_tiles;
    int k_tiles;
    if (lhs->rank != 2 || rhs->rank != 2 || output->rank != 2) return -1;
    if (lhs->shape[1] != rhs->shape[0] ||
        lhs->shape[0] != output->shape[0] ||
        rhs->shape[1] != output->shape[1]) return -1;
    m_tiles = lhs->tiles_per_dim[0];
    k_tiles = lhs->tiles_per_dim[1];
    n_tiles = rhs->tiles_per_dim[1];
    if (rhs->tiles_per_dim[0] != k_tiles ||
        output->tiles_per_dim[0] != m_tiles ||
        output->tiles_per_dim[1] != n_tiles) return -1;
    while (mi < m_tiles) {
        int ni = 0;
        while (ni < n_tiles) {
            int sp = 0;
            int mask = -1;
            int out_addr = output->tiles[mi * n_tiles + ni].base_addr;
            asm("scpad_ld %3, %1, %2\naddi_s %0, %3, 0" : "=r"(sp)
            : "r"(out_addr), "r"(ATALLA_SDMA_TILE32_SID1), "r"(sp));
            int ki = 0;
            while (ki < k_tiles) {
                int weight_addr = rhs->tiles[ki * n_tiles + ni].base_addr;
                int input_addr = lhs->tiles[mi * k_tiles + ki].base_addr;
                asm("scpad_ld %3, %1, %2\naddi_s %0, %3, 0" : "=r"(sp)
            : "r"(weight_addr), "r"(ATALLA_SDMA_TILE32_SID0), "r"(sp));
                int row = 0;
                while (row < 32) {
                    vec weights = vector_load(sp, row, 31, 0);
                    load_weights(weights);
                    row = row + 1;
                }
                asm("scpad_ld %3, %1, %2\naddi_s %0, %3, 0" : "=r"(sp)
            : "r"(input_addr), "r"(ATALLA_SDMA_TILE32_SID0), "r"(sp));
                row = 0;
                while (row < 32) {
                    vec a = vector_load(sp, row, 31, 0);
                    vec c = vector_load(sp, row, 31, 1);
                    vec result = gemm(a, c, mask);
                    vector_store(result, sp, row, 31, 1);
                    row = row + 1;
                }
                ki = ki + 1;
            }
            scpad_store(sp, out_addr, ATALLA_SDMA_TILE32_SID1);
            ni = ni + 1;
        }
        mi = mi + 1;
    }
    return 0;
}
