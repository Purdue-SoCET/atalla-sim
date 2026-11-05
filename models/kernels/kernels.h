#ifndef ATALLA_KERNELS_H
#define ATALLA_KERNELS_H

/* Shared graph descriptor ABI. Addresses are byte offsets in the graph DRAM image. */
#define ATALLA_MAX_RANK 8
#define ATALLA_TILE_HEIGHT 32
#define ATALLA_TILE_WIDTH 32
#define ATALLA_TILE_BYTES 2048
#ifndef NULL
#define NULL 0
#endif

typedef struct TileDesc32 {
    unsigned int base_addr;
} TileDesc32;

typedef struct GlobalTile {
    int rank;
    int shape[ATALLA_MAX_RANK];
    int tiles_per_dim[ATALLA_MAX_RANK];
    int count;
    TileDesc32 *tiles;
} GlobalTile;

/* Elementwise kernels accept equal-shaped, identically tiled BF16 tensors. */
int add_kernel(const GlobalTile *lhs, const GlobalTile *rhs,
               GlobalTile *output, float alpha);
int mul_kernel(const GlobalTile *lhs, const GlobalTile *rhs,
               GlobalTile *output);
int mul_scalar_kernel(const GlobalTile *input, GlobalTile *output,
                      float scalar);
int relu_kernel(const GlobalTile *input, GlobalTile *output,
                const GlobalTile *bias);

/* These signatures reserve the graph schedule ABI; capability-gated until implemented. */
int matmul_kernel(const GlobalTile *lhs, const GlobalTile *rhs,
                  GlobalTile *output);
int conv_kernel(const GlobalTile *input, const GlobalTile *weight,
                const GlobalTile *bias, GlobalTile *output,
                int stride_h, int stride_w, int pad_h, int pad_w,
                int dilation_h, int dilation_w, int groups);
int maxpool_kernel(const GlobalTile *input, GlobalTile *output,
                   int kernel_h, int kernel_w, int stride_h, int stride_w,
                   int pad_h, int pad_w, int dilation_h, int dilation_w,
                   int ceil_mode);
int avgpool_kernel(const GlobalTile *input, GlobalTile *output,
                   int kernel_h, int kernel_w, int stride_h, int stride_w,
                   int pad_h, int pad_w, int global_pool);

#endif
