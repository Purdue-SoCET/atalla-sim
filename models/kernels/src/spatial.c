#include "../kernels.h"
#include "../include/sdma.h"
#include "../include/spatial.h"

/* Keep scalar indexing inline: this compiler does not preserve all scalar
 * temporaries across nested function calls, including argument evaluation. */
#define NCHW_TILE(t,b,c,y,x) \
    ((((b) * (t)->shape[1] + (c)) * (t)->tiles_per_dim[2] + ((y)>>5)) * (t)->tiles_per_dim[3] + ((x)>>5))
#define NCHW_ADDRESS(t,b,c,y,x) \
    ((t)->tiles[NCHW_TILE(t,b,c,y,x)].base_addr + ((((y)&31))*32 + ((x)&31))*2)
#define READ_NCHW(result,t,b,c,y,x) do { \
    int *read_base = (int *)(t)->tiles[NCHW_TILE(t,b,c,y,x)].base_addr; \
    int read_index = (((y)&31))*32 + ((x)&31); \
    result = (read_base[(read_index>>1)] >> (((read_index&1))*16)) & 65535; \
} while (0)
#define write_nchw(t,b,c,y,x,value) \
    atalla_write_value(NCHW_ADDRESS(t,b,c,y,x), (value) << 16)
#define bf16_order(value) \
    (((value) & 0x8000) ? ((value) ^ 0xffff) : ((value) ^ 0x8000))

/* Generic convolution gathers one im2col row at a time through scalar reads,
 * but stages and computes it entirely in scratchpads. No dirty scalar buffer
 * is handed to DMA. Constant OIHW weights are packed into layout 2 by runner. */
int conv_kernel(const GlobalTile *input, const GlobalTile *weight,
                const GlobalTile *bias, GlobalTile *output,
                int stride_h, int stride_w, int pad_h, int pad_w,
                int dilation_h, int dilation_w, int groups) {
    int result_bits[32];
    int batch = 0;
    int in_per_group;
    int out_per_group;
    int kh;
    int kw;
    int positions;
    int k_total;
    int k_tiles;
    int n_tiles;
    if (input->rank != 4 || weight->rank != 4 || output->rank != 4 || weight->storage_layout != 2 || !output->gather_addr) return -1;
    if (groups < 1 || input->shape[1] % groups || weight->shape[0] % groups) return -1;
    if (stride_h < 1 || stride_w < 1 || dilation_h < 1 || dilation_w < 1) return -1;
    if (output->shape[0] != input->shape[0] || output->shape[1] != weight->shape[0]) return -1;
    if (bias && (bias->rank != 1 || bias->shape[0] != output->shape[1])) return -1;
    in_per_group = input->shape[1] / groups;
    if (weight->shape[1] != in_per_group) return -1;
    out_per_group = weight->shape[0] / groups;
    kh = weight->shape[2];
    kw = weight->shape[3];
    positions = output->shape[2] * output->shape[3];
    k_total = in_per_group * kh * kw;
    k_tiles = (k_total + 31) / 32;
    n_tiles = (out_per_group + 31) / 32;
    while (batch < output->shape[0]) {
        int group = 0;
        while (group < groups) {
            int ni = 0;
            while (ni < n_tiles) {
                int *gather = (int *)output->gather_addr;
                gather = gather + (batch * groups + group) * positions * k_tiles * 32;
                int channels = out_per_group - ni * 32;
                if (channels > 32) channels = 32;
                int position_block = 0;
                while (position_block < positions) {
                    int rows = positions - position_block;
                    if (rows > 32) rows = 32;
                    int ki = 0;
                    while (ki < k_tiles) {
                        int row = 0;
                        while (row < rows) {
                            atalla_stage_row(gather, row);
                            gather = gather + 32;
                            row = row + 1;
                        }
                        int tile = (group * k_tiles + ki) * n_tiles + ni;
                        int bias_address = -1;
                        if (bias) bias_address = bias->tiles[(group * out_per_group + ni * 32) / 32].base_addr;
                        /* Grouped channel offsets need a separate aligned bias stream;
                         * the allocator currently accepts aligned groups or no bias. */
                        if (bias && (group * out_per_group) % 32 && groups > 1) return -1;
                        atalla_compute_tile(weight->tiles[tile].base_addr, bias_address, rows, ki == 0, ki == k_tiles - 1);
                        ki = ki + 1;
                    }
                    int row = 0;
                    int out_y = position_block / output->shape[3];
                    int out_x = position_block % output->shape[3];
                    while (row < rows) {
                        atalla_extract_row(row, result_bits, channels);
                        int col = 0;
                        int position = position_block + row;
                        while (col < channels) {
                            int channel = group * out_per_group + ni * 32 + col;
                            write_nchw(output, batch, channel, out_y, out_x, result_bits[col] >> 16);
                            col = col + 1;
                        }
                        row = row + 1;
                        out_x = out_x + 1;
                        if (out_x == output->shape[3]) { out_x = 0; out_y = out_y + 1; }
                    }
                    position_block = position_block + 32;
                }
                ni = ni + 1;
            }
            group = group + 1;
        }
        batch = batch + 1;
    }
    return 0;
}

int maxpool_kernel(const GlobalTile *input, GlobalTile *output,
                   int kernel_h, int kernel_w, int stride_h, int stride_w,
                   int pad_h, int pad_w, int dilation_h, int dilation_w,
                   int ceil_mode) {
    int b = 0;
    if (input->rank != 4 || output->rank != 4 ||
        input->shape[0] != output->shape[0] || input->shape[1] != output->shape[1] ||
        kernel_h < 1 || kernel_w < 1 || stride_h < 1 || stride_w < 1 ||
        dilation_h != 1 || dilation_w != 1 || ceil_mode != 0) return -1;
    while (b < output->shape[0]) {
        int c = 0;
        while (c < output->shape[1]) {
            int oy = 0;
            while (oy < output->shape[2]) {
                int ox = 0;
                while (ox < output->shape[3]) {
                    int best = 0xff80u;
                    int ky = 0;
                    while (ky < kernel_h) {
                        int kx = 0;
                        while (kx < kernel_w) {
                            int iy = oy * stride_h - pad_h + ky;
                            int ix = ox * stride_w - pad_w + kx;
                            if (iy >= 0 && iy < input->shape[2] && ix >= 0 && ix < input->shape[3]) {
                                int v;
                                READ_NCHW(v, input, b, c, iy, ix);
                                if (bf16_order(v) > bf16_order(best)) best = v;
                            }
                            kx = kx + 1;
                        }
                        ky = ky + 1;
                    }
                    write_nchw(output, b, c, oy, ox, best);
                    ox = ox + 1;
                }
                oy = oy + 1;
            }
            c = c + 1;
        }
        b = b + 1;
    }
    return 0;
}

int flatten_kernel(const GlobalTile *input, GlobalTile *output) {
    int b = 0;
    if (input->rank != 4 || output->rank != 2 || input->shape[0] != output->shape[0] ||
        output->shape[1] != input->shape[1] * input->shape[2] * input->shape[3]) return -1;
    while (b < input->shape[0]) {
        int c = 0;
        while (c < input->shape[1]) {
            int y = 0;
            while (y < input->shape[2]) {
                int x = 0;
                while (x < input->shape[3]) {
                    int col = (c * input->shape[2] + y) * input->shape[3] + x;
                    int tile = (b >> 5) * output->tiles_per_dim[1] + (col >> 5);
                    int *dst = (int *)output->tiles[tile].base_addr;
                    int value;
                    READ_NCHW(value, input, b, c, y, x);
                    atalla_write_value((int)dst + ((b & 31) * 32 + (col & 31)) * 2, value << 16);
                    x = x + 1;
                }
                y = y + 1;
            }
            c = c + 1;
        }
        b = b + 1;
    }
    return 0;
}
