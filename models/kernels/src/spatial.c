#include "../kernels.h"
#include "../include/sdma.h"

/* NCHW tensors use one independently padded 32x32 tile per channel plane. */
static int *nchw_base(const GlobalTile *tensor,
                      int batch, int channel, int y, int x) {
    int th = tensor->tiles_per_dim[2];
    int tw = tensor->tiles_per_dim[3];
    int tile = ((batch * tensor->shape[1] + channel) * th + y / 32) * tw + x / 32;
    return (int *)tensor->tiles[tile].base_addr;
}

static int *weight_base(const GlobalTile *tensor,
                                       int out_channel, int in_channel,
                                       int y, int x) {
    int th = tensor->tiles_per_dim[2];
    int tw = tensor->tiles_per_dim[3];
    int tile = ((out_channel * tensor->shape[1] + in_channel) * th + y / 32) * tw + x / 32;
    return (int *)tensor->tiles[tile].base_addr;
}

static int conv_a[512];
static int conv_w[512];
static int conv_c[512];

static int read16(int *base, int index) {
    int word = base[index / 2];
    return (word >> ((index % 2) * 16)) & 65535;
}

static void write16(int *base, int index, int value) {
    int slot = index / 2;
    int shift = (index % 2) * 16;
    int mask = 65535 << shift;
    base[slot] = (base[slot] & ~mask) | ((value & 65535) << shift);
}

static int read_nchw(const GlobalTile *tensor, int b, int c, int y, int x) {
    return read16(nchw_base(tensor, b, c, y, x), (y % 32) * 32 + x % 32);
}

static void write_nchw(const GlobalTile *tensor, int b, int c, int y, int x, int value) {
    write16(nchw_base(tensor, b, c, y, x), (y % 32) * 32 + x % 32, value);
}

static int read_weight(const GlobalTile *tensor, int oc, int ic, int y, int x) {
    return read16(weight_base(tensor, oc, ic, y, x), (y % 32) * 32 + x % 32);
}

/* Pack a spatial block and a weight block into the validation GEMM's KxN
 * contract. This is the generic path; compiler contributors may replace its
 * pack/compute loops with an optimized direct convolution for common shapes. */
int conv_kernel(const GlobalTile *input, const GlobalTile *weight,
                const GlobalTile *bias, GlobalTile *output,
                int stride_h, int stride_w, int pad_h, int pad_w,
                int dilation_h, int dilation_w, int groups) {
    int batch;
    int out_channels;
    int in_per_group;
    int out_per_group;
    int kh;
    int kw;
    int positions;
    int k_total;
    if (input->rank != 4 || weight->rank != 4 || output->rank != 4) return -1;
    if (groups < 1 || input->shape[1] % groups || weight->shape[0] % groups) return -1;
    if (stride_h < 1 || stride_w < 1 || dilation_h < 1 || dilation_w < 1) return -1;
    if (output->shape[0] != input->shape[0] || output->shape[1] != weight->shape[0]) return -1;
    if (bias && (bias->rank != 1 || bias->shape[0] != output->shape[1])) return -1;
    in_per_group = input->shape[1] / groups;
    if (weight->shape[1] != in_per_group) return -1;
    out_per_group = weight->shape[0] / groups;
    out_channels = weight->shape[0];
    kh = weight->shape[2];
    kw = weight->shape[3];
    positions = output->shape[2] * output->shape[3];
    k_total = in_per_group * kh * kw;
    batch = 0;
    while (batch < output->shape[0]) {
        int group = 0;
        while (group < groups) {
            int position_block = 0;
            while (position_block < positions) {
                int channel_block = 0;
                while (channel_block < out_per_group) {
                    int idx = 0;
                    while (idx < 1024) {
                        int col = idx % 32;
                        int channel = group * out_per_group + channel_block + col;
                        write16(conv_c, idx, (bias && channel_block + col < out_per_group)
                            ? read16((int *)bias->tiles[channel / 32].base_addr, channel % 32) : 0);
                        idx = idx + 1;
                    }
                    int k_block = 0;
                    while (k_block < k_total) {
                        idx = 0;
                        while (idx < 1024) {
                            int row = idx / 32;
                            int col = idx % 32;
                            int k = k_block + col;
                            int position = position_block + row;
                            write16(conv_a, idx, 0);
                            if (k < k_total && position < positions) {
                                int local_ic = k / (kh * kw);
                                int rem = k % (kh * kw);
                                int ky = rem / kw;
                                int kx = rem % kw;
                                int oy = position / output->shape[3];
                                int ox = position % output->shape[3];
                                int iy = oy * stride_h - pad_h + ky * dilation_h;
                                int ix = ox * stride_w - pad_w + kx * dilation_w;
                                if (iy >= 0 && iy < input->shape[2] && ix >= 0 && ix < input->shape[3])
                                    write16(conv_a, idx, read_nchw(input, batch, group * in_per_group + local_ic, iy, ix));
                            }
                            write16(conv_w, idx, 0);
                            k = k_block + row;
                            if (k < k_total && channel_block + col < out_per_group) {
                                int local_ic = k / (kh * kw);
                                int rem = k % (kh * kw);
                                int ky = rem / kw;
                                int kx = rem % kw;
                                write16(conv_w, idx, read_weight(weight, group * out_per_group + channel_block + col, local_ic, ky, kx));
                            }
                            idx = idx + 1;
                        }
                        scpad_load(0, (int)conv_w, ATALLA_SDMA_TILE32_SID0);
                        int wi = 0;
                        while (wi < 32) {
                            vec w = vector_load(0, wi, 31, 0);
                            load_weights(w);
                            wi = wi + 1;
                        }
                        scpad_load(0, (int)conv_a, ATALLA_SDMA_TILE32_SID0);
                        scpad_load(0, (int)conv_c, ATALLA_SDMA_TILE32_SID1);
                        int row = 0;
                        while (row < 32) {
                            vec a = vector_load(0, row, 31, 0);
                            vec c = vector_load(0, row, 31, 1);
                            vec result = gemm(a, c, -1);
                            vector_store(result, 0, row, 31, 1);
                            row = row + 1;
                        }
                        scpad_store(0, (int)conv_c, ATALLA_SDMA_TILE32_SID1);
                        k_block = k_block + 32;
                    }
                    idx = 0;
                    while (idx < 1024) {
                        int row = idx / 32;
                        int col = idx % 32;
                        int position = position_block + row;
                        if (position < positions && channel_block + col < out_per_group) {
                            int oy = position / output->shape[3];
                            int ox = position % output->shape[3];
                            int channel = group * out_per_group + channel_block + col;
                            write_nchw(output, batch, channel, oy, ox, read16(conv_c, idx));
                        }
                        idx = idx + 1;
                    }
                    channel_block = channel_block + 32;
                }
                position_block = position_block + 32;
            }
            group = group + 1;
        }
        batch = batch + 1;
    }
    return 0;
}

/* BF16 bit patterns map monotonically to this unsigned comparison key. */
static int bf16_order(int value) {
    return (value & 0x8000u) ? (int)(value ^ 0xffffu) : (int)(value ^ 0x8000u);
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
                                int v = read_nchw(input, b, c, iy, ix);
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
                    int tile = b * output->tiles_per_dim[1] + col / 32;
                    int *dst = (int *)output->tiles[tile].base_addr;
                    write16(dst, col % 32, read_nchw(input, b, c, y, x));
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
