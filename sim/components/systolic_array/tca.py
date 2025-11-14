from grid import * 


class TCA:
    LANES = 32

    def __init__(self, stride: int = 1, pad: int = 0, dilation: int = 1,
                 dtype=np.float32):
        self.stride, self.pad, self.dilation = stride, pad, dilation
        self.dtype = dtype
        self.systolic = SystolicArray(self.LANES)

    def _footprint(self, No_h_eff: int, No_w_eff: int, Kh: int, Kw: int) -> tuple[int, int]:
        Ih = (No_h_eff - 1) * self.stride + (Kh - 1) * self.dilation + 1
        Iw = (No_w_eff - 1) * self.stride + (Kw - 1) * self.dilation + 1
        return Ih, Iw

    def _max_dim_by_footprint(self, Kdim: int) -> int:
        return max(1, ((self.LANES - ((Kdim - 1) * self.dilation + 1)) // self.stride) + 1)

    def _pow2_le(self, x: int) -> int:
        x = max(1, min(self.LANES, x))
        return 1 << (x.bit_length() - 1)

    def _mk_row_mask(self, Mt_eff: int) -> np.ndarray:
        m = np.zeros((self.LANES,), dtype=self.dtype)
        m[:Mt_eff] = 1
        return m

    def _mk_col_mask(self, Nt_eff: int) -> np.ndarray:
        n = np.zeros((self.LANES,), dtype=self.dtype)
        n[:Nt_eff] = 1
        return n

    def _tile_p2_steps(self, end: int, max_tile: int):
        pos = 0
        while pos < end:
            rem = end - pos
            eff = self._pow2_le(min(max_tile, rem))
            yield pos, eff
            pos += eff

    def _load_input_scpad(self, Xn_ci: np.ndarray, oh0: int, ow0: int,
                           No_h_eff: int, No_w_eff: int, H_in: int, W_in: int,
                           Kh: int, Kw: int) -> np.ndarray:
        Ih, Iw = self._footprint(No_h_eff, No_w_eff, Kh, Kw)
        sc = np.zeros((self.LANES, self.LANES), dtype=self.dtype)
        base_h = oh0 * self.stride - self.pad
        base_w = ow0 * self.stride - self.pad
        src_row = max(0, base_h)
        dst_row = max(0, -base_h)
        src_col = max(0, base_w)
        dst_col = max(0, -base_w)
        rows = min(Ih - dst_row, H_in - src_row)
        cols = min(Iw - dst_col, W_in - src_col)
        for r in range(rows):
            sc[dst_row + r, dst_col:dst_col + cols] = Xn_ci[src_row + r, src_col:src_col + cols]
        return sc

    def _emit_uv_tile_plan(self, uv: int, Kt_eff: int, Kh: int, Kw: int) -> tuple[np.ndarray, np.ndarray]:
        u_cols = np.empty((Kt_eff,), dtype=np.int32)
        v_cols = np.empty((Kt_eff,), dtype=np.int32)
        u = uv // Kw
        v = uv - u * Kw
        for j in range(Kt_eff):
            u_cols[j] = u
            v_cols[j] = v
            v += 1
            if v == Kw:
                v = 0
                u += 1
        return u_cols, v_cols

    def forward(self, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        N, Cin, H_in, W_in = X.shape
        Cout, Cin2, Kh, Kw = W.shape
        assert Cin == Cin2, "Cin mismatch"
        H_out, W_out = conv_output_shape(H_in, W_in, Kh, Kw,
                                         stride=self.stride, pad=self.pad,
                                         dilation=self.dilation)

        max_h = self._max_dim_by_footprint(Kh)
        max_w = self._max_dim_by_footprint(Kw)
        No_w = min(self.LANES, self._pow2_le(max_w))
        No_h = min(max_h, max(1, self.LANES // No_w))

        K_uv = Kh * Kw  
        Kt_uv = self._pow2_le(min(K_uv, self.LANES)) 

        Y = np.zeros((N, Cout, H_out, W_out), dtype=self.dtype)

        for n_idx in range(N):
            # accumulate full output in flattened C_full
            C_full = np.zeros((Cout, H_out * W_out), dtype=self.dtype)
            # iterate over output channel tiles
            for m0 in range(0, Cout, self.LANES):
                Mt_eff = min(self.LANES, Cout - m0)
                # iterate over output spatial tiles
                for oh0, No_h_eff in self._tile_p2_steps(H_out, No_h):
                    for ow0, No_w_eff in self._tile_p2_steps(W_out, No_w):
                        Nt_eff = No_h_eff * No_w_eff
                        # scratchpad for partial sums in this tile
                        C32 = np.zeros((self.LANES, Nt_eff), dtype=self.dtype)
                        # iterate over input channels
                        for ci in range(Cin):
                            Xn_ci = X[n_idx, ci]
                            # load input footprint for this (ci, tile)
                            input_scpad = self._load_input_scpad(
                                Xn_ci, oh0, ow0, No_h_eff, No_w_eff,
                                H_in, W_in, Kh, Kw
                            )
                            # iterate over kernel tap slices
                            uv0 = 0
                            while uv0 < K_uv:
                                Kt_eff = self._pow2_le(min(K_uv - uv0, Kt_uv))
                                # Build the weight tile Wt (shape Kt_eff x Mt_eff)
                                u_cols, v_cols = self._emit_uv_tile_plan(uv0, Kt_eff, Kh, Kw)
                                Wt = np.zeros((Kt_eff, Mt_eff), dtype=self.dtype)
                                for k_lin in range(Kt_eff):
                                    u = int(u_cols[k_lin]); v = int(v_cols[k_lin])
                                    for m_lin in range(Mt_eff):
                                        Wt[k_lin, m_lin] = W[m0 + m_lin, ci, u, v]
                                # Clear systolic array state and load weights
                                self.systolic._reset_state()  # reset activations, psums, FIFOs, output buffer but keep weights
                                self.systolic.preload_weights(Wt)
                                # Number of cycles: one per output spatial element plus flush cycles to account for shifting across M dimension
                                total_cycles = Nt_eff + Mt_eff - 1
                                # For each cycle t, feed inputs and run cycle
                                print(Kt_eff, Mt_eff, Nt_eff, Mt_eff, No_w_eff)

                                for t in range(total_cycles):
                                    # Prepare the input vector for this cycle (length = LANES)
                                    in_vec = np.zeros((self.LANES,), dtype=self.dtype)
                                    if t < Nt_eff:
                                        # Gather activations for spatial index n=t
                                        n_lin = t
                                        tile_h = n_lin // No_w_eff
                                        tile_w = n_lin %  No_w_eff
                                        # Build b_vec for Kt_eff rows
                                        b_vec = np.zeros((Kt_eff,), dtype=self.dtype)
                                        for k_lin in range(Kt_eff):
                                            u = int(u_cols[k_lin]); v = int(v_cols[k_lin])
                                            ih = tile_h * self.stride + u * self.dilation
                                            iw = tile_w * self.stride + v * self.dilation
                                            b_vec[k_lin] = input_scpad[ih, iw]
                                        # Assign b_vec into the input vector for the first Kt_eff rows
                                        in_vec[:Kt_eff] = b_vec
                                    else:
                                        # For flush cycles, feed zeros to all rows
                                        in_vec[:] = 0.0
                                    # Feed activations into the systolic array
                                    self.systolic.feed_inputs(in_vec.tolist())
                                    # Advance one cycle and get bottom row outputs
                                    bottom = self.systolic.cycle()
                                    # Map bottom outputs to C32 based on time and column
                                    for m_lin in range(Mt_eff):
                                        # Determine the spatial output index for this cycle and column.
                                        # Naming it ``n_out`` to avoid clobbering the batch index ``n_idx``.
                                        n_out = t - m_lin
                                        if 0 <= n_out < Nt_eff:
                                            C32[m_lin, n_out] += bottom[m_lin]
                                uv0 += Kt_eff
                        # commit C32 into C_full for this tile
                        base_row = oh0 * W_out + ow0
                        for m_lin in range(Mt_eff):
                            row_ptr = base_row
                            for th in range(No_h_eff):
                                start_c = th * No_w_eff
                                C_full[m0 + m_lin, row_ptr:row_ptr + No_w_eff] += C32[m_lin, start_c:start_c + No_w_eff]
                                row_ptr += W_out
            # reshape C_full into output tensor
            Y[n_idx] = C_full.reshape(Cout, H_out, W_out)
        return Y
    
def reference_conv2d(X: np.ndarray, W: np.ndarray, stride:int=1, pad:int=0, dilation:int=1) -> np.ndarray:
    N, Cin, H_in, W_in = X.shape
    Cout, Cin2, Kh, Kw = W.shape
    assert Cin == Cin2
    H_out, W_out = conv_output_shape(H_in, W_in, Kh, Kw, stride, pad, dilation)
    Y = np.zeros((N, Cout, H_out, W_out), dtype=X.dtype)

    for n in range(N):
        for co in range(Cout):
            for i_out in range(H_out):
                for j_out in range(W_out):
                    acc = 0.0
                    base_h = i_out*stride - pad
                    base_w = j_out*stride - pad
                    for ci in range(Cin):
                        for u in range(Kh):
                            h = base_h + u*dilation
                            for v in range(Kw):
                                w = base_w + v*dilation
                                if 0 <= h < H_in and 0 <= w < W_in:
                                    acc += X[n, ci, h, w] * W[co, ci, u, v]
                    Y[n, co, i_out, j_out] = acc
    return Y

