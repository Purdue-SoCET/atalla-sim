import numpy as np
from typing import Tuple

def conv_output_shape(H_in:int, W_in:int, Kh:int, Kw:int, stride:int=1, pad:int=0, dilation:int=1) -> Tuple[int,int]:
    H_out = (H_in + 2*pad - dilation*(Kh-1) - 1)//stride + 1
    W_out = (W_in + 2*pad - dilation*(Kw-1) - 1)//stride + 1
    if H_out <= 0 or W_out <= 0:
        raise ValueError(f"Non-positive output shape computed: H_out={H_out}, W_out={W_out}. Check params.")
    return H_out, W_out


class PE:
    """
    A simple Processing Element (PE) for a systolic array. Each PE stores an
    activation value, a weight value, and an accumulation string. Links
    represent the four directional connections: left, up, right, down.

    The activation and weight are stored as strings so that symbolic
    representations can be built up for debugging. Accumulation is also stored
    as a string of the form "a*b + ...".
    """
    def __init__(self, links: list | None = None) -> None:
        # Neighboring PEs: 0=left, 1=up, 2=right, 3=down
        if links is None:
            links = [None, None, None, None]
        self.links = links
        # Numeric state for the systolic array simulation
        self.activation: float = 0.0
        self.weight: float = 0.0
        self.partial_sum: float = 0.0
        # Human-readable strings for debugging/visualization
        self.weight_str: str = '--'
        self.activation_str: str = '0'
        self.accumulation_str: str = '0'

    def load_weight(self, weight: float) -> None:
        """Load a numeric weight into this PE and update the debug string."""
        self.weight = weight
        self.weight_str = f"{weight}"

    def set_activation(self, activation: float) -> None:
        """Set the current activation for this PE and update the debug string."""
        self.activation = activation
        self.activation_str = f"{activation}"

    def cycle(self, accum_in: float) -> float:
        """
        Perform one multiply-and-accumulate operation for this PE.

        Parameters
        ----------
        accum_in: float
            The partial sum from the PE above this one. For the top row,
            ``accum_in`` should be zero.

        Returns
        -------
        float
            The new partial sum computed by this PE. This value will be
            propagated to the PE in the next row (downwards) by the caller.
        """
        # Compute new partial sum
        psum = accum_in + self.weight * self.activation
        # Update internal numeric state
        self.partial_sum = psum
        # Update debug accumulation string
        mul_term = f"{self.activation_str}*{self.weight_str}" if self.activation_str not in ['0', '0.0'] and self.weight_str != '--' else '0'
        if accum_in == 0.0:
            # no previous accumulation
            self.accumulation_str = mul_term
        else:
            prev_str = self.accumulation_str
            if prev_str.strip() == '0' and mul_term == '0':
                self.accumulation_str = '0'
            elif prev_str.strip() == '0':
                self.accumulation_str = mul_term
            elif mul_term == '0':
                self.accumulation_str = prev_str
            else:
                self.accumulation_str = prev_str + ' + ' + mul_term
        return psum


class SystolicArray:
    """
    A 2D systolic array of processing elements. This array supports simple
    vertical accumulation: activations shift horizontally to the right each
    cycle while weights remain stationary. Accumulation flows downward.

    Parameters
    ----------
    size: int
        The dimension of the array (size x size).
    """
    def __init__(self, size: int) -> None:
        self.size = size
        # FIFO for incoming activations; one queue per row
        self.input_fifo: list[list[float]] = [[] for _ in range(size)]
        # Buffer to hold bottom-row outputs after each cycle (list of lists)
        self.output_buffer: list[list[float]] = []
        self.array = self._setup_array()
        # The activations and partial sums are stored directly on the PE objects.
        # We also keep a separate weight matrix for easier loading and resetting.
        self.weights = [[0.0 for _ in range(size)] for _ in range(size)]

    def _setup_array(self) -> list[list[PE]]:
        array: list[list[PE]] = [[PE() for _ in range(self.size)] for _ in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                left = array[i][j-1] if j > 0 else None
                up = array[i-1][j] if i > 0 else None
                right = array[i][j+1] if j < self.size - 1 else None
                down = array[i+1][j] if i < self.size - 1 else None
                array[i][j].links = [left, up, right, down]
        return array

    def _reset_state(self) -> None:
        """
        Helper to reset activations, partial sums and input/output buffers.

        The PE objects maintain their own activation and accumulation fields.
        This method zeroes those fields and clears the input FIFOs and output buffer.
        It does not modify the stored weights; those remain in place until
        explicitly reloaded via ``preload_weights``.
        """
        # Clear activations and partial sums on all PEs
        for i in range(self.size):
            for j in range(self.size):
                pe = self.array[i][j]
                # Reset numeric and debug states
                pe.activation = 0.0
                pe.activation_str = '0'
                pe.partial_sum = 0.0
                pe.accumulation_str = '0'
        # Clear input queues and output buffer
        self.input_fifo = [[] for _ in range(self.size)]
        self.output_buffer = []

    def clear(self) -> None:
        """
        Reset the systolic array state, including activations, accumulations,
        input FIFOs, output buffer and weights. Use this when starting a new
        computation with potentially different weights. If weights need to be
        preserved across cycles, call ``_reset_state`` instead.
        """
        # Clear weights, activations and accumulations
        for i in range(self.size):
            for j in range(self.size):
                pe = self.array[i][j]
                # Reset numeric and debug states
                pe.activation = 0.0
                pe.activation_str = '0'
                pe.weight = 0.0
                pe.weight_str = '--'
                pe.partial_sum = 0.0
                pe.accumulation_str = '0'
                self.weights[i][j] = 0.0
        # Clear queues and output buffer
        self.input_fifo = [[] for _ in range(self.size)]
        self.output_buffer = []

    def preload_weights(self, Wt: np.ndarray) -> None:
        """Load a weight tile into the array.

        The weight tile Wt must have shape (K_t, M_t) where K_t is the number
        of rows (the contraction dimension) and M_t is the number of columns
        (output channels). We load Wt[k, m] into the PE at row k and column m.
        We assume the array is large enough to accommodate Wt.
        """
        K_t, M_t = Wt.shape
        assert K_t <= self.size and M_t <= self.size, "Weight tile exceeds array size"
        # Clear previous weights and load numeric values into PE weights and into self.weights
        for i in range(self.size):
            for j in range(self.size):
                # Reset the PE's weight and debug fields
                self.array[i][j].load_weight(0.0)
                self.weights[i][j] = 0.0
        for k in range(K_t):
            for m in range(M_t):
                # Store numeric weight in both the PE and the weight matrix
                w_val = float(Wt[k, m])
                self.array[k][m].load_weight(w_val)
                self.weights[k][m] = w_val

    def feed_inputs(self, inputs: list[float]) -> None:
        """
        Enqueue one new activation for each row. The ``inputs`` list should
        have length equal to the array size. Each element is appended to the
        corresponding row's input FIFO. Values beyond the active rows (for
        example, when fewer than ``size`` rows are used) should be zero.
        """
        assert len(inputs) == self.size, "Input vector length must match array size"
        for row, val in enumerate(inputs):
            self.input_fifo[row].append(float(val))

    def cycle(self) -> list[float]:
        """
        Advance the systolic array by one cycle. This method shifts activations
        to the right, inserts new activations from the input FIFOs into the
        leftmost column, computes new partial sums (MAC operations) by
        vertically accumulating the products of activations and weights, and
        stores the bottom row partial sums into the output buffer.

        Returns
        -------
        list[float]
            A copy of the bottom row partial sums after this cycle. The list
            length equals the array size. Consumers can interpret the valid
            portion based on the dimensions of the currently loaded weight tile
            and the number of cycles executed.
        """
        size = self.size
        # Step 1: determine next activations for each PE by shifting and inserting new inputs
        # We'll store the new activations in a temporary matrix
        new_acts: list[list[float]] = [[0.0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            # Shift activations to the right across the row
            for j in range(size-1, 0, -1):
                new_acts[i][j] = self.array[i][j-1].activation
            # Insert new input from the FIFO at column 0
            if self.input_fifo[i]:
                new_acts[i][0] = self.input_fifo[i].pop(0)
            else:
                new_acts[i][0] = 0.0
        # Step 2: perform MAC operation on each PE row by row
        # We'll compute new partial sums and update the PE's activation and partial_sum fields
        # We also maintain a matrix of new partial sums to determine the accumulation input for the next row
        new_psums: list[list[float]] = [[0.0 for _ in range(size)] for _ in range(size)]
        for i in range(size):
            for j in range(size):
                # Update the PE's activation with the newly shifted value
                self.array[i][j].set_activation(new_acts[i][j])
            # Now compute partial sums across columns for this row
            for j in range(size):
                # The accumulation input is the partial sum from the PE above in the same column
                if i == 0:
                    accum_in = 0.0
                else:
                    accum_in = new_psums[i-1][j]
                psum = self.array[i][j].cycle(accum_in)
                new_psums[i][j] = psum
        # Step 3: save bottom row partial sums to the output buffer
        bottom_row = new_psums[size - 1][:]
        self.output_buffer.append(bottom_row)
        # Step 4: return the bottom row for convenience
        # The PE partial sums have been updated internally for the next cycle through ``cycle`` calls
        return bottom_row

    def vertical_psum(self, Wt: np.ndarray, b_vec: np.ndarray) -> np.ndarray:
        """
        **Deprecated.**

        This method previously performed a direct dot-product across the contraction
        dimension (k) to simulate a vertical accumulation without modelling the
        systolic dataflow. It has been superseded by the cycle-based simulation
        implemented in ``SystolicArray.cycle``. The new simulation preloads
        weights into the array and then feeds activations into the leftmost
        column on each cycle. Activations shift to the right while partial
        sums accumulate vertically and propagate downward. Results are emitted
        from the bottom row.

        The current implementation raises a ``NotImplementedError`` to warn
        against inadvertent use. Please use the cycle-based methods instead.
        """
        raise NotImplementedError(
            "vertical_psum() is deprecated; use SystolicArray.cycle() with preloaded weights "
            "and feed_inputs() to stream activations for accurate systolic simulation."
        )


class ImplicitGEMMConv32x32:
    """
    A specialized convolution kernel that maps operations onto a 32x32
    systolic array. The weight tile is loaded into the array and stays
    stationary, while activation windows are streamed and vertically
    accumulated. Partial sums are stored in a scratchpad for later reduction.

    This class integrates the SystolicArray class as a member to simulate
    the hardware behavior. It follows the same tiling logic as the user
    provided code but uses a simplified vertical reduction.
    """
    LANES = 32  # dimension of the systolic array

    def __init__(self, stride: int = 1, pad: int = 0, dilation: int = 1,
                 dtype=np.float32):
        self.stride, self.pad, self.dilation = stride, pad, dilation
        self.dtype = dtype
        # instantiate a 32x32 systolic array
        self.systolic = SystolicArray(self.LANES)

    def _footprint(self, No_h_eff: int, No_w_eff: int, Kh: int, Kw: int) -> tuple[int, int]:
        Ih = (No_h_eff - 1) * self.stride + (Kh - 1) * self.dilation + 1
        Iw = (No_w_eff - 1) * self.stride + (Kw - 1) * self.dilation + 1
        return Ih, Iw

    def _max_dim_by_footprint(self, Kdim: int) -> int:
        # Solve (No - 1)*S + (Kdim - 1)*D + 1 <= 32 => No <= floor((32 - (Kdim - 1)*D - 1)/S) + 1
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
        """Load a tile of input into a 32x32 scratchpad with padding and dilation."""
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
        """Generate lists of u and v indices for a contiguous chunk of kernel taps."""
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
        """
        Perform convolution using a weight-stationary systolic array simulation.

        Parameters
        ----------
        X: np.ndarray
            Input tensor of shape (N, C_in, H_in, W_in).
        W: np.ndarray
            Weight tensor of shape (C_out, C_in, K_h, K_w).

        Returns
        -------
        np.ndarray
            Output tensor of shape (N, C_out, H_out, W_out).
        """
        N, Cin, H_in, W_in = X.shape
        Cout, Cin2, Kh, Kw = W.shape
        assert Cin == Cin2, "Cin mismatch"
        # output spatial dimension
        H_out, W_out = conv_output_shape(H_in, W_in, Kh, Kw,
                                         stride=self.stride, pad=self.pad,
                                         dilation=self.dilation)

        # determine tile sizes for output spatial dimensions
        max_h = self._max_dim_by_footprint(Kh)
        max_w = self._max_dim_by_footprint(Kw)
        No_w = min(self.LANES, self._pow2_le(max_w))
        No_h = min(max_h, max(1, self.LANES // No_w))

        K_uv = Kh * Kw  # taps per input channel
        Kt_uv = self._pow2_le(min(K_uv, self.LANES))  # kernel taps per slice

        # prepare output tensor
        Y = np.zeros((N, Cout, H_out, W_out), dtype=self.dtype)

        # process batch
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

N = 2
for H_in, W_in, Kh, Kw, Cout, Cin in [
    (30, 30, 3, 3, 3 , 3), 
    (20, 20, 4, 4, 3, 1), 
    (16, 32, 11, 11, 2, 3)
]: 

    X = np.random.randn(N, Cin, H_in, W_in).astype(np.float32)
    W = np.random.randn(Cout, Cin, Kh, Kw).astype(np.float32)

    sim = ImplicitGEMMConv32x32(stride=1, pad=1, dilation=1, dtype=np.float32)
    Y_sim = sim.forward(X, W)
    Y_ref = reference_conv2d(X, W, stride=1, pad=1, dilation=1)

    print("Max abs diff:", np.max(np.abs(Y_sim - Y_ref)))
