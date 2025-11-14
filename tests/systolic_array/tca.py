from sim import TCA, reference_conv2d
import numpy as np
from sim.common import * 

if __name__ == "__main__":
    N = 2
    
    for H_in, W_in, Kh, Kw, Cout, Cin in [
        (30, 30, 3, 3, 3 , 3), 
        (20, 20, 4, 4, 3, 1), 
        (16, 32, 11, 11, 2, 3)
    ]: 

        X = np.random.randn(N, Cin, H_in, W_in).astype(np.float32)
        W = np.random.randn(Cout, Cin, Kh, Kw).astype(np.float32)

        sim = TCA(stride=1, pad=1, dilation=1, dtype=np.float32)
        Y_sim = sim.forward(X, W)
        Y_ref = reference_conv2d(X, W, stride=1, pad=1, dilation=1)

        print("Max abs diff:", np.max(np.abs(Y_sim - Y_ref)))
