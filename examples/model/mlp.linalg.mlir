// The atalla-graph PyTorch example's MLP (Linear 4->8, ReLU, Linear 8->3) in BF16,
// written as Linalg by hand in the shape torch-mlir's linalg-on-tensors output
// takes: transposed weights, a zero-filled matmul, a broadcast bias add, a ReLU.
// The weights are function arguments rather than constants, so the plan keeps
// them out of the IR (atalla-graph's plan does not carry constant data yet).
module {
  func.func @main(%x: tensor<2x4xbf16>, %w1: tensor<8x4xbf16>, %b1: tensor<8xbf16>,
                  %w2: tensor<3x8xbf16>, %b2: tensor<3xbf16>) -> tensor<2x3xbf16> {
    %zero = arith.constant 0.0 : bf16
    %e0 = tensor.empty() : tensor<4x8xbf16>
    %w1t = linalg.transpose ins(%w1 : tensor<8x4xbf16>) outs(%e0 : tensor<4x8xbf16>) permutation = [1, 0]
    %e1 = tensor.empty() : tensor<2x8xbf16>
    %z1 = linalg.fill ins(%zero : bf16) outs(%e1 : tensor<2x8xbf16>) -> tensor<2x8xbf16>
    %m1 = linalg.matmul ins(%x, %w1t : tensor<2x4xbf16>, tensor<4x8xbf16>) outs(%z1 : tensor<2x8xbf16>) -> tensor<2x8xbf16>
    %h1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%m1, %b1 : tensor<2x8xbf16>, tensor<8xbf16>) outs(%e1 : tensor<2x8xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %s = arith.addf %in, %in_0 : bf16
      linalg.yield %s : bf16
    } -> tensor<2x8xbf16>
    %r1 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%h1 : tensor<2x8xbf16>) outs(%e1 : tensor<2x8xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %m = arith.maximumf %in, %zero : bf16
      linalg.yield %m : bf16
    } -> tensor<2x8xbf16>
    %e2 = tensor.empty() : tensor<8x3xbf16>
    %w2t = linalg.transpose ins(%w2 : tensor<3x8xbf16>) outs(%e2 : tensor<8x3xbf16>) permutation = [1, 0]
    %e3 = tensor.empty() : tensor<2x3xbf16>
    %z2 = linalg.fill ins(%zero : bf16) outs(%e3 : tensor<2x3xbf16>) -> tensor<2x3xbf16>
    %m2 = linalg.matmul ins(%r1, %w2t : tensor<2x8xbf16>, tensor<8x3xbf16>) outs(%z2 : tensor<2x3xbf16>) -> tensor<2x3xbf16>
    %y = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%m2, %b2 : tensor<2x3xbf16>, tensor<3xbf16>) outs(%e3 : tensor<2x3xbf16>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: bf16):
      %s = arith.addf %in, %in_0 : bf16
      linalg.yield %s : bf16
    } -> tensor<2x3xbf16>
    return %y : tensor<2x3xbf16>
  }
}
