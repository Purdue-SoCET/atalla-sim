"""resnet_small: the BN and avgpool folds are exact, the traced graph uses only
kernels Atalla has, and the 2:4 twin is 2:4 where it should be."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "models"), str(ROOT / "models" / "atalla-graph")):
    if p not in sys.path:
        sys.path.insert(0, p)

from model.resnet_small import (ResNetSmall, ResNetSmallRef, example_input,  # noqa: E402
                                is_2to4)


def _pair(seed=0, **kw):
    torch.manual_seed(seed)
    ref = ResNetSmallRef().eval()
    return ref, ResNetSmall(ref=ref, **kw).eval()


def test_the_folded_network_computes_what_the_textbook_one_does():
    ref, folded = _pair()
    x = example_input()
    with torch.no_grad():
        want, got = ref(x), folded(x)
    assert got.shape == want.shape == (1, 10)
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-5)


def test_the_fold_has_no_batchnorm_or_average_pool_left():
    _, folded = _pair()
    kinds = {type(m) for m in folded.modules()}
    assert torch.nn.BatchNorm2d not in kinds
    assert torch.nn.AdaptiveAvgPool2d not in kinds


def test_every_pruned_conv_is_2to4_and_the_stem_and_fc_are_not_touched():
    _, dense = _pair(seed=1)
    _, sparse = _pair(seed=1, sparsity="2:4")
    for conv in sparse.pruned_convs():
        assert is_2to4(conv.weight)
        density = (conv.weight != 0).float().mean().item()
        assert density == pytest.approx(0.5)
    assert torch.equal(sparse.stem.weight, dense.stem.weight)
    assert torch.equal(sparse.fc.weight, dense.fc.weight)


def test_pruning_keeps_the_two_largest_of_each_group():
    _, dense = _pair(seed=2)
    _, sparse = _pair(seed=2, sparsity="2:4")
    d = dense.stage2.spatial.weight.reshape(dense.stage2.spatial.out_channels, -1, 4)
    s = sparse.stage2.spatial.weight.reshape(d.shape)
    kept = s.abs().sum(dim=2)
    top2 = d.abs().topk(2, dim=2).values.sum(dim=2)
    assert torch.allclose(kept, top2)


def test_the_traced_graph_is_fully_covered_by_atalla_kernels(tmp_path):
    pytest.importorskip("pydot")
    if not (ROOT / "third_party" / "aihw-ppci-compiler" / "ppci").is_dir():
        pytest.skip("the compiler submodule isn't checked out")
    import copy, json
    from run_schedule import build_graph, load_model, run_schedule
    torch.manual_seed(42)
    model, x = load_model("resnet_small")
    graph = build_graph(model, x)
    run_schedule(copy.deepcopy(graph), x, tmp_path)
    manifest = json.loads((tmp_path / "kernel_manifest.json").read_text())
    assert manifest["kernel_coverage"], manifest
