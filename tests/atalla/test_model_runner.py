"""The model runner (src/atalla/model_runner.py) on the example MLP plan."""
import json
from pathlib import Path

import numpy as np
import pytest

from atalla import model_runner as mr
from scheduler import golden

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "examples" / "model" / "mlp.plan.json"


def inputs(seed=0):
    plan = mr.Plan.load(PLAN)
    rng = np.random.default_rng(seed)
    return {"arg%d" % k: rng.normal(0, 0.5, size=shape).astype(np.float32)
            for k, (_, shape, _) in enumerate(plan.inputs)}


def test_types_and_plan_parse():
    assert mr.parse_type("tensor<2x8xbf16>") == ((2, 8), "bf16")
    assert mr.parse_type("bf16") == ((), "bf16")
    with pytest.raises(mr.PlanError):
        mr.parse_type("tensor<?x8xbf16>")
    plan = mr.Plan.load(PLAN)
    assert [s for _, s, _ in plan.inputs] == [(2, 4), (8, 4), (8,), (3, 8), (3,)]


def test_the_numpy_backend_is_the_mlp():
    x = inputs()
    y = mr.Runner(mr.Plan.load(PLAN), "numpy").run(x)[0]
    b = mr._bf16
    h = b(np.maximum(b(b(b(x["arg0"]) @ b(x["arg1"]).T) + b(x["arg2"])), 0))
    want = b(b(h @ b(x["arg3"]).T) + b(x["arg4"]))
    assert np.array_equal(y, want)


@pytest.mark.golden
@pytest.mark.skipif(not golden.HAVE_GOLDEN, reason="kernels come from the functional sim")
def test_the_mlp_runs_on_atalla_sim_and_matches_numpy_exactly():
    report = mr.run_model(PLAN, inputs(), config_summary="test")
    assert report["counts"] == {"accelerator": 5, "host": 2, "none": 7}
    assert report["output_vs_numpy"]["max_bf16_steps"] == 0
    for op in report["operations"]:
        if op["where"] == "accelerator":
            assert op["cycles"] > 0 and op["max_bf16_steps"] == 0, op
    assert "5 operations on the accelerator" in mr.summary(report)
    json.dumps(report, default=float)


def test_an_unsupported_operation_is_named(tmp_path):
    doc = json.loads(PLAN.read_text())
    doc["functions"][0]["operations"][5]["name"] = "linalg.batch_matmul"
    p = tmp_path / "bad.plan.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(mr.PlanError, match="op 5: linalg.batch_matmul"):
        mr.Runner(mr.Plan.load(p), "numpy").run(inputs())
