"""Model ELFs built by models/ and frozen by models/freeze.py, run on the
Atalla platform and checked against the PyTorch values stored beside them.
Needs neither torch nor the compiler."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import bench                                                     # noqa: E402
from atalla.atalla_platform import build_atalla_platform         # noqa: E402

HERE = Path(__file__).resolve().parent
MODELS = sorted(p.name[:-len(".expected.json")] for p in HERE.glob("*.expected.json"))


def read_tensor(dram, address, shape):
    """A tensor stored as row-major 32x32 BF16 tiles (models/ allocator)."""
    logical = tuple(shape)
    shape = (1, logical[0]) if len(logical) == 1 else logical
    height, width = shape[-2:]
    planes = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
    out = np.empty((planes, height, width), dtype=np.float32)
    for plane in range(planes):
        for y in range(0, height, 32):
            for x in range(0, width, 32):
                bits = np.frombuffer(dram.read(address, 2048), dtype="<u2").astype("<u4")
                tile = (bits << 16).view("<f4").reshape(32, 32)
                h, w = min(32, height - y), min(32, width - x)
                out[plane, y:y + h, x:x + w] = tile[:h, :w]
                address += 2048
    return out.reshape(logical)


@pytest.mark.parametrize("name", MODELS)
def test_frozen_model_matches_pytorch(name):
    doc = json.loads((HERE / (name + ".expected.json")).read_text())
    instr, data, regs = bench.load(doc["program"][:-len(".in.gz")])
    plat = build_atalla_platform(instr, data, init_sregs=regs, strict=True)
    plat.run_until_done(limit=20_000_000)
    assert plat.core.decode2.srf.storage.values[10] == 0, "run_graph returned nonzero status"
    assert doc["results"]
    for r in doc["results"]:
        want = (np.asarray(r["bf16"], dtype="<u4") << 16).view("<f4").reshape(r["shape"])
        got = read_tensor(plat.dram, r["address"], r["shape"])
        assert np.allclose(got, want, atol=doc["atol"], rtol=doc["rtol"]), \
            "%s (%s): max abs %g" % (r["node"], r["operation"], np.abs(got - want).max())
