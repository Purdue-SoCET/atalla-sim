"""atalla-sim sweep (tools/sweep.py)."""
import csv
import gzip
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import sweep  # noqa: E402
from atalla import config  # noqa: E402


def test_keys_resolve_by_path_unique_name_or_alias():
    cfg = config.load()
    assert sweep.resolve_key(cfg, "lanes") == ("core", "lane_count")
    assert sweep.resolve_key(cfg, "dram_latency") == ("memory", "dram_latency")
    assert sweep.resolve_key(cfg, "dcache.size_bytes") == ("dcache", "size_bytes")
    assert sweep.resolve_key(cfg, "ramulator.nCL") == ("memory", "ramulator", "nCL")
    with pytest.raises(ValueError, match="ambiguous"):
        sweep.resolve_key(cfg, "beat_bytes")             # dcache and axi_params
    with pytest.raises(ValueError, match="no configuration key"):
        sweep.resolve_key(cfg, "lane_cuont")


def test_values_parse():
    assert [sweep.parse_value(v) for v in ("8", "0x10", "2.5", "true", "tpu")] == \
        [8, 16, 2.5, True, "tpu"]


def test_a_sweep_runs_every_point_of_the_grid(tmp_path):
    prog = tmp_path / "relu.in"
    prog.write_bytes(gzip.decompress((ROOT / "tests/equivalence/programs/relu.in.gz").read_bytes()))
    out = tmp_path / "s.csv"
    status = sweep.main([str(prog), "--param", "latency=6,30", "--param", "lanes=8,16",
                         "-j", "2", "-o", str(out)])
    assert status == 0
    rows = list(csv.DictReader(open(out)))
    assert [(r["latency"], r["lanes"]) for r in rows] == \
        [("6", "8"), ("6", "16"), ("30", "8"), ("30", "16")]
    cycles = {(r["latency"], r["lanes"]): int(r["cycles"]) for r in rows}
    assert cycles[("30", "16")] > cycles[("6", "16")]
