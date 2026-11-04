"""Machine configurations (src/atalla/config.py, configs/*.toml)."""
import dataclasses
import gzip
import inspect
from pathlib import Path

import pytest

from atalla import config
from atalla.atalla_platform import build_atalla_platform
from memory.axi import AXIParams
from scheduler.core import SchedulerCore, load_program_text
from scheduler.dcache import DCacheConfig

PROGRAMS = Path(__file__).resolve().parents[1] / "equivalence" / "programs"


def _defaults(fn):
    return {k: p.default for k, p in inspect.signature(fn).parameters.items()
            if p.default is not inspect.Parameter.empty}


def test_the_default_file_is_the_code_s_defaults():
    cfg = config.load()
    plat, core = _defaults(build_atalla_platform), _defaults(SchedulerCore.__init__)
    assert cfg["core"]["lane_count"] == plat["lane_count"]
    assert cfg["core"]["systolic_array"] == plat["systolic_array"]
    assert cfg["core"]["warm_icache"] == plat["warm_icache"]
    assert cfg["memory"]["dram_latency"] == plat["dram_latency"]
    assert cfg["memory"]["burst_bytes"] == plat["burst_bytes"]
    for k in ("lsu_depth", "icache_first_beat_wait", "icache_beat_wait"):
        assert cfg["core"][k] == core[k], k
    assert cfg["dcache"] == dataclasses.asdict(DCacheConfig())
    assert cfg["memory"]["axi_params"] == dataclasses.asdict(AXIParams())
    assert cfg["memory"]["dram"] == "fixed" and cfg["memory"]["axi"] == "auto"


def test_a_platform_from_the_default_config_runs_like_a_plain_one():
    instr, data = load_program_text(
        gzip.decompress((PROGRAMS / "gemm_tiled.in.gz").read_bytes()).decode())
    a = build_atalla_platform(instr, data)
    b = build_atalla_platform(instr, data, **config.platform_kwargs(config.load()))
    assert a.run_until_done() == b.run_until_done()


def test_a_file_sets_only_what_it_names(tmp_path):
    p = tmp_path / "small.toml"
    p.write_text('[core]\nlane_count = 8\n[dcache]\nsize_bytes = 2048\n'
                 '[memory.ramulator]\nnCL = 22\n')
    cfg = config.load(p)
    assert cfg["name"] == "small" and cfg["core"]["lane_count"] == 8
    assert cfg["dcache"]["size_bytes"] == 2048 and cfg["dcache"]["ways"] == 4
    assert cfg["memory"]["ramulator"] == {"nCL": 22}
    assert "8 lanes" in config.summary(cfg)


def test_a_typo_is_an_error_not_a_silent_default(tmp_path):
    p = tmp_path / "typo.toml"
    p.write_text("[core]\nlanes = 8\n")
    with pytest.raises(config.ConfigError, match="core.lanes"):
        config.load(p)
    with pytest.raises(config.ConfigError, match="no config 'nope'"):
        config.load("nope")


def test_every_shipped_config_loads():
    for path in config.CONFIG_DIR.glob("*.toml"):
        cfg = config.load(path)
        assert config.summary(cfg).startswith(cfg["name"])
