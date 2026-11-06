"""Machine configurations: one file per design point, in configs/*.toml.

    cfg = config.load("atalla-0x01")              # a name in configs/, or a path
    cfg = config.override(cfg, {"core": {"lane_count": 8}})
    plat = build_atalla_platform(instr, data, **config.platform_kwargs(cfg))
    config.summary(cfg)                           # one line, for reports

A file sets any subset of the keys in configs/atalla-0x01.toml; the rest
keep that file's values, which are build_atalla_platform's own defaults
(tests/atalla/test_config.py keeps them equal). An unknown key is an
error, so a typo can't silently leave a parameter at its default.

Python 3.11 reads TOML itself; 3.9 and 3.10 need `pip install tomli`
(installed with atalla-sim).
"""

import copy
from pathlib import Path
from typing import Any, Dict, Union

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
DEFAULT = "atalla-0x01"


class ConfigError(ValueError):
    pass


def _toml(path: Path) -> Dict[str, Any]:
    try:
        import tomllib
    except ImportError:                                # Python < 3.11
        try:
            import tomli as tomllib
        except ImportError:
            raise ConfigError("reading %s needs Python 3.11, or `pip install tomli`" % path)
    with open(path, "rb") as f:
        try:
            return tomllib.load(f)
        except Exception as exc:                        # tomllib.TOMLDecodeError
            raise ConfigError("%s: %s" % (path, exc))


def resolve_path(name_or_path: Union[str, Path]) -> Path:
    p = Path(name_or_path)
    if p.suffix == ".toml" or p.exists():
        if not p.is_file():
            raise ConfigError("no config file %s" % p)
        return p
    p = CONFIG_DIR / ("%s.toml" % name_or_path)
    if not p.is_file():
        names = ", ".join(sorted(f.stem for f in CONFIG_DIR.glob("*.toml")))
        raise ConfigError("no config %r (in %s: %s)" % (str(name_or_path), CONFIG_DIR, names))
    return p


# Tables whose keys are free-form (any Ramulator parameter, any mnemonic).
_OPEN_TABLES = {("core", "ex_latency"), ("memory", "ramulator")}


def _merge(base: Dict[str, Any], over: Dict[str, Any], where: tuple = ()) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        here = where + (k,)
        if k not in out and where not in _OPEN_TABLES:
            raise ConfigError("unknown key %r (known: %s)"
                              % (".".join(here), ", ".join(sorted(out)) or "-"))
        if isinstance(v, dict) and here not in _OPEN_TABLES:
            if not isinstance(out.get(k), dict):
                raise ConfigError("%r is a value, not a table" % ".".join(here))
            out[k] = _merge(out[k], v, here)
        else:
            out[k] = copy.deepcopy(v)
    return out


def defaults() -> Dict[str, Any]:
    return _toml(CONFIG_DIR / ("%s.toml" % DEFAULT))


def load(name_or_path: Union[str, Path, None] = None) -> Dict[str, Any]:
    base = defaults()
    if name_or_path is None or str(name_or_path) == DEFAULT:
        return base
    path = resolve_path(name_or_path)
    cfg = _merge(base, _toml(path))
    if "name" not in _toml(path):
        cfg["name"] = path.stem
    return cfg


def override(cfg: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    """`cfg` with `changes` (same nesting) applied, checked like a file."""
    return _merge(cfg, changes)


def platform_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Keywords for build_atalla_platform. Builds the Ramulator channel when
    the config asks for one (raises RuntimeError if it isn't built)."""
    from memory.axi import AXIParams
    from scheduler.dcache import DCacheConfig
    core, mem = cfg["core"], cfg["memory"]
    kw: Dict[str, Any] = dict(
        lane_count=core["lane_count"], systolic_array=core["systolic_array"],
        lsu_depth=core["lsu_depth"], warm_icache=core["warm_icache"],
        icache_first_beat_wait=core["icache_first_beat_wait"],
        icache_beat_wait=core["icache_beat_wait"],
        dcache_config=DCacheConfig(**cfg["dcache"]),
        dram_latency=mem["dram_latency"], burst_bytes=mem["burst_bytes"],
        axi_params=AXIParams(**mem["axi_params"]))
    if core.get("ex_latency"):
        kw["ex_latency"] = dict(core["ex_latency"])
    axi = mem["axi"]
    if axi not in ("auto", True, False):
        raise ConfigError("memory.axi is true, false or \"auto\", not %r" % (axi,))
    kw["axi"] = None if axi == "auto" else bool(axi)
    if mem["dram"] == "ramulator":
        from memory.ramulator import RamulatorDRAM
        kw["dram_channel"] = RamulatorDRAM(mem["ramulator_config"], **dict(mem["ramulator"]))
    elif mem["dram"] != "fixed":
        raise ConfigError("memory.dram is \"fixed\" or \"ramulator\", not %r" % (mem["dram"],))
    return kw


def summary(cfg: Dict[str, Any]) -> str:
    core, mem = cfg["core"], cfg["memory"]
    if mem["dram"] == "ramulator":
        dram = "Ramulator %s" % mem["ramulator_config"]
        if mem["ramulator"]:
            dram += " (%s)" % ", ".join("%s=%s" % kv for kv in sorted(mem["ramulator"].items()))
    else:
        dram = "fixed %d-cycle DRAM" % mem["dram_latency"]
    axi = mem["axi"] if mem["axi"] != "auto" else mem["dram"] == "ramulator"
    return "%s: %s, %d lanes, %s, %s%s, %d B bursts" % (
        cfg.get("name", "?"), core["systolic_array"].upper(), core["lane_count"],
        "%d KB %d-way dcache" % (cfg["dcache"]["size_bytes"] // 1024, cfg["dcache"]["ways"]),
        dram, " behind the AXI bus" if axi else "", mem["burst_bytes"])
