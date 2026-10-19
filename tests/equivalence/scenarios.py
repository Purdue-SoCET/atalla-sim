"""Behaviour snapshot: fixed runs across the whole simulator, reduced to
numbers that must not change when the code is cleaned up.

Each scenario runs one piece of the simulator on fixed inputs and returns
a JSON-able dict: cycle counts, statistics, and SHA-256 fingerprints of
large state (registers, DRAM, output matrices). A refactor that changes
behaviour in any way -- a value, a cycle, a counter -- changes a number.

    make_snapshot.py      runs every scenario and writes snapshot.json
    test_equivalence.py   runs them again and compares with snapshot.json

To add a scenario: write a function returning a dict, add it to SCENARIOS
below, and regenerate snapshot.json on code you trust (see make_snapshot.py).
"""

import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "src", ROOT / "tools", ROOT / "tests" / "scheduler", ROOT / "tests" / "atalla"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def digest(obj) -> str:
    """A SHA-256 fingerprint of any JSON-able value (floats exactly)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=repr).encode()).hexdigest()


def jsonable(obj):
    """Round-trip through JSON, so snapshot and rerun compare the same way."""
    return json.loads(json.dumps(obj, sort_keys=True, default=repr))


# -- the scheduler ------------------------------------------------------------------------

def _core_state(core) -> dict:
    """Everything a scheduler run decides: when each packet issued, and the
    final scalar registers."""
    return {
        "issue_trace": digest([(cy, ip.pc) for cy, ip in core.issued]),
        "scalar_regs": digest([core.scalar_reg(r) for r in range(256)]),
        "flushes": core.flushes,
        "packets_fetched": core.packets_fetched,
        "dcache": dict(core.dcache.stats),
        "lsu": dict(core.lsu.stats),
    }


def _platform_state(plat, cycles: int) -> dict:
    out = _core_state(plat.core)
    out["cycles"] = cycles
    out["vector_regs"] = digest([plat.vc.read_vreg(v) for v in range(256)])
    out["mask_regs"] = digest([plat.core.decode2.mrf.read(m) for m in range(16)])
    out["dram"] = digest({k: bytes(v).hex() for k, v in sorted(plat.dram._blocks.items())})
    out["vector_side"] = dict(plat.core.vector.stats)
    out["perf"] = plat.perf.report() if plat.perf else None
    return out


def scalar_program(seed: int, small_cache: bool) -> dict:
    """A random scalar program on the bare scheduler core (stage 4 tests)."""
    from scheduler.core import SchedulerCore
    from scheduler.dcache import DCacheConfig
    from test_execute import DATA, random_program
    cfg = (DCacheConfig(size_bytes=512, ways=2, line_bytes=32, mshrs=2, targets_per_mshr=2)
           if small_cache else None)
    core = SchedulerCore(random_program(100 + seed if small_cache else seed,
                                        n=80 if small_cache else 60),
                         data=DATA, dcache_config=cfg)
    cycles = core.run_until_done(limit=200_000)
    out = _core_state(core)
    out["cycles"] = cycles
    out["memory"] = digest(sorted(core.memory.words.items()))
    return out


def vector_program(seed: int) -> dict:
    """A random vector program on the Atalla platform (stage 5 tests)."""
    from atalla.atalla_platform import build_atalla_platform
    from test_vector import DRAM_IN, LOAD_ROWS, random_vector_program, tile_data

    def vals(rr, col, _r=random.Random(seed)):
        return _r.choice([-3.0, -1.5, -0.5, 0.0, 0.25, 1.0, 2.5, 4.0])
    plat = build_atalla_platform(random_vector_program(seed),
                                 tile_data(DRAM_IN, LOAD_ROWS, vals), warm_icache=True)
    return _platform_state(plat, plat.run_until_done(limit=200_000))


#: Kernel programs, built once and kept here: the C compiler doesn't emit
#: the same assembly on every run, and this snapshot is about the simulator.
PROGRAMS = Path(__file__).resolve().parent / "programs"


def freeze_kernel(name: str) -> None:
    """Build a kernel's program (.in) and store it, gzipped, in PROGRAMS.

    A C program (name ending in .c) comes straight from the compiler
    (tools/c_kernels.py). The first four C scenarios were frozen from the
    compiler's old atalla-models branch, whose sources are gone: they are
    kept as they are."""
    import gzip
    import run_kernel
    frozen = PROGRAMS / (name + ".in.gz")
    if name.endswith(".c"):
        import c_kernels
        from scheduler.assembler import to_in_text
        sources = [d / name for d in (c_kernels.KERNELS, c_kernels.UNIT_TESTS)
                   if (d / name).is_file()]
        if not sources:
            if frozen.exists():
                return
            raise FileNotFoundError("no C source %s" % name)
        program, data, _ = c_kernels.build(sources[0])
        text = to_in_text(program.instr, data)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            text = run_kernel.build_kernel(name, [], Path(tmp)).read_text()
    PROGRAMS.mkdir(exist_ok=True)
    frozen.write_bytes(gzip.compress(text.encode(), mtime=0))


def kernel(name: str) -> dict:
    """A frozen kernel program (an assembly kernel from the functional sim's
    build scripts, or a C kernel, name ending in .c) on the Atalla platform."""
    import gzip
    from atalla.atalla_platform import build_atalla_platform, c_entry_sregs
    from scheduler.core import load_program_text
    frozen = PROGRAMS / (name + ".in.gz")
    if not frozen.exists():
        raise FileNotFoundError("no frozen program %s (make_snapshot.py builds it)" % frozen.name)
    instr, data = load_program_text(gzip.decompress(frozen.read_bytes()).decode())
    regs = c_entry_sregs(data) if name.endswith(".c") else None
    plat = build_atalla_platform(instr, data, init_sregs=regs, strict=False)
    return _platform_state(plat, plat.run_until_done(limit=5_000_000))


# -- the harness platform (TPUPlatform and friends) ----------------------------------------

def tpu_experiment(**overrides) -> dict:
    """atalla.sysarr_tpu_experiment: one GEMM tile through the harness."""
    from atalla.sysarr_tpu_experiment import SysArrTPUExperimentConfig, run_sysarr_tpu_experiment
    cfg = SysArrTPUExperimentConfig("snapshot", "manual", "none", "none")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return run_sysarr_tpu_experiment(cfg)


def tpu_system(size: int, systolic_array: str) -> dict:
    """atalla.sysarr_tpu_system.SysArrTPUSystem, as its main() runs it."""
    from atalla.sysarr_tpu_system import SysArrTPUSystem, _act_u16, _weights_u16
    system = SysArrTPUSystem(size=size, dtype="fp16", mirror=True, systolic_array=systolic_array)
    wgt = _weights_u16(size)
    system.load_inputs(_act_u16(size),
                       [[wgt[r][c] for r in range(size)] for c in range(size - 1, -1, -1)])
    got, mirror, cycles, metrics = system.run()
    return {"cycles": cycles, "output": digest(got), "mirror": digest(mirror),
            "flops": metrics.flops, "bytes_moved": metrics.bytes_moved}


def transpose_bench(kind: str) -> dict:
    from atalla.transpose_bench import run_spad_bench, run_vrf_bench
    r = run_vrf_bench() if kind == "vrf" else run_spad_bench()
    return {k: v for k, v in vars(r).items()}


def tiled_cosim(matrix: int, tile: int) -> dict:
    """The tiled-GEMM co-simulation from the tiled_1024 test, at a small size."""
    from test_scratchpad_vector_core_sysarr_tpu_tiled_1024 import run_tiled_tpu_cosim
    got, stats, _ = run_tiled_tpu_cosim(matrix, tile)
    return {"output": digest(got.tolist()), "stats": stats}


def blocked_cosim(matrix: int, tile: int, reuse: int) -> dict:
    """The blocked M/N-reuse co-simulation from the blocked_mn test, small."""
    from test_scratchpad_vector_core_sysarr_tpu_tiled_1024_blocked_mn import (
        run_blocked_mn_tpu_cosim)
    got, stats, _ = run_blocked_mn_tpu_cosim(matrix, tile, weight_reuse_m=reuse,
                                             activation_reuse_n=reuse)
    return {"output": digest(got.tolist()), "stats": stats}


# -- the list -------------------------------------------------------------------------------

ASM_KERNELS = ["add", "relu", "sigmoid", "softmax", "layernorm_param", "maxpool",
               "gemm", "gemm_tiled", "conv_tiled"]
C_KERNELS = ["add_4x32.c", "softmax_row32.c", "layernorm_4x4_active.c",
             "conv_h4_w4_r3_s3_k4_baseline.c",
             # compiled by the atalla-shaunak branch's packetizer
             "add.c", "relu.c", "maxpool.c"]

SCENARIOS = {}
for _s in range(3):
    SCENARIOS["scalar_program_%d" % _s] = lambda s=_s: scalar_program(s, False)
    SCENARIOS["scalar_program_small_cache_%d" % _s] = lambda s=_s: scalar_program(s, True)
    SCENARIOS["vector_program_%d" % _s] = lambda s=_s: vector_program(s)
for _k in ASM_KERNELS + C_KERNELS:
    SCENARIOS["kernel_" + _k] = lambda k=_k: kernel(k)
SCENARIOS.update({
    "tpu_experiment_default": lambda: tpu_experiment(),
    "tpu_experiment_tile16": lambda: tpu_experiment(tile=16),
    "tpu_experiment_bf16_lat6": lambda: tpu_experiment(dtype="bf16", backend_dram_latency=6),
    "tpu_system_32_tpu": lambda: tpu_system(32, "tpu"),
    "tpu_system_16_meissa": lambda: tpu_system(16, "meissa"),
    "transpose_bench_vrf": lambda: transpose_bench("vrf"),
    "transpose_bench_spad": lambda: transpose_bench("spad"),
    "tiled_cosim_64": lambda: tiled_cosim(64, 32),
    "blocked_cosim_64_reuse2": lambda: blocked_cosim(64, 32, 2),
})


def run(name: str) -> dict:
    return jsonable(SCENARIOS[name]())
