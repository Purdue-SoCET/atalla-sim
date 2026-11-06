"""The opt-in SDMA start/end log (src/atalla/sdma_log.py)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import bench                                                     # noqa: E402
from atalla.atalla_platform import build_atalla_platform         # noqa: E402


def _run(name, **kw):
    instr, data, regs = bench.load(name)
    plat = build_atalla_platform(instr, data, init_sregs=regs, strict=False, **kw)
    return plat, plat.run_until_done()


def test_off_by_default_and_attaches_nothing():
    plat, _ = _run("add.c")
    assert plat.sdma_log is None
    assert plat.core.vector.on_sdma is None


def test_logs_every_sdma_start_and_end_without_changing_timing():
    plain, cycles = _run("add.c")
    plat, logged = _run("add.c", sdma_log=True)
    assert logged == cycles
    recs = plat.sdma_log.records
    s = plain.core.vector.stats
    assert len(recs) == s["sdma_loads"] + s["sdma_stores"] > 0
    assert [r["id"] for r in recs] == list(range(len(recs)))
    for r in recs:
        assert r["end"] is not None and r["end"] > r["start"]
        assert r["op"] in ("scpad.ld", "scpad.st")
        assert r["bytes"] == r["rows"] * r["cols"] * 2
    assert sum(r["op"] == "scpad.st" for r in recs) == s["sdma_stores"]
    lines = plat.sdma_log.text().splitlines()
    assert len(lines) == 2 * len(recs)
    assert sum(" start #" in l for l in lines) == sum(" end   #" in l for l in lines) == len(recs)
    starts = [int(l.split()[1]) for l in lines]
    assert starts == sorted(starts)


def test_run_flag(tmp_path, capsys):
    import gzip
    import run_kernel
    prog = tmp_path / "add.in"
    prog.write_bytes(gzip.decompress((bench.PROGRAMS / "add.c.in.gz").read_bytes()))
    out = tmp_path / "sdma.log"
    assert run_kernel.main([str(prog), "--sdma-log", str(out)]) in (0, None)
    text = out.read_text()
    assert "start #0 scpad.ld" in text and "end   #0" in text
    run_kernel.main([str(prog), "--sdma-log", "-"])
    assert "SDMAs (" in capsys.readouterr().out
