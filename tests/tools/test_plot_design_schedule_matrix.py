"""The 2x2 design-by-schedule plotter."""
from pathlib import Path

import pytest

from tools.plot_design_schedule_matrix import (
    DESIGNS, SCHEDULES, Matrix, _burst_utilisation, plot_decomposition,
    plot_dram, plot_interaction, plot_kernel_ecdf, plot_matrix_cycles,
    plot_useful_work)

HEADER = "[gantt] tag ti tj tk slot path start_cycle end_cycle duration_cycles touches"


def _run_dir(tmp: Path, name: str, *, cycles: int, array: str, pads: int,
             bytes_tx: int, kernel: int) -> Path:
    d = tmp / name
    d.mkdir(parents=True)
    (d / "stats.log").write_text(
        "[stats] cycles %d\n[stats] systolic_array %s\n[stats] spad_pads %d\n"
        "[stats] bytes_transmitted %d\n[stats] backend_dram_burst_bytes 32\n"
        "[stats] active_pe_sum %d\n"
        % (cycles, array, pads, bytes_tx, 1024 ** 3))
    lines = [HEADER]
    for ti in range(4):
        lines.append("[gantt] ti%02d_tj00_tk00 %d 0 0 0 kernel_total %d %d %d 1"
                     % (ti, ti, ti * 10, ti * 10 + kernel, kernel))
    (d / "gantt.log").write_text("\n".join(lines) + "\n")
    return d


@pytest.fixture
def matrix(tmp_path):
    spec = {
        ("tiled", "old"): dict(cycles=22_000_000, array="tpu", pads=2,
                               bytes_tx=268_435_456, kernel=1300),
        ("tiled", "new"): dict(cycles=21_800_000, array="meissa", pads=4,
                               bytes_tx=268_435_456, kernel=1290),
        ("reuse", "old"): dict(cycles=13_250_000, array="tpu", pads=2,
                               bytes_tx=218_103_808, kernel=820),
        ("reuse", "new"): dict(cycles=11_600_000, array="meissa", pads=4,
                               bytes_tx=218_103_808, kernel=580),
    }
    paths = {k: _run_dir(tmp_path, "%s_%s" % k, **v) for k, v in spec.items()}
    return Matrix(paths)


def test_the_matrix_indexes_by_schedule_and_design(matrix):
    assert matrix[("reuse", "new")].cycles == 11_600_000
    assert matrix[("tiled", "old")].label == "TPU / 2 pads"
    assert matrix[("reuse", "new")].label == "MEISSA / 4 pads"


def test_burst_utilisation_is_a_duty_cycle(matrix):
    """One burst launch per cycle is the limit, so this must land in (0, 1]."""
    for s in SCHEDULES:
        for d in DESIGNS:
            u = _burst_utilisation(matrix, (s, d))
            assert 0.0 < u <= 1.0, "%s/%s gave %r" % (s, d, u)
    # reuse moves fewer bytes in fewer cycles, so it works the channel harder
    assert (_burst_utilisation(matrix, ("reuse", "new"))
            > _burst_utilisation(matrix, ("reuse", "old"))
            > _burst_utilisation(matrix, ("tiled", "old")))


def test_dram_traffic_is_identical_between_designs(matrix):
    """The control for the whole comparison: the design cannot change how many
    bytes cross the DRAM boundary, only the schedule can."""
    for s in SCHEDULES:
        assert (matrix.stat((s, "old"), "bytes_transmitted")
                == matrix.stat((s, "new"), "bytes_transmitted"))
    assert (matrix.stat(("tiled", "old"), "bytes_transmitted")
            > matrix.stat(("reuse", "old"), "bytes_transmitted"))


def test_every_figure_is_written(matrix, tmp_path):
    out = tmp_path / "figs"
    out.mkdir()
    for fn in (plot_matrix_cycles, plot_interaction, plot_decomposition,
               plot_dram, plot_useful_work, plot_kernel_ecdf):
        p = fn(matrix, out)
        assert p.exists() and p.stat().st_size > 0, fn.__name__
