"""The comparison plotter, on a pair of tiny synthetic runs."""
from pathlib import Path

import pytest

from tools.plot_arch_comparison import (
    Run, _contiguous_cluster, plot_bottleneck, plot_cycles,
    plot_kernel_speedup, plot_phase_breakdown)

HEADER = "[gantt] tag ti tj tk slot path start_cycle end_cycle duration_cycles touches"


def _write_run(tmp: Path, name: str, array: str, pads: int, scale: float) -> Path:
    d = tmp / name
    d.mkdir(parents=True)
    (d / "stats.log").write_text(
        "[stats] cycles %d\n[stats] systolic_array %s\n[stats] spad_pads %d\n"
        % (int(4000 * scale), array, pads))
    lines = [HEADER]
    cycle = 0
    for block in range(2):
        for ti in range(3):
            tag = "ti%02d_tj00_tk%02d" % (ti, block)
            dur = int(100 * scale)
            for path, off, ln in (("kernel_total", 0, dur),
                                  ("sdma_wgt", 0, dur // 2),
                                  ("systolic_array", dur // 2, dur // 4)):
                lines.append("[gantt] %s %d 0 %d 0 %s %d %d %d 1"
                             % (tag, ti, block, path, cycle + off,
                                cycle + off + ln, ln))
            cycle += dur // 3
        cycle += 100_000          # a gap: the next reuse block
    (d / "gantt.log").write_text("\n".join(lines) + "\n")
    return d


@pytest.fixture
def runs(tmp_path):
    return [Run(_write_run(tmp_path, "old", "tpu", 2, 1.0)),
            Run(_write_run(tmp_path, "new", "meissa", 4, 0.5))]


def test_run_reads_stats_and_spans(runs):
    old, new = runs
    assert old.cycles == 4000 and new.cycles == 2000
    assert old.label == "TPU / 2 pads"
    assert new.label == "MEISSA / 4 pads"
    assert len(old.tag_durations()) == 6
    # envelopes are excluded from the phase totals or they double count
    assert "kernel_total" not in old.path_totals()
    assert old.path_totals()["sdma_wgt"] > 0


def test_cluster_stops_at_the_gap(runs):
    """A reuse block is contiguous; the next one is millions of cycles later
    and must not be pulled into the same picture."""
    chosen = _contiguous_cluster(runs[0].rows, max_tags=10)
    assert len(chosen) == 3, "should stop at the block boundary, got %s" % chosen
    assert all(t.endswith("tk00") for t in chosen)


def test_cluster_respects_max_tags(runs):
    assert len(_contiguous_cluster(runs[0].rows, max_tags=2)) == 2


def test_every_figure_is_written(runs, tmp_path):
    out = tmp_path / "figs"
    out.mkdir()
    for fn in (plot_cycles, plot_phase_breakdown, plot_kernel_speedup,
               plot_bottleneck):
        path = fn(runs, out)
        assert path.exists() and path.stat().st_size > 0, fn.__name__
