"""Compare runs of the same harness on different architectures.

The harness is the control: both runs decompose the work identically, so the
gantt logs carry the same 32,768 tags and the same span kinds, and only the
cycle numbers differ. That makes a direct comparison possible -- per tile-pair
kernel, per phase, and side by side on the reuse timeline.

    python tools/plot_arch_comparison.py \\
        --runs logs/blocked1024_tpu_2pad logs/blocked1024_meissa_4pad \\
        --out docs/results/arch-comparison

Each --runs entry is a log directory written by one of the tiled harnesses.
The first is treated as the baseline.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.plot_tiled_sysarr_tpu_gantt import (  # noqa: E402
    ENVELOPE_PATHS, PATH_COLORS, PATH_LABELS, PATH_ORDER, _filter_rows,
    _read_rows, plot_gantt, read_stats_log)

BASE_COLOR = "#4c78a8"
NEW_COLOR = "#f58518"


class Run:
    """One log directory, loaded lazily."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.stats = read_stats_log(self.path / "stats.log")
        self._rows: List[Dict[str, str]] | None = None

    @property
    def rows(self) -> List[Dict[str, str]]:
        if self._rows is None:
            self._rows = _read_rows(self.path / "gantt.log")
        return self._rows

    @property
    def cycles(self) -> int:
        return int(self.stats.get("cycles", 0))

    @property
    def label(self) -> str:
        arr = self.stats.get("systolic_array", "tpu")
        pads = self.stats.get("spad_pads", "?")
        return "%s / %s pads" % (str(arr).upper(), pads)

    def tag_durations(self) -> Dict[str, int]:
        """kernel_total span per tile-pair, which is that kernel's wall time."""
        out: Dict[str, int] = {}
        for row in self.rows:
            if row["path"] != "kernel_total":
                continue
            out[row["tag"]] = int(row["duration_cycles"])
        return out

    def path_totals(self) -> Dict[str, int]:
        """Total occupancy per span kind, envelopes excluded -- they are sums
        of the others and would double count."""
        totals: Dict[str, int] = defaultdict(int)
        for row in self.rows:
            if row["path"] in ENVELOPE_PATHS:
                continue
            totals[row["path"]] += int(row["duration_cycles"])
        return dict(totals)


def _pct(new: float, base: float) -> float:
    return 100.0 * (new - base) / base if base else 0.0


def plot_cycles(runs: Sequence[Run], out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(runs)), 5))
    labels = [r.label for r in runs]
    values = [r.cycles for r in runs]
    colors = [BASE_COLOR] + [NEW_COLOR] * (len(runs) - 1)
    bars = ax.bar(labels, values, color=colors)
    base = values[0]
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, "{:,}".format(v),
                ha="center", va="bottom", fontsize=11)
        if v != base:
            ax.text(bar.get_x() + bar.get_width() / 2, v / 2,
                    "%+.1f%%" % _pct(v, base), ha="center", va="center",
                    fontsize=15, fontweight="bold", color="white")
    ax.set_ylabel("cycles to complete the GEMM", fontsize=13)
    ax.set_title("Total cycles, same harness and same decomposition", fontsize=14)
    ax.margins(y=0.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / "cycles.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_phase_breakdown(runs: Sequence[Run], out: Path) -> Path:
    """Where the time goes, per span kind. Spans overlap, so these are
    occupancies rather than a partition of the runtime -- the point is which
    kinds shrank."""
    totals = [r.path_totals() for r in runs]
    kinds = [p for p in PATH_ORDER if any(t.get(p) for t in totals)]
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.8 / len(runs)
    xs = range(len(kinds))
    for i, (run, tot) in enumerate(zip(runs, totals)):
        ax.bar([x + i * width for x in xs], [tot.get(k, 0) for k in kinds],
               width, label=run.label,
               color=BASE_COLOR if i == 0 else NEW_COLOR)
    ax.set_xticks([x + width * (len(runs) - 1) / 2 for x in xs])
    ax.set_xticklabels([PATH_LABELS.get(k, k) for k in kinds],
                       rotation=35, ha="right", fontsize=11)
    ax.set_ylabel("total span occupancy (cycles)", fontsize=13)
    ax.set_title("Where the time goes, by span kind", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / "phase_breakdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_kernel_speedup(runs: Sequence[Run], out: Path) -> Path:
    """Both runs carry the same tags, so each tile-pair kernel can be compared
    against itself."""
    base = runs[0]
    base_d = base.tag_durations()
    fig, axes = plt.subplots(1, len(runs) - 1, figsize=(6 * (len(runs) - 1), 5),
                             squeeze=False)
    for ax, run in zip(axes[0], runs[1:]):
        other = run.tag_durations()
        shared = [t for t in base_d if t in other and base_d[t] > 0]
        deltas = [_pct(other[t], base_d[t]) for t in shared]
        ax.hist(deltas, bins=60, color=NEW_COLOR, edgecolor="none")
        med = sorted(deltas)[len(deltas) // 2] if deltas else 0.0
        ax.axvline(0, color="#444", lw=1)
        ax.axvline(med, color="#e45756", lw=2,
                   label="median %+.1f%%" % med)
        ax.set_xlabel("per-kernel change vs %s (%%)" % base.label, fontsize=12)
        ax.set_ylabel("tile-pair kernels", fontsize=12)
        ax.set_title("%s, %d kernels compared" % (run.label, len(shared)),
                     fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out / "kernel_speedup.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


SDMA_PATHS = ("sdma_act", "sdma_wgt")


def plot_bottleneck(runs: Sequence[Run], out: Path) -> Path:
    """Why the end-to-end gain is smaller than the per-kernel gain.

    Occupancy divided by wall time is how many spans of a kind are live on an
    average cycle. The compute side speeds up and the DRAM side does not,
    so the DRAM share of the critical path grows and the kernel gain cannot be
    cashed in full. Note this is about the DRAM fill being a fixed cost per
    block, NOT about the burst channel being saturated -- it runs at 38-59%
    of its one-burst-per-cycle limit. See plot_dram_utilisation.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    base = runs[0]
    base_kt = base.tag_durations()
    base_mean = sum(base_kt.values()) / len(base_kt)
    names, kernel_pct, wall_pct = [], [], []
    for run in runs[1:]:
        kt = run.tag_durations()
        names.append(run.label)
        kernel_pct.append(_pct(sum(kt.values()) / len(kt), base_mean))
        wall_pct.append(_pct(run.cycles, base.cycles))
    xs = range(len(names))
    ax1.bar([x - 0.2 for x in xs], kernel_pct, 0.4, label="per-kernel",
            color=NEW_COLOR)
    ax1.bar([x + 0.2 for x in xs], wall_pct, 0.4, label="whole GEMM",
            color=BASE_COLOR)
    for x, (k, w) in enumerate(zip(kernel_pct, wall_pct)):
        ax1.text(x - 0.2, k, "%+.1f%%" % k, ha="center", va="top", fontsize=12)
        ax1.text(x + 0.2, w, "%+.1f%%" % w, ha="center", va="top", fontsize=12)
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels(names, fontsize=11)
    ax1.axhline(0, color="#444", lw=1)
    ax1.set_ylabel("change vs %s (%%)" % base.label, fontsize=12)
    ax1.set_title("The kernels got much faster than the GEMM did", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(axis="y", alpha=0.3)

    cats = [("kernels in flight", None), ("SDMA (DRAM side)", SDMA_PATHS),
            ("systolic array", ("systolic_array",))]
    width = 0.8 / len(runs)
    for i, run in enumerate(runs):
        tot = run.path_totals()
        kt = run.tag_durations()
        vals = []
        for label, paths in cats:
            if paths is None:
                vals.append(sum(kt.values()) / run.cycles)
            else:
                vals.append(sum(tot.get(p, 0) for p in paths) / run.cycles)
        ax2.bar([x + i * width for x in range(len(cats))], vals, width,
                label=run.label, color=BASE_COLOR if i == 0 else NEW_COLOR)
    ax2.set_xticks([x + width * (len(runs) - 1) / 2 for x in range(len(cats))])
    ax2.set_xticklabels([c[0] for c in cats], fontsize=11)
    ax2.set_ylabel("spans live per cycle", fontsize=12)
    ax2.set_title("DRAM demand held while compute demand fell", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = out / "bottleneck.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _contiguous_cluster(rows: Sequence[Dict[str, str]], max_tags: int
                        ) -> List[str]:
    """Tags that actually run together, earliest first.

    A filter on (tj, tk) alone spans the whole GEMM at 1024 -- the same weight
    column is revisited in every reuse block, millions of cycles apart -- and
    an 800-cycle span is invisible on an 11M-cycle axis. Take the first block
    instead: the earliest tag, plus every tag starting before the last of them
    ends.
    """
    starts: Dict[str, int] = {}
    ends: Dict[str, int] = {}
    for row in rows:
        if row["path"] != "kernel_total":
            continue
        tag = row["tag"]
        starts[tag] = int(row["start_cycle"])
        ends[tag] = int(row["end_cycle"])
    if not starts:
        return []
    order = sorted(starts, key=lambda t: starts[t])
    chosen = [order[0]]
    horizon = ends[order[0]]
    for tag in order[1:]:
        if len(chosen) >= max_tags:
            break
        if starts[tag] > horizon:
            break                      # a gap: the next block, not this one
        chosen.append(tag)
        horizon = max(horizon, ends[tag])
    return chosen


def plot_reuse_side_by_side(runs: Sequence[Run], out: Path, *,
                            kind: str, tj: int, tk: int, ti: int,
                            max_tags: int) -> List[Path]:
    """One reuse block per run, zoomed, so the packing difference is visible.

    Both panels use time_mode="relative", so each starts at zero and the axes
    are directly comparable even though the blocks occur at different absolute
    cycles in the two runs.
    """
    written: List[Path] = []
    for run in runs:
        if kind == "weight":
            pre = _filter_rows(run.rows, ti=None, tj=tj, tk=tk, tags=(),
                               max_tags=None, include_envelopes=True)
            head = "weight reuse, tj=%d tk=%d" % (tj, tk)
            stem = "reuse_weight"
        else:
            pre = _filter_rows(run.rows, ti=ti, tj=None, tk=tk, tags=(),
                               max_tags=None, include_envelopes=True)
            head = "activation reuse, ti=%d tk=%d" % (ti, tk)
            stem = "reuse_activation"
        tags = _contiguous_cluster(pre, max_tags)
        if not tags:
            continue
        rows = _filter_rows(pre, ti=None, tj=None, tk=None, tags=tags,
                            max_tags=None, include_envelopes=True)
        span = max(int(r["end_cycle"]) for r in rows) - \
            min(int(r["start_cycle"]) for r in rows)
        title = "%s -- %s (%d kernels, %d cycles)" % (
            run.label, head, len(tags), span)
        name = "%s_%s.png" % (stem, run.path.name)
        written.append(plot_gantt(rows, out / name, title,
                                  row_mode="tag", time_mode="relative"))
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="log directories; the first is the baseline")
    ap.add_argument("--out", required=True, help="directory for the figures")
    ap.add_argument("--tj", type=int, default=0)
    ap.add_argument("--tk", type=int, default=0)
    ap.add_argument("--ti", type=int, default=0)
    ap.add_argument("--max-tags", type=int, default=8)
    args = ap.parse_args()

    runs = [Run(Path(p)) for p in args.runs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("loaded:")
    for r in runs:
        print("  %-24s %12s cycles  %s" % (r.path.name, "{:,}".format(r.cycles),
                                           r.label))

    written = [plot_cycles(runs, out)]
    written.append(plot_phase_breakdown(runs, out))
    if len(runs) > 1:
        written.append(plot_kernel_speedup(runs, out))
        written.append(plot_bottleneck(runs, out))
    written += plot_reuse_side_by_side(runs, out, kind="weight", tj=args.tj,
                                       tk=args.tk, ti=args.ti,
                                       max_tags=args.max_tags)
    written += plot_reuse_side_by_side(runs, out, kind="activation", tj=args.tj,
                                       tk=args.tk, ti=args.ti,
                                       max_tags=args.max_tags)
    print("\nwrote:")
    for p in written:
        print("  %s" % p)


if __name__ == "__main__":
    main()
