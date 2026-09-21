"""Two designs x two schedules, on the same 1024x1024 GEMM.

    design    = the hardware: TPU/2 pads (old) or MEISSA/4 pads (new)
    schedule  = the software: plain tiled, or blocked M/N weight+activation reuse

Four runs make the interaction visible, which neither axis shows alone: the
design only pays off once the schedule has cut enough DRAM traffic for compute
to matter.

    python tools/plot_design_schedule_matrix.py \\
        --tiled-old  logs/tiled1024_tpu_2pad \\
        --tiled-new  logs/tiled1024_meissa_4pad \\
        --reuse-old  logs/blocked1024_tpu_2pad \\
        --reuse-new  logs/blocked1024_meissa_4pad \\
        --out docs/results/arch-comparison/matrix

WHAT IS AND IS NOT COMPARABLE
-----------------------------
Comparable across all four: cycles, bytes_transmitted, external bandwidth, and
active_pe_sum -- which is exactly 1024^3 in every run, i.e. the same useful
work, differently scheduled.

NOT comparable across designs: pe_mac_ops, pe_mul_ops, pe_add_ops, flops_micro,
mac_utilization and the avg-active-PE figures. The two arrays count them by
different conventions -- the TPU counts every cell it clocks, bubbles included,
while MEISSA counts only live columns -- so a bar chart of them would show a
difference in bookkeeping, not in hardware. Nothing here plots them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.plot_arch_comparison import Run, _pct  # noqa: E402

OLD_COLOR = "#4c78a8"
NEW_COLOR = "#f58518"
SCHEDULES = ("tiled", "reuse")
DESIGNS = ("old", "new")
DESIGN_LABEL = {"old": "TPU / 2 pads", "new": "MEISSA / 4 pads"}
SCHEDULE_LABEL = {"tiled": "plain tiled\n(no reuse)",
                  "reuse": "blocked M/N\n(weight + act reuse)"}
COLOR = {"old": OLD_COLOR, "new": NEW_COLOR}

#: Bytes per cycle the DRAM side sustains while a transfer is in flight. The
#: same in all four runs -- one burst launch per cycle, shared by every backend.
DRAM_ROOF_KEY = "external_bandwidth_active_bytes_per_cycle"


def _burst_utilisation(m: "Matrix", key: Tuple[str, str]) -> float:
    """Fraction of cycles in which the shared channel launched a burst.

    SharedDRAMBurstChannel permits one launch per cycle across every backend,
    so bytes / burst_bytes / cycles is that channel's duty cycle -- the one
    resource more pads cannot widen.
    """
    b = m.stat(key, "bytes_transmitted")
    burst = m.stat(key, "backend_dram_burst_bytes", 32)
    cycles = m[key].cycles
    return (b / burst) / cycles if cycles else 0.0


class Matrix:
    def __init__(self, paths: Dict[Tuple[str, str], Path]):
        self.runs = {k: Run(v) for k, v in paths.items()}

    def __getitem__(self, key: Tuple[str, str]) -> Run:
        return self.runs[key]

    def stat(self, key: Tuple[str, str], name: str, default=0):
        return self.runs[key].stats.get(name, default)


def _annotate(ax, x, y, text, **kw):
    ax.annotate(text, (x, y), ha="center", fontsize=11, **kw)


def plot_matrix_cycles(m: Matrix, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.35
    xs = range(len(SCHEDULES))
    for i, design in enumerate(DESIGNS):
        vals = [m[(s, design)].cycles for s in SCHEDULES]
        bars = ax.bar([x + (i - 0.5) * width for x in xs], vals, width,
                      label=DESIGN_LABEL[design], color=COLOR[design])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, "{:,}".format(v),
                    ha="center", va="bottom", fontsize=10)
    for x, s in zip(xs, SCHEDULES):
        old, new = m[(s, "old")].cycles, m[(s, "new")].cycles
        ax.text(x, max(old, new) * 0.45, "%+.1f%%" % _pct(new, old),
                ha="center", fontsize=15, fontweight="bold", color="#333")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([SCHEDULE_LABEL[s] for s in SCHEDULES], fontsize=12)
    ax.set_ylabel("cycles", fontsize=13)
    ax.set_title("1024x1024 GEMM: two designs, two schedules", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.margins(y=0.12)
    fig.tight_layout()
    p = out / "matrix_cycles.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def plot_interaction(m: Matrix, out: Path) -> Path:
    """Non-parallel lines mean the two factors interact: the design's value
    depends on the schedule."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs = range(len(SCHEDULES))
    for design in DESIGNS:
        vals = [m[(s, design)].cycles / 1e6 for s in SCHEDULES]
        ax.plot(list(xs), vals, "o-", lw=2.5, ms=10,
                color=COLOR[design], label=DESIGN_LABEL[design])
        # The two lines nearly touch on the left, so push the labels apart
        # rather than letting them overprint each other.
        va = "bottom" if design == "old" else "top"
        pad = 0.12 if design == "old" else -0.12
        for x, v in zip(xs, vals):
            ax.annotate("%.2fM" % v, (x, v + pad), ha="center", va=va,
                        fontsize=11, color=COLOR[design])
    gap_t = m[("tiled", "old")].cycles - m[("tiled", "new")].cycles
    gap_r = m[("reuse", "old")].cycles - m[("reuse", "new")].cycles
    ax.set_xticks(list(xs))
    ax.set_xticklabels([SCHEDULE_LABEL[s] for s in SCHEDULES], fontsize=12)
    ax.set_ylabel("cycles (millions)", fontsize=13)
    ax.set_title("The design is worth %.2fM cycles with reuse, %.2fM without"
                 % (gap_r / 1e6, gap_t / 1e6), fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.margins(x=0.18, y=0.15)
    fig.tight_layout()
    p = out / "interaction.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def plot_decomposition(m: Matrix, out: Path) -> Path:
    """Both orders of applying the two changes, to show neither is additive."""
    base = m[("tiled", "old")].cycles
    best = m[("reuse", "new")].cycles
    routes = [
        ("schedule first", [("baseline", base),
                            ("+ reuse", m[("reuse", "old")].cycles),
                            ("+ new design", best)]),
        ("design first", [("baseline", base),
                          ("+ new design", m[("tiled", "new")].cycles),
                          ("+ reuse", best)]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, (title, steps) in zip(axes, routes):
        labels = [s[0] for s in steps]
        vals = [s[1] / 1e6 for s in steps]
        colors = ["#999999", "#54a24b", "#54a24b"]
        bars = ax.bar(labels, vals, color=colors)
        for i, (bar, v) in enumerate(zip(bars, vals)):
            ax.text(bar.get_x() + bar.get_width() / 2, v, "%.2fM" % v,
                    ha="center", va="bottom", fontsize=11)
            if i:
                ax.text(bar.get_x() + bar.get_width() / 2, v / 2,
                        "%+.1f%%" % _pct(steps[i][1], steps[i - 1][1]),
                        ha="center", va="center", fontsize=13,
                        fontweight="bold", color="white")
        ax.set_title(title, fontsize=13)
        ax.grid(axis="y", alpha=0.3)
        ax.margins(y=0.12)
    axes[0].set_ylabel("cycles (millions)", fontsize=13)
    fig.suptitle("Same endpoint, different split: the steps are not additive",
                 fontsize=14)
    fig.tight_layout()
    p = out / "decomposition.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def plot_dram(m: Matrix, out: Path) -> Path:
    """DRAM traffic is set by the schedule alone; the design cannot touch it."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    width = 0.35
    xs = range(len(SCHEDULES))
    for i, design in enumerate(DESIGNS):
        mb = [m.stat((s, design), "bytes_transmitted") / 2 ** 20 for s in SCHEDULES]
        ax1.bar([x + (i - 0.5) * width for x in xs], mb, width,
                label=DESIGN_LABEL[design], color=COLOR[design])
        for x, v in zip(xs, mb):
            ax1.text(x + (i - 0.5) * width, v, "%.0f MB" % v, ha="center",
                     va="bottom", fontsize=10)
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels([SCHEDULE_LABEL[s] for s in SCHEDULES], fontsize=11)
    ax1.set_ylabel("bytes moved to/from DRAM (MB)", fontsize=12)
    ax1.set_title("Traffic is the schedule's to change, not the design's",
                  fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    ax1.margins(y=0.12)

    # The shared channel launches at most one burst per cycle, so
    # bursts/cycle IS the utilisation of the one resource the pad count
    # cannot widen.
    width2 = 0.35
    for i, design in enumerate(DESIGNS):
        util = [100.0 * _burst_utilisation(m, (s, design)) for s in SCHEDULES]
        ax2.bar([x + (i - 0.5) * width2 for x in xs], util, width2,
                label=DESIGN_LABEL[design], color=COLOR[design])
        for x, v in zip(xs, util):
            ax2.text(x + (i - 0.5) * width2, v, "%.1f%%" % v, ha="center",
                     va="bottom", fontsize=10)
    ax2.axhline(100.0, color="#e45756", ls="--", lw=2,
                label="one burst per cycle, the hard limit")
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels([SCHEDULE_LABEL[s] for s in SCHEDULES], fontsize=11)
    ax2.set_ylabel("DRAM burst-channel utilisation (%)", fontsize=12)
    ax2.set_title("Rising, but never saturated", fontsize=13)
    ax2.set_ylim(0, 112)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = out / "dram.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def plot_kernel_ecdf(m: Matrix, out: Path) -> Path:
    """Per-kernel wall time, all four runs. Same 32,768 tags everywhere."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for s in SCHEDULES:
        for design in DESIGNS:
            durs = sorted(m[(s, design)].tag_durations().values())
            ys = [i / len(durs) for i in range(len(durs))]
            ax.step(durs, ys, where="post",
                    color=COLOR[design],
                    ls="-" if s == "reuse" else "--", lw=2,
                    label="%s, %s" % (s, DESIGN_LABEL[design]))
    ax.set_xlabel("tile-pair kernel duration (cycles)", fontsize=12)
    ax.set_ylabel("fraction of kernels at or below", fontsize=12)
    ax.set_title("Per-kernel wall time, 32,768 kernels per run", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out / "kernel_ecdf.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def plot_useful_work(m: Matrix, out: Path) -> Path:
    """The control: all four runs do exactly the same useful work."""
    fig, ax = plt.subplots(figsize=(9, 5))
    names, vals = [], []
    for s in SCHEDULES:
        for d in DESIGNS:
            names.append("%s\n%s" % (s, d))
            vals.append(m.stat((s, d), "active_pe_sum") / 1024 ** 3)
    ax.bar(names, vals, color=[COLOR[d] for _ in SCHEDULES for d in DESIGNS])
    ax.axhline(1.0, color="#e45756", ls="--", lw=2,
               label="1024$^3$ MACs, the algorithmic minimum")
    for i, v in enumerate(vals):
        ax.text(i, v, "%.3f" % v, ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("active PE-cycles / 1024$^3$", fontsize=12)
    ax.set_title("Same useful work in every run -- only the timing differs",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.margins(y=0.15)
    fig.tight_layout()
    p = out / "useful_work.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for s in SCHEDULES:
        for d in DESIGNS:
            ap.add_argument("--%s-%s" % (s, d), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = {(s, d): Path(getattr(args, "%s_%s" % (s, d)))
             for s in SCHEDULES for d in DESIGNS}
    m = Matrix(paths)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("%-10s %-18s %14s %12s" % ("schedule", "design", "cycles", "DRAM MB"))
    for s in SCHEDULES:
        for d in DESIGNS:
            print("%-10s %-18s %14s %12.0f" % (
                s, DESIGN_LABEL[d], "{:,}".format(m[(s, d)].cycles),
                m.stat((s, d), "bytes_transmitted") / 2 ** 20))

    written = [plot_matrix_cycles(m, out), plot_interaction(m, out),
               plot_decomposition(m, out), plot_dram(m, out),
               plot_useful_work(m, out), plot_kernel_ecdf(m, out)]
    print("\nwrote:")
    for p in written:
        print("  %s" % p)


if __name__ == "__main__":
    main()
