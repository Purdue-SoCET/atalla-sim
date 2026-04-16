#!/usr/bin/env python3

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with `python3.11 -m pip install matplotlib`."
    ) from exc


PHASE_COLUMNS = [
    "phase_weight_preload",
    "phase_activation_stream",
    "phase_compute_ramp",
    "phase_steady_compute",
    "phase_store_tail",
    "phase_drain",
    "phase_idle",
]


QUEUE_COLUMNS = [
    "queue_max_gsau_rd_queue",
    "queue_max_vlsu_dst_fifo",
    "queue_max_wb_buffer",
]


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _num(value: str) -> float:
    return float(value)


def _label_for_value(value: float) -> str:
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:g}"


def _has_column(rows: List[Dict[str, str]], column: str) -> bool:
    return any(str(row.get(column, "")).strip() not in {"", "None"} for row in rows)


def plot_sweep(rows: List[Dict[str, str]], sweep_name: str, output_dir: Path) -> Path:
    ordered = sorted(rows, key=lambda row: _num(row["param_value"]))
    x = [_num(row["param_value"]) for row in ordered]
    labels = [_label_for_value(value) for value in x]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"TPU sweep story: {sweep_name}")

    axes[0, 0].plot(x, [_num(row["cycles"]) for row in ordered], marker="o", linewidth=2)
    axes[0, 0].set_title("Time To Finish")
    axes[0, 0].set_xlabel(ordered[0]["param_name"])
    axes[0, 0].set_ylabel("cycles")
    axes[0, 0].grid(True, alpha=0.3)

    bottom = [0.0 for _ in x]
    phase_labels = [name.replace("phase_", "") for name in PHASE_COLUMNS]
    phase_colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#9d755d"]
    for color, phase_col, phase_label in zip(phase_colors, PHASE_COLUMNS, phase_labels):
        shares = []
        for row in ordered:
            cycles = max(_num(row["cycles"]), 1.0)
            shares.append(100.0 * _num(row.get(phase_col, "0")) / cycles)
        axes[0, 1].bar(labels, shares, bottom=bottom, label=phase_label, color=color)
        bottom = [a + b for a, b in zip(bottom, shares)]
    axes[0, 1].set_title("Where The Time Goes")
    axes[0, 1].set_xlabel(ordered[0]["param_name"])
    axes[0, 1].set_ylabel("share of total cycles (%)")
    axes[0, 1].legend(fontsize=8)

    ax_util = axes[1, 0]
    ax_tp = ax_util.twinx()
    ax_util.plot(x, [_num(row["mac_utilization"]) for row in ordered], marker="o", color="#54a24b", label="overall utilization")
    if _has_column(ordered, "mac_utilization_sysarr_active"):
        ax_util.plot(
            x,
            [_num(row.get("mac_utilization_sysarr_active", "0")) for row in ordered],
            marker="^",
            linestyle="--",
            color="#4c78a8",
            label="utilization while active",
        )
    ax_tp.plot(
        x,
        [_num(row["throughput_float_operations_per_cycle"]) for row in ordered],
        marker="s",
        color="#f58518",
        label="throughput",
    )
    ax_util.set_title("How Busy The Array Is")
    ax_util.set_xlabel(ordered[0]["param_name"])
    ax_util.set_ylabel("utilization")
    ax_tp.set_ylabel("FLOPs/cycle")
    ax_util.grid(True, alpha=0.3)
    lines = ax_util.get_lines() + ax_tp.get_lines()
    ax_util.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)

    reuse_columns = [
        ("reuse_weight_internal_over_external", "weight reuse", "#f28e2b"),
        ("reuse_act_internal_over_external", "activation reuse", "#4e79a7"),
        ("reuse_psum_internal_over_external", "psum reuse", "#59a14f"),
    ]
    if any(_has_column(ordered, column) for column, _, _ in reuse_columns):
        for column, label, color in reuse_columns:
            if not _has_column(ordered, column):
                continue
            axes[1, 1].plot(x, [_num(row.get(column, "0")) for row in ordered], marker="o", color=color, label=label)
        axes[1, 1].set_title("How Much Data Gets Reused")
        axes[1, 1].set_xlabel(ordered[0]["param_name"])
        axes[1, 1].set_ylabel("internal / external traffic (x)")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True, alpha=0.3, which="both")
        axes[1, 1].legend(fontsize=8)
    else:
        for column, color in zip(QUEUE_COLUMNS, ["#e45756", "#72b7b2", "#9d755d"]):
            axes[1, 1].plot(x, [_num(row.get(column, "0")) for row in ordered], marker="o", color=color, label=column.replace("queue_max_", ""))
        axes[1, 1].set_title("Key Queue Max Depths")
        axes[1, 1].set_xlabel(ordered[0]["param_name"])
        axes[1, 1].set_ylabel("entries")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    output_path = output_dir / f"{sweep_name}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot sysarr TPU sweep results from results.csv.")
    parser.add_argument("--input", default=None, help="Path to the sweep CSV. Defaults to logs/sysarr_tpu_sweeps/results.csv")
    parser.add_argument("--output-dir", default=None, help="Directory to write the PNG plots")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    input_path = Path(args.input) if args.input else repo_root / "logs" / "sysarr_tpu_sweeps" / "results.csv"
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(input_path)
    if not rows:
        raise SystemExit(f"no rows found in {input_path}")
    rows = [row for row in rows if row.get("status", "ok") == "ok"]
    if not rows:
        raise SystemExit(f"no successful rows found in {input_path}")

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["sweep"], []).append(row)

    for sweep_name, sweep_rows in grouped.items():
        output_path = plot_sweep(sweep_rows, sweep_name, output_dir)
        print(f"wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
